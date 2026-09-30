"""Analysis API endpoints."""

import logging
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session, get_current_user, get_optional_user
from app.config import settings
from app.models import Analysis, User, ExerciseType
from app.schemas import (
    AnalysisCreateResponse,
    AnalysisStatusResponse,
    AnalysisResponse,
    GenerateImageRequest,
)
from app.services.ml_pipeline import MLPipeline
from app.services.storage import get_storage_service
from app.services.auth import create_refresh_token_db, get_user_by_email, get_password_hash

logger = logging.getLogger(__name__)

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

async def read_file_with_size_limit(file: UploadFile, max_size_mb: int) -> bytes:
    """Read file in chunks with size limit to prevent memory exhaustion."""
    max_bytes = max_size_mb * 1024 * 1024
    content = bytearray()
    total_read = 0

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            raise HTTPException(400, f"File too large (> {max_size_mb}MB)")
        content.extend(chunk)

    return bytes(content)


@router.post("/analyze", response_model=AnalysisCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(
    background_tasks: BackgroundTasks,
    trainer_video: UploadFile = File(...),
    user_video: UploadFile = File(...),
    exercise_name: str = Form(...),
    email: str = Form(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Create analysis job with trainer and user videos."""
    # Validate videos
    for video, name in [(trainer_video, "trainer_video"), (user_video, "user_video")]:
        if not video.filename:
            raise HTTPException(400, f"{name}: No filename provided")

        ext = Path(video.filename).suffix.lower()
        if ext not in settings.supported_video_formats_list:
            raise HTTPException(400, f"{name}: Unsupported format. Use: {settings.supported_video_formats_list}")

    # Check file sizes - stream and validate to prevent memory exhaustion
    max_size_mb = settings.MAX_VIDEO_SIZE_MB
    trainer_content = await read_file_with_size_limit(trainer_video, max_size_mb)
    user_content = await read_file_with_size_limit(user_video, max_size_mb)

    # Validate exercise_name against enum and use the canonical value from here on
    try:
        exercise_name = ExerciseType.parse(exercise_name).value
    except ValueError:
        valid_exercises = [e.value for e in ExerciseType]
        raise HTTPException(400, f"Invalid exercise_name. Valid options: {valid_exercises}")

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Save videos to temporary files
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    trainer_temp = temp_dir / f"{task_id}_trainer{Path(trainer_video.filename).suffix}"
    user_temp = temp_dir / f"{task_id}_user{Path(user_video.filename).suffix}"

    trainer_temp.write_bytes(trainer_content)
    user_temp.write_bytes(user_content)

    # Get or create user
    user = current_user
    if not user:
        user = await get_user_by_email(db, email)
        if not user:
            # Create user
            user = User(
                email=email,
                hashed_password=get_password_hash(str(uuid.uuid4())),  # Random password for temp user
                is_verified=True,
            )
            db.add(user)
            await db.flush()

    # Upload to S3
    storage = get_storage_service()
    trainer_key = storage.upload_bytes(trainer_content, f"videos/{task_id}_trainer{Path(trainer_video.filename).suffix}")
    user_key = storage.upload_bytes(user_content, f"videos/{task_id}_user{Path(user_video.filename).suffix}")

    # Create analysis record - store only S3 keys, generate presigned URLs on-demand
    analysis = Analysis(
        task_id=task_id,
        user_id=user.id,
        exercise_name=exercise_name,
        trainer_video_url="",
        user_video_url="",
        trainer_video_key=trainer_key,
        user_video_key=user_key,
        status="pending",
    )
    db.add(analysis)
    await db.commit()

    # Start background processing
    background_tasks.add_task(
        process_analysis_task,
        task_id=task_id,
        trainer_video_path=str(trainer_temp),
        user_video_path=str(user_temp),
        exercise_name=exercise_name,
        user_id=user.id,
    )

    return AnalysisCreateResponse(
        task_id=task_id,
        status="pending",
        message="Analysis job created. Processing started.",
    )


async def process_analysis_task(
    task_id: str,
    trainer_video_path: str,
    user_video_path: str,
    exercise_name: str,
    user_id: str,
):
    """Background task to process analysis."""
    from app.database import get_db_context
    from app.models import Analysis, AnalysisStatus

    async with get_db_context() as db:
        # Get analysis record
        from sqlalchemy import select
        result = await db.execute(select(Analysis).where(Analysis.task_id == task_id))
        analysis = result.scalar_one_or_none()

        if not analysis:
            logger.error(f"Analysis {task_id} not found")
            return

        try:
            analysis.status = AnalysisStatus.PROCESSING
            analysis.started_at = datetime.utcnow()
            await db.flush()

            # Define progress callback (sync - ML pipeline is synchronous)
            def progress_callback(progress: int, stage: str):
                analysis.progress = progress
                analysis.current_stage = stage
                # Note: Progress updates are persisted when db.flush() is called
                # in the main loop or at completion. For real-time updates, use WebSocket/Redis.

            # Run ML pipeline
            pipeline = MLPipeline()
            result_data = pipeline.analyze_with_trainer(
                user_video_path,
                trainer_video_path,
                exercise_name,
                progress_callback,
                task_id=task_id,
            )

            # Save frame analyses
            from app.models import AnalysisFrame
            for frame_data in result_data["analysis"]:
                frame = AnalysisFrame(
                    analysis_id=analysis.id,
                    frame_id=frame_data["frame_id"],
                    error_score=frame_data["error_score"],
                    feedback=frame_data["feedback"],
                    technical_observation=frame_data["technical_observation"],
                    user_image_key=frame_data.get("user_image_key"),
                    trainer_image_key=frame_data.get("trainer_image_key"),
                    joint_angles=frame_data.get("joint_angles"),
                    pose_landmarks=frame_data.get("pose_landmarks"),
                )
                db.add(frame)

            # Update analysis with results
            analysis.status = AnalysisStatus.COMPLETED
            analysis.completed_at = datetime.utcnow()
            analysis.processing_time_seconds = result_data.get("processing_time", 0)
            analysis.reps = result_data["reps"]
            analysis.feedback_summary = result_data["feedback_summary"]
            analysis.technical_details = result_data["technical_details"]
            analysis.frames_data = result_data["analysis"]
            analysis.progress = 100
            analysis.current_stage = "Complete"

            await db.commit()
            logger.info(f"Analysis {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Analysis {task_id} failed: {e}", exc_info=True)
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(e)
            analysis.completed_at = datetime.utcnow()
            await db.commit()
        finally:
            # Clean up temp files
            for path in [trainer_video_path, user_video_path]:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


@router.post("/analyze/no-trainer", response_model=AnalysisCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_analysis_no_trainer(
    background_tasks: BackgroundTasks,
    user_video: UploadFile = File(...),
    exercise_name: str = Form(...),
    email: str = Form(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Create analysis job without trainer video."""
    # Validate video
    if not user_video.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(user_video.filename).suffix.lower()
    if ext not in settings.supported_video_formats_list:
        raise HTTPException(400, f"Unsupported format. Use: {settings.supported_video_formats_list}")

    # Stream and validate size to prevent memory exhaustion
    user_content = await read_file_with_size_limit(user_video, settings.MAX_VIDEO_SIZE_MB)

    # Validate exercise_name against enum and use the canonical value from here on
    try:
        exercise_name = ExerciseType.parse(exercise_name).value
    except ValueError:
        valid_exercises = [e.value for e in ExerciseType]
        raise HTTPException(400, f"Invalid exercise_name. Valid options: {valid_exercises}")

    task_id = str(uuid.uuid4())
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    user_temp = temp_dir / f"{task_id}_user{Path(user_video.filename).suffix}"
    user_temp.write_bytes(user_content)

    # Get or create user
    user = current_user
    if not user:
        user = await get_user_by_email(db, email)
        if not user:
            user = User(
                email=email,
                hashed_password=get_password_hash(str(uuid.uuid4())),
                is_verified=True,
            )
            db.add(user)
            await db.flush()

    # Upload to S3
    storage = get_storage_service()
    user_key = storage.upload_bytes(user_content, f"videos/{task_id}_user{Path(user_video.filename).suffix}")

    # Create analysis record - store only S3 key, generate presigned URL on-demand
    analysis = Analysis(
        task_id=task_id,
        user_id=user.id,
        exercise_name=exercise_name,
        user_video_url="",
        user_video_key=user_key,
        status="pending",
    )
    db.add(analysis)
    await db.commit()

    background_tasks.add_task(
        process_analysis_no_trainer_task,
        task_id=task_id,
        user_video_path=str(user_temp),
        exercise_name=exercise_name,
        user_id=user.id,
    )

    return AnalysisCreateResponse(
        task_id=task_id,
        status="pending",
        message="Analysis job created. Processing started.",
    )


async def process_analysis_no_trainer_task(
    task_id: str,
    user_video_path: str,
    exercise_name: str,
    user_id: str,
):
    """Background task for analysis without trainer."""
    from app.database import get_db_context
    from app.models import Analysis, AnalysisStatus

    async with get_db_context() as db:
        from sqlalchemy import select
        result = await db.execute(select(Analysis).where(Analysis.task_id == task_id))
        analysis = result.scalar_one_or_none()

        if not analysis:
            return

        try:
            analysis.status = AnalysisStatus.PROCESSING
            analysis.started_at = datetime.utcnow()
            await db.flush()

            def progress_callback(progress: int, stage: str):
                analysis.progress = progress
                analysis.current_stage = stage

            pipeline = MLPipeline()
            result_data = pipeline.analyze_without_trainer(
                user_video_path,
                exercise_name,
                progress_callback,
                task_id=task_id,
            )

            from app.models import AnalysisFrame
            for frame_data in result_data["analysis"]:
                frame = AnalysisFrame(
                    analysis_id=analysis.id,
                    frame_id=frame_data["frame_id"],
                    error_score=frame_data["error_score"],
                    feedback=frame_data["feedback"],
                    technical_observation=frame_data["technical_observation"],
                    user_image_key=frame_data.get("user_image_key"),
                    joint_angles=frame_data.get("joint_angles"),
                )
                db.add(frame)

            analysis.status = AnalysisStatus.COMPLETED
            analysis.completed_at = datetime.utcnow()
            analysis.processing_time_seconds = result_data.get("processing_time", 0)
            analysis.reps = result_data["reps"]
            analysis.feedback_summary = result_data["feedback_summary"]
            analysis.technical_details = result_data["technical_details"]
            analysis.frames_data = result_data["analysis"]
            analysis.progress = 100
            analysis.current_stage = "Complete"

            await db.commit()
            logger.info(f"Analysis {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Analysis {task_id} failed: {e}", exc_info=True)
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(e)
            analysis.completed_at = datetime.utcnow()
            await db.commit()
        finally:
            try:
                Path(user_video_path).unlink(missing_ok=True)
            except Exception:
                pass


@router.get("/analyze/{task_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    task_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get analysis job status."""
    from sqlalchemy import select

    result = await db.execute(
        select(Analysis).where(Analysis.task_id == task_id).where(Analysis.user_id == current_user.id)
    )
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(404, "Analysis not found")

    # Collect all S3 keys and generate presigned URLs in one parallel batch
    storage = get_storage_service()

    all_keys: list[str] = []
    if analysis.trainer_video_key:
        all_keys.append(analysis.trainer_video_key)
    if analysis.user_video_key:
        all_keys.append(analysis.user_video_key)

    # Collect frame image keys from stored results
    if analysis.status.value == "completed" and analysis.frames_data:
        for frame in analysis.frames_data:
            row = dict(frame)
            u_key = row.get("user_image_key")
            t_key = row.get("trainer_image_key")
            if u_key:
                all_keys.append(u_key)
            if t_key:
                all_keys.append(t_key)

    # Single parallel batch — O(1) network round-trips instead of O(n)
    presigned = storage.generate_presigned_urls_batch(all_keys) if all_keys else {}

    trainer_video_url = presigned.get(analysis.trainer_video_key, "")
    user_video_url = presigned.get(analysis.user_video_key, "")

    result_data = None
    if analysis.status.value == "completed" and analysis.frames_data:
        # Stream frame images from object storage via short-lived presigned URLs.
        # Legacy rows carry base64 and no keys -> those pass through unchanged.
        frames = []
        for frame in analysis.frames_data:
            row = dict(frame)
            user_key = row.get("user_image_key")
            trainer_key = row.get("trainer_image_key")
            if user_key:
                row["user_image_url"] = presigned.get(user_key, "")
            if trainer_key:
                row["trainer_image_url"] = presigned.get(trainer_key, "")
            frames.append(row)

        result_data = AnalysisResponse(
            analysis=frames,
            reps=analysis.reps,
            feedback_summary=analysis.feedback_summary or "",
            technical_details=analysis.technical_details or [],
        )

    return AnalysisStatusResponse(
        task_id=analysis.task_id,
        status=analysis.status.value,
        progress=analysis.progress,
        current_stage=analysis.current_stage,
        error_message=analysis.error_message,
        result=result_data,
        trainer_video_url=trainer_video_url,
        user_video_url=user_video_url,
    )


@router.get("/analyze/history")
async def get_analysis_history(
    page: int = 1,
    size: int = 20,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get user's analysis history."""
    from sqlalchemy import select, desc
    from sqlalchemy.orm import selectinload

    offset = (page - 1) * size
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == current_user.id)
        .order_by(desc(Analysis.created_at))
        .offset(offset)
        .limit(size)
    )
    analyses = result.scalars().all()

    return [
        {
            "task_id": a.task_id,
            "exercise_name": a.exercise_name.value,
            "status": a.status.value,
            "reps": a.reps,
            "created_at": a.created_at.isoformat(),
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "feedback_summary": a.feedback_summary,
        }
        for a in analyses
    ]


@router.get("/analyze/frames/{s3_key:path}")
async def proxy_frame_image(
    s3_key: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Proxy endpoint for frame images stored in S3.

    Presigned URLs expire after 1 hour. This endpoint streams images through the
    backend so old analyses still display correctly.
    """
    from botocore.exceptions import ClientError
    from sqlalchemy import select

    # Security: verify the user owns an analysis that references this key.
    # The key format is frames/<task_id>/<frame_id>_<side>.jpg
    # We extract the task_id prefix to check ownership.
    parts = s3_key.split("/")
    if len(parts) < 2 or parts[0] != "frames":
        raise HTTPException(400, "Invalid frame key format")

    task_id = parts[1]
    result = await db.execute(
        select(Analysis)
        .where(Analysis.task_id == task_id)
        .where(Analysis.user_id == current_user.id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(404, "Analysis not found or access denied")

    storage = get_storage_service()
    try:
        response = storage.client.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        content_length = response.get("ContentLength", 0)
        if content_length and content_length > 5 * 1024 * 1024:  # 5MB safety limit
            raise HTTPException(413, "Frame image too large")
        body = response["Body"].read()
    except ClientError:
        raise HTTPException(404, "Frame image not found")

    return StreamingResponse(
        iter([body]),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )
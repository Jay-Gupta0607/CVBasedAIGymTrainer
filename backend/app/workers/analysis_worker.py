"""Celery workers for async analysis processing.

Note: These workers use asyncio.run() which is an anti-pattern in Celery.
For production, consider using celery[async] with a proper async worker,
or running the ML pipeline synchronously in a thread pool.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from celery import shared_task

from app.config import settings
from app.database import get_db_context
from app.models import Analysis, AnalysisFrame, AnalysisStatus

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_analysis(
    self,
    task_id: str,
    trainer_video_path: str,
    user_video_path: str,
    exercise_name: str,
    user_id: str,
):
    """Celery task to process analysis with trainer video."""
    from app.services.ml_pipeline import MLPipeline

    async def _process():
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

                # Progress callback - note: async flush won't work in Celery sync task
                # For progress updates in Celery, use a separate mechanism (Redis pub/sub)
                def progress_callback(progress: int, stage: str):
                    analysis.progress = progress
                    analysis.current_stage = stage

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
                raise
            finally:
                # Clean up temp files
                for path in [trainer_video_path, user_video_path]:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass

    # Run async function (anti-pattern in Celery - use proper async worker in production)
    import asyncio
    asyncio.run(_process())


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_analysis_no_trainer(
    self,
    task_id: str,
    user_video_path: str,
    exercise_name: str,
    user_id: str,
):
    """Celery task to process analysis without trainer video."""
    from app.services.ml_pipeline import MLPipeline

    async def _process():
        async with get_db_context() as db:
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
                raise
            finally:
                try:
                    Path(user_video_path).unlink(missing_ok=True)
                except Exception:
                    pass

    import asyncio
    asyncio.run(_process())


@shared_task
def cleanup_temp_files():
    """Periodic task to clean up old temp files."""
    temp_dir = Path(settings.TEMP_DIR)
    if not temp_dir.exists():
        return

    import time
    now = time.time()
    cleaned = 0

    for file_path in temp_dir.iterdir():
        try:
            # Delete files older than 24 hours
            if file_path.is_file() and (now - file_path.stat().st_mtime) > 86400:
                file_path.unlink()
                cleaned += 1
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

    logger.info(f"Cleaned up {cleaned} temp files")
    return {"cleaned_files": cleaned}
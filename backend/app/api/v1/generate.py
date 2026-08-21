"""Generate image API endpoints."""

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from app.api.deps import get_db_session, get_optional_user
from app.config import settings
from app.models import User
from app.schemas import GenerateImageRequest

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


@router.post("/generate", response_class=StreamingResponse)
async def generate_corrected_image(
    image: UploadFile = File(...),
    prompt: str = Form(default="rod"),
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Generate corrected form image using Modal/ComfyUI."""
    # Validate image
    if not image.filename:
        raise HTTPException(400, "No filename provided")

    content = await image.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "Image too large")

    # Check if Modal endpoint is configured
    if not settings.MODAL_IMAGE_GEN_ENDPOINT:
        raise HTTPException(503, "Image generation service not configured")

    try:
        # Call Modal endpoint
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"image": (image.filename, content, image.content_type)}
            data = {"prompt": prompt}

            response = await client.post(
                settings.MODAL_IMAGE_GEN_ENDPOINT,
                files=files,
                data=data,
            )

            if response.status_code != 200:
                logger.error(f"Modal API error: {response.status_code} - {response.text}")
                raise HTTPException(502, "Image generation service error")

            # Return generated image
            return StreamingResponse(
                iter([response.content]),
                media_type="image/jpeg",
                headers={"Content-Disposition": 'attachment; filename="corrected_form.jpg"'},
            )

    except httpx.TimeoutException:
        raise HTTPException(504, "Image generation timed out")
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(500, f"Image generation failed: {str(e)}")


@router.post("/generate/pose-transfer", response_class=StreamingResponse)
async def generate_pose_transfer(
    user_image: UploadFile = File(...),
    trainer_image: UploadFile = File(...),
    exercise_name: str = Form(...),
    frame_id: int = Form(default=0),
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Generate pose-transfer image using Flux + ControlNet + InstantID + IP-Adapter.

    Takes a user frame and trainer reference image, extracts poses using DWPose,
    aligns directions, and generates the user's body in the trainer's pose.
    """
    # Validate images
    for img, name in [(user_image, "user_image"), (trainer_image, "trainer_image")]:
        if not img.filename:
            raise HTTPException(400, f"{name}: No filename provided")

        content = await img.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(400, f"{name}: File too large (> 10MB)")

        # Reset file pointer
        await img.seek(0)

    # Check if Modal endpoint is configured
    if not settings.MODAL_POSE_TRANSFER_ENDPOINT:
        raise HTTPException(503, "Pose transfer service not configured")

    try:
        user_content = await user_image.read()
        trainer_content = await trainer_image.read()

        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {
                "user_image": ("user.jpg", user_content, "image/jpeg"),
                "trainer_image": ("trainer.jpg", trainer_content, "image/jpeg"),
            }
            data = {
                "prompt": f"professional {exercise_name.lower()} form, perfect technique, gym lighting, high quality fitness photography",
            }

            response = await client.post(
                f"{settings.MODAL_POSE_TRANSFER_ENDPOINT}/pose-transfer",
                files=files,
                data=data,
            )

            if response.status_code != 200:
                logger.error(f"Modal pose-transfer error: {response.status_code} - {response.text}")
                raise HTTPException(502, f"Pose transfer generation failed: {response.text}")

            return StreamingResponse(
                iter([response.content]),
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="pose_transfer_frame_{frame_id}.jpg"',
                    "X-Frame-ID": str(frame_id),
                    "X-Exercise": exercise_name,
                },
            )

    except httpx.TimeoutException:
        raise HTTPException(504, "Pose transfer generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pose transfer generation failed: {e}")
        raise HTTPException(500, f"Pose transfer generation failed: {str(e)}")
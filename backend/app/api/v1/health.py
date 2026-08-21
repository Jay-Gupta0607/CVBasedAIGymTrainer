"""Health check API endpoints."""

import logging
from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.config import settings
from app.schemas import HealthResponse, ReadyResponse
from app.services import get_pose_estimator, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db_session),
):
    """Liveness probe - basic health check."""
    checks = {
        "database": False,
        "redis": False,
        "ml_models": False,
        "storage": False,
    }

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    # Check Redis
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        checks["redis"] = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    # Check ML models
    try:
        pose_estimator = get_pose_estimator()
        checks["ml_models"] = pose_estimator.is_healthy()
    except Exception as e:
        logger.warning(f"ML models health check failed: {e}")

    # Check storage
    try:
        storage = get_storage_service()
        storage.client.list_buckets()
        checks["storage"] = True
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.APP_VERSION,
        ml_models_loaded=checks["ml_models"],
        database_connected=checks["database"],
        redis_connected=checks["redis"],
        s3_connected=checks["storage"],
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check(
    db: AsyncSession = Depends(get_db_session),
):
    """Readiness probe - detailed readiness check."""
    checks = {}
    ready = True

    # Database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"failed: {e}"
        ready = False

    # Redis
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"failed: {e}"
        ready = False

    # ML Models
    try:
        pose_estimator = get_pose_estimator()
        checks["ml_models"] = "ok" if pose_estimator.is_healthy() else "not initialized"
    except Exception as e:
        checks["ml_models"] = f"failed: {e}"
        ready = False

    # Storage
    try:
        storage = get_storage_service()
        storage.client.list_buckets()
        checks["storage"] = "ok"
    except Exception as e:
        checks["storage"] = f"failed: {e}"
        ready = False

    # Modal endpoint
    if settings.MODAL_IMAGE_GEN_ENDPOINT:
        checks["modal_endpoint"] = "configured"
    else:
        checks["modal_endpoint"] = "not configured"

    return ReadyResponse(ready=ready, checks=checks)
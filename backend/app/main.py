"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import structlog

from app.api.deps import get_db_session
from app.api.v1 import health, analysis, auth, chat, generate
from app.config import settings
from app.database import close_db, init_db
from app.services.storage import get_storage_service, init_storage


# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(level=settings.LOG_LEVEL)
logger = structlog.get_logger()


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up CVBasedAIGymTrainer backend...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize storage (S3/MinIO)
    init_storage()
    logger.info("Storage initialized")

    # Initialize ML models (lazy - on first request)
    logger.info("ML models will be initialized on first request")

    logger.info("Startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="CV-Based AI Gym Trainer - Biomechanical exercise form analysis API",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # Include routers
    app.include_router(health.router, prefix=settings.API_PREFIX, tags=["health"])
    app.include_router(auth.router, prefix=settings.API_PREFIX, tags=["auth"])
    app.include_router(analysis.router, prefix=settings.API_PREFIX, tags=["analysis"])
    app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["chat"])
    app.include_router(generate.router, prefix=settings.API_PREFIX, tags=["generate"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception", error=str(exc), path=request.url.path, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
    )
"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationInfo


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "CVBasedAIGymTrainer"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # API
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Allowed CORS origins (comma-separated, no wildcards)",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS string to list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://trainer:password@localhost:5432/gym_trainer",
        description="Async PostgreSQL connection string",
    )
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for caching and Celery",
    )
    REDIS_MAX_CONNECTIONS: int = 20

    # S3/MinIO Storage
    S3_ENDPOINT: str = Field(default="http://localhost:9000", description="S3 endpoint URL")
    S3_ACCESS_KEY: str = Field(default="minioadmin", description="S3 access key")
    S3_SECRET_KEY: str = Field(default="minioadmin", description="S3 secret key")
    S3_BUCKET: str = Field(default="gym-trainer", description="S3 bucket name")
    S3_REGION: str = Field(default="us-east-1", description="S3 region")
    S3_USE_SSL: bool = False

    # JWT Authentication
    JWT_SECRET: str = Field(
        default="your-super-secret-jwt-key-change-in-production-min-32-chars",
        description="JWT signing secret (min 32 chars)",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ML Models
    MODELS_DIR: str = Field(default="/app/models", description="Directory containing ONNX models")
    POSE_MODEL_PATH: str = Field(default="/app/models/pose_landmarker.onnx", description="MediaPipe pose ONNX model")
    POSE_MODEL_TRT_PATH: Optional[str] = Field(default=None, description="TensorRT engine path (optional)")
    SCORER_MODEL_PATH: str = Field(default="/app/models/form_scorer.onnx", description="LightGBM scorer ONNX model")
    ONNX_PROVIDERS: str = Field(
        default="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
        description="ONNX Runtime execution providers in priority order (comma-separated)",
    )

    @property
    def onnx_providers_list(self) -> List[str]:
        """Parse ONNX_PROVIDERS string to list."""
        return [p.strip() for p in self.ONNX_PROVIDERS.split(",")]

    # DWPose (for pose transfer)
    DWPOSE_MODEL_PATH: str = Field(default="/app/models/dwpose.onnx", description="DWPose ONNX model")
    DWPOSE_ONNX_PROVIDERS: str = Field(
        default="CUDAExecutionProvider,CPUExecutionProvider",
        description="ONNX Runtime execution providers for DWPose (comma-separated)",
    )

    @property
    def dwpose_onnx_providers_list(self) -> List[str]:
        """Parse DWPOSE_ONNX_PROVIDERS string to list."""
        return [p.strip() for p in self.DWPOSE_ONNX_PROVIDERS.split(",")]

    # MediaPipe Pose Config
    POSE_MIN_DETECTION_CONFIDENCE: float = 0.5
    POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
    POSE_MODEL_COMPLEXITY: int = 1  # 0=lite, 1=full, 2=heavy

    # Video Processing
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_VIDEO_DURATION_SECONDS: int = 300
    SUPPORTED_VIDEO_FORMATS: str = Field(
        default=".mp4,.mov,.avi,.mkv",
        description="Allowed video file extensions (comma-separated)",
    )

    @property
    def supported_video_formats_list(self) -> List[str]:
        """Parse SUPPORTED_VIDEO_FORMATS string to list."""
        return [ext.strip() for ext in self.SUPPORTED_VIDEO_FORMATS.split(",")]
    FRAME_EXTRACTION_FPS: int = 30
    TEMP_DIR: str = "/tmp/gym_trainer"

    # Celery
    CELERY_BROKER_URL: Optional[str] = None  # Defaults to REDIS_URL
    CELERY_RESULT_BACKEND: Optional[str] = None  # Defaults to REDIS_URL
    CELERY_TASK_ACKS_LATE: bool = True
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    CELERY_TASK_TIME_LIMIT: int = 600  # 10 minutes max per task
    CELERY_TASK_SOFT_TIME_LIMIT: int = 540

    # External Services
    MODAL_API_KEY: Optional[str] = Field(default=None, description="Modal API key for FLUX image generation")
    MODAL_IMAGE_GEN_ENDPOINT: Optional[str] = Field(default=None, description="Modal webhook URL for image generation")
    MODAL_POSE_TRANSFER_ENDPOINT: Optional[str] = Field(default=None, description="Modal webhook URL for pose transfer generation")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console

    # Health Checks
    HEALTH_CHECK_INTERVAL: int = 30

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v: str, info: ValidationInfo) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")

        default_secret = "your-super-secret-jwt-key-change-in-production-min-32-chars"
        if v == default_secret:
            env = info.data.get("ENVIRONMENT", "development")
            if env == "production":
                raise ValueError("JWT_SECRET must be changed in production (cannot use default)")
            import warnings
            warnings.warn(
                "Using default JWT_SECRET - set JWT_SECRET environment variable for production!",
                UserWarning,
                stacklevel=2
            )
        return v

    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
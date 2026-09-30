"""API schemas matching frontend expectations."""

from datetime import datetime
from typing import List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, ConfigDict


# Analysis Frame (matches frontend AnalysisFrame interface)
class AnalysisFrame(BaseModel):
    """Single frame analysis result."""
    frame_id: int
    error_score: int = Field(ge=0, le=100)
    feedback: str
    technical_observation: str
    user_image: Optional[str] = ""  # legacy base64 (pre-S3 analyses)
    trainer_image: Optional[str] = ""  # legacy base64 (pre-S3 analyses)
    user_image_url: Optional[str] = None  # presigned MinIO URL (current analyses)
    trainer_image_url: Optional[str] = None  # presigned MinIO URL (current analyses)


# Analysis Response (matches frontend AnalysisResponse interface)
class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    analysis: List[AnalysisFrame]
    reps: int
    feedback_summary: str
    technical_details: List[dict]  # {title: str, description: str}


# Request/Response schemas for API endpoints

class AnalysisCreateRequest(BaseModel):
    """Request to create analysis job."""
    exercise_name: str
    email: EmailStr


class AnalysisCreateResponse(BaseModel):
    """Response after creating analysis job."""
    task_id: str
    status: str
    message: str


class AnalysisStatusResponse(BaseModel):
    """Analysis job status response."""
    task_id: str
    status: str
    progress: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[AnalysisResponse] = None
    trainer_video_url: Optional[str] = None
    user_video_url: Optional[str] = None


class GenerateImageRequest(BaseModel):
    """Request for image generation."""
    prompt: str = Field(default="rod", description="Prompt for form correction")


class UserLoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str = Field(min_length=8)


class UserSignupRequest(BaseModel):
    """User signup request."""
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    """User profile response."""
    id: UUID
    email: EmailStr
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class ChatMessageRequest(BaseModel):
    """Chat message request."""
    message: str


class ChatMessageResponse(BaseModel):
    """Chat message response."""
    response: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    ml_models_loaded: bool
    database_connected: bool
    redis_connected: bool
    s3_connected: bool


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    checks: dict


# WebSocket message types
class WSProgressMessage(BaseModel):
    """WebSocket progress message."""
    type: str = "progress"
    task_id: str
    stage: str
    progress: int
    message: str


class WSCompleteMessage(BaseModel):
    """WebSocket completion message."""
    type: str = "complete"
    task_id: str
    result: AnalysisResponse


class WSErrorMessage(BaseModel):
    """WebSocket error message."""
    type: str = "error"
    task_id: str
    error: str
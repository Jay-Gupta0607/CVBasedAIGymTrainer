"""Database models."""

import enum
import re
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Enum,
    Index,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class UserRole(str, enum.Enum):
    """User role enumeration."""
    USER = "user"
    ADMIN = "admin"
    TRAINER = "trainer"


class AnalysisStatus(str, enum.Enum):
    """Analysis job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExerciseType(str, enum.Enum):
    """Supported exercise types."""
    BARBELL_BICEPS_CURL = "Barbell Biceps Curl"
    BENCH_PRESS = "Bench Press"
    CHEST_FLY_MACHINE = "Chest Fly Machine"
    DEADLIFT = "Deadlift"
    DECLINE_BENCH_PRESS = "Decline Bench Press"
    HAMMER_CURL = "Hammer Curl"
    HIP_THRUST = "Hip Thrust"
    INCLINE_BENCH_PRESS = "Incline Bench Press"
    LAT_PULLDOWN = "Lat Pulldown"
    LATERAL_RAISE = "Lateral Raise"
    LEG_EXTENSION = "Leg Extension"
    LEG_RAISES = "Leg Raises"
    PLANK = "Plank"
    PULL_UP = "Pull Up"
    PUSH_UP = "Push-up"
    ROMANIAN_DEADLIFT = "Romanian Deadlift"
    RUSSIAN_TWIST = "Russian Twist"
    SHOULDER_PRESS = "Shoulder Press"
    SQUAT = "Squat"
    T_BAR_ROW = "T Bar Row"
    TRICEP_DIPS = "Tricep Dips"
    TRICEP_PUSHDOWN = "Tricep Pushdown"

    @classmethod
    def parse(cls, name: str) -> "ExerciseType":
        """Resolve a user-supplied name to a member, ignoring case, spaces, '-' and '_'.

        "squat", "Squat", "push_up" and "Push-up" all resolve; raises ValueError otherwise.
        """
        def key(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", text.lower())

        wanted = key(name)
        for member in cls:
            if wanted in (key(member.value), key(member.name)):
                return member
        raise ValueError(f"Unknown exercise: {name!r}")


class User(Base):
    """User model."""
    __tablename__ = "users"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis", back_populates="user", cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_created_at", "created_at"),
    )


class RefreshToken(Base):
    """Refresh token model for JWT token rotation."""
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    token_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")

    __table_args__ = (
        Index("ix_refresh_tokens_user_id", "user_id"),
        Index("ix_refresh_tokens_expires_at", "expires_at"),
    )


class Analysis(Base):
    """Analysis job model."""
    __tablename__ = "analyses"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)

    # Input
    exercise_name: Mapped[ExerciseType] = mapped_column(Enum(ExerciseType), nullable=False)
    trainer_video_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_video_url: Mapped[str] = mapped_column(Text, nullable=False)
    trainer_video_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    user_video_key: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_stage: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results
    reps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    feedback_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    technical_details: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)
    frames_data: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")

    __table_args__ = (
        Index("ix_analyses_user_id_status", "user_id", "status"),
        Index("ix_analyses_created_at", "created_at"),
    )


class AnalysisFrame(Base):
    """Individual frame analysis results (for detailed queries)."""
    __tablename__ = "analysis_frames"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    error_score: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-100
    feedback: Mapped[str] = mapped_column(Text, nullable=False)
    technical_observation: Mapped[str] = mapped_column(Text, nullable=False)
    user_image_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    trainer_image_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    joint_angles: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pose_landmarks: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_analysis_frames_analysis_id", "analysis_id"),
        Index("ix_analysis_frames_frame_id", "analysis_id", "frame_id"),
    )
"""Services package exports."""

from app.services.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    create_refresh_token_db,
    decode_token,
    get_user_by_email,
    get_user_by_id,
    validate_refresh_token,
    revoke_refresh_token,
    revoke_all_user_tokens,
)
from app.services.pose_estimator import PoseEstimator, get_pose_estimator
from app.services.dtw_aligner import DTWAligner, compute_joint_angles, compute_angle_differences
from app.services.form_scorer import FormScorer, RuleBasedScorer, create_scorer
from app.services.video_processor import VideoProcessor, get_video_processor
from app.services.storage import StorageService, get_storage_service
from app.services.ml_pipeline import MLPipeline, run_full_analysis
from app.services.chat import ChatService, get_chat_service

__all__ = [
    # Auth
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "create_refresh_token_db",
    "decode_token",
    "get_user_by_email",
    "get_user_by_id",
    "validate_refresh_token",
    "revoke_refresh_token",
    "revoke_all_user_tokens",
    # ML
    "PoseEstimator",
    "get_pose_estimator",
    "DTWAligner",
    "compute_joint_angles",
    "compute_angle_differences",
    "FormScorer",
    "RuleBasedScorer",
    "create_scorer",
    "VideoProcessor",
    "get_video_processor",
    "StorageService",
    "get_storage_service",
    "MLPipeline",
    "run_full_analysis",
    # Chat
    "ChatService",
    "get_chat_service",
]
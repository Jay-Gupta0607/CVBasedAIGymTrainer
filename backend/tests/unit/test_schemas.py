"""Unit tests for schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.analysis import (
    AnalysisFrame,
    AnalysisResponse,
    AnalysisCreateResponse,
    AnalysisStatusResponse,
)
from app.schemas.auth import UserLoginRequest, UserSignupRequest, TokenResponse
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage


class TestAnalysisSchemas:
    """Tests for analysis schemas."""

    def test_analysis_frame_valid(self):
        """Test valid AnalysisFrame."""
        frame = AnalysisFrame(
            frame_id=0,
            error_score=15.5,
            feedback="Good form!",
            technical_observation="Knees tracking well",
            user_image_key="s3://bucket/frame.jpg",
            trainer_image_key="s3://bucket/trainer_frame.jpg",
            joint_angles={"left_knee": 95.0, "right_knee": 93.0},
            pose_landmarks=[{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33,
        )
        assert frame.frame_id == 0
        assert frame.error_score == 15.5

    def test_analysis_frame_minimal(self):
        """Test AnalysisFrame with minimal fields."""
        frame = AnalysisFrame(
            frame_id=0,
            error_score=15.5,
            feedback="Good form!",
            technical_observation="Knees tracking well",
        )
        assert frame.user_image_key is None
        assert frame.trainer_image_key is None
        assert frame.joint_angles is None
        assert frame.pose_landmarks is None

    def test_analysis_response_valid(self):
        """Test valid AnalysisResponse."""
        response = AnalysisResponse(
            analysis=[
                AnalysisFrame(
                    frame_id=0,
                    error_score=15.5,
                    feedback="Good form!",
                    technical_observation="Knees tracking well",
                )
            ],
            reps=3,
            feedback_summary="Overall good form",
            technical_details=["Rep 1: Depth 95°", "Rep 2: Depth 92°"],
        )
        assert len(response.analysis) == 1
        assert response.reps == 3

    def test_analysis_create_response(self):
        """Test AnalysisCreateResponse."""
        response = AnalysisCreateResponse(
            task_id="test-task-id",
            status="pending",
            message="Analysis job created",
        )
        assert response.task_id == "test-task-id"
        assert response.status == "pending"

    def test_analysis_status_response(self):
        """Test AnalysisStatusResponse."""
        response = AnalysisStatusResponse(
            task_id="test-task-id",
            status="completed",
            progress=100,
            current_stage="Complete",
            error_message=None,
            result=None,
        )
        assert response.status == "completed"
        assert response.progress == 100


class TestAuthSchemas:
    """Tests for auth schemas."""

    def test_user_login_request_valid(self):
        """Test valid UserLoginRequest."""
        request = UserLoginRequest(email="test@example.com", password="password123")
        assert request.email == "test@example.com"
        assert request.password == "password123"

    def test_user_login_request_invalid_email(self):
        """Test UserLoginRequest with invalid email."""
        with pytest.raises(ValidationError):
            UserLoginRequest(email="invalid-email", password="password123")

    def test_user_signup_request_valid(self):
        """Test valid UserSignupRequest."""
        request = UserSignupRequest(
            email="test@example.com",
            password="Password123!",
            full_name="Test User"
        )
        assert request.email == "test@example.com"
        assert request.password == "Password123!"
        assert request.full_name == "Test User"

    def test_user_signup_request_short_password(self):
        """Test UserSignupRequest with short password."""
        with pytest.raises(ValidationError):
            UserSignupRequest(
                email="test@example.com",
                password="Short1",
                full_name="Test User"
            )

    def test_token_response(self):
        """Test TokenResponse."""
        response = TokenResponse(
            access_token="access_token",
            refresh_token="refresh_token",
            user={"id": "123", "email": "test@example.com", "full_name": "Test User", "is_active": True, "is_verified": True, "created_at": "2024-01-01T00:00:00Z"}
        )
        assert response.access_token == "access_token"
        assert response.refresh_token == "refresh_token"


class TestChatSchemas:
    """Tests for chat schemas."""

    def test_chat_request_valid(self):
        """Test valid ChatRequest."""
        request = ChatRequest(message="How do I improve my squat form?")
        assert request.message == "How do I improve my squat form?"

    def test_chat_request_empty_message(self):
        """Test ChatRequest with empty message."""
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_message(self):
        """Test ChatMessage."""
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_chat_response(self):
        """Test ChatResponse."""
        response = ChatResponse(response="Here's how to improve your squat...")
        assert response.response == "Here's how to improve your squat..."
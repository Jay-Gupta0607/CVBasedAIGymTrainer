"""Unit tests for schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.analysis import (
    AnalysisFrame,
    AnalysisResponse,
    AnalysisCreateResponse,
    AnalysisStatusResponse,
)
from app.schemas.auth import UserLoginRequest, UserSignupRequest, TokenResponse, UserResponse
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage


class TestAnalysisSchemas:
    """Tests for analysis schemas."""

    def test_analysis_frame_valid(self):
        """Test valid AnalysisFrame."""
        frame = AnalysisFrame(
            frame_id=0,
            error_score=15,
            feedback="Good form!",
            technical_observation="Knees tracking well",
            user_image_url="http://minio/frames/t/0_user.jpg",
            trainer_image_url="http://minio/frames/t/0_trainer.jpg",
        )
        assert frame.frame_id == 0
        assert frame.error_score == 15
        assert frame.user_image_url.endswith("0_user.jpg")
        assert frame.trainer_image_url.endswith("0_trainer.jpg")

    def test_analysis_frame_minimal(self):
        """Test AnalysisFrame with minimal fields."""
        frame = AnalysisFrame(
            frame_id=0,
            error_score=15,
            feedback="Good form!",
            technical_observation="Knees tracking well",
        )
        assert frame.user_image_url is None
        assert frame.trainer_image_url is None
        # legacy base64 fields default to empty (pre-S3 analyses)
        assert frame.user_image == ""
        assert frame.trainer_image == ""

    def test_analysis_frame_score_bounds(self):
        for bad in (-1, 101):
            with pytest.raises(ValidationError):
                AnalysisFrame(frame_id=0, error_score=bad, feedback="x", technical_observation="y")

    def test_analysis_response_valid(self):
        """Test valid AnalysisResponse."""
        response = AnalysisResponse(
            analysis=[
                AnalysisFrame(
                    frame_id=0,
                    error_score=15,
                    feedback="Good form!",
                    technical_observation="Knees tracking well",
                )
            ],
            reps=3,
            feedback_summary="Overall good form",
            technical_details=[
                {"title": "Left Knee", "description": "Insufficient depth (occurred in 2 rep(s))"},
            ],
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
            user={
                "id": "6f1f8f0e-5b0a-4a43-9f3e-1d2f6f7a9c11",
                "email": "test@example.com",
                "full_name": "Test User",
                "role": "user",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z",
            },
        )
        assert response.access_token == "access_token"
        assert response.refresh_token == "refresh_token"
        assert response.user.email == "test@example.com"

    def test_token_response_user_is_optional(self):
        response = TokenResponse(access_token="a", refresh_token="r")
        assert response.user is None


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
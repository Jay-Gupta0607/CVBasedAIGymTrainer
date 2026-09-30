"""Unit tests for analysis endpoints."""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


class TestAnalysisEndpoints:
    """Tests for analysis endpoints."""

    @pytest.mark.asyncio
    async def test_create_analysis_without_trainer(self, async_client: AsyncClient, mock_storage):
        """Test creating analysis without trainer video."""
        with patch("app.api.v1.analysis.get_storage_service", return_value=mock_storage):
            with patch("app.api.v1.analysis.process_analysis_no_trainer_task") as mock_task:
                # Create a fake video file
                video_content = b"fake video content"

                response = await async_client.post(
                    "/api/v1/analyze/no-trainer",
                    files={"user_video": ("test.mp4", video_content, "video/mp4")},
                    data={"exercise_name": "squat", "email": "test@example.com"}
                )

                assert response.status_code == 202
                data = response.json()
                assert "task_id" in data
                assert data["status"] == "pending"
                assert "message" in data
                mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_analysis_without_trainer_invalid_format(self, async_client: AsyncClient):
        """Test creating analysis with invalid video format."""
        video_content = b"fake video content"

        response = await async_client.post(
            "/api/v1/analyze/no-trainer",
            files={"user_video": ("test.txt", video_content, "text/plain")},
            data={"exercise_name": "squat", "email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_video_format_is_matched_exactly_not_by_substring(self, async_client: AsyncClient):
        """".mp" is a substring of ".mp4,.mov,..." but is not a supported format."""
        response = await async_client.post(
            "/api/v1/analyze/no-trainer",
            files={"user_video": ("test.mp", b"fake", "video/mp4")},
            data={"exercise_name": "Squat", "email": "test@example.com"},
        )
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_exercise_name_rejected(self, async_client: AsyncClient):
        response = await async_client.post(
            "/api/v1/analyze/no-trainer",
            files={"user_video": ("test.mp4", b"fake", "video/mp4")},
            data={"exercise_name": "burpee", "email": "test@example.com"},
        )
        assert response.status_code == 400
        assert "Invalid exercise_name" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_analysis_without_trainer_too_large(self, async_client: AsyncClient):
        """Test creating analysis with video too large."""
        # Create a large fake video (>500MB)
        large_content = b"x" * (501 * 1024 * 1024)

        response = await async_client.post(
            "/api/v1/analyze/no-trainer",
            files={"user_video": ("test.mp4", large_content, "video/mp4")},
            data={"exercise_name": "squat", "email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_analysis_status_not_found(self, async_client: AsyncClient, db_session):
        """Test getting status of non-existent analysis."""
        # First create a user and login to get token
        await async_client.post(
            "/api/v1/auth/signup",
            json={"email": "test@example.com", "password": "TestPass123!", "full_name": "Test User"}
        )
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "TestPass123!"}
        )
        access_token = login_response.json()["access_token"]

        response = await async_client.get(
            "/api/v1/analyze/non-existent-task-id/status",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 404
        assert "Analysis not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_analysis_history(self, async_client: AsyncClient, db_session):
        """Test getting analysis history."""
        # Create user and login
        await async_client.post(
            "/api/v1/auth/signup",
            json={"email": "test@example.com", "password": "TestPass123!", "full_name": "Test User"}
        )
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "TestPass123!"}
        )
        access_token = login_response.json()["access_token"]

        response = await async_client.get(
            "/api/v1/analyze/history",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
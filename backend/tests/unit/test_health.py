"""Unit tests for health endpoints."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test liveness probe."""
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "ml_models_loaded" in data
        assert "database_connected" in data
        assert "redis_connected" in data
        assert "s3_connected" in data

    @pytest.mark.asyncio
    async def test_readiness_check(self, async_client: AsyncClient):
        """Test readiness probe."""
        response = await async_client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)
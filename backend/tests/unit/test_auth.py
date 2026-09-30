"""Unit tests for auth endpoints."""

import pytest
from httpx import AsyncClient


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_signup_success(self, async_client: AsyncClient):
        """Test successful user signup."""
        response = await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == "test@example.com"
        assert data["user"]["full_name"] == "Test User"

    @pytest.mark.asyncio
    async def test_signup_duplicate_email(self, async_client: AsyncClient):
        """Test signup with duplicate email fails."""
        # First signup
        await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
        )

        # Second signup with same email
        response = await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User 2"
            }
        )
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient):
        """Test successful login."""
        # Create user first
        await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
        )

        # Login
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: AsyncClient):
        """Test login with invalid credentials."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPassword123!"
            }
        )
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient):
        """Test token refresh."""
        # Signup and login
        await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
        )
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!"
            }
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["refresh_token"] != refresh_token  # Should be rotated

    @pytest.mark.asyncio
    async def test_get_current_user(self, async_client: AsyncClient):
        """Test getting current user info."""
        # Signup and login
        await async_client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
        )
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!"
            }
        )
        access_token = login_response.json()["access_token"]

        # Get current user
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"


class TestRefreshTokenPersistence:
    """Refresh/logout must commit: they run on a per-request session that is rolled back."""

    @staticmethod
    async def _signup(client: AsyncClient) -> dict:
        response = await client.post(
            "/api/v1/auth/signup",
            json={"email": "rot@example.com", "password": "TestPass123!", "full_name": "Rot Ate"},
        )
        assert response.status_code == 201
        return response.json()

    @pytest.mark.asyncio
    async def test_signup_returns_user(self, isolated_client: AsyncClient):
        data = await self._signup(isolated_client)
        assert data["user"]["email"] == "rot@example.com"

    @pytest.mark.asyncio
    async def test_refresh_rotates_and_persists(self, isolated_client: AsyncClient):
        first = (await self._signup(isolated_client))["refresh_token"]

        rotated = await isolated_client.post("/api/v1/auth/refresh", json={"refresh_token": first})
        assert rotated.status_code == 200
        second = rotated.json()["refresh_token"]
        assert second != first

        # The old token is revoked and the new one is actually stored.
        replay = await isolated_client.post("/api/v1/auth/refresh", json={"refresh_token": first})
        assert replay.status_code == 401
        again = await isolated_client.post("/api/v1/auth/refresh", json={"refresh_token": second})
        assert again.status_code == 200

    @pytest.mark.asyncio
    async def test_logout_revokes_refresh_token(self, isolated_client: AsyncClient):
        tokens = await self._signup(isolated_client)

        out = await isolated_client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert out.status_code == 200

        reuse = await isolated_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        )
        assert reuse.status_code == 401

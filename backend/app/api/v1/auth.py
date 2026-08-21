"""Auth API endpoints."""

import logging
from datetime import datetime
from jwt import ExpiredSignatureError, InvalidTokenError

from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db_session, get_current_user
from app.models import User, RefreshToken
from app.schemas import (
    UserLoginRequest,
    UserSignupRequest,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
)
from app.services.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    create_refresh_token_db,
    decode_token,
    revoke_refresh_token,
    validate_refresh_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    response: Response,
    credentials: UserLoginRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """User login."""
    # Get user by email
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    # Create tokens
    access_token = create_access_token({"sub": user.email, "user_id": user.id})
    refresh_token = await create_refresh_token_db(db, user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    response: Response,
    user_data: UserSignupRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """User signup."""
    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        is_verified=True,  # Skip email verification for demo
    )
    db.add(user)
    await db.flush()

    # Create tokens
    access_token = create_access_token({"sub": user.email, "user_id": user.id})
    refresh_token = await create_refresh_token_db(db, user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Refresh access token."""
    user = await validate_refresh_token(db, request.refresh_token)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Revoke old refresh token
    await revoke_refresh_token(db, request.refresh_token)

    # Create new tokens
    access_token = create_access_token({"sub": user.email, "user_id": user.id})
    refresh_token = await create_refresh_token_db(db, user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/auth/logout")
async def logout(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Logout - revoke refresh token."""
    await revoke_refresh_token(db, request.refresh_token)
    return {"message": "Logged out successfully"}


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get current user info."""
    return UserResponse.model_validate(current_user)
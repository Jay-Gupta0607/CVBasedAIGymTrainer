"""Authentication service - JWT handling, password hashing."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import User, RefreshToken
from app.schemas.auth import TokenData, UserInDB


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


async def create_refresh_token_db(
    db: AsyncSession,
    user_id: UUID,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> str:
    """Create and store a refresh token in database."""
    import secrets

    # Generate a secure random token
    token = secrets.token_urlsafe(32)
    token_hash = get_password_hash(token)

    expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    refresh_token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address,
    )
    db.add(refresh_token)
    await db.flush()
    return token


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return TokenData(**payload)
    except JWTError:
        return None


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[UserInDB]:
    """Get user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[UserInDB]:
    """Get user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def validate_refresh_token(db: AsyncSession, token: str) -> Optional[UserInDB]:
    """Validate refresh token and return user if valid."""
    # Find the token by user_id would be better, but we hash and compare
    # Since we used bcrypt in create_refresh_token_db, we need to check all unrevoked tokens
    result = await db.execute(
        select(RefreshToken)
        .where(RefreshToken.revoked_at.is_(None))
        .where(RefreshToken.expires_at > datetime.utcnow())
    )
    tokens = result.scalars().all()

    # Verify against each stored bcrypt hash
    for refresh_token in tokens:
        if verify_password(token, refresh_token.token_hash):
            # Get associated user
            result = await db.execute(select(User).where(User.id == refresh_token.user_id))
            return result.scalar_one_or_none()

    return None


async def revoke_refresh_token(db: AsyncSession, token: str) -> bool:
    """Revoke a refresh token."""
    # Find and verify the token
    result = await db.execute(
        select(RefreshToken)
        .where(RefreshToken.revoked_at.is_(None))
        .where(RefreshToken.expires_at > datetime.utcnow())
    )
    tokens = result.scalars().all()

    for refresh_token in tokens:
        if verify_password(token, refresh_token.token_hash):
            refresh_token.revoked_at = datetime.utcnow()
            return True
    return False


async def revoke_all_user_tokens(db: AsyncSession, user_id: UUID) -> int:
    """Revoke all refresh tokens for a user."""
    result = await db.execute(
        select(RefreshToken)
        .where(RefreshToken.user_id == user_id)
        .where(RefreshToken.revoked_at.is_(None))
    )
    tokens = result.scalars().all()
    for token in tokens:
        token.revoked_at = datetime.utcnow()
    return len(tokens)
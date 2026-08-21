"""FastAPI dependencies."""

from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory, get_db
from app.services.auth import decode_token, get_user_by_id

from app.models import User


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise credentials_exception

    token = authorization.split(" ")[1]
    token_data = decode_token(token)

    if token_data is None or token_data.type != "access":
        raise credentials_exception

    user = await get_user_by_id(db, token_data.user_id)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_optional_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session),
) -> Optional[User]:
    """Get current user if authenticated, otherwise None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ")[1]
    token_data = decode_token(token)

    if token_data is None or token_data.type != "access":
        return None

    user = await get_user_by_id(db, token_data.user_id)
    if user is None or not user.is_active:
        return None

    return user


def verify_rate_limit():
    """Rate limiting dependency placeholder."""
    # Actual rate limiting handled by slowapi middleware
    pass


class CommonQueryParams:
    """Common query parameters for pagination."""

    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ):
        self.page = max(1, page)
        self.size = min(100, max(1, size))
        self.sort_by = sort_by
        self.sort_order = sort_order.lower()

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        return self.size
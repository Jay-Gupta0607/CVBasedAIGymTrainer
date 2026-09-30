"""Auth schemas."""

from typing import Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    sub: Optional[str] = None
    user_id: Optional[UUID] = None
    email: Optional[EmailStr] = None
    type: str = "access"
    exp: Optional[int] = None


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(min_length=8)


class UserUpdate(BaseModel):
    """User update schema."""
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User schema with DB fields."""
    id: UUID
    hashed_password: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class UserLoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserSignupRequest(BaseModel):
    """User signup request."""
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class UserResponse(UserBase):
    """User response schema."""
    id: UUID
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(Token):
    """Token response with user info."""
    user: Optional[UserResponse] = None

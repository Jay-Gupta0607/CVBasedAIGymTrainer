"""Chat schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request."""
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Chat response."""
    response: str


class ChatMessage(BaseModel):
    """Chat message for history."""
    role: str  # "user" or "assistant"
    content: str
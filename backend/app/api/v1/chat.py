"""Chat API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db_session, get_current_user, get_optional_user
from app.database import get_db_context
from app.models import User
from app.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.chat import get_chat_service
from app.services.auth import decode_token

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Send message to AI coach."""
    chat_service = get_chat_service()

    # Build context from user's recent analyses
    context = {}
    if current_user:
        from sqlalchemy import select, desc
        from app.models import Analysis
        result = await db.execute(
            select(Analysis)
            .where(Analysis.user_id == current_user.id)
            .where(Analysis.status == "completed")
            .order_by(desc(Analysis.created_at))
            .limit(3)
        )
        recent_analyses = result.scalars().all()
        if recent_analyses:
            context["recent_analyses"] = [
                {
                    "exercise": a.exercise_name.value,
                    "reps": a.reps,
                    "summary": a.feedback_summary,
                }
                for a in recent_analyses
            ]

    response = await chat_service.send_message(
        request.message,
        user_id=str(current_user.id) if current_user else None,
        context=context,
    )

    return ChatResponse(response=response)


# WebSocket chat
class ConnectionManager:
    """WebSocket connection manager."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_message(self, client_id: str, message: dict):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket, str(id(websocket)))
    client_id = str(id(websocket))

    chat_service = get_chat_service()
    user_id = None
    context = {}
    authenticated = False

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")

            # Handle first message with authentication token
            if not authenticated and data.get("token"):
                token = data.get("token")
                token_data = decode_token(token)
                if token_data:
                    user_id = str(token_data.user_id)
                    async with get_db_context() as db:
                        result = await db.execute(select(User).where(User.id == token_data.user_id))
                        user = result.scalar_one_or_none()
                        if user:
                            context["user"] = {"email": user.email, "name": user.full_name}
                authenticated = True
                # Acknowledge authentication
                await manager.send_message(client_id, {
                    "type": "auth",
                    "status": "authenticated" if user_id else "unauthorized"
                })
                continue

            if not message:
                continue

            response = await chat_service.send_message(message, user_id, context)

            await manager.send_message(client_id, {
                "type": "message",
                "role": "assistant",
                "content": response,
            })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)
"""Chat service for AI coach."""

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service for AI coach interactions."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def send_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        """Send message to AI coach and get response.

        Args:
            message: User message
            user_id: Optional user ID for personalization
            context: Optional context (exercise, analysis results, etc.)

        Returns:
            AI response string
        """
        # For now, return a placeholder response
        # In production, integrate with LLM (OpenAI, Anthropic, local model, etc.)
        return self._generate_response(message, context)

    def _generate_response(self, message: str, context: Optional[dict] = None) -> str:
        """Generate response (placeholder - integrate with real LLM)."""
        message_lower = message.lower()

        # Simple keyword-based responses for demo
        if any(word in message_lower for word in ["form", "technique", "correct"]):
            return (
                "Good form is crucial for preventing injury and maximizing gains. "
                "Key principles: maintain neutral spine, engage core, control the eccentric phase, "
                "and move through full range of motion. Want me to analyze a specific exercise?"
            )

        if any(word in message_lower for word in ["rep", "repetition", "set"]):
            return (
                "Quality over quantity! Focus on controlled reps with good form. "
                "Typical rep ranges: 8-12 for hypertrophy, 3-6 for strength, 12-20 for endurance. "
                "Rest 60-120s between sets for hypertrophy, 2-5min for strength."
            )

        if any(word in message_lower for word in ["squat", "deadlift", "bench", "press"]):
            return (
                f"Great compound movement! For {message.strip('?')}, "
                "focus on: 1) Setup and bracing, 2) Controlled descent, 3) Explosive but controlled ascent, "
                "4) Lockout without hyperextension. Upload a video for detailed analysis!"
            )

        if any(word in message_lower for word in ["pain", "hurt", "injury", "sore"]):
            return (
                "I'm not a medical professional. If you're experiencing sharp pain, "
                "stop the exercise and consult a healthcare provider. "
                "Normal soreness (DOMS) peaks 24-48h post-workout. "
                "Sharp joint pain during movement = red flag."
            )

        if any(word in message_lower for word in ["warmup", "warm up", "mobility"]):
            return (
                "Good warmup = better performance + lower injury risk. "
                "Try: 5-10min light cardio, dynamic stretches (leg swings, arm circles), "
                "movement-specific warmup with lighter weights. "
                "Save static stretching for after workout."
            )

        # Default response
        return (
            "I'm your AI gym coach! I can help with form analysis, exercise technique, "
            "programming questions, and motivation. "
            "Upload a video for form analysis or ask me anything about your training!"
        )

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
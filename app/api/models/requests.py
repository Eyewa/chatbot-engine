"""
Pydantic models for API requests.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatbotRequest(BaseModel):
    """Request model for chatbot chat endpoint."""

    input: str = Field(..., description="User's message to the chatbot")
    chat_history: List[str] = Field(
        default_factory=list, description="Conversation history"
    )
    summarize: bool = Field(
        default=False, description="If true, response will be shortened if too long"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        alias="conversationId",
        description="ID used to fetch previous messages",
    )


class MonitoringRequest(BaseModel):
    """Request model for monitoring endpoints."""

    enabled: Optional[bool] = Field(
        default=None, description="Enable/disable monitoring"
    )


class ConfigReloadRequest(BaseModel):
    """Request model for configuration reload."""

    force: bool = Field(
        default=False, description="Force reload even if no changes detected"
    )

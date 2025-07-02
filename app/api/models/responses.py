"""
Pydantic models for API responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatbotResponse(BaseModel):
    """Response model for chatbot chat endpoint."""

    conversation_message: Optional[str] = Field(
        default=None, description="Human-readable message contextualizing the output"
    )
    output: Any = Field(
        ..., description="Chatbot's concise reply (can be a dict or a list of dicts)"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Standard success response model."""

    message: str = Field(..., description="Success message")
    status: str = Field(default="success", description="Response status")


class MonitoringStatsResponse(BaseModel):
    """Response model for monitoring statistics."""

    status: str = Field(..., description="Response status")
    monitoring_active: bool = Field(..., description="Whether monitoring is active")
    statistics: Dict[str, Any] = Field(..., description="Token usage statistics")
    timestamp: float = Field(..., description="Response timestamp")


class MonitoringSummaryResponse(BaseModel):
    """Response model for detailed monitoring summary."""

    status: str = Field(..., description="Response status")
    monitoring_active: bool = Field(..., description="Whether monitoring is active")
    summary: Dict[str, Any] = Field(..., description="Detailed summary")
    timestamp: float = Field(..., description="Response timestamp")


class ConfigReloadResponse(BaseModel):
    """Response model for configuration reload."""

    message: str = Field(..., description="Reload message")
    status: str = Field(..., description="Reload status")
    changes_detected: bool = Field(..., description="Whether changes were detected")
    reloaded_files: List[str] = Field(
        default_factory=list, description="List of reloaded files"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: float = Field(..., description="Response timestamp")

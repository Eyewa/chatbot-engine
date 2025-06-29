"""
Admin API routes.
Handles health checks, configuration reload, and administrative endpoints.
"""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status

from ..models.requests import ConfigReloadRequest
from ..models.responses import (
    HealthResponse,
    ConfigReloadResponse,
    SuccessResponse,
    ErrorResponse
)
from ...core.config import get_settings
from agent.reload_config import router as reload_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    try:
        settings = get_settings()
        return HealthResponse(
            status="healthy",
            version=settings.version,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.get("/ping")
async def ping() -> dict:
    """
    Simple ping endpoint to check if the server is running.
    
    Returns:
        Simple status response
    """
    return {"status": "ok", "timestamp": time.time()}


@router.post("/config/reload", response_model=ConfigReloadResponse)
async def reload_configuration(
    request: ConfigReloadRequest
) -> ConfigReloadResponse:
    """
    Reload application configuration.
    
    Args:
        request: Configuration reload request
        
    Returns:
        Reload status response
    """
    try:
        # This would integrate with the existing reload_config router
        # For now, return a placeholder response
        logger.info(f"Configuration reload requested (force={request.force})")
        
        return ConfigReloadResponse(
            message="Configuration reload initiated",
            status="success",
            changes_detected=True,  # Placeholder
            reloaded_files=["config/templates/response_types.yaml"]  # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Configuration reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration reload failed: {str(e)}"
        )


@router.get("/config/status")
async def get_config_status() -> dict:
    """
    Get current configuration status.
    
    Returns:
        Configuration status information
    """
    try:
        settings = get_settings()
        return {
            "status": "success",
            "config": {
                "app_name": settings.app_name,
                "version": settings.version,
                "debug": settings.debug,
                "monitoring_enabled": settings.monitoring.enabled,
                "chat_history_enabled": settings.chat.include_chat_history,
                "llm_model": settings.llm.model,
                "database_configured": bool(settings.database.live_db_uri and settings.database.common_db_uri)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting config status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting configuration status: {str(e)}"
        ) 
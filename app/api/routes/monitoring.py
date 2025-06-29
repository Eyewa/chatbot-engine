"""
Monitoring API routes.
Handles token monitoring and statistics endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from ..models.responses import (
    MonitoringStatsResponse,
    MonitoringSummaryResponse,
    SuccessResponse,
    ErrorResponse
)
from ...services.monitoring_service import MonitoringService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


def get_monitoring_service() -> MonitoringService:
    """Dependency to get monitoring service instance."""
    return MonitoringService()


@router.get("/stats", response_model=MonitoringStatsResponse)
async def get_token_stats(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
) -> MonitoringStatsResponse:
    """
    Get current token usage statistics.
    
    Args:
        monitoring_service: Monitoring service instance
        
    Returns:
        Token usage statistics
    """
    try:
        result = monitoring_service.get_stats()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return MonitoringStatsResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while getting token statistics: {str(e)}"
        )


@router.get("/summary", response_model=MonitoringSummaryResponse)
async def get_token_summary(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
) -> MonitoringSummaryResponse:
    """
    Get detailed token usage summary with breakdown.
    
    Args:
        monitoring_service: Monitoring service instance
        
    Returns:
        Detailed token usage summary
    """
    try:
        result = monitoring_service.get_detailed_summary()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return MonitoringSummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while getting token summary: {str(e)}"
        )


@router.post("/start", response_model=SuccessResponse)
async def start_token_monitoring(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
) -> SuccessResponse:
    """
    Start token monitoring.
    
    Args:
        monitoring_service: Monitoring service instance
        
    Returns:
        Success response
    """
    try:
        result = monitoring_service.start_monitoring()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return SuccessResponse(
            message=result["message"],
            status=result["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while starting monitoring: {str(e)}"
        )


@router.post("/stop", response_model=SuccessResponse)
async def stop_token_monitoring(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
) -> SuccessResponse:
    """
    Stop token monitoring.
    
    Args:
        monitoring_service: Monitoring service instance
        
    Returns:
        Success response
    """
    try:
        result = monitoring_service.stop_monitoring()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return SuccessResponse(
            message=result["message"],
            status=result["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while stopping monitoring: {str(e)}"
        ) 
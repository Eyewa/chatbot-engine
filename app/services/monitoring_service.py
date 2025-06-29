"""
Monitoring service for token usage tracking.
Handles token monitoring, statistics, and dashboard functionality.
"""

import logging
import time
from typing import Dict, Any, Optional

from ..core.config import get_settings

# Import token monitoring with fallback
try:
    from monitor_tokens import start_monitoring, stop_monitoring, get_stats
    TOKEN_MONITORING_AVAILABLE = True
except ImportError:
    TOKEN_MONITORING_AVAILABLE = False
    logging.warning("Token monitoring not available - monitor_tokens module not found")

logger = logging.getLogger(__name__)


class MonitoringService:
    """Service for handling token monitoring and statistics."""
    
    def __init__(self):
        self.settings = get_settings()
        self._monitoring_active = False
    
    @property
    def monitoring_active(self) -> bool:
        """Check if monitoring is currently active."""
        return self._monitoring_active
    
    @property
    def monitoring_available(self) -> bool:
        """Check if monitoring is available."""
        return TOKEN_MONITORING_AVAILABLE
    
    def start_monitoring(self) -> Dict[str, Any]:
        """
        Start token monitoring.
        
        Returns:
            Status response
        """
        if not self.monitoring_available:
            return {
                "error": "Token monitoring not available",
                "status": "unavailable"
            }
        
        if self._monitoring_active:
            return {
                "message": "Monitoring is already active",
                "status": "already_running"
            }
        
        try:
            start_monitoring()
            self._monitoring_active = True
            logger.info("Token monitoring started")
            return {
                "message": "Token monitoring started",
                "status": "started"
            }
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop token monitoring.
        
        Returns:
            Status response
        """
        if not self.monitoring_available:
            return {
                "error": "Token monitoring not available",
                "status": "unavailable"
            }
        
        if not self._monitoring_active:
            return {
                "message": "Monitoring is not active",
                "status": "not_running"
            }
        
        try:
            stop_monitoring()
            self._monitoring_active = False
            logger.info("Token monitoring stopped")
            return {
                "message": "Token monitoring stopped",
                "status": "stopped"
            }
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current token usage statistics.
        
        Returns:
            Statistics response
        """
        if not self.monitoring_available:
            return {
                "error": "Token monitoring not available",
                "status": "unavailable"
            }
        
        try:
            stats = get_stats()
            return {
                "status": "success",
                "monitoring_active": self._monitoring_active,
                "statistics": stats,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting token stats: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_detailed_summary(self) -> Dict[str, Any]:
        """
        Get detailed token usage summary with breakdown.
        
        Returns:
            Detailed summary response
        """
        if not self.monitoring_available:
            return {
                "error": "Token monitoring not available",
                "status": "unavailable"
            }
        
        try:
            from token_tracker import TokenTracker
            tracker = TokenTracker()
            
            # Get basic stats
            stats = get_stats()
            
            # Get detailed breakdown
            breakdown = {}
            for usage in tracker.usage_log:
                call_type = usage.call_type
                if call_type not in breakdown:
                    breakdown[call_type] = {
                        "calls": 0,
                        "total_tokens": 0,
                        "total_cost": 0,
                        "avg_tokens_per_call": 0
                    }
                
                breakdown[call_type]["calls"] += 1
                breakdown[call_type]["total_tokens"] += usage.total_tokens
                breakdown[call_type]["total_cost"] += usage.cost_estimate
            
            # Calculate averages
            for call_type in breakdown:
                calls = breakdown[call_type]["calls"]
                if calls > 0:
                    breakdown[call_type]["avg_tokens_per_call"] = breakdown[call_type]["total_tokens"] / calls
            
            return {
                "status": "success",
                "monitoring_active": self._monitoring_active,
                "summary": {
                    "statistics": stats,
                    "breakdown": breakdown,
                    "recent_calls": [
                        {
                            "timestamp": usage.timestamp,
                            "call_type": usage.call_type,
                            "function_name": usage.function_name,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                            "cost": usage.cost_estimate,
                            "model": usage.model
                        }
                        for usage in tracker.usage_log[-10:]  # Last 10 calls
                    ]
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting token summary: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """
        Get dashboard configuration.
        
        Returns:
            Dashboard configuration
        """
        return {
            "port": self.settings.monitoring.dashboard_port,
            "log_file": self.settings.monitoring.log_file,
            "enabled": self.settings.monitoring.enabled,
            "available": self.monitoring_available,
            "active": self._monitoring_active
        } 
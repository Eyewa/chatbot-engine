"""
API routes package.
Contains all route handlers for the application.
"""

from .chat import router as chat_router
from .monitoring import router as monitoring_router
from .admin import router as admin_router

__all__ = ["chat_router", "monitoring_router", "admin_router"] 
"""
API routes package.
Contains all route handlers for the application.
"""

from .admin import router as admin_router
from .chat import router as chat_router

__all__ = ["chat_router", "admin_router"]

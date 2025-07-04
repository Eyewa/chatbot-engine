"""
API middleware package.
Contains middleware components for the application.
"""

from .error_handler import register_exception_handlers

__all__ = ["register_exception_handlers"]

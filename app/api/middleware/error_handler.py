"""
Error handling middleware for the API.
Provides centralized error handling and logging.
"""

import logging
import traceback
from typing import Any, Dict

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...core.config import get_settings
from ..models.responses import ErrorResponse

logger = logging.getLogger(__name__)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    error_detail = "Validation error"
    if exc.errors():
        error_detail = f"Validation error: {exc.errors()[0]['msg']}"

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail=error_detail, error_code="VALIDATION_ERROR"
        ).dict(),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail), error_code=f"HTTP_{exc.status_code}"
        ).dict(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    settings = get_settings()

    # Log the full error with traceback
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # In production, don't expose internal errors
    if settings.debug:
        error_detail = f"Internal server error: {str(exc)}"
        error_traceback = traceback.format_exc()
    else:
        error_detail = "An internal server error occurred"
        error_traceback = None

    response_content: Dict[str, Any] = {
        "detail": error_detail,
        "error_code": "INTERNAL_SERVER_ERROR",
    }

    if error_traceback and settings.debug:
        response_content["traceback"] = error_traceback

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=response_content
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

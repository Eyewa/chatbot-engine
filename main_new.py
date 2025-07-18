"""
Main application entry point for the Chatbot Engine.
Production-ready, modular FastAPI application.
"""

import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langsmith import trace

from agent.reload_config import router as reload_router
from app.api.middleware.error_handler import register_exception_handlers
from app.api.routes import admin_router, chat_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

# LangSmith integration: set environment variables if not already set
if not os.environ.get("LANGCHAIN_API_KEY"):
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print(
            "[LangSmith] Warning: LANGCHAIN_API_KEY not set. Tracing will not be enabled."
        )
    else:
        os.environ["LANGCHAIN_API_KEY"] = api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "chatbot-engine")

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Chatbot Engine...")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Chatbot Engine...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    # Create FastAPI app with lifespan management
    app = FastAPI(
        title=settings.app_name,
        description="A sophisticated, multi-agent chatbot with dynamic data routing and token monitoring.",
        version=settings.version,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(chat_router)
    app.include_router(admin_router)

    # Include existing reload router
    try:
        app.include_router(reload_router)
        logger.info("✅ Configuration reload router included")
    except ImportError as e:
        logger.warning(f"⚠️ Could not include reload router: {e}")

    # Add LangSmith tracing middleware
    @app.middleware("http")
    async def langsmith_tracing_middleware(request: Request, call_next):
        conversation_id = "unknown"
        if request.url.path == "/chat/" and request.method == "POST":
            try:
                request_body = await request.json()
                conversation_id = request_body.get("conversationId", "unknown")
            except Exception:
                conversation_id = request.headers.get("X-Conversation-Id", "unknown")
        else:
            conversation_id = request.headers.get("X-Conversation-Id", "unknown")
        with trace(
            "api-request",
            project_name=os.environ.get("LANGSMITH_PROJECT", "chatbot-engine"),
            metadata={"conversation_id": conversation_id},
        ):
            if conversation_id == "unknown":
                logger.warning(
                    "[LangSmith] conversation_id not passed in request body or header; using 'unknown'."
                )
            response = await call_next(request)
        return response

    return app


def run_startup_tests():
    """Run startup tests to verify dependencies and connections."""
    logger.info("🧪 Running startup tests...")

    try:
        # Run pytest with basic tests
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info("✅ Startup tests passed")
        else:
            logger.warning("⚠️ Startup tests failed:")
            logger.warning(result.stdout)
            logger.warning(result.stderr)

    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Startup tests timed out")
    except FileNotFoundError:
        logger.warning("⚠️ pytest not found, skipping startup tests")
    except Exception as e:
        logger.warning(f"⚠️ Error running startup tests: {e}")


def main():
    """Main application entry point."""
    settings = get_settings()

    # Run startup tests
    run_startup_tests()

    # Create app
    app = create_app()

    # Log startup information
    logger.info(f"🟢 Starting {settings.app_name} v{settings.version}")
    logger.info(f"📍 Server: {settings.host}:{settings.port}")
    logger.info(f"🔧 Debug mode: {settings.debug}")

    # Import uvicorn here to avoid import issues
    import uvicorn

    # Start server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()

"""
Configuration management for the chatbot engine.
Uses Pydantic for type safety and environment variable support.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    live_db_uri: str = Field(
        default="mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_live",
        description="Live database URI",
    )
    common_db_uri: str = Field(
        default="mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_common",
        description="Common database URI",
    )
    pool_size: int = Field(default=10, description="Database pool size")
    max_overflow: int = Field(default=20, description="Database max overflow")
    pool_timeout: int = Field(default=30, description="Database pool timeout")


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    model: str = Field(default="gpt-4o", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, description="LLM max tokens")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    organization: Optional[str] = Field(default=None, description="OpenAI organization")


class MonitoringSettings(BaseModel):
    """Token monitoring configuration."""

    enabled: bool = Field(default=True, description="Enable token monitoring")
    log_file: str = Field(default="token_usage.log", description="Token log file")
    dashboard_port: int = Field(default=8080, description="Dashboard port")


class ChatSettings(BaseModel):
    """Chat-specific configuration."""

    include_chat_history: bool = Field(
        default=False, description="Include chat history"
    )
    max_history_tokens: int = Field(default=8000, description="Max history tokens")
    conversation_ttl: int = Field(
        default=86400, description="Conversation TTL"
    )  # 24 hours


class Settings(BaseModel):
    """Main application settings."""

    # Application
    app_name: str = Field(default="Chatbot Engine", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Auto reload")

    # CORS
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    cors_credentials: bool = Field(default=True, description="CORS credentials")

    # Database
    database: DatabaseSettings = DatabaseSettings()

    # LLM
    llm: LLMSettings = LLMSettings()

    # Monitoring
    monitoring: MonitoringSettings = MonitoringSettings()

    # Chat
    chat: ChatSettings = ChatSettings()

    # Paths
    config_dir: Path = Field(default=Path("config"), description="Config directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")


def load_settings() -> Settings:
    """Load settings from environment variables."""
    max_tokens_str = os.getenv("LLM_MAX_TOKENS")
    max_tokens = int(max_tokens_str) if max_tokens_str else None

    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        llm=LLMSettings(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
        ),
        database=DatabaseSettings(
            live_db_uri=os.getenv(
                "LIVE_DB_URI",
                "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_live",
            ),
            common_db_uri=os.getenv(
                "COMMON_DB_URI",
                "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_common",
            ),
        ),
        monitoring=MonitoringSettings(
            enabled=os.getenv("TOKEN_MONITORING_ENABLED", "true").lower() == "true",
            log_file=os.getenv("TOKEN_LOG_FILE", "token_usage.log"),
            dashboard_port=int(os.getenv("DASHBOARD_PORT", "8080")),
        ),
        chat=ChatSettings(
            include_chat_history=os.getenv("INCLUDE_CHAT_HISTORY", "false").lower()
            == "true",
            max_history_tokens=int(os.getenv("MAX_HISTORY_TOKENS", "8000")),
            conversation_ttl=int(os.getenv("CONVERSATION_TTL", "86400")),
        ),
    )


# Global settings instance
settings = load_settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def get_database_uri(db_key: str) -> str:
    """Get database URI for the specified database key."""
    if db_key == "live":
        return settings.database.live_db_uri
    elif db_key == "common":
        return settings.database.common_db_uri
    else:
        raise ValueError(f"Unknown database key: {db_key}")


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration as a dictionary."""
    return {
        "model": settings.llm.model,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
        "api_key": settings.llm.api_key,
        "organization": settings.llm.organization,
    }

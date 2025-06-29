"""
Unit tests for configuration management.
"""

import os
import pytest
from unittest.mock import patch

from app.core.config import (
    get_settings,
    get_database_uri,
    get_llm_config,
    DatabaseSettings,
    LLMSettings,
    MonitoringSettings,
    ChatSettings,
    Settings
)


class TestDatabaseSettings:
    """Test database settings configuration."""
    
    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()
        
        assert settings.live_db_uri == "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_live"
        assert settings.common_db_uri == "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_common"
        assert settings.pool_size == 10
        assert settings.max_overflow == 20
        assert settings.pool_timeout == 30
    
    def test_custom_values(self):
        """Test custom database settings."""
        settings = DatabaseSettings(
            live_db_uri="custom_live_uri",
            common_db_uri="custom_common_uri",
            pool_size=5,
            max_overflow=10,
            pool_timeout=15
        )
        
        assert settings.live_db_uri == "custom_live_uri"
        assert settings.common_db_uri == "custom_common_uri"
        assert settings.pool_size == 5
        assert settings.max_overflow == 10
        assert settings.pool_timeout == 15


class TestLLMSettings:
    """Test LLM settings configuration."""
    
    def test_default_values(self):
        """Test default LLM settings."""
        settings = LLMSettings(api_key="test_key")
        
        assert settings.model == "gpt-4o"
        assert settings.temperature == 0.0
        assert settings.max_tokens is None
        assert settings.api_key == "test_key"
        assert settings.organization is None
    
    def test_custom_values(self):
        """Test custom LLM settings."""
        settings = LLMSettings(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000,
            api_key="custom_key",
            organization="custom_org"
        )
        
        assert settings.model == "gpt-3.5-turbo"
        assert settings.temperature == 0.5
        assert settings.max_tokens == 1000
        assert settings.api_key == "custom_key"
        assert settings.organization == "custom_org"


class TestMonitoringSettings:
    """Test monitoring settings configuration."""
    
    def test_default_values(self):
        """Test default monitoring settings."""
        settings = MonitoringSettings()
        
        assert settings.enabled is True
        assert settings.log_file == "token_usage.log"
        assert settings.dashboard_port == 8080
    
    def test_custom_values(self):
        """Test custom monitoring settings."""
        settings = MonitoringSettings(
            enabled=False,
            log_file="custom.log",
            dashboard_port=9000
        )
        
        assert settings.enabled is False
        assert settings.log_file == "custom.log"
        assert settings.dashboard_port == 9000


class TestChatSettings:
    """Test chat settings configuration."""
    
    def test_default_values(self):
        """Test default chat settings."""
        settings = ChatSettings()
        
        assert settings.include_chat_history is False
        assert settings.max_history_tokens == 8000
        assert settings.conversation_ttl == 86400
    
    def test_custom_values(self):
        """Test custom chat settings."""
        settings = ChatSettings(
            include_chat_history=True,
            max_history_tokens=4000,
            conversation_ttl=3600
        )
        
        assert settings.include_chat_history is True
        assert settings.max_history_tokens == 4000
        assert settings.conversation_ttl == 3600


class TestSettings:
    """Test main settings configuration."""
    
    def test_default_values(self):
        """Test default settings."""
        settings = Settings(
            llm=LLMSettings(api_key="test_key")
        )
        
        assert settings.app_name == "Chatbot Engine"
        assert settings.version == "1.0.0"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.reload is False
        assert settings.cors_origins == ["*"]
        assert settings.cors_credentials is True
    
    def test_custom_values(self):
        """Test custom settings."""
        settings = Settings(
            app_name="Custom App",
            version="2.0.0",
            debug=True,
            host="127.0.0.1",
            port=9000,
            reload=True,
            cors_origins=["http://localhost:3000"],
            cors_credentials=False,
            llm=LLMSettings(api_key="test_key")
        )
        
        assert settings.app_name == "Custom App"
        assert settings.version == "2.0.0"
        assert settings.debug is True
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.reload is True
        assert settings.cors_origins == ["http://localhost:3000"]
        assert settings.cors_credentials is False


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    @patch('app.core.config.settings')
    def test_get_settings(self, mock_settings):
        """Test get_settings function."""
        result = get_settings()
        assert result == mock_settings
    
    @patch('app.core.config.settings')
    def test_get_database_uri_live(self, mock_settings):
        """Test get_database_uri for live database."""
        mock_settings.database.live_db_uri = "live_uri"
        mock_settings.database.common_db_uri = "common_uri"
        
        result = get_database_uri("live")
        assert result == "live_uri"
    
    @patch('app.core.config.settings')
    def test_get_database_uri_common(self, mock_settings):
        """Test get_database_uri for common database."""
        mock_settings.database.live_db_uri = "live_uri"
        mock_settings.database.common_db_uri = "common_uri"
        
        result = get_database_uri("common")
        assert result == "common_uri"
    
    def test_get_database_uri_invalid(self):
        """Test get_database_uri with invalid key."""
        with pytest.raises(ValueError, match="Unknown database key: invalid"):
            get_database_uri("invalid")
    
    @patch('app.core.config.settings')
    def test_get_llm_config(self, mock_settings):
        """Test get_llm_config function."""
        mock_settings.llm.model = "gpt-4o"
        mock_settings.llm.temperature = 0.0
        mock_settings.llm.max_tokens = 1000
        mock_settings.llm.api_key = "test_key"
        mock_settings.llm.organization = "test_org"
        
        result = get_llm_config()
        
        expected = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 1000,
            "api_key": "test_key",
            "organization": "test_org",
        }
        
        assert result == expected

def test_main_new_entrypoint_importable():
    """Test that main_new.py can be imported and exposes a FastAPI app."""
    import importlib.util
    import sys
    import os
    
    main_new_path = os.path.join(os.path.dirname(__file__), '../../main_new.py')
    spec = importlib.util.spec_from_file_location("main_new", main_new_path)
    assert spec is not None
    assert spec.loader is not None
    main_new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_new)
    
    # The app should be created by create_app()
    app = main_new.create_app()
    assert app is not None
    assert hasattr(app, 'openapi') 
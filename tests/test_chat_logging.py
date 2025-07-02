"""
Test chat logging system
Senior AI Engineer Design: Comprehensive testing of logging functionality
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from app.services.chat_logger import ChatLogger, LLMCall, LLMCallTracker


class TestChatLogger:
    """Test the chat logger functionality."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        with patch('app.services.chat_logger.create_engine') as mock_create:
            mock_engine = Mock()
            # Add context manager support
            mock_engine.begin.return_value.__enter__ = Mock()
            mock_engine.begin.return_value.__exit__ = Mock()
            mock_create.return_value = mock_engine
            yield mock_engine
    
    @pytest.fixture
    def logger(self, mock_engine):
        """Create logger instance with mocked engine."""
        with patch.dict('os.environ', {'SQL_DATABASE_URI_LIVE_WRITE': 'mock://test'}):
            return ChatLogger()
    
    def test_init_creates_tables(self, logger, mock_engine):
        """Test that logger initialization creates tables."""
        # Verify that begin() was called (for table creation)
        mock_engine.begin.assert_called()
    
    def test_start_conversation_message(self, logger, mock_engine):
        """Test starting conversation message tracking."""
        conversation_id = "test-conv-id"
        user_id = "test-user-id"
        message_text = "Hello, world!"
        message_id = logger.start_conversation_message(conversation_id, user_id, message_text)
        assert isinstance(message_id, str)
    
    def test_complete_conversation_message(self, logger, mock_engine):
        """Test completing conversation message."""
        message_id = str(uuid.uuid4())
        conversation_id = "test-conv-id"
        user_id = "test-user-id"
        message_text = "Hi, this is the bot."
        intent = "greeting"
        context = {"foo": "bar"}
        logger.complete_conversation_message(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            message_text=message_text,
            intent=intent,
            context=context,
            sender='bot'
        )
        # No assertion needed, just ensure no error


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
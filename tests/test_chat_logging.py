"""
Test chat logging system
Senior AI Engineer Design: Comprehensive testing of logging functionality
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from agent.chat_logger import ChatLogger, LLMCall, LLMCallTracker
from agent.token_tracker import TokenTracker, TokenUsage


class TestChatLogger:
    """Test the chat logger functionality."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        with patch('agent.chat_logger.create_engine') as mock_create:
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
        message_id = "test-message-id"
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


class TestTokenTracker:
    """Test the token tracker functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create token tracker instance."""
        return TokenTracker()
    
    def test_count_tokens_string(self, tracker):
        """Test counting tokens in string."""
        text = "Hello world, this is a test."
        tokens = tracker.count_tokens(text, "gpt-4o")
        
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_count_tokens_list(self, tracker):
        """Test counting tokens in list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        tokens = tracker.count_tokens(messages, "gpt-4o")
        
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_estimate_cost(self, tracker):
        """Test cost estimation."""
        input_tokens = 100
        output_tokens = 50
        cost = tracker.estimate_cost(input_tokens, output_tokens, "gpt-4o")
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_track_llm_call(self, tracker):
        """Test tracking complete LLM call."""
        input_text = "Test input"
        output_text = "Test output"
        
        usage = tracker.track_llm_call(
            input_text=input_text,
            output_text=output_text,
            model="gpt-4o",
            function_name="test_function"
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens > 0
        assert usage.cost_estimate > 0
        assert usage.model == "gpt-4o"
    
    def test_get_model_info(self, tracker):
        """Test getting model information."""
        info = tracker.get_model_info("gpt-4o")
        
        assert isinstance(info, dict)
        assert "model" in info
        assert "input_cost_per_1k" in info
        assert "output_cost_per_1k" in info
        assert "encoder_available" in info


class TestIntegration:
    """Integration tests for the chat logging system."""
    
    @pytest.fixture
    def mock_chat_logger(self):
        """Mock chat logger for integration tests."""
        with patch('agent.chat_logger.get_chat_logger') as mock_get:
            mock_logger = Mock()
            mock_get.return_value = mock_logger
            yield mock_logger
    
    @pytest.fixture
    def mock_token_tracker(self):
        """Mock token tracker for integration tests."""
        with patch('agent.token_tracker.get_token_tracker') as mock_get:
            mock_tracker = Mock()
            mock_get.return_value = mock_tracker
            yield mock_tracker
    
    def test_chat_service_integration(self, mock_chat_logger, mock_token_tracker):
        """Test integration between chat service and logging."""
        from app.services.chat_service import ChatService
        
        # Mock the agent and LLM
        with patch('app.services.chat_service.build_chatbot_agent') as mock_build_agent, \
             patch('app.services.chat_service.ChatOpenAI') as mock_chat_openai:
            
            mock_agent = Mock()
            mock_agent.invoke.return_value = {"type": "text_response", "message": "Test response"}
            mock_build_agent.return_value = mock_agent
            
            mock_llm = Mock()
            mock_llm.invoke.return_value.content = "Test conversation message"
            mock_chat_openai.return_value = mock_llm
            
            # Create service
            service = ChatService()
            
            # Mock the message ID generation
            mock_chat_logger.start_conversation_message.return_value = "test-message-id"
            
            # Test chat method
            import asyncio
            result = asyncio.run(service.chat("Test user input", "test-conversation-id"))
            
            # Verify logging was called
            mock_chat_logger.start_conversation_message.assert_called_once()
            mock_chat_logger.complete_conversation_message.assert_called_once()
            
            # Verify result structure
            assert "conversation_message" in result
            assert "output" in result
            assert "conversation_id" in result
            assert "message_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
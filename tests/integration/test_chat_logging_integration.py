import uuid
from unittest.mock import Mock, patch

import pytest


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the chat logging system."""

    @pytest.fixture
    def mock_chat_logger(self):
        """Mock chat logger for integration tests."""
        with patch("app.services.chat_service.get_chat_logger") as mock_get:
            mock_logger = Mock()
            mock_get.return_value = mock_logger
            yield mock_logger

    def test_chat_service_integration(self, mock_chat_logger):
        """Test integration between chat service and logging."""
        from app.services.chat_service import ChatService

        # Mock the agent and LLM
        with patch(
            "app.services.chat_service.build_chatbot_agent"
        ) as mock_build_agent, patch(
            "app.services.chat_service.ChatOpenAI"
        ) as mock_chat_openai:

            mock_agent = Mock()
            mock_agent.invoke.return_value = {
                "type": "text_response",
                "message": "Test response",
            }
            mock_build_agent.return_value = mock_agent

            mock_llm = Mock()
            mock_llm.invoke.return_value.content = "Test conversation message"
            mock_chat_openai.return_value = mock_llm

            # Create service
            service = ChatService()

            # Mock the message ID generation for user and bot
            user_message_id = str(uuid.uuid4())
            mock_chat_logger.start_conversation_message.return_value = user_message_id
            mock_chat_logger.complete_conversation_message.side_effect = (
                lambda **kwargs: None
            )

            # Test chat method
            import asyncio

            result = asyncio.run(
                service.chat("Test user input", "test-conversation-id")
            )

            # Verify logging was called
            mock_chat_logger.start_conversation_message.assert_called_once()
            mock_chat_logger.complete_conversation_message.assert_called_once()

            # Verify result structure
            assert "conversation_message" in result
            assert "output" in result
            assert "conversation_id" in result
            assert "message_id" in result

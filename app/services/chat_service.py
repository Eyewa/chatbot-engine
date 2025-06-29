"""
Chat service for orchestrating chatbot functionality.
Handles agent invocation, response processing, and history management.
"""

import logging
import os
import yaml
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from agent.agent import build_chatbot_agent
from agent.chat_history_repository import ChatHistoryRepository
from agent.utils import generate_llm_message
from ..core.config import get_settings
from ..utils.response_formatter import (
    enforce_response_schema,
    parse_agent_output,
    RESPONSE_TYPES
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._agent = None
        self._llm = None
        self._history_repo = ChatHistoryRepository()
    
    @property
    def agent(self):
        """Get or create the chatbot agent."""
        if self._agent is None:
            try:
                self._agent = build_chatbot_agent()
                logger.info("Chatbot agent initialized successfully")
            except Exception as e:
                logger.error(f"Error creating agent: {e}")
                raise
        return self._agent
    
    @property
    def llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            llm_config = get_settings().llm
            self._llm = ChatOpenAI(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                api_key=llm_config.api_key,
                organization=llm_config.organization,
            )
        return self._llm
    
    def get_chat_history(self, conversation_id: Optional[str]) -> List[str]:
        """
        Get chat history for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chat history messages
        """
        if not conversation_id or not self.settings.chat.include_chat_history:
            logger.debug("Chat history is disabled by config.")
            return []
        
        try:
            history = self._history_repo.fetch_history_with_token_limit(
                conversation_id, 
                max_tokens=self.settings.chat.max_history_tokens
            )
            logger.debug(f"Loaded chat history for conversation_id={conversation_id}: {len(history)} messages")
            return history
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []
    
    def save_chat_history(self, conversation_id: str, user_input: str, response: str) -> None:
        """
        Save chat message to history.
        
        Args:
            conversation_id: Conversation identifier
            user_input: User's input message
            response: Bot's response
        """
        if not conversation_id:
            return
        
        try:
            self._history_repo.save_message(conversation_id, user_input, response)
            logger.debug(f"Saved message to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
    
    def process_agent_result(self, agent_result: Any) -> Any:
        """
        Process and format agent result.
        
        Args:
            agent_result: Raw agent result
            
        Returns:
            Processed and formatted result
        """
        logger.debug(f"[CHAT] Raw agent result: {agent_result}")
        
        final_response_data = agent_result
        response_type = None
        
        # Handle different types of agent results
        if isinstance(agent_result, dict):
            # If it's a dict where all values are dicts (summary objects), convert to list
            if all(isinstance(v, dict) for v in agent_result.values()):
                agent_result = list(agent_result.values())
                final_response_data = [
                    enforce_response_schema(item, RESPONSE_TYPES) if isinstance(item, dict) else item 
                    for item in agent_result
                ]
            else:
                response_type = agent_result.get("type")
                final_response_data = enforce_response_schema(agent_result, RESPONSE_TYPES)
        
        elif isinstance(agent_result, list) and agent_result:
            # If it's a list of responses, process all of them
            processed_responses = []
            for result in agent_result:
                if isinstance(result, dict):
                    response_type = result.get("type")
                    processed_result = enforce_response_schema(result, RESPONSE_TYPES)
                    processed_responses.append(processed_result)
                else:
                    processed_responses.append(result)
            
            # If we have multiple responses, return them as a list
            if len(processed_responses) > 1:
                final_response_data = processed_responses
            else:
                # If only one response, return it directly
                final_response_data = processed_responses[0] if processed_responses else agent_result
        
        else:
            # If not a dict or list, it's likely plain text - try to determine the appropriate schema
            # Try to determine what type of data this is based on the content
            if isinstance(agent_result, str):
                # Load response type detection configuration
                config_path = os.path.join("config", "query_routing.yaml")
                try:
                    with open(config_path, 'r') as f:
                        routing_config = yaml.safe_load(f)
                    content_keywords = routing_config.get('response_type_detection', {}).get('content_keywords', {})
                except Exception as e:
                    logger.warning(f"Could not load response type detection config: {e}")
                    content_keywords = {}
                
                # Determine response type based on configuration
                agent_result_lower = agent_result.lower()
                for response_type, keywords in content_keywords.items():
                    if any(keyword in agent_result_lower for keyword in keywords):
                        response_type = routing_config.get('response_type_detection', {}).get('type_mapping', {}).get(response_type, response_type)
                        break
                else:
                    response_type = "text_response"
            
            final_response_data = generate_llm_message(
                agent_result,
                self.llm,
                schema=RESPONSE_TYPES,
                response_type=response_type
            )

        # If after enforcement, still not a dict or missing required fields, try to fix with LLM
        # Only run LLM fallback if not a dict or a list of dicts with 'type'
        if (
            not isinstance(final_response_data, dict)
            and not (
                isinstance(final_response_data, list)
                and all(isinstance(item, dict) and "type" in item for item in final_response_data)
            )
        ):
            final_response_data = generate_llm_message(
                agent_result,
                self.llm,
                schema=RESPONSE_TYPES,
                response_type="text_response"
            )

        logger.debug(f"[CHAT] Enforced response: {final_response_data}")
        return final_response_data
    
    async def chat(
        self, 
        user_input: str, 
        conversation_id: Optional[str] = None,
        chat_history: Optional[List[str]] = None,
        summarize: bool = False
    ) -> Any:
        """
        Process a chat message.
        
        Args:
            user_input: User's input message
            conversation_id: Optional conversation identifier
            chat_history: Optional chat history
            summarize: Whether to summarize the response
            
        Returns:
            Processed chat response
        """
        try:
            # Get chat history if not provided
            if chat_history is None:
                chat_history = self.get_chat_history(conversation_id)
            
            # Prepare input for agent
            input_dict = {
                "input": user_input,
                "chat_history": chat_history,
            }
            
            # Invoke agent
            agent_result = self.agent.invoke(input_dict)
            
            # Process and format the result
            final_response = self.process_agent_result(agent_result)
        except Exception as e:
            logger.error(f"Error in chat service: {e}", exc_info=True)
            return {
                "conversation_message": None,
                "output": str(e)
            }

        # Generate a human-readable message using the LLM
        conversation_message = await self._generate_conversation_message(user_input, final_response)
        return {
            "conversation_message": conversation_message,
            "output": final_response
        }

    async def _generate_conversation_message(self, user_input, final_response):
        """
        Use the LLM to generate a human-readable message summarizing the output.
        """
        try:
            # Compose a prompt for the LLM
            prompt = (
                "Given the following user request and structured data output, "
                "write a short, friendly, human-readable summary message for the user. "
                "Do not repeat the user's question verbatim."
                "\nUser request: " + str(user_input) +
                "\nStructured output: " + str(final_response) +
                "\nMessage:"
            )
            response = self.llm.invoke(prompt)
            
            # Extract only the content from the LLM response, excluding metadata
            if hasattr(response, 'content'):
                # LangChain AIMessage or similar
                return response.content
            elif isinstance(response, dict):
                # Dictionary response
                return response.get("message") or response.get("content") or str(response)
            elif isinstance(response, str):
                # String response
                return response
            else:
                # Fallback: try to get string representation but clean it
                response_str = str(response)
                # Remove common metadata patterns
                if "response_metadata" in response_str:
                    # Try to extract just the content part
                    import re
                    # Look for content field in the response
                    content_match = re.search(r"'content':\s*'([^']*)'", response_str)
                    if content_match:
                        return content_match.group(1)
                    # If no content field, try to get the first part before metadata
                    parts = response_str.split("response_metadata")
                    if parts:
                        return parts[0].strip()
                return response_str
                
        except Exception as e:
            logger.warning(f"Failed to generate conversation message: {e}")
            return None 
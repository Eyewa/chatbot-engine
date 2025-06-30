"""
Chat service for orchestrating chatbot functionality.
Handles agent invocation, response processing, and history management.
Now with enhanced comprehensive logging!
"""

import logging
import os
import yaml
import time
import uuid
from typing import Any, Dict, List, Optional, Union
import inspect

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from agent.agent import build_chatbot_agent
from agent.chat_logger import get_chat_logger
from agent.token_tracker import get_token_tracker
from agent.utils import generate_llm_message
from ..core.config import get_settings
from ..utils.response_formatter import (
    enforce_response_schema,
    parse_agent_output,
    RESPONSE_TYPES
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations with enhanced logging."""
    
    def __init__(self):
        self.settings = get_settings()
        self._agent = None
        self._llm = None
        self._chat_logger = get_chat_logger()
        self._token_tracker = get_token_tracker()
    
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
    
    def _extract_sql_queries(self, debug_info: Optional[Dict[str, Any]]) -> List[str]:
        """Extract SQL queries from debug info"""
        sql_queries = []
        if debug_info:
            # Look for SQL queries in various debug info locations
            for key in ['sql_queries', 'queries', 'sql', 'executed_sql']:
                if key in debug_info:
                    queries = debug_info[key]
                    if isinstance(queries, list):
                        sql_queries.extend(queries)
                    elif isinstance(queries, str):
                        sql_queries.append(queries)
            
            # Also check for SQL in tool calls
            if 'tool_calls' in debug_info:
                for tool_call in debug_info['tool_calls']:
                    if isinstance(tool_call, dict) and 'args' in tool_call:
                        args = tool_call['args']
                        if isinstance(args, dict) and 'query' in args:
                            sql_queries.append(args['query'])
        
        return sql_queries
    
    def _extract_sql_results_count(self, debug_info: Optional[Dict[str, Any]]) -> Optional[int]:
        """Extract SQL results count from debug info"""
        if debug_info:
            # Look for result counts in various locations
            for key in ['sql_results_count', 'results_count', 'row_count', 'count']:
                if key in debug_info:
                    count = debug_info[key]
                    if isinstance(count, int):
                        return count
        return None
    
    def _extract_debug_info(self, agent_result: Any) -> Optional[Dict[str, Any]]:
        """Extract debug information from agent result"""
        debug_info = {}
        
        if isinstance(agent_result, dict):
            # Extract common debug fields
            for key in ['intent', 'classification', 'database_used', 'debug_info', 'metadata']:
                if key in agent_result:
                    debug_info[key] = agent_result[key]
            
            # Extract tool calls if available
            if 'intermediate_steps' in agent_result:
                debug_info['tool_calls'] = agent_result['intermediate_steps']
        
        return debug_info if debug_info else None
    
    def get_chat_history(self, conversation_id: Optional[str]) -> List[str]:
        """
        Get chat history for a conversation.
        Now uses enhanced logging system for better performance and limited history.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chat history messages (limited to 0 messages for now)
        """
        if not conversation_id or not self.settings.chat.include_chat_history:
            logger.debug("Chat history is disabled by config.")
            return []
        
        try:
            # Get 0 recent messages for now (no history)
            history = self._chat_logger.get_conversation_details(conversation_id)
            if history and 'messages' in history:
                messages = history['messages']
                # Take 0 recent messages (no history)
                recent_messages = messages[-0:] if len(messages) > 0 else []
                
                # Convert to flat list
                flat_history = []
                for msg in recent_messages:
                    flat_history.append(msg['user_input'])
                    flat_history.append(msg['final_output'])
                
                logger.debug(f"Loaded recent chat history for conversation_id={conversation_id}: {len(flat_history)} messages")
                return flat_history
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []
    
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
        Process a chat message with enhanced comprehensive logging.
        
        Args:
            user_input: User's input message
            conversation_id: Optional conversation identifier
            chat_history: Optional chat history
            summarize: Whether to summarize the response
            
        Returns:
            Processed chat response with comprehensive logging
        """
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Start tracking this conversation message
        message_id = self._chat_logger.start_conversation_message(conversation_id, user_input)
        
        # Track overall conversation timing
        conversation_start_time = time.time()
        
        try:
            # Get chat history if not provided
            if chat_history is None:
                chat_history = self.get_chat_history(conversation_id)
            
            # Prepare input for agent
            input_dict = {
                "input": user_input,
                "chat_history": chat_history,
            }

            # Prepare metadata for LangSmith
            metadata = {
                "conversation_id": conversation_id,
                "message_id": message_id
            }
            # Optionally add user_id if available in input_dict or context
            user_id = input_dict.get("user_id") or None
            if user_id:
                metadata["user_id"] = user_id

            # Track agent invocation
            agent_start_time = time.time()
            with self._chat_logger.track_llm_call(
                conversation_id=conversation_id,
                message_id=message_id,
                model="gpt-4o",  # The model used by the agent
                function_name="agent_invoke",
                input_text=str(input_dict)
            ) as call_tracker:
                try:
                    invoke_sig = inspect.signature(self.agent.invoke)
                    if 'config' in invoke_sig.parameters:
                        agent_result = self.agent.invoke(input_dict, config={"metadata": metadata})
                    else:
                        agent_result = self.agent.invoke(input_dict)
                    agent_duration_ms = (time.time() - agent_start_time) * 1000
                except TypeError:
                    agent_result = self.agent.invoke(input_dict)
                agent_duration_ms = (time.time() - agent_start_time) * 1000
                
                # Set the output details for tracking
                # Note: We don't have exact token counts from the agent, so we'll estimate
                output_text = str(agent_result)
                estimated_input_tokens = len(str(input_dict).split()) * 1.3  # Rough estimate
                estimated_output_tokens = len(output_text.split()) * 1.3  # Rough estimate
                estimated_cost = (estimated_input_tokens * 0.000005) + (estimated_output_tokens * 0.000015)  # GPT-4o pricing
                
                call_tracker.set_output(
                    output_text=output_text,
                    input_tokens=int(estimated_input_tokens),
                    output_tokens=int(estimated_output_tokens),
                    cost_estimate=estimated_cost,
                    metadata={"agent_duration_ms": agent_duration_ms}
                )
            
            # Process and format the result
            final_response = self.process_agent_result(agent_result)
            
            # Generate conversation message
            conversation_message = await self._generate_conversation_message(user_input, final_response)
            
            # Extract debug info and SQL queries
            debug_info = self._extract_debug_info(agent_result)
            sql_queries = self._extract_sql_queries(debug_info)
            sql_results_count = self._extract_sql_results_count(debug_info)
            
            # Complete conversation message logging
            conversation_duration_ms = (time.time() - conversation_start_time) * 1000
            
            # Get LLM call summary from the logger
            conversation_details = self._chat_logger.get_conversation_details(conversation_id)
            llm_calls = conversation_details.get('llm_calls', [])
            total_llm_calls = len(llm_calls)
            total_tokens_used = sum(call.get('total_tokens', 0) for call in llm_calls)
            total_cost = sum(call.get('cost_estimate', 0.0) for call in llm_calls)
            
            self._chat_logger.complete_conversation_message(
                message_id=message_id,
                final_output=str(final_response),
                conversation_message=conversation_message,
                intent=debug_info.get('intent') if debug_info else None,
                classification=debug_info.get('classification') if debug_info else None,
                database_used=debug_info.get('database_used') if debug_info else None,
                sql_queries=sql_queries,
                sql_results_count=sql_results_count,
                success=True,
                debug_info=debug_info
            )
            
            # Update conversation message with LLM call summary
            with self._chat_logger.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE chatbot_conversation_messages 
                    SET total_llm_calls = :total_llm_calls,
                        total_tokens_used = :total_tokens_used,
                        total_cost = :total_cost,
                        total_duration_ms = :total_duration_ms
                    WHERE message_id = :message_id
                """), {
                    "message_id": message_id,
                    "total_llm_calls": total_llm_calls,
                    "total_tokens_used": total_tokens_used,
                    "total_cost": total_cost,
                    "total_duration_ms": conversation_duration_ms
                })
            
            logger.info(f"Chat completed with enhanced logging - Conversation: {conversation_id}, Duration: {conversation_duration_ms:.2f}ms")
            
            return {
                "conversation_message": conversation_message,
                "output": final_response,
                "conversation_id": conversation_id,
                "message_id": message_id
            }
            
        except Exception as e:
            # Log error
            conversation_duration_ms = (time.time() - conversation_start_time) * 1000
            self._chat_logger.complete_conversation_message(
                message_id=message_id,
                final_output=str(e),
                success=False,
                error_message=str(e),
                debug_info={"error": str(e), "traceback": str(e.__traceback__)}
            )
            
            logger.error(f"Chat failed with enhanced logging - Conversation: {conversation_id}, Error: {e}", exc_info=True)
            return {
                "conversation_message": None,
                "output": str(e),
                "conversation_id": conversation_id,
                "message_id": message_id
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
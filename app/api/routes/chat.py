"""
Chat API routes.
Handles chat endpoints and conversation management.
Now with enhanced comprehensive logging!
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from ..models.requests import ChatbotRequest
from ..models.responses import ChatbotResponse, ErrorResponse
from ...services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chatbot"])


def get_chat_service() -> ChatService:
    """Dependency to get chat service instance."""
    return ChatService()


@router.post(
    "/",
    response_model=ChatbotResponse,
    responses={500: {"model": ErrorResponse}}
)
async def chat_endpoint(
    request: ChatbotRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatbotResponse:
    """
    Process a chat message and return a response.
    Now with enhanced comprehensive logging!
    
    This endpoint now tracks every detail of the conversation:
    - User input and final output
    - All LLM calls with token usage and costs
    - SQL queries executed
    - Performance metrics
    - Error details and debugging info
    - Chat history based on conversation_id
    
    Args:
        request: Chat request containing user input and context
        chat_service: Chat service instance
        
    Returns:
        Chatbot response with comprehensive logging
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.debug(f"[CHAT] Received request: {request.dict()}")
        
        # Process the chat request with enhanced logging
        response_data = await chat_service.chat(
            user_input=request.input,
            conversation_id=request.conversation_id,
            chat_history=request.chat_history,
            summarize=request.summarize
        )
        
        # Extract conversation_message and output
        if isinstance(response_data, dict) and "conversation_message" in response_data and "output" in response_data:
            conversation_message = response_data.get("conversation_message")
            output = response_data.get("output")
        else:
            conversation_message = None
            output = response_data
        
        # Create response object
        response_obj = ChatbotResponse(conversation_message=conversation_message, output=output)
        
        # Note: Chat history is now automatically saved by the enhanced logging system
        # No need to manually save it here anymore
        
        logger.debug(f"[CHAT] Response: {response_data}")
        return response_obj
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        ) 
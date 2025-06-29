"""
Chat API routes.
Handles chat endpoints and conversation management.
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
    
    Args:
        request: Chat request containing user input and context
        chat_service: Chat service instance
        
    Returns:
        Chatbot response
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.debug(f"[CHAT] Received request: {request.dict()}")
        
        # Process the chat request
        response_data = await chat_service.chat(
            user_input=request.input,
            conversation_id=request.conversation_id,
            chat_history=request.chat_history,
            summarize=request.summarize
        )
        
        # Create response object
        response_obj = ChatbotResponse(output=response_data)
        
        # Save chat history if conversation_id is present
        if request.conversation_id:
            chat_service.save_chat_history(
                request.conversation_id,
                request.input,
                response_obj.model_dump_json()
            )
        
        logger.debug(f"[CHAT] Response: {response_data}")
        return response_obj
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        ) 
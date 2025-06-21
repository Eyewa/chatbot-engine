# main.py

import os
import logging
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import build_chatbot_agent

# Define request/response schema
class ChatbotRequest(BaseModel):
    input: str = Field(..., description="User's message to the chatbot")
    chat_history: List[str] = Field(default_factory=list, description="List of previous messages")

class ChatbotResponse(BaseModel):
    output: str = Field(..., description="Chatbot's response")

class ErrorResponse(BaseModel):
    detail: str

# Create FastAPI app
def create_app() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    env = os.getenv("ENV", "local")
    logging.info(f"\U0001F7E2 Starting chatbot API in '{env}' environment")

    app = FastAPI(
        title="Eyewear Chatbot API",
        version="1.0.0",
        description="LangChain-powered FastAPI chatbot service"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = build_chatbot_agent()

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logging.exception("Unhandled Exception occurred")
        return JSONResponse(
            status_code=500,
            content={"detail": "Oops! Something went wrong on the server. We're fixing it!"}
        )

    @app.post("/chat", response_model=ChatbotResponse, responses={500: {"model": ErrorResponse}})
    async def chat_endpoint(request: ChatbotRequest):
        try:
            if agent is None:
                raise RuntimeError("Agent not available. Check DB or initialization errors.")

            greetings = ["hello", "hi", "hey", "yo", "hola"]
            if request.input.lower().strip() in greetings:
                return ChatbotResponse(output="👋 Hi there! I'm Winkly — your assistant for everything eyewear. How can I help today?")

            response = agent.invoke({
                "input": request.input,
                "chat_history": request.chat_history
            })

            output = response.get("output", None)
            if not isinstance(output, str):
                raise ValueError("Response 'output' is not a valid string.")

            return ChatbotResponse(output=output)

        except Exception as e:
            logging.error(f"💥 Error during chat processing: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Winkly encountered an error while processing your request: {str(e)}"}
            )

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=os.getenv("ENV") == "local")

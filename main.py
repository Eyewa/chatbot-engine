import os
import logging
from typing import List
from fastapi import FastAPI
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

# Create FastAPI app
def create_app() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    env = os.getenv("ENV", "local")
    logging.info(f"🟢 Starting chatbot API in '{env}' environment")

    app = FastAPI(
        title="Eyewear Chatbot API",
        version="1.0.0",
        description="LangChain-powered FastAPI chatbot service"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Update for prod if needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = build_chatbot_agent()

    @app.post("/chat", response_model=ChatbotResponse)
    async def chat_endpoint(request: ChatbotRequest):
        response = agent.invoke({
            "input": request.input,
            "chat_history": request.chat_history
        })
        # LangChain agent usually returns a dict with 'output' key
        return ChatbotResponse(output=response.get("output", str(response)))

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=os.getenv("ENV") == "local")

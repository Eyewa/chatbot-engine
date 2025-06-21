# main.py

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

from agent import build_chatbot_agent


# Define input/output manually to avoid schema issues with langchain
class ChatbotRequest(BaseModel):
    input: str = Field(..., description="User's message")
    chat_history: List[object] = Field(default_factory=list, description="Chat history")

class ChatbotResponse(BaseModel):
    output: str = Field(..., description="Chatbot reply")

# Build app
def create_app() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    env = os.getenv("ENV", "local")
    logging.info(f"🟢 Starting in '{env}' environment")

    app = FastAPI(
        title="Eyewear Chatbot API",
        version="1.0.0",
        description="LangChain chatbot API for Eyewa"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent_executor = build_chatbot_agent()

    @app.post("/chatbot", response_model=ChatbotResponse)
    async def chatbot_endpoint(payload: ChatbotRequest):
        result = await agent_executor.ainvoke(payload.dict())
        return {"output": result}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=(os.getenv("ENV") == "local"))

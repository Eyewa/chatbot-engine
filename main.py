import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langserve import add_routes
from langserve.validation import ChatbotBatchRequest
from pydantic import __version__ as pydantic_version

from agent import build_agent


def create_app() -> FastAPI:
    # Load environment vars
    load_dotenv()
    env = os.getenv("ENV", "local")
    logging.basicConfig(level=logging.INFO)
    logging.info(f"🟢 Starting Eyewear Chatbot API in '{env}' environment")

    # Fix for Swagger/OpenAPI schema
    major_version = int(pydantic_version.split(".")[0])
    if major_version >= 2:
        ChatbotBatchRequest.model_rebuild()
    else:
        ChatbotBatchRequest.update_forward_refs()

    app = FastAPI(
        title="Eyewear Chatbot API",
        version="1.0.0",
        description="LangServe microservice for Eyewa POS chatbot"
    )

    # CORS (adjust origins in staging/production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Build and mount the LangChain agent
    agent = build_agent()
    add_routes(app, agent, path="/chatbot")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=(os.getenv("ENV", "local") == "local")
    )

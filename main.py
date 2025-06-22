import os
import re
import ast
import logging
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent import build_chatbot_agent
from langchain_openai import ChatOpenAI

# ------------------------
# Request and Response Models
# ------------------------

class ChatbotRequest(BaseModel):
    input: str = Field(..., description="User's message to the chatbot")
    chat_history: List[str] = Field(default_factory=list, description="Conversation history")
    summarize: bool = Field(default=False, description="If true, response will be shortened if too long")

class ChatbotResponse(BaseModel):
    output: str = Field(..., description="Chatbot's concise reply")

class ErrorResponse(BaseModel):
    detail: str

# ------------------------
# Utilities
# ------------------------

def shorten_if_needed(output: str, max_tokens: int = 300) -> str:
    """Shorten or rephrase output if it's too long."""
    if len(output) > 1500:  # character threshold (~300 tokens)
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            system_msg = "Please rephrase the following response to be shorter, while keeping all important information intact."
            final = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": output}
            ])
            return final.content if hasattr(final, "content") else output
        except Exception as e:
            logging.warning("⚠️ Could not shorten output: %s", e)
    return output

def extract_final_output(raw_output: str) -> str:
    """Extract the last usable 'output' from raw response string."""
    try:
        matches = re.findall(r"\{.*?\}", raw_output, re.DOTALL)
        for item in reversed(matches):
            parsed = ast.literal_eval(item)
            if "output" in parsed and "Agent stopped" not in parsed["output"]:
                return parsed["output"]
    except Exception as e:
        logging.warning("⚠️ Could not parse structured output: %s", e)
    return raw_output.strip()  # fallback

# ------------------------
# FastAPI App Setup
# ------------------------

def create_app() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"🟢 Starting chatbot API in '{os.getenv('ENV', 'local')}' environment")

    app = FastAPI(title="Eyewa Chatbot API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = build_chatbot_agent()

    @app.exception_handler(Exception)
    async def handle_error(request: Request, exc: Exception):
        logging.exception("Unhandled Exception occurred")
        return JSONResponse(
            status_code=500,
            content={"detail": "Oops! Something went wrong. Please try again later."}
        )

    @app.post("/chat", response_model=ChatbotResponse, responses={500: {"model": ErrorResponse}})
    async def chat_endpoint(request: ChatbotRequest):
        try:
            if request.input.strip().lower() in ["hi", "hello", "hey"]:
                return ChatbotResponse(output="👋 Hi there! I'm Winkly. How can I help you today?")

            logging.info("🧠 Processing input: %s", request.input)
            result = agent.invoke({"input": request.input, "chat_history": request.chat_history})
            logging.info("✅ Agent responded successfully")

            # Safely extract usable text
            raw_output = str(result)
            cleaned_output = extract_final_output(raw_output)

            # Optionally summarize
            final_output = shorten_if_needed(cleaned_output) if request.summarize else cleaned_output

            return ChatbotResponse(output=final_output)

        except Exception as e:
            logging.error(f"💥 Error during chat: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"detail": str(e)})

    return app

# ------------------------
# Run Application
# ------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

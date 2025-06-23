# main.py
import os
import re
import ast
import json
import logging
from typing import List, Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent.agent import build_chatbot_agent
from agent.agent_router import _create_classifier_chain, _classify_query, deep_clean_json_blocks
from agent.chat_history_repository import ChatHistoryRepository
from agent.reload_config import router as reload_router

from langchain_openai import ChatOpenAI
from simple_yaml import safe_load

# Load response types config
with open(os.path.join("config", "templates", "response_types.yaml")) as f:
    RESPONSE_TYPES = safe_load(f)

# Add this mapping at the top (after RESPONSE_TYPES is loaded)
OUTPUT_TO_YAML_KEY_MAP = {
    "orders": "orders_summary",
    "loyalty_card": "loyalty_summary",
    # Add more mappings as needed
}

def filter_response_by_type(response_json: dict) -> dict:
    response_type = response_json.get("type")
    if not response_type or response_type not in RESPONSE_TYPES:
        return response_json
    allowed_fields = RESPONSE_TYPES[response_type].get("fields", [])
    filtered = {"type": response_type}
    data = response_json.get("data", {})
    # Always include allowed fields from both top-level and data
    for key in allowed_fields:
        if key in data:
            filtered[key] = data[key]
        elif key in response_json:
            filtered[key] = response_json[key]
    return filtered

def merge_and_filter_responses(live, common, llm=None):
    merged = {}
    if isinstance(live, dict):
        merged.update(live)
    if isinstance(common, dict):
        merged.update(common)
    # Prefer 'mixed_summary' if both have data and LLM is available
    if llm and live and common:
        prompt = f"""
        You are a response merger for a customer support chatbot. Merge the following two JSON objects into a single, concise response for the user, following the schema for 'mixed_summary' in response_types.yaml. Only include fields defined in the schema.
        Output only a valid JSON object, with no explanation or code block formatting.

        LIVE DATA:
        {json.dumps(live, indent=2)}

        COMMON DATA:
        {json.dumps(common, indent=2)}
        """
        try:
            llm_resp = llm.invoke([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
            # Clean and parse the LLM's response robustly
            merged_candidate = deep_clean_json_blocks(str(llm_resp))
            if isinstance(merged_candidate, dict):
                merged = merged_candidate
        except Exception as e:
            logging.warning(f"⚠️ LLM merge failed: {e}")
    response_type = merged.get('type') or (live.get('type') if isinstance(live, dict) else None) or (common.get('type') if isinstance(common, dict) else None)
    merged['type'] = response_type
    return filter_response_by_type(merged)

def robust_clean(obj):
    if isinstance(obj, str):
        # Remove code block
        match = re.search(r"```(?:json)?\\n?(.*?)```", obj, re.DOTALL)
        if match:
            obj = match.group(1).strip()
        # Try to parse as JSON
        try:
            parsed = json.loads(obj)
            return robust_clean(parsed)
        except Exception:
            return None
    elif isinstance(obj, dict):
        return {k: robust_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [robust_clean(v) for v in obj]
    else:
        return obj

def parse_agent_output(obj):
    if isinstance(obj, str):
        # Remove code block
        match = re.search(r"```(?:json)?\\n?(.*?)```", obj, re.DOTALL)
        if match:
            obj = match.group(1).strip()
        # Try to parse as JSON
        try:
            parsed = json.loads(obj)
            return parse_agent_output(parsed)
        except Exception:
            return None
    elif isinstance(obj, list):
        return {"data": [parse_agent_output(v) for v in obj]}
    elif isinstance(obj, dict):
        return {k: parse_agent_output(v) for k, v in obj.items()}
    else:
        return obj

def merge_outputs(live, common):
    # Both are dicts
    if isinstance(live, dict) and isinstance(common, dict):
        merged = {**live, **common}
    elif isinstance(live, dict):
        merged = live
    elif isinstance(common, dict):
        merged = common
    else:
        merged = None
    # Fallback if nothing
    if not merged or not isinstance(merged, dict) or not merged.keys():
        return {"type": "text_response", "message": "No data found from any source."}
    # Ensure type
    if "type" not in merged or not isinstance(merged["type"], str) or not merged["type"]:
        merged["type"] = "text_response"
    return filter_response_by_type(merged)

# ------------------------
# Request and Response Models
# ------------------------

class ChatbotRequest(BaseModel):
    input: str = Field(..., description="User's message to the chatbot")
    chat_history: List[str] = Field(default_factory=list, description="Conversation history")
    summarize: bool = Field(default=False, description="If true, response will be shortened if too long")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId", description="ID used to fetch previous messages")

class ChatbotResponse(BaseModel):
    output: Any = Field(..., description="Chatbot's concise reply")

class ErrorResponse(BaseModel):
    detail: str

# ------------------------
# Utilities
# ------------------------

def shorten_if_needed(output: str, max_tokens: int = 300) -> str:
    if len(output) > 1500:
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            system_msg = "Please rephrase the following response to be shorter, while keeping all important information intact."
            final = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": output}
            ])
            if hasattr(final, "content"):
                content = final.content
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                return content
            else:
                return output
        except Exception as e:
            logging.warning("⚠️ Could not shorten output: %s", e)
    return output

def extract_final_output(raw_output: str) -> str:
    """Return the clean JSON block from agent output, filtered by response_types.yaml."""
    try:
        parsed = json.loads(raw_output)
        filtered = filter_response_by_type(parsed)
        return json.dumps(filtered)
    except Exception as e:
        logging.warning("⚠️ Could not parse structured output: %s", e)
        return raw_output.strip()

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

    # Optional router to reload schema/config
    app.include_router(reload_router)

    # Init core components
    agent = build_chatbot_agent()
    repo = ChatHistoryRepository()
    classifier_chain = _create_classifier_chain()

    @app.exception_handler(Exception)
    async def handle_error(request: Request, exc: Exception):
        logging.exception("Unhandled Exception occurred")
        return JSONResponse(
            status_code=500,
            content={"detail": "Oops! Something went wrong. Please try again later."}
        )

    @app.get("/ping", tags=["Health"])
    async def ping():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatbotResponse, tags=["Chatbot"], responses={500: {"model": ErrorResponse}})
    async def chat_endpoint(request: ChatbotRequest):
        try:
            if request.input.strip().lower() in ["hi", "hello", "hey"]:
                return ChatbotResponse(output="👋 Hi there! I'm Winkly. How can I help you today?")

            history = request.chat_history
            if request.conversation_id:
                try:
                    history = repo.fetch_history(request.conversation_id, limit=0)
                except Exception as e:
                    logging.warning("⚠️ Could not fetch chat history: %s", e)

            logging.info("🧠 Processing input: %s", request.input)
            result = agent.invoke({"input": request.input, "chat_history": history})
            logging.info("🔍 Raw agent result: %s", result)
            intent = _classify_query(request.input, classifier_chain)

            # --- Robust parsing, merging, and filtering ---
            if isinstance(result, dict) and "live" in result and "common" in result:
                live = parse_agent_output(result["live"])
                common = parse_agent_output(result["common"])
                final = merge_outputs(live, common)
            else:
                final = parse_agent_output(result)
                if not isinstance(final, dict):
                    final = {"type": "text_response", "data": final}
                else:
                    try:
                        if "type" not in final or not isinstance(final["type"], str) or not final["type"]:
                            raise Exception("Invalid type key")
                    except Exception:
                        final = {"type": "text_response", "data": str(final)}
                final = filter_response_by_type(final)

            # Save to history
            try:
                result_str = json.dumps(final)
                if request.conversation_id:
                    repo.save_message(
                        request.conversation_id,
                        request.input,
                        result_str,
                        intent=intent,
                        debug_info={"raw_output": result_str},
                    )
            except Exception as e:
                logging.warning("⚠️ Could not save chat history: %s", e)

            return ChatbotResponse(output=final)

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
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)

# main.py
import os
import re
import ast
import json
import logging
from typing import List, Optional, Any, Dict
import subprocess

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
from agent.utils import filter_response_by_type, generate_llm_message

logging.basicConfig(level=logging.DEBUG)

# Load response types config
with open(os.path.join("config", "templates", "response_types.yaml")) as f:
    RESPONSE_TYPES = safe_load(f)

# build a reverse index: field → list of types that include it
FIELD_TO_TYPES: Dict[str, List[str]] = {}
for type_name, spec in RESPONSE_TYPES.items():
    for f in spec.get("fields", []):
        FIELD_TO_TYPES.setdefault(f, []).append(type_name)

# Add this mapping at the top (after RESPONSE_TYPES is loaded)
OUTPUT_TO_YAML_KEY_MAP = {
    "orders": "orders_summary",
    "loyalty_card": "loyalty_summary",
    # Add more mappings as needed
}

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
        match = re.search(r"```(?:json)?\s*(.*?)```", obj, re.DOTALL)
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

def parse_agent_output(obj, response_type=None):
    global RESPONSE_TYPES
    logging.debug(f"[parse_agent_output] Called with obj={repr(obj)} (type={type(obj).__name__}), response_type={response_type}")
    if isinstance(obj, str):
        match = re.search(r"```(?:json)?\s*(.*?)```", obj, re.DOTALL)
        if match:
            obj = match.group(1).strip()
        try:
            parsed = json.loads(obj)
            logging.debug(f"[parse_agent_output] Parsed JSON: {repr(parsed)} (type={type(parsed).__name__})")
            return parse_agent_output(parsed, response_type)
        except Exception:
            logging.debug(f"[parse_agent_output] Could not parse as JSON, returning message: {repr(obj)}")
            return {"message": obj}
    elif isinstance(obj, list):
        logging.debug(f"[parse_agent_output] Handling list of length {len(obj)}")
        result = {"data": [parse_agent_output(v, response_type) for v in obj]}
        logging.debug(f"[parse_agent_output] Returning for list: {repr(result)}")
        return result
    elif isinstance(obj, dict):
        this_type = response_type
        if "type" in obj and isinstance(obj["type"], str):
            this_type = obj["type"]
        allowed_fields = set(RESPONSE_TYPES.get(this_type, {}).get("fields", [])) if this_type else set()
        out = {}
        for k, v in obj.items():
            logging.debug(f"[parse_agent_output] Dict key={k}, value={repr(v)} (type={type(v).__name__}), allowed_fields={allowed_fields}")
            if k == "type":
                out[k] = v
            elif k in allowed_fields:
                if isinstance(v, (dict, list)):
                    out[k] = parse_agent_output(v, this_type)
                else:
                    out[k] = v
            else:
                out[k] = parse_agent_output(v, this_type)
        logging.debug(f"[parse_agent_output] Returning for dict: {repr(out)}")
        return out
    else:
        logging.debug(f"[parse_agent_output] Returning primitive as-is: {repr(obj)} (type={type(obj).__name__})")
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

def unwrap_message_dicts(obj):
    if isinstance(obj, dict):
        # If the dict is exactly {'message': value}, unwrap it
        if set(obj.keys()) == {"message"}:
            return unwrap_message_dicts(obj["message"])
        # Otherwise, recursively unwrap all values
        return {k: unwrap_message_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unwrap_message_dicts(v) for v in obj]
    else:
        return obj

def flatten_orders_field(obj):
    if isinstance(obj, dict) and "orders" in obj:
        orders_val = obj["orders"]
        if isinstance(orders_val, dict) and "data" in orders_val:
            obj["orders"] = orders_val["data"]
    return obj

def flatten_allowed_fields(obj, allowed_fields):
    """
    Recursively flatten any nested dicts whose keys are in allowed_fields.
    Promotes fields from nested dicts to the top level if they match allowed_fields.
    """
    if not isinstance(obj, dict):
        return obj
    flat = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            # If any keys in v are in allowed_fields, promote them
            for subk, subv in v.items():
                if subk in allowed_fields:
                    flat[subk] = flatten_allowed_fields(subv, allowed_fields)
            # Also keep the original key if it's allowed and not already set
            if k in allowed_fields and k not in flat:
                flat[k] = flatten_allowed_fields(v, allowed_fields)
        else:
            flat[k] = v
    return flat

def filter_order_fields(orders, allowed_fields):
    if isinstance(orders, list):
        return [
            {k: v for k, v in order.items() if k in allowed_fields}
            for order in orders
        ]
    return orders

# ------------------------
# Request and Response Models
# ------------------------

class ChatbotRequest(BaseModel):
    input: str = Field(..., description="User's message to the chatbot")
    chat_history: List[str] = Field(default_factory=list, description="Conversation history")
    summarize: bool = Field(default=False, description="If true, response will be shortened if too long")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId", description="ID used to fetch previous messages")

class ChatbotResponse(BaseModel):
    output: Any = Field(..., description="Chatbot's concise reply (can be a dict or a list of dicts)")

class ErrorResponse(BaseModel):
    detail: str

# Robust schema enforcement for all response types

def enforce_response_schema(final, response_types):
    response_type = final.get("type")
    allowed_fields = set(response_types.get(response_type, {}).get("fields", []))
    result = {"type": response_type}
    extras = {}
    for field, value in final.items():
        if field == "type":
            continue
        if field in allowed_fields:
            result[field] = value
        else:
            extras[field] = value
    if extras:
        result["extras"] = extras
    # Special handling for orders_summary: filter fields in each order using schema
    if response_type == "orders_summary" and "orders" in result:
        # Dynamically get allowed fields for orders from the schema
        allowed_order_fields = set()
        if "orders" in response_types.get("orders_summary", {}).get("fields", []):
            # Try to infer allowed fields for each order by inspecting the first order, or document this as a config extension point
            # For now, do not filter fields (or optionally, add a config for allowed order fields)
            pass
        # If you want to filter, you could add a config section for order fields
        # result["orders"] = filter_order_fields(result["orders"], allowed_order_fields)
    logging.debug(f"[enforce_response_schema] Returning: {result}")
    return result

# ------------------------
# FastAPI App Setup
# ------------------------

def create_app() -> FastAPI:
    """Initializes and configures the FastAPI application."""

    # Load environment variables from .env file
    load_dotenv()

    # Initialize agent (if not already done)
    # Lazy initialization for agent
    agent = None

    def get_agent():
        nonlocal agent
        if agent is None:
            agent = build_chatbot_agent()
        return agent

    app = FastAPI(
        title="Chatbot Engine",
        description="A sophisticated, multi-agent chatbot with dynamic data routing.",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add the reload router
    app.include_router(reload_router)

    # ------------------------
    # API Endpoints
    # ------------------------

    @app.exception_handler(Exception)
    async def handle_error(request: Request, exc: Exception):
        logging.error(f"💥 Error during request: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "An internal server error occurred."})

    @app.get("/ping", tags=["Health"])
    async def ping():
        """A simple endpoint to check if the server is running."""
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatbotResponse, tags=["Chatbot"], responses={500: {"model": ErrorResponse}})
    async def chat_endpoint(request: ChatbotRequest):
        logging.debug(f"[CHAT] Received request: {request.dict()}")
        
        history_repo = ChatHistoryRepository()
        chat_history = []
        if request.conversation_id:
            chat_history = history_repo.fetch_history(request.conversation_id)
            logging.debug(f"Loaded chat history for conversation_id={request.conversation_id}: {chat_history}")

        input_dict = {
            "input": request.input,
            "chat_history": chat_history,
        }

        agent_instance = get_agent()
        agent_result = agent_instance.invoke(input_dict)
        logging.debug(f"[CHAT] Raw agent result: {agent_result}")

        final_response = agent_result
        
        if request.conversation_id and isinstance(final_response, dict) and "message" in final_response:
            history_repo.save_message(
                request.conversation_id, 
                request.input,
                final_response["message"]
            )
        
        logging.debug(f"[CHAT] Final output structure before return: {final_response}")
        return ChatbotResponse(output=final_response)

    return app

# Main entry point
if __name__ == "__main__":
    # Run startup test to check dependencies and connections
    try:
        subprocess.run(["pytest"], check=True, capture_output=True, text=True)
        print("\n[Startup Test] All tests passed.")
    except subprocess.CalledProcessError as e:
        print("\n[Startup Test] Pytest failed:")
        print(e.stdout)
        print(e.stderr)
        exit(1)
        
    import uvicorn
    app = create_app()

    # Load the database URIs from environment variables or .env file
    SQL_DATABASE_URI_LIVE = os.getenv("SQL_DATABASE_URI_LIVE")
    if SQL_DATABASE_URI_LIVE:
        logging.debug(f"SQL_DATABASE_URI_LIVE = {SQL_DATABASE_URI_LIVE}")

    logging.info("🟢 Starting chatbot API in 'local' environment")
    uvicorn.run(app, host="0.0.0.0", port=8000)

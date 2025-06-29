# main.py
import os
import re
import ast
import json
import logging
import threading
from typing import List, Optional, Any, Dict
import subprocess
import time

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

# Import token monitoring
try:
    from monitor_tokens import start_monitoring, stop_monitoring, get_stats
    TOKEN_MONITORING_AVAILABLE = True
except ImportError:
    TOKEN_MONITORING_AVAILABLE = False
    logging.warning("Token monitoring not available - monitor_tokens module not found")

logging.basicConfig(level=logging.DEBUG)

# Load response types config
RESPONSE_TYPES = safe_load("config/templates/response_types.yaml")

# build a reverse index: field → list of types that include it
FIELD_TO_TYPES: Dict[str, List[str]] = {}
for type_name, spec in RESPONSE_TYPES.items():
    for f in spec.get("fields", []):
        FIELD_TO_TYPES.setdefault(f, []).append(type_name)

# Global token monitoring state
token_monitoring_active = False

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

def enforce_response_schema(response: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures the response conforms to the defined schema.
    Any fields not in the schema are moved to an 'extras' dictionary.
    """
    response_type = response.get("type")
    if not response_type or response_type not in schema:
        # If the type is unknown, return as-is but log a warning.
        logging.warning(f"Unknown response type: {response_type}. Passing through without enforcement.")
        return response

    allowed_fields = set(schema[response_type].get("fields", []))
    
    # Always include 'type'
    allowed_fields.add("type")

    result = {}
    extras = {}

    for key, value in response.items():
        if key in allowed_fields:
            result[key] = value
        else:
            extras[key] = value
    
    if extras:
        result["extras"] = extras

    return result

# ------------------------
# FastAPI App Setup
# ------------------------

def create_app() -> FastAPI:
    """Initializes and configures the FastAPI application."""

    # Load environment variables from .env file
    load_dotenv()

    app = FastAPI(
        title="Chatbot API",
        description="A sophisticated, multi-agent chatbot with dynamic data routing and token monitoring.",
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

    def get_agent():
        """Get or create the chatbot agent"""
        try:
            agent = build_chatbot_agent()
            return agent
        except Exception as e:
            logging.error(f"Error creating agent: {e}")
            raise

    # ------------------------
    # Startup/Shutdown Events
    # ------------------------

    @app.on_event("startup")
    async def startup_event():
        """Startup event - initialize token monitoring"""
        global token_monitoring_active
        logging.info("🚀 Starting chatbot API...")
        
        if TOKEN_MONITORING_AVAILABLE:
            try:
                start_monitoring()
                token_monitoring_active = True
                logging.info("🔍 Token monitoring started automatically")
            except Exception as e:
                logging.error(f"Failed to start token monitoring: {e}")
        else:
            logging.warning("⚠️ Token monitoring not available")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event - stop token monitoring"""
        global token_monitoring_active
        logging.info("🛑 Shutting down chatbot API...")
        
        if TOKEN_MONITORING_AVAILABLE and token_monitoring_active:
            try:
                stop_monitoring()
                token_monitoring_active = False
                logging.info("🔍 Token monitoring stopped")
            except Exception as e:
                logging.error(f"Failed to stop token monitoring: {e}")

    # ------------------------
    # Monitoring Endpoints
    # ------------------------

    @app.get("/monitor/stats", tags=["Monitoring"])
    async def get_token_stats():
        """Get current token usage statistics"""
        if not TOKEN_MONITORING_AVAILABLE:
            return {"error": "Token monitoring not available"}
        
        try:
            stats = get_stats()
            return {
                "status": "success",
                "monitoring_active": token_monitoring_active,
                "statistics": stats,
                "timestamp": time.time()
            }
        except Exception as e:
            logging.error(f"Error getting token stats: {e}")
            return {"error": str(e)}

    @app.get("/monitor/summary", tags=["Monitoring"])
    async def get_token_summary():
        """Get detailed token usage summary"""
        if not TOKEN_MONITORING_AVAILABLE:
            return {"error": "Token monitoring not available"}
        
        try:
            from token_tracker import TokenTracker
            tracker = TokenTracker()
            
            # Get basic stats
            stats = get_stats()
            
            # Get detailed breakdown
            breakdown = {}
            for usage in tracker.usage_log:
                call_type = usage.call_type
                if call_type not in breakdown:
                    breakdown[call_type] = {
                        "calls": 0,
                        "total_tokens": 0,
                        "total_cost": 0,
                        "avg_tokens_per_call": 0
                    }
                
                breakdown[call_type]["calls"] += 1
                breakdown[call_type]["total_tokens"] += usage.total_tokens
                breakdown[call_type]["total_cost"] += usage.cost_estimate
            
            # Calculate averages
            for call_type in breakdown:
                calls = breakdown[call_type]["calls"]
                if calls > 0:
                    breakdown[call_type]["avg_tokens_per_call"] = breakdown[call_type]["total_tokens"] / calls
            
            return {
                "status": "success",
                "monitoring_active": token_monitoring_active,
                "summary": {
                    "statistics": stats,
                    "breakdown": breakdown,
                    "recent_calls": [
                        {
                            "timestamp": usage.timestamp,
                            "call_type": usage.call_type,
                            "function_name": usage.function_name,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                            "cost": usage.cost_estimate,
                            "model": usage.model
                        }
                        for usage in tracker.usage_log[-10:]  # Last 10 calls
                    ]
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logging.error(f"Error getting token summary: {e}")
            return {"error": str(e)}

    @app.post("/monitor/start", tags=["Monitoring"])
    async def start_token_monitoring():
        """Start token monitoring"""
        global token_monitoring_active
        if not TOKEN_MONITORING_AVAILABLE:
            return {"error": "Token monitoring not available"}
        
        if token_monitoring_active:
            return {"message": "Monitoring is already active", "status": "already_running"}
        
        try:
            start_monitoring()
            token_monitoring_active = True
            return {"message": "Token monitoring started", "status": "started"}
        except Exception as e:
            logging.error(f"Error starting monitoring: {e}")
            return {"error": str(e)}

    @app.post("/monitor/stop", tags=["Monitoring"])
    async def stop_token_monitoring():
        """Stop token monitoring"""
        global token_monitoring_active
        if not TOKEN_MONITORING_AVAILABLE:
            return {"error": "Token monitoring not available"}
        
        if not token_monitoring_active:
            return {"message": "Monitoring is not active", "status": "not_running"}
        
        try:
            stop_monitoring()
            token_monitoring_active = False
            return {"message": "Token monitoring stopped", "status": "stopped"}
        except Exception as e:
            logging.error(f"Error stopping monitoring: {e}")
            return {"error": str(e)}

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
        # Make chat history inclusion configurable
        include_history = os.getenv("INCLUDE_CHAT_HISTORY", "false").lower() == "true"
        if request.conversation_id and include_history:
            # Use token-based limiting to prevent context explosion - limit to 8000 tokens
            chat_history = history_repo.fetch_history_with_token_limit(request.conversation_id, max_tokens=8000)
            logging.debug(f"Loaded chat history for conversation_id={request.conversation_id}: {len(chat_history)} messages")
        else:
            logging.debug("Chat history is disabled by config.")

        input_dict = {
            "input": request.input,
            "chat_history": chat_history,
        }

        agent_instance = get_agent()
        agent_result = agent_instance.invoke(input_dict)
        logging.debug(f"[CHAT] Raw agent result: {agent_result}")

        final_response_data = agent_result
        response_type = None
        
        # Handle different types of agent results
        if isinstance(agent_result, dict):
            response_type = agent_result.get("type")
            final_response_data = enforce_response_schema(agent_result, RESPONSE_TYPES)
        elif isinstance(agent_result, list) and agent_result:
            # If it's a list of responses, take the first one
            first_result = agent_result[0]
            if isinstance(first_result, dict):
                response_type = first_result.get("type")
                final_response_data = enforce_response_schema(first_result, RESPONSE_TYPES)
            else:
                final_response_data = first_result
        else:
            # If not a dict or list, it's likely plain text - try to determine the appropriate schema
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
            # Try to determine what type of data this is based on the content
            if isinstance(agent_result, str):
                if "order" in agent_result.lower():
                    response_type = "orders_summary"
                elif "loyalty" in agent_result.lower() or "card" in agent_result.lower():
                    response_type = "loyalty_summary"
                elif "wallet" in agent_result.lower() or "balance" in agent_result.lower():
                    response_type = "wallet_summary"
                else:
                    response_type = "text_response"
            
            final_response_data = generate_llm_message(
                agent_result,
                llm,
                schema=RESPONSE_TYPES,
                response_type=response_type
            )

        # If after enforcement, still not a dict or missing required fields, try to fix with LLM
        if not isinstance(final_response_data, dict) or not response_type or response_type not in RESPONSE_TYPES:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            final_response_data = generate_llm_message(
                agent_result,
                llm,
                schema=RESPONSE_TYPES,
                response_type="text_response"
            )

        logging.debug(f"[CHAT] Enforced response: {final_response_data}")

        response_obj = ChatbotResponse(output=final_response_data)
        
        # Save history if conversation_id is present
        if request.conversation_id:
            history_repo.save_message(request.conversation_id, request.input, response_obj.model_dump_json())

        return response_obj

    return app

app = create_app()

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
    logging.info("🟢 Starting chatbot API in 'local' environment")
    uvicorn.run(app, host="0.0.0.0", port=8000)

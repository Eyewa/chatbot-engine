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
    load_dotenv()
    print("DEBUG: SQL_DATABASE_URI_LIVE =", os.getenv("SQL_DATABASE_URI_LIVE"))
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

    # Only run tests in the main process (avoid double run with reload)
    if os.environ.get("RUN_MAIN") == "true" or os.environ.get("SERVER_MAIN") == "true" or not os.environ.get("RUN_MAIN"):
        try:
            print("[Startup Test] Running all tests with pytest...")
            result = subprocess.run(["pytest", "--maxfail=1", "--disable-warnings"], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("[Startup Test] Some tests failed:")
                print(result.stderr)
            else:
                print("[Startup Test] All tests passed.")
        except Exception as e:
            print(f"[Startup Test] Could not run pytest: {e}")

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
            logging.debug(f"[CHAT] Received request: {request}")
            if request.input.strip().lower() in ["hi", "hello", "hey"]:
                logging.debug("[CHAT] Greeting detected, returning welcome message.")
                return ChatbotResponse(output="👋 Hi there! I'm Winkly. How can I help you today?")

            history = request.chat_history
            logging.debug(f"[CHAT] Initial chat_history: {history}")
            if request.conversation_id:
                try:
                    history = repo.fetch_history(request.conversation_id, limit=0)
                    logging.debug(f"[CHAT] Loaded chat history for conversation_id={request.conversation_id}: {history}")
                except Exception as e:
                    logging.warning(f"⚠️ Could not fetch chat history: {e}")

            logging.info(f"[CHAT] Input: {request.input}")
            intent = _classify_query(request.input, classifier_chain)
            logging.info(f"[CHAT] Classifier prediction: {intent}")
            result = agent.invoke({"input": request.input, "chat_history": history})
            logging.info(f"[CHAT] Raw agent result: {result}")
            logging.debug(f"[CHAT] Raw agent result type: {type(result)}")
            # --- ADDED: Print full agent result before parsing ---
            logging.debug(f"[CHAT] FULL agent result BEFORE parsing: {json.dumps(result, default=str)}")
            # --- Robust parsing, routing, and filtering ---
            summaries = []
            logging.debug(f"[CHAT] Entering summary parsing block. Result type: {type(result)}")
            logging.debug(f"[CHAT] Raw agent result (repr): {repr(result)}")
            if isinstance(result, dict) and "live" in result and "common" in result:
                logging.debug(f"[CHAT] Raw result['live']: {repr(result['live'])} (type={type(result['live'])})")
                logging.debug(f"[CHAT] Raw result['common']: {repr(result['common'])} (type={type(result['common'])})")
                # Always clean both before parsing
                cleaned_live = deep_clean_json_blocks(result["live"])
                cleaned_common = deep_clean_json_blocks(result["common"])
                logging.debug(f"[CHAT] Cleaned live: {repr(cleaned_live)} (type={type(cleaned_live)})")
                logging.debug(f"[CHAT] Cleaned common: {repr(cleaned_common)} (type={type(cleaned_common)})")
                live = parse_agent_output(cleaned_live)
                logging.debug(f"[CHAT] Parsed live: {live} (type={type(live)})")
                logging.debug(f"[CHAT] Parsed live keys: {list(live.keys()) if isinstance(live, dict) else 'N/A'}")
                common = parse_agent_output(cleaned_common)
                logging.debug(f"[CHAT] Parsed common: {common} (type={type(common)})")
                logging.debug(f"[CHAT] Parsed common keys: {list(common.keys()) if isinstance(common, dict) else 'N/A'}")
                for summary in (live, common):
                    merged = summary if isinstance(summary, dict) else {}
                    logging.debug(f"[CHAT] Merged summary: {merged} (type={type(merged)})")
                    # Extra debug for orders_summary
                    if isinstance(merged, dict) and merged.get("type") == "orders_summary":
                        logging.debug(f"[CHAT][ORDERS] Before flatten_orders_field: {merged}")
                    merged = flatten_orders_field(merged)
                    if isinstance(merged, dict) and merged.get("type") == "orders_summary":
                        logging.debug(f"[CHAT][ORDERS] After flatten_orders_field: {merged}")
                    allowed_fields = set(RESPONSE_TYPES.get(merged.get("type"), {}).get("fields", [])) if isinstance(merged, dict) else set()
                    logging.debug(f"[CHAT] Allowed fields (type={merged.get('type', None)}): {allowed_fields}")
                    merged = flatten_allowed_fields(merged, allowed_fields)
                    logging.debug(f"[CHAT] After flatten_allowed_fields: {merged}")
                    final = unwrap_message_dicts(merged)
                    logging.debug(f"[CHAT] After unwrap_message_dicts: {final}")
                    if isinstance(final, dict) and final.get("type") == "orders_summary":
                        logging.debug(f"[CHAT][ORDERS] Before enforce_response_schema: {final}")
                    final = enforce_response_schema(final, RESPONSE_TYPES)
                    if isinstance(final, dict) and final.get("type") == "orders_summary":
                        logging.debug(f"[CHAT][ORDERS] After enforce_response_schema: {final}")
                    logging.debug(f"[CHAT] Final schema-enforced response: {final}")
                    if final and isinstance(final, dict) and final.get("type"):
                        summaries.append(final)
                    else:
                        logging.debug(f"[CHAT] Skipping summary as it did not pass final checks: {final}")
                if not summaries:
                    logging.debug("[CHAT] No valid summaries found after processing live/common. Triggering fallback message.")
                    summaries = [{"type": "text_response", "message": "No data found from either source."}]
                output = summaries if len(summaries) > 1 else summaries[0]
                logging.debug(f"[CHAT] Output after processing live/common: {output}")
            else:
                if isinstance(result, dict) and "output" in result:
                    single = parse_agent_output(result["output"])
                    logging.debug(f"[CHAT] Parsed single from 'output': {single} (type={type(single)})")
                else:
                    single = parse_agent_output(result)
                    logging.debug(f"[CHAT] Parsed single from result: {single} (type={type(single)})")
                logging.debug(f"[CHAT] Single keys: {list(single.keys()) if isinstance(single, dict) else 'N/A'}")

                # --- UNWRAP PATCH: unwrap nested 'data' if present ---
                if (
                    isinstance(single, dict)
                    and "data" in single
                    and isinstance(single["data"], dict)
                    and "data" in single["data"]
                    and isinstance(single["data"]["data"], list)
                ):
                    logging.debug("[CHAT] Unwrapping nested 'data' key in single.")
                    single["data"] = single["data"]["data"]

                # --- PATCH START: handle list of summaries in 'data' key ---
                if isinstance(single, dict) and "data" in single and isinstance(single["data"], list):
                    logging.debug(f"[CHAT] Detected 'data' key with list of summaries: {single['data']} (type={type(single['data'])})")
                    summaries = []
                    for idx, summary in enumerate(single["data"]):
                        logging.debug(f"[CHAT] Processing summary #{idx}: {summary} (type={type(summary)})")
                        merged = summary if isinstance(summary, dict) else {}
                        logging.debug(f"[CHAT] Merged summary #{idx}: {merged}")
                        merged = flatten_orders_field(merged)
                        logging.debug(f"[CHAT] After flatten_orders_field for summary #{idx}: {merged}")
                        allowed_fields = set(RESPONSE_TYPES.get(merged.get("type"), {}).get("fields", [])) if isinstance(merged, dict) else set()
                        logging.debug(f"[CHAT] Allowed fields for summary #{idx} (type={merged.get('type')}): {allowed_fields}")
                        missing_fields = allowed_fields - set(merged.keys()) if isinstance(merged, dict) else set()
                        logging.debug(f"[CHAT] Missing allowed fields for summary #{idx}: {missing_fields}")
                        merged = flatten_allowed_fields(merged, allowed_fields)
                        logging.debug(f"[CHAT] After flatten_allowed_fields for summary #{idx}: {merged}")
                        final = unwrap_message_dicts(merged)
                        logging.debug(f"[CHAT] After unwrap_message_dicts for summary #{idx}: {final}")
                        final = enforce_response_schema(final, RESPONSE_TYPES)
                        logging.debug(f"[CHAT] Final schema-enforced response for summary #{idx}: {final}")
                        if final and isinstance(final, dict) and final.get("type"):
                            summaries.append(final)
                        else:
                            logging.debug(f"[CHAT] Skipping summary #{idx} as it did not pass final checks: {final}")
                    if not summaries:
                        logging.debug("[CHAT] No valid summaries found after processing all items in 'data'. Triggering fallback message.")
                        logging.debug(f"[CHAT] Context at fallback: single={single}, summaries={summaries}")
                        summaries = [{"type": "text_response", "message": "No data found from either source."}]
                    logging.debug(f"[CHAT] Output after processing all summaries: {summaries if len(summaries) > 1 else summaries[0]}")
                    output = summaries if len(summaries) > 1 else summaries[0]
                else:
                    logging.debug(f"[CHAT] No 'data' key with list found. Processing as single summary.")
                    merged = single if isinstance(single, dict) else {}
                    logging.debug(f"[CHAT] Merged single: {merged}")
                    merged = flatten_orders_field(merged)
                    logging.debug(f"[CHAT] After flatten_orders_field: {merged}")
                    allowed_fields = set(RESPONSE_TYPES.get(merged.get("type"), {}).get("fields", [])) if isinstance(merged, dict) else set()
                    logging.debug(f"[CHAT] Allowed fields for single (type={merged.get('type')}): {allowed_fields}")
                    merged = flatten_allowed_fields(merged, allowed_fields)
                    logging.debug(f"[CHAT] After flatten_allowed_fields: {merged}")
                    final = unwrap_message_dicts(merged)
                    logging.debug(f"[CHAT] After unwrap_message_dicts: {final}")
                    final = enforce_response_schema(final, RESPONSE_TYPES)
                    logging.debug(f"[CHAT] Final schema-enforced response: {final}")
                    if not final or not isinstance(final, dict) or not final.get("type"):
                        logging.debug(f"[CHAT] Fallback triggered: final output is missing or has no type. Context: merged={merged}, final={final}, single={single}")
                    output = final

            # Save to history
            try:
                result_str = json.dumps(output)
                if request.conversation_id:
                    repo.save_message(
                        request.conversation_id,
                        request.input,
                        result_str,
                        intent=intent,
                        debug_info={"raw_output": result_str},
                    )
                    logging.debug(f"[CHAT] Saved message to history for conversation_id={request.conversation_id}")
            except Exception as e:
                logging.warning(f"⚠️ Could not save chat history: {e}")

            # After processing all summaries
            # --- ADDED: Print summaries list before output construction ---
            logging.debug(f"[CHAT] Summaries list before output construction: {summaries}")
            if len(summaries) > 1:
                message = generate_llm_message(summaries, ChatOpenAI(model="gpt-4o", temperature=0))
                output = {"message": message, "data": summaries}
            elif len(summaries) == 1:
                message = generate_llm_message(summaries[0], ChatOpenAI(model="gpt-4o", temperature=0))
                output = {"message": message, **summaries[0]}
            else:
                output = {"type": "text_response", "message": "No data found from either source."}
            logging.debug(f"[CHAT] Final output structure before return: {output} (type={type(output)})")
            return ChatbotResponse(output=output)

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

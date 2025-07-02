import os
import yaml
import re
import json
import logging


def safe_load(file_path):
    if isinstance(file_path, str):
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        # If a file object is passed, just use yaml.safe_load directly
        return yaml.safe_load(file_path)


def filter_response_by_type(response_json: dict) -> dict:
    # Import RESPONSE_TYPES locally to avoid circular import
    RESPONSE_TYPES = None
    if os.path.exists(os.path.join("config", "templates", "response_types.yaml")):
        with open(os.path.join("config", "templates", "response_types.yaml")) as f:
            RESPONSE_TYPES = safe_load(f)
    if RESPONSE_TYPES is None:
        return response_json
    response_type = response_json.get("type")
    if not response_type or response_type not in RESPONSE_TYPES:
        return response_json
    allowed_fields = RESPONSE_TYPES[response_type].get("fields", [])
    filtered = {"type": response_type}
    extras = {}
    for key, value in response_json.items():
        if key == "type":
            continue
        if key in allowed_fields:
            filtered[key] = value
        else:
            extras[key] = value
    if extras:
        filtered["extras"] = extras
    return filtered


# --- Added for extracting 'last N' or 'recent N' from user queries ---

WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_last_n(query: str):
    pattern = r"(last|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(order|orders|record|records)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        n_str = match.group(2).lower()
        n = int(n_str) if n_str.isdigit() else WORD_TO_NUM.get(n_str, None)
        return n
    return None


def generate_llm_message(output, llm, schema=None, response_type=None):
    prompt = "Given the following data, output valid JSON."
    if schema and response_type and response_type in schema:
        allowed_fields = schema[response_type]["fields"]
        prompt += f" The response must match this schema: {json.dumps(allowed_fields, indent=2)}. "
        if "instructions" in schema[response_type]:
            prompt += schema[response_type]["instructions"] + " "
    prompt += (
        "\nDATA:\n"
        + json.dumps(output, indent=2)
        + "\nOutput only valid JSON. No text, no markdown."
    )
    try:
        metadata = {
            "conversation_id": os.environ.get("CONVERSATION_ID", "test-conv-id"),
            "message_id": os.environ.get("MESSAGE_ID", "test-msg-id"),
        }
        logging.info(f"Invoking LLM with metadata: {metadata}")
        resp = llm.invoke(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            metadata=metadata,
        )
        message = str(resp.content) if hasattr(resp, "content") else str(resp)
        # Try to parse as JSON
        try:
            parsed = json.loads(message)
            return parsed
        except Exception:
            return {"type": "text_response", "message": message}
    except Exception as e:
        logging.warning(f"⚠️ Could not generate LLM message: {e}")
        return {
            "type": "text_response",
            "message": "I'm sorry, I couldn't summarize the data.",
        }

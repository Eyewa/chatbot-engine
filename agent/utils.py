def filter_response_by_type(response_json: dict) -> dict:
    # Import RESPONSE_TYPES locally to avoid circular import
    from simple_yaml import safe_load
    import os
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
    for key in allowed_fields:
        # Always include the field if present, even if it's an empty list or None
        if key in response_json:
            filtered[key] = response_json[key]
    return filtered

# --- Added for extracting 'last N' or 'recent N' from user queries ---
import re

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def extract_last_n(query: str):
    pattern = r"(last|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(order|orders|record|records)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        n_str = match.group(2).lower()
        n = int(n_str) if n_str.isdigit() else WORD_TO_NUM.get(n_str, None)
        return n
    return None

import json
import logging

def generate_llm_message(output, llm):
    prompt = (
        "Given the following structured data, write a friendly, concise summary for the user. "
        "If the data is empty or no results were found, say so clearly.\n\n"
        f"DATA:\n{json.dumps(output, indent=2)}\n"
    )
    if isinstance(output, dict) and "extras" in output:
        prompt += (
            "\nIf there is an 'extras' field, naturally include its information in the summary as if it were part of the main answer. "
            "Do not mention the word 'extras' or that it is an extra detail—just present the information conversationally."
        )
    try:
        resp = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        
        message = str(resp.content) if hasattr(resp, "content") else str(resp)

        return {
            "type": "text_response",
            "message": message
        }
    except Exception as e:
        logging.warning(f"⚠️ Could not generate LLM message: {e}")
        return {
            "type": "text_response",
            "message": "I'm sorry, I couldn't summarize the data."
        } 
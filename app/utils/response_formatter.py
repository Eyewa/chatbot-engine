"""
Response formatting and validation utilities.
Handles response schema enforcement and formatting.
"""

import json
import logging
import re
import yaml
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Load response types config
def safe_load_yaml(file_path: str) -> Dict[str, Any]:
    """Safely load YAML file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {e}")
        return {}

RESPONSE_TYPES = safe_load_yaml("config/templates/response_types.yaml")


def deep_clean_json_blocks(text: str) -> Any:
    """
    Clean and parse JSON blocks from text.
    
    Args:
        text: Text containing JSON blocks
        
    Returns:
        Parsed JSON object or original text
    """
    if not isinstance(text, str):
        return text
    
    # Remove code block markers
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def filter_response_by_type(response: Dict[str, Any], response_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Filter response fields based on response type schema.
    
    Args:
        response: Response dictionary
        response_type: Type of response to filter for
        
    Returns:
        Filtered response dictionary
    """
    if not isinstance(response, dict):
        return response
    
    if not response_type:
        response_type = response.get("type")
    
    if not response_type or response_type not in RESPONSE_TYPES:
        logger.warning(f"Unknown response type: {response_type}")
        return response
    
    allowed_fields = set(RESPONSE_TYPES[response_type].get("fields", []))
    allowed_fields.add("type")  # Always include type
    
    filtered = {}
    for key, value in response.items():
        if key in allowed_fields:
            filtered[key] = value
    
    return filtered


def enforce_response_schema(response: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure response conforms to defined schema.
    Any fields not in schema are moved to 'extras' dictionary.
    
    Args:
        response: Response dictionary
        schema: Schema definition
        
    Returns:
        Schema-compliant response
    """
    response_type = response.get("type")
    if not response_type or response_type not in schema:
        logger.warning(f"Unknown response type: {response_type}. Passing through without enforcement.")
        return response

    allowed_fields = set(schema[response_type].get("fields", []))
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


def parse_agent_output(obj: Any, response_type: Optional[str] = None) -> Any:
    """
    Parse and clean agent output.
    
    Args:
        obj: Raw agent output
        response_type: Expected response type
        
    Returns:
        Parsed and cleaned output
    """
    logger.debug(f"[parse_agent_output] Called with obj={repr(obj)} (type={type(obj).__name__}), response_type={response_type}")
    
    if isinstance(obj, str):
        # Try to extract JSON from code blocks
        parsed = deep_clean_json_blocks(obj)
        if isinstance(parsed, dict):
            return parse_agent_output(parsed, response_type)
        else:
            return {"message": obj}
    
    elif isinstance(obj, list):
        logger.debug(f"[parse_agent_output] Handling list of length {len(obj)}")
        result = {"data": [parse_agent_output(v, response_type) for v in obj]}
        logger.debug(f"[parse_agent_output] Returning for list: {repr(result)}")
        return result
    
    elif isinstance(obj, dict):
        this_type = response_type
        if "type" in obj and isinstance(obj["type"], str):
            this_type = obj["type"]
        
        allowed_fields = set(RESPONSE_TYPES.get(this_type, {}).get("fields", [])) if this_type else set()
        out = {}
        
        for k, v in obj.items():
            logger.debug(f"[parse_agent_output] Dict key={k}, value={repr(v)} (type={type(v).__name__}), allowed_fields={allowed_fields}")
            if k == "type":
                out[k] = v
            elif k in allowed_fields:
                if isinstance(v, (dict, list)):
                    out[k] = parse_agent_output(v, this_type)
                else:
                    out[k] = v
            else:
                out[k] = parse_agent_output(v, this_type)
        
        logger.debug(f"[parse_agent_output] Returning for dict: {repr(out)}")
        return out
    
    else:
        logger.debug(f"[parse_agent_output] Returning primitive as-is: {repr(obj)} (type={type(obj).__name__})")
        return obj


def merge_and_filter_responses(live: Any, common: Any, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Merge live and common responses and filter by schema.
    
    Args:
        live: Live data response
        common: Common data response
        llm: Optional LLM instance for merging
        
    Returns:
        Merged and filtered response
    """
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
            logger.warning(f"⚠️ LLM merge failed: {e}")
    
    response_type = merged.get('type') or (live.get('type') if isinstance(live, dict) else None) or (common.get('type') if isinstance(common, dict) else None)
    merged['type'] = response_type
    
    return filter_response_by_type(merged)


def unwrap_message_dicts(obj: Any) -> Any:
    """
    Unwrap nested message dictionaries.
    
    Args:
        obj: Object to unwrap
        
    Returns:
        Unwrapped object
    """
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


def flatten_orders_field(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested orders field structure.
    
    Args:
        obj: Object with orders field
        
    Returns:
        Object with flattened orders field
    """
    if isinstance(obj, dict) and "orders" in obj:
        orders_val = obj["orders"]
        if isinstance(orders_val, dict) and "data" in orders_val:
            obj["orders"] = orders_val["data"]
    return obj 
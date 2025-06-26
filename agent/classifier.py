from __future__ import annotations
import re
from typing import Dict, List, Any
import yaml
import os
import logging
import json
from openai import OpenAI

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
except Exception:
    ChatOpenAI = None  # type: ignore
    Runnable = None  # type: ignore

# Load the schema to provide context to the LLM
from agent.dynamic_sql_builder import load_schema


INTENT_HIERARCHY: Dict[str, List[str]] = {
    "order": ["recent_orders", "order_amount", "order_status", "order_items", "order_date"],
    "payment": ["amount_paid", "payment_method", "payment_status"],
    "loyalty": ["card_number", "balance", "ledger_history", "expiry"],
    "customer_profile": ["name", "email", "mobile", "id"],
    "address": ["shipping_address", "billing_address", "city", "phone"],
}


_RULES: Dict[str, Dict[str, str]] = {
    "order": {
        "recent_orders": r"\b(last|recent)\b.*order",
        "order_amount": r"\b(amount|total)\b",
        "order_status": r"\bstatus\b",
        "order_items": r"\bitem",
        "order_date": r"\bdate\b",
    },
    "payment": {
        "amount_paid": r"\b(paid|payment amount)\b",
        "payment_method": r"\b(method|card|visa|paypal)\b",
        "payment_status": r"\bpayment status\b",
    },
    "loyalty": {
        "card_number": r"card number",
        "balance": r"(loyalty balance|points)",
        "ledger_history": r"ledger|history",
        "expiry": r"expiry|expiration",
    },
    "customer_profile": {
        "name": r"name",
        "email": r"email",
        "mobile": r"mobile|phone",
        "id": r"customer\s+\d+|customer id",
    },
    "address": {
        "shipping_address": r"shipping",
        "billing_address": r"billing",
        "city": r"city",
        "phone": r"phone",
    },
}


def _load_schema_for_prompt():
    """Loads and formats the schema for the LLM prompt."""
    path = os.path.join("config", "schema", "schema.yaml")
    with open(path) as f:
        schema = yaml.safe_load(f)
    
    # Format for prompt - just tables and fields are enough for the classifier
    prompt_schema = {}
    for db_key, db_schema in schema.items():
        if "tables" in db_schema:
            prompt_schema[db_key] = {}
            for table_name, table_info in db_schema["tables"].items():
                fields = table_info.get("fields", [])
                aliases = list(table_info.get("field_aliases", {}).keys())
                prompt_schema[db_key][table_name] = fields + aliases
    return yaml.dump(prompt_schema, default_flow_style=False)


def _rule_based(query: str) -> Dict[str, List[str]]:
    q = query.lower()
    result: Dict[str, List[str]] = {}
    for intent, rules in _RULES.items():
        subs = []
        for sub, pattern in rules.items():
            if re.search(pattern, q):
                subs.append(sub)
        if subs:
            result[intent] = subs
    return result


prompt_template = """
You are an expert at understanding user queries and extracting the necessary information to build a SQL query.
Your goal is to identify the main table, all the fields the user is asking for, and any filters to apply.

**Instructions:**
1.  Identify the main subject of the query to determine the `main_table`.
2.  List ALL fields the user is asking for. Be comprehensive. For example, if they ask for "customer name", you MUST include "customer_name" in the fields list. If they ask for "order amount", include "order_amount".
3.  Extract any filters the user has specified. For example, "for customer 123" should become `{{"customer_id": 123}}`.
4.  If the user specifies a number of records, extract it as the `limit`. For example, "last 5 orders" means a `limit` of 5. If not specified, default to 10.

Now, provide the JSON output for the user query above.
"""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=3)


def classify(query: str) -> Dict[str, Any]:
    """
    Uses an LLM to classify the user's query and extract key information.
    """
    schema = load_schema(db_key="live")
    schema_str = str(schema)

    # This is a placeholder for a more robust prompt formatting solution
    # that injects the user query and schema into the template.
    # For now, we simulate this by replacing placeholders.
    user_prompt = f"User Query: {query}\n\nSchema:\n{schema_str}"
    
    full_prompt = f"{prompt_template}\n\n{user_prompt}"

    logging.debug(f"[Classifier] Prompt to LLM:\n{full_prompt}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns JSON."},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        llm_output = response.choices[0].message.content
        logging.debug(f"[Classifier] Raw LLM Output: {llm_output}")
        if not llm_output:
            raise ValueError("LLM returned an empty response.")
        return json.loads(llm_output)
    except Exception as e:
        logging.error(f"[Classifier] Error during LLM call: {e}", exc_info=True)
        return {"error": "Failed to classify query."}

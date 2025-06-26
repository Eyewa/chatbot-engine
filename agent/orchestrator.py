from __future__ import annotations
import json
import logging
from typing import Dict, Any, List

from .classifier import classify
from .config_loader import config_loader
from tools.sql_toolkit_factory import (
    get_live_query_tool,
    get_common_query_tool,
)
from .prompt_builder import PromptBuilder
from agent.dynamic_sql_builder import load_schema, build_dynamic_sql, get_field_alias

try:
    import yaml
except Exception:
    import simple_yaml as yaml


def _build_sql(table: str, fields: List[str], filters: Dict[str, Any] = {}) -> str:
    cols = ", ".join(fields)
    sql = f"SELECT {cols} FROM {table}"
    if filters:
        where_clause = " AND ".join(f"{k} = {repr(v)}" for k, v in filters.items())
        sql += f" WHERE {where_clause}"
    sql += " LIMIT 5"
    return sql


# Stub for DB execution (replace with your real DB toolkit)
def run_sql_query(sql: str, db: str = "live") -> list:
    # TODO: Replace with actual DB execution logic
    logging.info(f"[DB] Would execute on db={db}: {sql}")
    # Return dummy data for demo
    return [
        {"order_id": 2000581581, "order_amount": 149.0, "status": "closed", "created_at": "2025-06-22T13:35:40", "firstname": "John", "lastname": "Doe"},
        {"order_id": 17000117627, "order_amount": 159.0, "status": "closed", "created_at": "2024-08-05T12:19:07", "firstname": "Jane", "lastname": "Smith"}
    ]


def orchestrate(query: str) -> Dict[str, Any]:
    logging.info(f"🧠 Received query: {query}")
    classification: Dict[str, Any] = classify(query)
    logging.info(f"🏷️ Classifier output: {classification}")

    registry = config_loader.get_intent_registry()
    schema = load_schema(db_key="live")
    llm_struct = classification.get('llm_struct')
    if not llm_struct:
        return {"type": "text_response", "message": "LLM did not extract fields. Please try again."}

    main_table = llm_struct['main_table']
    user_fields = llm_struct['fields']
    filters = llm_struct.get('filters', {})
    limit = llm_struct.get('limit', 10)

    # Map user-friendly fields to real fields using schema
    expanded_fields = []
    for field in user_fields:
        expanded_fields.extend(get_field_alias(main_table, field, schema))
    logging.info(f"[Field Mapping] User fields: {user_fields} -> Expanded fields: {expanded_fields}")

    sql = build_dynamic_sql(user_fields, main_table, filters, limit, schema)
    logging.info(f"[Dynamic SQL]: {sql}")

    try:
        results = run_sql_query(sql, db="live")
        # Post-process results: if customer_name was requested, combine firstname/lastname
        for row in results:
            if "firstname" in row and "lastname" in row:
                row["customer_name"] = f"{row['firstname']} {row['lastname']}"
        # Build response
        response = {
            "type": "orders_summary",
            "orders": [
                {k: v for k, v in row.items() if k in expanded_fields or k == "customer_name" or k in ["order_id", "order_amount", "status", "created_at"]}
                for row in results
            ]
        }
        return response
    except Exception as exc:
        logging.error(f"[DB ERROR] {exc}")
        return {"type": "text_response", "message": f"Query failed: {exc}"}

# --- Usage Example ---
if __name__ == "__main__":
    # Simulate LLM output for: "Show last two orders and customer name for customer 1338787"
    llm_struct = {
        'main_table': 'sales_order',
        'fields': ['order_id', 'order_amount', 'customer_name'],
        'filters': {'customer_id': 1338787},
        'limit': 2
    }
    schema = load_schema(db_key="live")
    sql = build_dynamic_sql(llm_struct['fields'], llm_struct['main_table'], llm_struct['filters'], llm_struct['limit'], schema)
    print("Generated SQL:")
    print(sql)
    # Demo DB call
    results = run_sql_query(sql, db="live")
    print("Results:")
    print(results)
    # Demo orchestrate
    print("Orchestrate output:")
    print(orchestrate("Show last two orders and customer name for customer 1338787"))

from __future__ import annotations
import json
import logging
from typing import Dict, Any, List
import ast

from .classifier import classify
from .config_loader import config_loader
from tools.sql_toolkit_factory import (
    get_live_query_tool,
    get_common_query_tool,
)
from .prompt_builder import PromptBuilder
from agent.core.dynamic_sql_builder import load_schema, build_dynamic_sql, get_field_alias

import yaml


def _build_sql(table: str, fields: List[str], filters: Dict[str, Any] = {}) -> str:
    cols = ", ".join(fields)
    sql = f"SELECT {cols} FROM {table}"
    if filters:
        where_clause = " AND ".join(f"{k} = {repr(v)}" for k, v in filters.items())
        sql += f" WHERE {where_clause}"
    sql += " LIMIT 5"
    return sql


def run_sql_query(sql: str, db: str = "live") -> list:
    if db == "live":
        toolkit = get_live_query_tool()
    else:
        toolkit = get_common_query_tool()

    if not toolkit:
        logging.error(f"Could not get toolkit for db={db}")
        return []

    logging.info(f"Executing SQL on {db}: {sql}")
    try:
        # The result from toolkit.db.run is a string representation of a list of dicts
        result_str = toolkit.db.run(sql, include_columns=True)
        logging.info(f"SQL result (raw string): {result_str}")
        # The result string looks like '[{\'key\': \'value\'}]'. We need to evaluate it.
        return ast.literal_eval(result_str)
    except (ValueError, SyntaxError, AttributeError) as e:
        logging.error(f"Could not execute or parse SQL result: {e}")
        # Attempt to run without include_columns as a fallback for some versions
        try:
            result_str = toolkit.db.run(sql)
            logging.info(f"SQL result (fallback raw string): {result_str}")
            return ast.literal_eval(result_str)
        except Exception as fallback_e:
            logging.error(f"Fallback SQL execution failed: {fallback_e}")
            return []


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
        # Build response based on original user fields, plus key identifiers
        response_fields = set(user_fields) | {"order_id", "order_amount", "status", "created_at"}
        response = {
            "type": "orders_summary",
            "orders": [
                {k: v for k, v in row.items() if k in response_fields}
                for row in results
            ]
        }
        return response
    except Exception as exc:
        logging.error(f"[DB ERROR] {exc}")
        return {"type": "text_response", "message": f"Query failed: {exc}"}

# --- Usage Example ---
if __name__ == "__main__":
    # Demo orchestrate
    print("Orchestrate output:")
    print(orchestrate("Show last two orders and customer name for customer 1338787"))

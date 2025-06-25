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


def orchestrate(query: str) -> Dict[str, Any]:
    logging.info(f"🧠 Received query: {query}")
    classification: Dict[str, Any] = classify(query)
    logging.info(f"🏷️ Classifier output: {classification}")

    # Extract relevant tables from intent registry
    registry = config_loader.get_intent_registry()
    relevant_tables = set()
    for intent in classification.get("intent", []):
        subs = classification.get("sub_intents", {}).get(intent, [])
        for sub in subs:
            cfg = registry.get(intent, {}).get(sub)
            if cfg and "table" in cfg:
                relevant_tables.add(cfg["table"])
    relevant_tables = list(relevant_tables)

    # Build mini-schema prompt
    builder = PromptBuilder()
    mini_prompt = builder.build_system_prompt_with_mini_schema(
        db="eyewa_live", relevant_tables=relevant_tables
    )
    logging.info(f"[Mini-schema prompt for LLM]:\n{mini_prompt}")

    live_tool = get_live_query_tool()
    common_tool = get_common_query_tool()

    results: Dict[str, Any] = {}
    raw_sqls: Dict[str, str] = {}

    # Get intent registry from configuration
    schema = config_loader.get_schema()
    db_tables = schema.get('tables', {})

    for intent in classification.get("intent", []):
        subs = classification.get("sub_intents", {}).get(intent, [])
        for sub in subs:
            cfg = registry.get(intent, {}).get(sub)
            if not cfg:
                logging.warning(f"⚠️ No config for {intent}.{sub}")
                continue

            table = cfg.get("table")
            db = cfg.get("db", "live")
            # Only run query if table is allowed in the target DB
            if db == "live":
                allowed_tables = ["sales_order", "customer_entity", "order_meta_data", "sales_order_address", "sales_order_payment"]
            else:
                allowed_tables = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
            if table not in allowed_tables:
                logging.error(f"Table {table} not allowed in db {db}, skipping {intent}.{sub}")
                continue

            sql = _build_sql(table, cfg.get("fields", []))
            tool = live_tool if db == "live" else common_tool

            if tool is None:
                logging.error(f"❌ Tool not available for db: {db}")
                continue

            try:
                logging.info(f"🚀 Running query for {intent}.{sub}: {sql}")
                res = tool.run({"query": sql})
                results[f"{intent}.{sub}"] = res
                raw_sqls[f"{intent}.{sub}"] = sql
            except Exception as exc:
                logging.error(f"❌ Query failed for {intent}.{sub}: {exc}")
                results[f"{intent}.{sub}"] = {"error": str(exc)}

    # Format response into mixed_summary
    orders_data = {}
    loyalty_data = {}

    for k, v in results.items():
        if k.startswith("order."):
            orders_data[k.split(".")[1]] = v
        elif k.startswith("loyalty."):
            loyalty_data[k.split(".")[1]] = v

    if orders_data or loyalty_data:
        response = {
            "type": "mixed_summary",
            "orders": orders_data,
            "loyalty": loyalty_data,
        }
        logging.info(f"✅ Assembled mixed_summary: {json.dumps(response, indent=2)}")
        return response

    logging.warning(f"⚠️ No valid data found in any result: {results}")
    return {"type": "text_response", "message": "No data found from either source"}

from __future__ import annotations
import json
import logging
from typing import Dict, Any, List

from .classifier import classify
from tools.sql_toolkit_factory import (
    get_live_query_tool,
    get_common_query_tool,
)

try:
    import yaml
except Exception:
    import simple_yaml as yaml


_INTENT_REG_PATH = "config/intent_registry.yaml"


def _load_registry() -> Dict[str, Any]:
    try:
        with open(_INTENT_REG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.error(f"❌ Failed to load intent registry: {e}")
        return {}


_REGISTRY = _load_registry()


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

    live_tool = get_live_query_tool()
    common_tool = get_common_query_tool()

    results: Dict[str, Any] = {}
    raw_sqls: Dict[str, str] = {}

    for intent in classification.get("intent", []):
        subs = classification.get("sub_intents", {}).get(intent, [])
        for sub in subs:
            cfg = _REGISTRY.get(intent, {}).get(sub)
            if not cfg:
                logging.warning(f"⚠️ No config for {intent}.{sub}")
                continue

            sql = _build_sql(cfg.get("table"), cfg.get("fields", []))
            db = cfg.get("db", "live")
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

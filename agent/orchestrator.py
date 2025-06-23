from __future__ import annotations
import json
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
    except Exception:
        return {}


_REGISTRY = _load_registry()


def _build_sql(table: str, fields: List[str]) -> str:
    cols = ", ".join(fields)
    return f"SELECT {cols} FROM {table} LIMIT 5"


def orchestrate(query: str) -> Dict[str, Any]:
    classification = classify(query)
    live_tool = get_live_query_tool()
    common_tool = get_common_query_tool()

    results: Dict[str, Any] = {}

    for intent in classification.get("intent", []):
        subs = classification.get("sub_intents", {}).get(intent, [])
        for sub in subs:
            cfg = _REGISTRY.get(intent, {}).get(sub)
            if not cfg:
                continue
            sql = _build_sql(cfg.get("table"), cfg.get("fields", []))
            db = cfg.get("db", "live")
            tool = live_tool if db == "live" else common_tool
            if tool is None:
                continue
            try:
                res = tool.run({"query": sql})
            except Exception as exc:
                res = str(exc)
            results[f"{intent}.{sub}"] = res

    return results

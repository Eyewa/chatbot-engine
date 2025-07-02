# agent/reload.py
import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

from agent.core.prompt_builder import PromptBuilder

router = APIRouter()

# --- Shared cache for tools ---
RELOADABLE_STATE: Dict[str, Optional[List[Any]]] = {
    "live": None,
    "common": None,
}

# --- LLM instance ---
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception:
    llm = None


@router.get("/admin/reload-config", tags=["Admin"])
def reload_config():
    """Clear and reload prompt configuration cache."""
    PromptBuilder._cache.clear()
    PromptBuilder._timestamps.clear()
    logging.info("✅ Config cache cleared via /admin/reload-config")
    return {"status": "✅ Config reloaded"}


@router.get("/admin/reload-response-types", tags=["Admin"])
def reload_response_types():
    """Reload response types configuration."""
    try:
        # Load response types directly from config
        response_types = safe_load("config/templates/response_types.yaml")

        logging.info("✅ Response types reloaded via /admin/reload-response-types")
        return {
            "status": "✅ Response types reloaded",
            "types": list(response_types.keys()),
        }
    except Exception as e:
        logging.error(f"❌ Failed to reload response types: {e}")
        return {"status": "❌ Failed to reload response types", "error": str(e)}


def safe_load(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rename_tools(tools, suffix: str):
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools


def rebuild_sql_tools(db_type: str):
    if llm is None:
        return None

    builder = PromptBuilder()
    custom_info = builder.build_custom_table_info()

    if db_type == "live":
        uri = os.getenv("SQL_DATABASE_URI_LIVE")
        # Load allowed tables from schema
        schema_path = os.path.join("config", "schema", "schema.yaml")
        try:
            with open(schema_path, "r") as f:
                schema = yaml.safe_load(f)
            allowed = list(schema.get("live", {}).get("tables", {}).keys())
        except Exception as e:
            logging.error(f"Could not load schema config: {e}")
            allowed = []
    elif db_type == "common":
        uri = os.getenv("SQL_DATABASE_URI_COMMON")
        # Load allowed tables from schema
        schema_path = os.path.join("config", "schema", "schema.yaml")
        try:
            with open(schema_path, "r") as f:
                schema = yaml.safe_load(f)
            allowed = list(schema.get("common", {}).get("tables", {}).keys())
        except Exception as e:
            logging.error(f"Could not load schema config: {e}")
            allowed = []
    else:
        return None

    if not uri:
        logging.error(f"No database URI found for {db_type}")
        return None

    db = SQLDatabase.from_uri(
        uri,
        include_tables=[t for t in allowed if t in custom_info],
        sample_rows_in_table_info=0,
        custom_table_info=custom_info,
    )
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    RELOADABLE_STATE[db_type] = _rename_tools(tools, db_type)
    logging.info(f"🔄 Reloaded tools for {db_type}")
    return RELOADABLE_STATE[db_type]


@router.get("/admin/reload-tools/{db_type}", tags=["Admin"])
def reload_sql_tools_endpoint(db_type: str):
    """Trigger rebuild of live/common SQL tools."""
    tools = rebuild_sql_tools(db_type)
    if not tools:
        return {"status": "❌ Invalid DB", "tools": []}
    return {"status": "✅ Reloaded", "tools": [t.name for t in tools]}

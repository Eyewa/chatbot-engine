# agent/reload.py
import os
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter
from agent.prompt_builder import PromptBuilder
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

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
        # Import and reload the response types
        from simple_yaml import safe_load
        import main
        
        # Reload the response types
        main.RESPONSE_TYPES = safe_load("config/templates/response_types.yaml")
        
        logging.info("✅ Response types reloaded via /admin/reload-response-types")
        return {"status": "✅ Response types reloaded", "types": list(main.RESPONSE_TYPES.keys())}
    except Exception as e:
        logging.error(f"❌ Failed to reload response types: {e}")
        return {"status": "❌ Failed to reload response types", "error": str(e)}


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
        allowed = [
            "sales_order",
            "customer_entity",
            "order_meta_data",
            "sales_order_address",
            "sales_order_payment",
        ]
    elif db_type == "common":
        uri = os.getenv("SQL_DATABASE_URI_COMMON")
        allowed = [
            "customer_loyalty_card",
            "customer_loyalty_ledger",
            "customer_wallet",
        ]
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

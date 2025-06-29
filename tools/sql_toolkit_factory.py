import os
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from sqlalchemy import text

import yaml

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from langchain_community.utilities import SQLDatabase  # type: ignore
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain.tools import Tool  # <--- Add this import
except ImportError:
    SQLDatabase = None
    SQLDatabaseToolkit = None
    def ChatOpenAI(*args, **kwargs):
        raise ModuleNotFoundError("LangChain dependencies not installed")

# Load environment variables
load_dotenv()

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception:
    llm = None

from agent.prompt_builder import PromptBuilder

# ------------------------------------------
# Query Validator
# ------------------------------------------

def _validate_query(query: str | dict, schema_map: Dict[str, List[str]]) -> None:
    if isinstance(query, dict):
        query = query.get("query", "")
    if not isinstance(query, str):
        query = str(query)
    alias_map = {}
    alias_matches = re.findall(r'\b(from|join)\s+([a-zA-Z_][\w]*)\s+(?:as\s+)?([a-zA-Z_][\w]*)', query, flags=re.I)
    for _, table, alias in alias_matches:
        alias_map[alias] = table
    column_pairs = re.findall(r'([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)', query)
    for alias, column in column_pairs:
        table = alias_map.get(alias, alias)
        fields = schema_map.get(table)
        if fields is None:
            logging.warning(f"⚠️ Unknown table or alias: '{table}' (from alias '{alias}')")
            continue
        if column not in fields:
            logging.warning(f"🛑 Invalid column '{column}' in table '{table}' — allowed: {fields}")
            raise ValueError(f"Invalid column: {table}.{column}")

# ------------------------------------------
# SQL tools factory for live and common DBs
# ------------------------------------------

def _rename_tools(tools, suffix: str):
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools

def make_strict_sql_db_query(db):
    def strict_sql_db_query(query: str) -> list:
        """
        Execute a SQL query and return real DB results as a list of dicts.
        """
        with db._engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                return []
            return [dict(row._mapping) for row in rows]
    # Return as a LangChain Tool object
    return Tool(
        name="strict_sql_db_query",
        func=strict_sql_db_query,
        description="Executes a SQL query and returns real DB results as a list of dicts."
    )

def _create_sql_tools(uri: str, allowed_tables: List[str], suffix: str) -> List[Any]:
    builder = PromptBuilder()
    custom_info = builder.build_custom_table_info(allowed_tables)
    schema_map = {
        table: builder.schema_cfg.get("tables", {}).get(table, {}).get("fields", [])
        for table in allowed_tables
    }
    if SQLDatabase is None:
        logging.error("SQLDatabase is not available")
        return []
    db = SQLDatabase.from_uri(
        uri,
        include_tables=list(custom_info.keys()),
        sample_rows_in_table_info=0,
        custom_table_info=custom_info,
    )
    logging.info(f"✅ Loaded {suffix} DB tables: %s", db.get_usable_table_names())
    # Only use the strict function tool for querying
    return [make_strict_sql_db_query(db)]

@lru_cache
def get_live_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_LIVE")
    if not uri:
        logging.error("SQL_DATABASE_URI_LIVE is not set")
        return []
    allowed = ["sales_order", "customer_entity", "order_meta_data", "sales_order_address", "sales_order_payment"]
    return _create_sql_tools(uri, allowed, "live")

@lru_cache
def get_common_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_COMMON")
    if not uri:
        logging.error("SQL_DATABASE_URI_COMMON is not set")
        return []
    allowed = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    return _create_sql_tools(uri, allowed, "common")


def get_live_query_tool():
    """Return the sql_db_query tool for the live DB or None."""
    for tool in get_live_sql_tools():
        if tool.name.startswith("sql_db_query"):
            return tool
    return None


def get_common_query_tool():
    """Return the sql_db_query tool for the common DB or None."""
    for tool in get_common_sql_tools():
        if tool.name.startswith("sql_db_query"):
            return tool
    return None

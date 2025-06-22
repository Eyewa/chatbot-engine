import os
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import yaml
except Exception:
    import simple_yaml as yaml

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

try:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
except ImportError:
    SQLDatabase = None
    SQLDatabaseToolkit = None
    def ChatOpenAI(*args, **kwargs):
        raise ModuleNotFoundError("LangChain dependencies not installed")

# Load environment variables
load_dotenv()

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except Exception:
    llm = None

from agent.prompt_builder import PromptBuilder

# ------------------------------------------
# Query Validator
# ------------------------------------------

def _validate_query(query: str | dict, schema_map: Dict[str, List[str]]) -> None:
    if isinstance(query, dict):
        query = query.get("query", "")

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

def _create_sql_tools(uri: str, allowed_tables: List[str], suffix: str) -> List[Any]:
    builder = PromptBuilder()
    custom_info = builder.build_custom_table_info(allowed_tables)
    schema_map = {
        table: builder.schema_cfg.get("tables", {}).get(table, {}).get("fields", [])
        for table in allowed_tables
    }

    db = SQLDatabase.from_uri(
        uri,
        include_tables=list(custom_info.keys()),
        sample_rows_in_table_info=0,
        custom_table_info=custom_info,
    )
    logging.info(f"✅ Loaded {suffix} DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    for tool in tools:
        if tool.name.startswith("sql_db_query"):
            original = tool.run
            def run(query: str | dict = "", **kwargs):
                q = query.get("query") if isinstance(query, dict) else query
                _validate_query(q, schema_map)
                return original(q, **kwargs)
            try:
                object.__setattr__(tool, "run", run)
                object.__setattr__(tool, "invoke", run)
            except Exception as e:
                logging.error(f"Failed to patch run for {tool.name}: {e}")

        if tool.name.startswith("sql_db_schema"):
            original = tool.run
            def schema_run(table_names: Optional[str] = None, **kwargs):
                if isinstance(table_names, dict):
                    table_names = table_names.get("table_names")

                valid_tables = list(schema_map.keys())

                if not isinstance(table_names, str):
                    names = valid_tables
                else:
                    raw = str(table_names).split(",")
                    names = [n.strip() for n in raw if n.strip() in valid_tables]
                    if not names:
                        names = valid_tables

                return json.dumps({n: schema_map.get(n, []) for n in names})
            try:
                object.__setattr__(tool, "run", schema_run)
                object.__setattr__(tool, "invoke", schema_run)
            except Exception as e:
                logging.error(f"Failed to patch schema_run for {tool.name}: {e}")

    return _rename_tools(tools, suffix)

@lru_cache
def get_live_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_LIVE")
    allowed = ["sales_order", "customer_entity", "order_meta_data", "sales_order_address", "sales_order_payment"]
    return _create_sql_tools(uri, allowed, "live")

@lru_cache
def get_common_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_COMMON")
    allowed = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    return _create_sql_tools(uri, allowed, "common")

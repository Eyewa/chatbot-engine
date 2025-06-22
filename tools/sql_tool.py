import os
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:  # Optional: environment loader
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

try:  # Optional: LangChain and DB agent dependencies
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
except Exception:
    SQLDatabase = None
    SQLDatabaseToolkit = None
    def ChatOpenAI(*args, **kwargs):
        raise ModuleNotFoundError("langchain not installed")

# Load environment variables
load_dotenv()

# LLM client
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except Exception:
    llm = None


# ------------------------------------------
# PromptBuilder class for schema/prompt prep
# ------------------------------------------

class PromptBuilder:
    """Load prompt configuration from YAML files and build system prompts."""

    def __init__(self, base_dir: str = "config"):
        self.response_cfg = self._load_yaml(Path(base_dir) / "templates" / "response_types.yaml")
        self.schema_cfg = self._load_yaml(Path(base_dir) / "schema" / "schema.yaml")
        self._validate_schema()

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logging.warning(f"⚠️ YAML file not found: {path}")
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"❌ Failed to load YAML: {path} — {e}")
            return {}

    def _validate_schema(self):
        tables = self.schema_cfg.get("tables", {})
        if not isinstance(tables, dict):
            raise ValueError("Schema YAML must contain a 'tables' dictionary")
        for table, meta in tables.items():
            if not isinstance(meta, dict):
                logging.warning(f"⚠️ Table '{table}' metadata is not a dictionary. Skipping.")
                continue
            if "fields" not in meta:
                raise ValueError(f"Missing 'fields' key in table definition for '{table}'")
            if not isinstance(meta["fields"], list):
                raise ValueError(f"'fields' for table '{table}' must be a list")

    def build_system_prompt(self, db: str = "", allowed_tables: List[str] = None) -> str:
        types = ", ".join(self.response_cfg.keys())
        lines = [
            "You are Winkly — an intelligent, structured response assistant.",
            f"Always respond using JSON with a top-level 'type'. Valid types are: {types}.",
        ]

        if allowed_tables:
            lines.append(f"You are using the `{db}` database with access to: {', '.join(allowed_tables)}.")
        else:
            allowed_tables = []

        lines.append("⚠️ Tables with `_live` and `_common` suffixes belong to separate databases. Never join across them.")

        join_lines = []
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if not isinstance(meta, dict):
                continue
            if allowed_tables and table not in allowed_tables:
                continue
            for join in meta.get("joins", []):
                if join.get("to_table") in allowed_tables:
                    join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
        if join_lines:
            lines.append("Use these known joins when needed:")
            lines.extend(join_lines)

        lines.append("Do NOT hallucinate tables or fields. Only use those explicitly listed.")
        return "\n".join(lines)

    def build_custom_table_info(self) -> Dict[str, str]:
        table_info = {}
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if not isinstance(meta, dict):
                logging.warning(f"⚠️ Skipping malformed metadata for table: {table}")
                continue
            description = meta.get("description", f"{table} table.")
            fields = meta.get("fields", [])
            if not isinstance(fields, list):
                logging.warning(f"⚠️ 'fields' must be a list in table: {table}. Skipping.")
                continue
            field_list = ", ".join(fields)
            table_info[table] = f"{table}: {description}\nColumns: {field_list}"
        return table_info

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}


# -------------------------
# SQL tools factory methods
# -------------------------

def _rename_tools(tools, suffix: str):
    """Suffix tool names to prevent conflicts in multi-agent chains."""
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools


@lru_cache
def get_live_sql_tools():
    """Return LangChain tools for eyewa_live DB."""
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []

    builder = PromptBuilder()
    allowed_tables = [
        "sales_order",
        "customer_entity",
        "order_meta_data",
        "sales_order_address",
        "sales_order_payment",
    ]

    custom_info = {
        t: info for t, info in builder.build_custom_table_info().items()
        if t in allowed_tables
    }

    uri = os.getenv("SQL_DATABASE_URI_LIVE")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=list(custom_info.keys()),
        sample_rows_in_table_info=0,
        custom_table_info=custom_info,
    )
    logging.info("✅ Loaded live DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "live")


@lru_cache
def get_common_sql_tools():
    """Return LangChain tools for eyewa_common DB."""
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []

    builder = PromptBuilder()
    allowed_tables = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]

    custom_info = {
        t: info for t, info in builder.build_custom_table_info().items()
        if t in allowed_tables
    }

    uri = os.getenv("SQL_DATABASE_URI_COMMON")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=list(custom_info.keys()),
        sample_rows_in_table_info=0,
        custom_table_info=custom_info,
    )
    logging.info("✅ Loaded common DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "common")

import os
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML isn't installed
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

# Load env variables
load_dotenv()

# Initialize LLM
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
        self.base_dir = Path(base_dir)
        self.response_cfg = self._load_yaml(self.base_dir / "templates" / "response_types.yaml")
        self.schema_cfg = self._load_yaml(self.base_dir / "schema" / "schema.yaml")
        self._validate_schema()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logging.warning(f"⚠️ YAML file not found: {path}")
            return {}
        try:
            if yaml is not None:
                with path.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            json_path = path.with_suffix(".json")
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ Failed to load YAML: {path} — {e}")
            return {}

    def _validate_schema(self):
        tables = self.schema_cfg.get("tables", {})
        if not isinstance(tables, dict):
            raise ValueError("Schema YAML must contain a 'tables' dictionary")
        for table, meta in tables.items():
            if not isinstance(meta, dict):
                continue
            if "fields" not in meta or not isinstance(meta["fields"], list):
                raise ValueError(f"Invalid or missing 'fields' in table '{table}'")
            if "joins" in meta:
                for join in meta["joins"]:
                    if not all(k in join for k in ("from_field", "to_table", "to_field")):
                        raise ValueError(f"Incomplete join definition in table '{table}'")

    def build_system_prompt(self, db: str = "", allowed_tables: Optional[List[str]] = None) -> str:
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
            if not isinstance(meta, dict) or (allowed_tables and table not in allowed_tables):
                continue
            for join in meta.get("joins", []):
                if join.get("to_table") in allowed_tables:
                    join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
        if join_lines:
            lines.append("Use these known joins when needed:")
            lines.extend(join_lines)

        lines.append("Do NOT hallucinate tables or fields. Only use those explicitly listed.")
        return "\n".join(lines)

    def build_custom_table_info(self, allowed_tables: Optional[List[str]] = None) -> Dict[str, str]:
        info = {}
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if not isinstance(meta, dict) or (allowed_tables and table not in allowed_tables):
                continue
            description = meta.get("description", f"{table} table.")
            fields = meta.get("fields", [])
            if not isinstance(fields, list):
                continue
            info[table] = f"{table}: {description}\nColumns: {', '.join(fields)}"
        return info

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}


def _validate_query(query: str, schema_map: Dict[str, List[str]]) -> None:
    """Ensure all table.column references in the query exist in the schema."""
    alias_map = {}
    for kw in ["from", "join"]:
        for tbl, alias in re.findall(fr"{kw}\s+([a-zA-Z_][\w]*)\s+(?:as\s+)?([a-zA-Z_][\w]*)", query, flags=re.I):
            alias_map[alias] = tbl

    pairs = re.findall(r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)", query)
    for table, column in pairs:
        actual = alias_map.get(table, table)
        fields = schema_map.get(actual)
        if fields is not None and column not in fields:
            raise ValueError(f"Invalid column: {actual}.{column}")


# ------------------------------------------
# SQL tools factory for live and common DBs
# ------------------------------------------

def _rename_tools(tools, suffix: str):
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools


@lru_cache
def get_live_sql_tools():
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

    custom_info = builder.build_custom_table_info(allowed_tables)

    schema_map = {
        table: builder.schema_cfg.get("tables", {}).get(table, {}).get("fields", [])
        for table in allowed_tables
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
    for tool in tools:
        if tool.name == "sql_db_query":
            original = tool.run

            def run(query: str):
                _validate_query(query, schema_map)
                return original(query)

            tool.run = run
        if tool.name == "sql_db_schema":
            original = tool.run

            def schema_run(table_names: Optional[str] = None):
                names = (
                    [n.strip() for n in table_names.split(",")]
                    if table_names
                    else list(schema_map.keys())
                )
                return json.dumps({n: schema_map.get(n, []) for n in names})

            tool.run = schema_run
    return _rename_tools(tools, "live")


@lru_cache
def get_common_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []

    builder = PromptBuilder()
    allowed_tables = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    custom_info = builder.build_custom_table_info(allowed_tables)

    schema_map = {
        table: builder.schema_cfg.get("tables", {}).get(table, {}).get("fields", [])
        for table in allowed_tables
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
    for tool in tools:
        if tool.name == "sql_db_query":
            original = tool.run

            def run(query: str):
                _validate_query(query, schema_map)
                return original(query)

            tool.run = run
        if tool.name == "sql_db_schema":
            original = tool.run

            def schema_run(table_names: Optional[str] = None):
                names = (
                    [n.strip() for n in table_names.split(",")]
                    if table_names
                    else list(schema_map.keys())
                )
                return json.dumps({n: schema_map.get(n, []) for n in names})

            tool.run = schema_run
    return _rename_tools(tools, "common")

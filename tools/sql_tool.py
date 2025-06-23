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


# ------------------------------------------
# PromptBuilder class for schema/prompt prep
# ------------------------------------------

class PromptBuilder:
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
                continue
            if "fields" not in meta or not isinstance(meta["fields"], list):
                raise ValueError(f"Invalid or missing 'fields' in table '{table}'")
            if "joins" in meta:
                for join in meta["joins"]:
                    if not all(k in join for k in ("from_field", "to_table", "to_field")):
                        raise ValueError(f"Incomplete join definition in table '{table}'")

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


# ------------------------------------------
# Query Validator
# ------------------------------------------

def _validate_query(query: str | dict, schema_map: Dict[str, List[str]]) -> None:
    """Validates that all columns used in the query exist in the schema (resolves aliases)."""
    if isinstance(query, dict):
        query = query.get("query", "")

    # Step 1: Map aliases to actual tables
    alias_map = {}
    alias_matches = re.findall(
        r'\b(from|join)\s+([a-zA-Z_][\w]*)\s+(?:as\s+)?([a-zA-Z_][\w]*)',
        query, flags=re.I
    )
    for _, table, alias in alias_matches:
        alias_map[alias] = table

    # Step 2: Extract all alias.column usages from query
    column_pairs = re.findall(r'([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)', query)

    for alias, column in column_pairs:
        table = alias_map.get(alias, alias)
        fields = schema_map.get(table)
        if fields is None:
            logging.warning(
                f"⚠️ Unknown table or alias: '{table}' (from alias '{alias}')"
            )
            continue  # Optionally raise here if strict
        if column not in fields:
            logging.warning(
                f"🛑 Invalid column '{column}' in table '{table}' — allowed: {fields}"
            )
            raise ValueError(f"Invalid column: {table}.{column}")


def extract_table_names(sql: str) -> List[str]:
    """Return a list of table names referenced in the SQL query."""
    names: List[str] = []
    pattern = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", re.IGNORECASE)
    for match in pattern.finditer(sql):
        table = match.group(1)
        if table not in names:
            names.append(table)
    return names


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
        if tool.name == "sql_db_query":
            original = tool.run

            def run(query: str | dict = "", **kwargs):
                q = query.get("query") if isinstance(query, dict) else query
                _validate_query(q, schema_map)
                return original(q, **kwargs)

            try:
                tool.run = run
            except Exception:
                object.__setattr__(tool, "run", run)

        if tool.name == "sql_db_schema":
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
                tool.run = schema_run
            except Exception:
                object.__setattr__(tool, "run", schema_run)

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

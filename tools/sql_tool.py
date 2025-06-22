import os
import logging
from functools import lru_cache

try:  # pragma: no cover - optional dependencies for runtime
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback for tests
    def load_dotenv():
        return None

try:  # pragma: no cover - optional dependencies for runtime
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - fallback for tests
    SQLDatabase = None  # type: ignore
    SQLDatabaseToolkit = None  # type: ignore

    def ChatOpenAI(*args, **kwargs):  # type: ignore
        raise ModuleNotFoundError("langchain not installed")

load_dotenv()
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except Exception:  # pragma: no cover - missing dependency
    llm = None

def _rename_tools(tools, suffix: str):
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools

@lru_cache
def get_live_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_LIVE")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=[
            "sales_order",
            "customer_entity",
            "order_meta_data",
            "sales_order_address",
            "sales_order_payment",
        ],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ Loaded live DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "live")

@lru_cache
def get_common_sql_tools():
    if SQLDatabase is None or SQLDatabaseToolkit is None or llm is None:
        return []
    uri = os.getenv("SQL_DATABASE_URI_COMMON")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=["customer_loyalty_card", "customer_loyalty_ledger"],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ Loaded common DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "common")

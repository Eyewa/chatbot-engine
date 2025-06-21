# tools/sql_tool.py

import os
import logging
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from functools import lru_cache


def _rename_tools(tools, suffix: str):
    """Return tools with their names suffixed by the given database label."""
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@lru_cache
def get_live_sql_tools():
    db_uri = os.getenv("SQL_DATABASE_URI_LIVE")
    db_live = SQLDatabase.from_uri(
        db_uri,
        include_tables=["sales_order", "customer_entity", "order_meta_data","sales_order_payment"],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ Live DB tables: %s", db_live.get_usable_table_names())
    toolkit = SQLDatabaseToolkit(db=db_live, llm=llm)
    tools = toolkit.get_tools()
    return _rename_tools(tools, "live")

@lru_cache
def get_common_sql_tools():
    db_uri = os.getenv("SQL_DATABASE_URI_COMMON")
    db_common = SQLDatabase.from_uri(
        db_uri,
        include_tables=[
            "customer_loyalty_card",
            "customer_loyalty_ledger",
        ],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ common DB tables: %s", db_common.get_usable_table_names())
    toolkit = SQLDatabaseToolkit(db=db_common, llm=llm)
    tools = toolkit.get_tools()
    return _rename_tools(tools, "common")

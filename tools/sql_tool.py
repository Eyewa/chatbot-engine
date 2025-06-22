import os
import logging
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from functools import lru_cache

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def _rename_tools(tools, suffix: str):
    for tool in tools:
        tool.name = f"{tool.name}_{suffix}"
    return tools

@lru_cache
def get_live_sql_tools():
    uri = os.getenv("SQL_DATABASE_URI_LIVE")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=["sales_order", "customer_entity", "order_meta_data", "sales_order_address","sales_order_payment"],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ Loaded live DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "live")

@lru_cache
def get_common_sql_tools():
    uri = os.getenv("SQL_DATABASE_URI_COMMON")
    db = SQLDatabase.from_uri(
        uri,
        include_tables=["customer_loyalty_card", "customer_loyalty_ledger", ],
        sample_rows_in_table_info=5,
    )
    logging.info("✅ Loaded common DB tables: %s", db.get_usable_table_names())
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    return _rename_tools(tools, "common")

# tools/sql_tool.py

import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_live_sql_tools():
    db_uri = os.getenv("SQL_DATABASE_URI_LIVE")
    db_live = SQLDatabase.from_uri(db_uri, include_tables=[], sample_rows_in_table_info=0)
    toolkit = SQLDatabaseToolkit(db=db_live, llm=llm)
    return toolkit.get_tools()

def get_common_sql_tools():
    db_uri = os.getenv("SQL_DATABASE_URI_COMMON")
    db_common = SQLDatabase.from_uri(db_uri, include_tables=[], sample_rows_in_table_info=0)
    toolkit = SQLDatabaseToolkit(db=db_common, llm=llm)
    return toolkit.get_tools()

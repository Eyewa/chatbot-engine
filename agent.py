import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.sql_database import SQLDatabase
from langchain.utilities import SQLDatabaseChain
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

from tools.sql_tool import query_order_status
from tools.es_tool import search_products_by_filters


def build_agent() -> BaseTool:
    load_dotenv()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

    # Placeholder SQLDatabaseToolkit and Elasticsearch tool can be extended later
    # For now, we directly use the dummy tools defined above
    tools = [query_order_status, search_products_by_filters]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

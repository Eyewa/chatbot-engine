import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# Import your @tool-decorated functions
from tools.sql_tool import query_order_status
from tools.es_tool import search_products_by_filters


def build_agent() -> BaseTool:
    # Load env vars
    load_dotenv()

    # Validate OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.startswith("sk-"):
        raise ValueError("❌ OPENAI_API_KEY is missing or invalid in .env file.")

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        temperature=0,
        verbose=(os.getenv("ENV", "local") != "production")
    )

    # Define available structured tools
    tools = [
        query_order_status,
        search_products_by_filters
    ]

    # Build the agent with function-calling support
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    return agent

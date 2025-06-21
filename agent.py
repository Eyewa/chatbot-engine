from typing import List
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Import tools
from tools.sql_tool import query_order_status
from tools.es_tool import search_products_by_filters

# Load environment and validate key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key or not openai_key.startswith("sk-"):
    raise ValueError("❌ Missing or invalid OPENAI_API_KEY in environment")

# System instructions
SYSTEM_PROMPT = (
    "You are a helpful assistant with tools to answer questions about orders and products. "
    "Use `query_order_status` to check order details, and `search_products_by_filters` for product queries."
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Register tools
TOOLS = [
    query_order_status,
    search_products_by_filters
]

# Setup the LLM and agent
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_key)
agent_chain = create_openai_functions_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent_chain, tools=TOOLS, verbose=True)

# Build function for FastAPI to use
def build_chatbot_agent():
    return agent_executor

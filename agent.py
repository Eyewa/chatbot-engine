# agent.py

from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import tools
from tools.sql_tool import query_order_status
from tools.es_tool import search_products_by_filters
from dotenv import load_dotenv
load_dotenv()

# Tools your agent can use
TOOLS = [
    query_order_status,
    search_products_by_filters,
]

# System instruction for the assistant
SYSTEM_PROMPT = (
    "You are a helpful assistant for Eyewa's POS chatbot.\n"
    "Use 'query_order_status' to get order info and 'search_products_by_filters' to help with product discovery.\n"
    "Call tools only when needed and reply clearly to the customer."
)

# Construct the full prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ChatOpenAI setup (uses OPENAI_API_KEY from .env)
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

# Bind prompt + tools to LLM agent
agent_chain = create_openai_functions_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent_chain, tools=TOOLS, verbose=True)

# Pydantic input/output types (for internal use or FastAPI validation)
class Input(BaseModel):
    input: str = Field(..., description="User's latest message")
    chat_history: List[object] = Field(default_factory=list, description="Chat history")

class Output(BaseModel):
    output: str = Field(..., description="Bot's response")

# Function to return the agent instance
def build_chatbot_agent() -> AgentExecutor:
    return agent_executor

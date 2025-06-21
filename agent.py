# agent.py

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from tools.sql_tool import get_live_sql_tools, get_common_sql_tools

def build_chatbot_agent():
    try:
        all_tools = get_live_sql_tools() + get_common_sql_tools()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        return initialize_agent(
            tools=all_tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Failed to build chatbot agent: {e}")
        return None

# agent.py

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.sql_tool import get_live_sql_tools, get_common_sql_tools

def build_chatbot_agent():

    """
    Construct and return the chatbot agent.

    SQL tools are loaded from both the ``eyewa_live`` and ``eyewa_common``
    databases. Each tool is automatically suffixed with the database label
    (e.g. ``sql_db_query_live``) to avoid name collisions when registered with
    the agent.
    """
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # SQL tools already have unique names like ``sql_db_query_live`` to prevent
    # collisions across databases.
    tools = get_live_sql_tools() + get_common_sql_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Winkly, a helpful assistant for Eyewa customers. Respond politely and informatively."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=False
    )

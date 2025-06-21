# agent.py

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.sql_tool import get_live_sql_tools, get_common_sql_tools

def build_chatbot_agent():

    """
    Constructs and returns a LangChain AgentExecutor for Winkly, the Eyewa assistant.
    The agent uses tools from the eyewa_live and eyewa_common SQL databases and
    responds based on a custom ChatPromptTemplate including chat history and scratchpad.
    """
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

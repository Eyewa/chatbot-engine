from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging
import re

from tools.sql_tool import get_live_sql_tools, get_common_sql_tools
from typing import Optional


def _extract_customer_id(query: str) -> Optional[str]:
    """Return customer ID if present in the query."""
    match = re.search(r"customer\s+(\d+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _classify_query(query: str, classifier_chain) -> str:
    """Classify a query using simple heuristics then an LLM chain."""
    q = query.lower()
    if "order" in q and "loyalty" in q:
        return "both"
    try:
        return classifier_chain.invoke({"input": query}).strip().lower()
    except Exception as exc:  # pragma: no cover - LLM errors
        logging.error("Classifier error: %s", exc)
        return "both"

def _build_agent(tools, system_message):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def _create_live_agent():
    tools = get_live_sql_tools()
    return _build_agent(tools, "You are Winkly with access to eyewa_live DB including sales_order, customer_entity, order_meta_data, sales_order_payment.")

def _create_common_agent():
    tools = get_common_sql_tools()
    return _build_agent(tools, "You are Winkly with access to eyewa_common DB including customer_loyalty_card, customer_loyalty_ledger.")

def _combine_responses(resp_live, resp_common):
    parts = []
    if resp_live:
        parts.append(str(getattr(resp_live, "content", resp_live)))
    if resp_common:
        parts.append(str(getattr(resp_common, "content", resp_common)))
    return "\n".join(parts)


def _create_classifier_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
        You are an intent classifier for a chatbot that handles eyewear customer queries.
        Determine the source of data needed for the query:
        - If it concerns order details, customers, order_meta_data, sales_order_payment: respond "live"
        - If it concerns loyalty cards, loyalty ledger: respond "common"
        - If the query relates to both (e.g., orders + loyalty, customer + ledger), respond "both"
        Respond only with: live, common, or both.
        """,
        ),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()

def get_routed_agent():
    live_agent = _create_live_agent()
    common_agent = _create_common_agent()

    classifier_chain = _create_classifier_chain()

    def _classify(input_dict):
        query = input_dict.get("input", "")
        intent = _classify_query(query, classifier_chain)
        logging.info("🏷️ Classifier prediction: %s", intent)
        return intent

    def _handle_both(input_dict):
        logging.info("🔀 Handling BOTH agent path")
        try:
            # Step 1: invoke live agent to extract order + customer data
            live_resp = live_agent.invoke(input_dict)

            # Step 2: invoke common agent with same query
            common_resp = common_agent.invoke(input_dict)
            return _combine_responses(live_resp, common_resp)
        except Exception as e:
            logging.error("_handle_both error: %s", e)
            return "⚠️ There was an error fetching data from both sources."

    router = (
        RunnablePassthrough()
        .assign(intent=RunnableLambda(_classify))
        | RunnableBranch(
            (lambda x: x["intent"] == "live", live_agent),
            (lambda x: x["intent"] == "common", common_agent),
            (lambda x: x["intent"] == "both", RunnableLambda(_handle_both)),
            RunnableLambda(_handle_both),  # fallback
        )
    )

    return router

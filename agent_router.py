# agent_router.py

"""Routing logic for Winkly chatbot.

This module builds separate agents for the ``eyewa_live`` and
``eyewa_common`` databases and routes user queries to the
appropriate agent using an LLM-based intent classifier.
"""

import logging
from typing import Dict, Any

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
from langchain_openai import ChatOpenAI

from tools.sql_tool import get_live_sql_tools, get_common_sql_tools


def _build_agent(tools, system_message: str) -> AgentExecutor:
    """Helper to create a database-specific agent."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def _create_live_agent() -> AgentExecutor:
    tools = get_live_sql_tools()
    system_msg = (
        "You are Winkly, an assistant with access to the 'eyewa_live' database. "
        "Use the provided tools to query tables such as 'sales_order', "
        "'customer_entity', and 'order_meta_data'. "
        "Order information resides in `eyewa_live.sales_order` and should be "
        "queried via `sql_db_query_live`. "
        "Customer IDs correspond to the `entity_id` column in `customer_entity`."
    )
    return _build_agent(tools, system_msg)


def _create_common_agent() -> AgentExecutor:
    tools = get_common_sql_tools()
    system_msg = (
        "You are Winkly, an assistant with access to the 'eyewa_common' database. "
        "Use the provided tools to query tables such as 'customer_loyalty_card', "
        "'sales_order_payment', and 'customer_loyalty_ledger'. "
        "Loyalty card data is stored in `eyewa_common.customer_loyalty_card` "
        "and is keyed by `customer_id`."
    )
    return _build_agent(tools, system_msg)


def _combine_responses(resp_live: Any, resp_common: Any) -> str:
    parts = []
    if resp_live:
        parts.append(str(getattr(resp_live, "content", resp_live)))
    if resp_common:
        parts.append(str(getattr(resp_common, "content", resp_common)))
    return "\n".join(parts)


def _classify_query(query: str, llm_chain) -> str:
    """Return destination database(s) for the given query.

    The function first applies simple heuristics based on keywords and falls
    back to the provided ``llm_chain`` when no direct match is found.
    """
    lowered = query.lower()
    has_order = "order" in lowered
    has_loyalty = any(word in lowered for word in ["loyalty", "loyalty card", "points", "card"])

    if has_order and has_loyalty:
        return "both"
    if has_order:
        return "live"
    if has_loyalty:
        return "common"

    try:
        return llm_chain.invoke({"input": query}).strip().lower()
    except Exception as exc:  # pragma: no cover - log errors
        logging.error("Classifier error: %s", exc)
        return "both"


def get_routed_agent() -> RunnableBranch:
    """Return a runnable router that dispatches to the correct database agent."""
    live_agent = _create_live_agent()
    common_agent = _create_common_agent()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    classifier_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Decide whether the user question relates to the live order database or "
            "the common loyalty database. Respond with one of: 'live', 'common', "
            "or 'both'.",
        ),
        ("user", "{input}"),
    ])
    classifier_chain = classifier_prompt | llm | StrOutputParser()

    def _classify(input_dict: Dict[str, Any]) -> str:
        query = input_dict.get("input", "")
        dest = _classify_query(query, classifier_chain)
        logging.info("🏷️ Classifier prediction: %s", dest)
        return dest

    def _handle_both(input_dict: Dict[str, Any]):
        logging.info("🔀 Handling query across both databases")
        resp_live = None
        resp_common = None
        try:
            resp_live = live_agent.invoke(input_dict)
        except Exception as exc:  # pragma: no cover
            logging.error("Live agent error: %s", exc)
        try:
            resp_common = common_agent.invoke(input_dict)
        except Exception as exc:  # pragma: no cover
            logging.error("Common agent error: %s", exc)
        return _combine_responses(resp_live, resp_common)

    def _handle_unknown(input_dict: Dict[str, Any]):
        logging.warning(
            "⚠️ Unknown intent '%s'. Falling back to both agents", input_dict.get("intent")
        )
        return _handle_both(input_dict)

    router = (
        RunnablePassthrough()
        .assign(intent=RunnableLambda(_classify))
        | RunnableBranch(
            (lambda x: x["intent"] == "live", live_agent),
            (lambda x: x["intent"] == "common", common_agent),
            (lambda x: x["intent"] == "both", RunnableLambda(_handle_both)),
            RunnableLambda(_handle_unknown),
        )
    )
    return router

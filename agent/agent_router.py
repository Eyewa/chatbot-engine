"""Agent router for directing queries to the correct data source."""

import json
import logging
import re
from typing import Optional

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import (
        RunnableBranch,
        RunnableLambda,
        RunnablePassthrough,
    )
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

    class RunnableLambda:
        def __init__(self, func):
            self.func = func

        def invoke(self, input_dict):
            return self.func(input_dict)

    class AgentExecutor:
        def __init__(self, agent=None, tools=None, verbose=False):
            self.agent = agent

        def invoke(self, input_dict):
            if callable(self.agent):
                return self.agent(input_dict)
            return None

    def ChatOpenAI(*args, **kwargs):
        raise ModuleNotFoundError("LangChain not installed")


from tools.sql_toolkit_factory import get_live_sql_tools, get_common_sql_tools
from agent.prompt_builder import PromptBuilder

# -------------------------
# CLASSIFIER & INTENT
# -------------------------


def _extract_customer_id(query: str) -> Optional[str]:
    match = re.search(r"customer\s+(\d+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _classify_query(query: str, classifier_chain) -> str:
    q = query.lower()
    if "order" in q and "loyalty" in q:
        return "both"
    try:
        return classifier_chain.invoke({"input": query}).strip().lower()
    except Exception as exc:
        logging.error("Classifier error: %s", exc)
        return "both"


# -------------------------
# AGENT CONSTRUCTION
# -------------------------


def _build_agent(
    tools, db_key: str, allowed_tables: list[str], max_iterations: int = 5
):
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        db=db_key, allowed_tables=allowed_tables
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=max_iterations
    )


def _create_live_agent():
    tools = get_live_sql_tools()
    allowed = [
        "sales_order",
        "customer_entity",
        "order_meta_data",
        "sales_order_address",
        "sales_order_payment",
    ]
    return _build_agent(tools, db_key="eyewa_live", allowed_tables=allowed)


def _create_common_agent():
    tools = get_common_sql_tools()
    allowed = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    # Limit iterations to minimize repeated invalid SQL retries
    return _build_agent(
        tools, db_key="eyewa_common", allowed_tables=allowed, max_iterations=1
    )


def _combine_responses(resp_live, resp_common):
    """Merge two agent responses with schema validation and namespacing."""

    builder = PromptBuilder()

    # Build allowed field sets per agent based on schema
    def _fields_for_tables(tables: list[str]) -> set[str]:
        fields = set()
        for t in tables:
            fields.update(
                builder.schema_cfg.get("tables", {}).get(t, {}).get("fields", [])
            )
        return fields

    live_tables = [
        "sales_order",
        "customer_entity",
        "order_meta_data",
        "sales_order_address",
        "sales_order_payment",
    ]
    common_tables = [
        "customer_loyalty_card",
        "customer_loyalty_ledger",
        "customer_wallet",
    ]
    live_allowed = _fields_for_tables(live_tables)
    common_allowed = _fields_for_tables(common_tables)

    def _extract_json_blob(text: str) -> str:
        """Return the first valid JSON object found in text."""
        try:
            json.loads(text)
            return text
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return match.group(0)
        return text

    def filter_fields(d: dict, allowed_fields: set[str]) -> dict:
        """Return dictionary with only allowed fields."""
        return {k: v for k, v in d.items() if k in allowed_fields}

    def _parse(resp, allowed_fields: set[str]):
        if not resp:
            return None
        raw = str(getattr(resp, "content", resp))
        cleaned_raw = _extract_json_blob(raw)
        data = builder.translate_freeform(cleaned_raw)
        resp_type = data.get("type")
        payload = data.get("data")
        if payload is None and isinstance(data, dict):
            payload = {k: v for k, v in data.items() if k != "type"}
        if resp_type != "text_response" and isinstance(payload, dict):
            cleaned = filter_fields(payload, allowed_fields)
            dropped = set(payload) - set(cleaned)
            for key in dropped:
                logging.warning("Dropping unexpected key '%s' from %s", key, resp_type)
            return {"type": resp_type, "data": cleaned}
        return data

    live_data = _parse(resp_live, live_allowed)
    common_data = _parse(resp_common, common_allowed)

    merged = {
        "type": "mixed_summary",
        "data": {"orders": None, "loyalty_card": None},
    }

    if live_data and live_data.get("type") != "text_response" and live_data.get("data"):
        merged["data"]["orders"] = live_data["data"]

    if (
        common_data
        and common_data.get("type") != "text_response"
        and common_data.get("data")
    ):
        merged["data"]["loyalty_card"] = common_data["data"]

    if merged["data"]["orders"] is not None or merged["data"]["loyalty_card"] is not None:
        return json.dumps(merged)

    return json.dumps(
        {"type": "text_response", "message": "No data found from either source"}
    )


# -------------------------
# ADVANCED BOTH HANDLER
# -------------------------


def _handle_both(input_dict):
    logging.info("🔀 Handling BOTH agent path")
    input_text = input_dict.get("input", "")
    history = input_dict.get("chat_history", [])

    # Split into scoped prompts
    sub_inputs = {
        "live": {
            "input": input_text + " (only fetch from orders, payments, customers)",
            "chat_history": history,
        },
        "common": {
            "input": input_text + " (only fetch from loyalty, wallet, ledger)",
            "chat_history": history,
        },
    }

    live_resp = None
    common_resp = None

    try:
        live_resp = live_agent.invoke(sub_inputs["live"])
    except Exception as exc:
        logging.error("Live agent error: %s", exc)

    attempts = 0
    cid = _extract_customer_id(input_text)
    fallback_applied = False
    while attempts < 2 and common_resp is None:
        try:
            if attempts == 1 and cid and not fallback_applied:
                sub_inputs["common"]["input"] = (
                    "SELECT clc.card_number FROM customer_loyalty_card clc "
                    "JOIN customer_loyalty_ledger cll ON clc.wallet_id = cll.wallet_id "
                    f"WHERE clc.customer_id = {cid};"
                )
                fallback_applied = True
            common_resp = common_agent.invoke(sub_inputs["common"])
        except Exception as exc:
            attempts += 1
            logging.warning("Common agent failed (attempt %s): %s", attempts, exc)
            if attempts >= 2:
                break

    if not live_resp and not common_resp:
        return "⚠️ There was an error fetching data from both sources."

    return _combine_responses(live_resp, common_resp)


# -------------------------
# ROUTED AGENT
# -------------------------


def _create_classifier_chain():
    if not LANGCHAIN_AVAILABLE:

        class Dummy:
            def invoke(self, payload):
                return "both"

        return Dummy()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
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
        ]
    )
    return prompt | llm | StrOutputParser()


def get_routed_agent():
    global live_agent, common_agent
    live_agent = _create_live_agent()
    common_agent = _create_common_agent()
    classifier_chain = _create_classifier_chain()

    def _classify(input_dict):
        query = input_dict.get("input", "")
        intent = _classify_query(query, classifier_chain)
        logging.info("🏷️ Classifier prediction: %s", intent)
        return intent

    if not LANGCHAIN_AVAILABLE:

        class SimpleRouter:
            def invoke(self, input_dict):
                intent = _classify(input_dict)
                if intent == "live":
                    return live_agent.invoke(input_dict)
                if intent == "common":
                    return common_agent.invoke(input_dict)
                return _handle_both(input_dict)

        return SimpleRouter()

    router = RunnablePassthrough().assign(
        intent=RunnableLambda(_classify)
    ) | RunnableBranch(
        (lambda x: x["intent"] == "live", live_agent),
        (lambda x: x["intent"] == "common", common_agent),
        (lambda x: x["intent"] == "both", RunnableLambda(_handle_both)),
        RunnableLambda(_handle_both),
    )

    return router

"""Agent router for directing queries to the correct data source."""

import json
import logging
import re
from typing import Optional, Any
import os

LANGCHAIN_AVAILABLE = False
try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
    from langchain_core.runnables import (
        RunnableBranch,  # type: ignore
        RunnableLambda,  # type: ignore
        RunnablePassthrough,  # type: ignore
    )
    from langchain_openai import ChatOpenAI  # type: ignore
    from sentence_transformers import SentenceTransformer, util
    LANGCHAIN_AVAILABLE = True
except Exception:
    class RunnableLambda:  # type: ignore
        def __init__(self, func):
            self.func = func

    class AgentExecutor:  # type: ignore
        def __init__(self, agent=None, tools=None, verbose=False):
            self.agent = agent

        def invoke(self, input_dict):
            if callable(self.agent):
                return self.agent(input_dict)
            return None

    def ChatOpenAI(*args, **kwargs):  # type: ignore
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


def load_join_examples(path="config/join_examples.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_relevant_examples(allowed_tables, examples, user_query=None, top_k=2):
    # Try semantic search if possible
    if user_query:
        try:
            query_emb = SentenceTransformer('all-MiniLM-L6-v2').encode(user_query, convert_to_tensor=True)
            example_texts = [ex['description'] + ' ' + ex['example'] for ex in examples]
            example_embs = SentenceTransformer('all-MiniLM-L6-v2').encode(example_texts, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_emb, example_embs)[0]
            top_indices = similarities.argsort(descending=True)[:top_k]
            return [examples[i]['example'] for i in top_indices]
        except Exception as e:
            logging.warning(f"Semantic search failed, falling back to tag match: {e}")
    # Fallback: tag-based match
    scored = []
    allowed = set([t.lower() for t in allowed_tables])
    for ex in examples:
        score = sum(1 for tag in ex["tags"] if tag in allowed)
        if score > 0:
            scored.append((score, ex))
    scored.sort(reverse=True)
    return [ex["example"] for _, ex in scored[:top_k]]


def _build_agent(
    tools, db_key: str, allowed_tables: list[str], max_iterations: int = 5
):
    builder = PromptBuilder()
    # Load and inject only relevant join examples
    join_examples = load_join_examples()
    # Try to get user_query from context if available (for semantic search)
    import inspect
    user_query = None
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        parent_locals = frame.f_back.f_locals
        if parent_locals and 'input_dict' in parent_locals:
            user_query = parent_locals['input_dict'].get('input')
    relevant_examples = find_relevant_examples(allowed_tables, join_examples, user_query=user_query)
    system_prompt = builder.build_system_prompt(
        db=db_key, allowed_tables=allowed_tables, extra_examples=relevant_examples
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
        agent=agent, tools=tools, verbose=True
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


def clean_agent_output(output):
    if not output:
        return None
    # Remove triple backticks and language tag
    if isinstance(output, str):
        match = re.search(r"```(?:json)?\\n?(.*?)```", output, re.DOTALL)
        if match:
            output = match.group(1).strip()
        # Try to parse as JSON
        try:
            return json.loads(output)
        except Exception:
            pass
    # If already dict/list, return as is
    if isinstance(output, (dict, list)):
        return output
    return output


def deep_clean_json_blocks(obj):
    # Recursively clean and parse all string values, including repeated code block unwrapping
    if isinstance(obj, str):
        prev = obj
        cleaned = clean_agent_output(prev)
        # If cleaning returns a string, try again (handle nested code blocks)
        while isinstance(cleaned, str) and cleaned != prev:
            prev = cleaned
            cleaned = clean_agent_output(prev)
        return cleaned
    elif isinstance(obj, dict):
        return {k: deep_clean_json_blocks(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_json_blocks(v) for v in obj]
    else:
        return obj


def _combine_responses(resp_live, resp_common):
    """Robustly merge two agent responses. Return any real data, or a generic message if both are empty."""
    builder = PromptBuilder()

    def _parse(resp):
        if not resp:
            return None
        # If it's a dict with 'output', clean it
        if isinstance(resp, dict) and 'output' in resp:
            return clean_agent_output(resp['output'])
        # If it's a string, try to clean and parse
        if isinstance(resp, str):
            return clean_agent_output(resp)
        # If already dict/list, return as is
        if isinstance(resp, (dict, list)):
            return resp
        return None

    live_data = _parse(resp_live)
    common_data = _parse(resp_common)

    # If both are empty or None
    if not live_data and not common_data:
        return {"type": "text_response", "message": "No data found from either source"}
    # If only live has data
    if live_data and not common_data:
        return deep_clean_json_blocks(live_data)
    # If only common has data
    if common_data and not live_data:
        return deep_clean_json_blocks(common_data)
    # If both have data, return both in a dict
    merged = {"live": live_data, "common": common_data}
    return deep_clean_json_blocks(merged)


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

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
        intent=RunnableLambda(_classify)  # type: ignore
    ) | RunnableBranch(
        (lambda x: x["intent"] == "live", live_agent),  # type: ignore
        (lambda x: x["intent"] == "common", common_agent),  # type: ignore
        (lambda x: x["intent"] == "both", RunnableLambda(_handle_both)),  # type: ignore
        RunnableLambda(_handle_both),  # type: ignore  # default branch, not a tuple!
    )  # type: ignore

    return router

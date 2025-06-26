"""Agent router for directing queries to the correct data source."""

import json
import logging
import re
from typing import Optional, Any
import os

logging.basicConfig(level=logging.DEBUG)

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
from agent.config_loader import config_loader
from agent.utils import filter_response_by_type, extract_last_n, generate_llm_message

# -------------------------
# CLASSIFIER & INTENT
# -------------------------


def _extract_customer_id(query: str) -> Optional[str]:
    match = re.search(r"customer\s+(\d+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _classify_query(query: str, classifier_chain) -> str:
    q = query.lower()
    
    # Get configuration-driven classification rules
    live_only_keywords = config_loader.get_keywords("live_only")
    common_only_keywords = config_loader.get_keywords("common_only")
    
    # Check if query contains keywords for both databases
    has_live_keywords = any(keyword in q for keyword in live_only_keywords)
    has_common_keywords = any(keyword in q for keyword in common_only_keywords)
    
    if has_live_keywords and has_common_keywords:
        return "both"
    
    # Default to classifier chain
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

    logging.info(f"[AGENT] About to build agent for db={db_key}")
    logging.info(f"[AGENT] Allowed tables: {allowed_tables}")
    logging.info(f"[AGENT] System prompt to LLM:\n{system_prompt}")

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

    def logging_wrapper(input, config=None):
        logging.info(f"[AGENT] Invoking agent with input: {input}")
        result = agent.invoke(input, config)
        logging.info(f"[AGENT] Raw LLM output: {result}")
        try:
            import json
            parsed = None
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = result
            else:
                parsed = result
            logging.info(f"[AGENT] Parsed LLM output: {parsed}")
        except Exception as e:
            logging.warning(f"[AGENT] Could not parse LLM output: {e}")
        return result

    wrapped_agent = RunnableLambda(logging_wrapper)

    return AgentExecutor(
        agent=wrapped_agent, tools=tools, verbose=True
    )


def _create_live_agent():
    tools = get_live_sql_tools()
    allowed = config_loader.get_database_tables("live")
    logging.info(f"[AGENT] Loaded allowed tables from config for eyewa_live: {allowed}")
    if not allowed:
        logging.error("[AGENT] No allowed tables found for eyewa_live in config! LLM will hallucinate.")
    return _build_agent(tools, db_key="eyewa_live", allowed_tables=allowed)


def _create_common_agent():
    tools = get_common_sql_tools()
    # Use only tables that exist in eyewa_common
    allowed = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    # Build a strict prompt
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        db="eyewa_common",
        allowed_tables=allowed,
        extra_examples=None
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
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


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
    logging.debug(f"[_combine_responses] resp_live: {resp_live}, resp_common: {resp_common}")
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
    logging.debug(f"[_combine_responses] Parsed live_data: {live_data}, common_data: {common_data}")

    cleaned_live = deep_clean_json_blocks(live_data)
    if isinstance(cleaned_live, dict):
        cleaned_live = filter_response_by_type(cleaned_live)
    cleaned_common = deep_clean_json_blocks(common_data)
    if isinstance(cleaned_common, dict):
        # --- GENERIC PATCH: flatten any list of dicts for allowed fields ---
        summary_type = cleaned_common.get("type")
        allowed_fields = set()
        if summary_type:
            try:
                from simple_yaml import safe_load
                import os
                schema_path = os.path.join("config", "templates", "response_types.yaml")
                with open(schema_path) as f:
                    response_types = safe_load(f)
                allowed_fields = set(response_types.get(summary_type, {}).get("fields", []))
            except Exception as e:
                logging.warning(f"[GENERIC FLATTEN] Could not load response_types.yaml: {e}")
        # For each key in cleaned_common, if it's a list of dicts, flatten the first dict's allowed fields
        for key, value in list(cleaned_common.items()):
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for field in allowed_fields:
                    if field in value[0]:
                        cleaned_common[field] = value[0][field]
                cleaned_common.pop(key, None)
        cleaned_common = filter_response_by_type(cleaned_common)

    # If both are empty or None
    if not cleaned_live and not cleaned_common:
        logging.info("[_combine_responses] No data from either source.")
        return {"type": "text_response", "message": "No data found from either source"}
    # If only live has data
    if cleaned_live and not cleaned_common:
        return cleaned_live
    # If only common has data
    if cleaned_common and not cleaned_live:
        return cleaned_common
    # If both have data, return both in a dict
    merged = {
        "live": cleaned_live,
        "common": cleaned_common,
    }
    logging.info(f"[_combine_responses] Returning merged: {merged}")
    return merged


# -------------------------
# ADVANCED BOTH HANDLER
# -------------------------


def is_structured_response(resp, expected_types=None):
    """
    Returns True if resp is a dict with a known type and at least one required field present.
    expected_types: list of allowed types (from response_types.yaml), or None for all.
    """
    if not isinstance(resp, dict):
        return False
    t = resp.get("type")
    if not t:
        return False
    if expected_types and t not in expected_types:
        return False
    from main import RESPONSE_TYPES  # avoid circular import at top
    allowed_fields = RESPONSE_TYPES.get(t, {}).get("fields", [])
    return any(resp.get(f) is not None for f in allowed_fields)


def extract_focused_prompt(user_query, db):
    """
    Extract the part of the user query relevant to the given db (live/common).
    Use regex to extract loyalty/wallet/ledger for common, order/payment/customer for live.
    If no relevant phrase is found, return None.
    """
    customer_id_match = re.findall(r'customer\s*(\d+)', user_query)
    customer_id_str = f" of customer {''.join(customer_id_match)}" if customer_id_match else ""
    if db == "common":
        m = re.search(r"(loyalty card|wallet|ledger|points|balance|expiry|loyalty)", user_query, re.I)
        if m:
            return m.group(0) + customer_id_str
        return None
    else:
        m = re.search(r"(order|payment|customer|address|shipping|billing)", user_query, re.I)
        if m:
            return m.group(0) + customer_id_str
        return None


def inject_limit_phrase(prompt, limit):
    import re
    # Only inject if not already present
    if not re.search(r"(last|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(order|orders|record|records)", prompt, re.IGNORECASE):
        # Try to keep the rest of the prompt, just prepend
        if "order" in prompt:
            # If 'of' is present, keep the rest
            return f"last {limit} orders {prompt[prompt.find('of'):].strip()}" if "of" in prompt else f"last {limit} orders {prompt}"
        else:
            return f"last {limit} records {prompt}"
    return prompt


def _handle_both(input_dict):
    logging.info(f"[_handle_both] input_dict: {input_dict}")
    logging.info("🔀 Handling BOTH agent path")
    input_text = input_dict.get("input", "")
    history = input_dict.get("chat_history", [])

    # Pre-process for 'last N' or 'recent N'
    limit_n = extract_last_n(input_text)

    # Extract focused prompts for each DB
    live_prompt = extract_focused_prompt(input_text, db="live")
    common_prompt = extract_focused_prompt(input_text, db="common")

    if not live_prompt and not common_prompt:
        return {"type": "text_response", "message": "Could not determine which data to fetch for either database. Please rephrase your query."}
    if not live_prompt:
        return {"type": "text_response", "message": "Could not determine which live (order/customer) data to fetch. Please rephrase your query."}
    if not common_prompt:
        return {"type": "text_response", "message": "Could not determine which common (loyalty/wallet/ledger) data to fetch. Please rephrase your query."}

    sub_inputs = {
        "live": {
            "input": live_prompt,
            "chat_history": history,
        },
        "common": {
            "input": common_prompt,
            "chat_history": history,
        },
    }
    if limit_n:
        sub_inputs["live"]["limit"] = limit_n
        sub_inputs["common"]["limit"] = limit_n
        # Rewrite input to include 'last N orders' if not already present
        sub_inputs["live"]["input"] = inject_limit_phrase(sub_inputs["live"]["input"], limit_n)
        sub_inputs["common"]["input"] = inject_limit_phrase(sub_inputs["common"]["input"], limit_n)

    try:
        live_resp = live_agent.invoke(sub_inputs["live"])
    except Exception as exc:
        logging.error("Live agent error: %s", exc)
        live_resp = None
    try:
        common_resp = common_agent.invoke(sub_inputs["common"])
    except Exception as exc:
        logging.error("Common agent error: %s", exc)
        common_resp = None

    if not live_resp and not common_resp:
        logging.error("[_handle_both] Error fetching data from both sources.")
        return {"type": "text_response", "message": "There was an error fetching data from both sources."}

    result = _combine_responses(live_resp, common_resp)

    # If both live and common are present, merge for summary and data
    if isinstance(result, dict) and 'live' in result and 'common' in result:
        data = []
        if isinstance(result['live'], dict):
            data.append(result['live'])
        if isinstance(result['common'], dict):
            data.append(result['common'])
        # Generate summary for both
        summary = generate_llm_message(data, ChatOpenAI(model="gpt-4o", temperature=0))
        return {"message": summary, "data": data}
    else:
        # Handle single branch as before
        if isinstance(result, dict) and 'type' in result:
            summary = generate_llm_message([result], ChatOpenAI(model="gpt-4o", temperature=0))
            return {"message": summary, "data": [result]}
        return result


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

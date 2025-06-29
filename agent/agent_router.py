"""Agent router for directing queries to the correct data source."""

import json
import logging
import re
from typing import Optional, Any
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import yaml

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
from agent.dynamic_sql_builder import load_schema, build_dynamic_sql, get_field_alias

# Global agent instances
live_agent = None
common_agent = None

# -------------------------
# CLASSIFIER & INTENT
# -------------------------


def _extract_customer_id(query: str) -> Optional[str]:
    match = re.search(r"customer\s+(\d+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _classify_query(query: str, classifier_chain) -> str:
    """Classify the intent of a query."""
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
        result = classifier_chain.invoke({"input": query})
        
        return str(result).strip().lower()
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
    if not LANGCHAIN_AVAILABLE:
        def dummy_agent(input_dict):
            return {"type": "text_response", "message": "LangChain not available"}
        return dummy_agent

    from agent.prompt_builder import PromptBuilder
    from langchain.agents import create_openai_functions_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    # Enforce strict JSON output
    system_prompt += "\n\nALWAYS respond with a valid JSON object. Do NOT include any text, explanation or markdown code blocks. Only output the JSON."

    logging.info(f"[AGENT] About to build agent for db={db_key}")
    logging.info(f"[AGENT] Allowed tables: {allowed_tables}")
    logging.info(f"[AGENT] System prompt to LLM (JSON enforcement={'ALWAYS respond with a valid JSON object' in system_prompt}):\n{system_prompt}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    logging.debug(f"[AGENT] Full prompt object: {prompt}")

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def _create_live_agent():
    tools = get_live_sql_tools()
    allowed = config_loader.get_database_tables("live")
    logging.info(f"[AGENT] Loaded allowed tables from config for eyewa_live: {allowed}")
    if not allowed:
        logging.error("[AGENT] No allowed tables found for eyewa_live in config! LLM will hallucinate.")
    return _build_agent(tools, db_key="eyewa_live", allowed_tables=allowed)


def _create_common_agent():
    tools = get_common_sql_tools()
    # Load allowed tables from schema
    schema_path = os.path.join("config", "schema", "schema.yaml")
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        allowed = list(schema.get('common', {}).get('tables', {}).keys())
    except Exception as e:
        logging.error(f"Could not load schema config: {e}")
        # Fallback to hardcoded list
        allowed = ["customer_loyalty_card", "customer_loyalty_ledger", "customer_wallet"]
    
    # Build a strict prompt
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        db="eyewa_common",
        allowed_tables=allowed,
        extra_examples=None
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3)
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
    """
    Recursively finds and parses JSON blocks from strings.
    Handles nested structures and repeated markdown fences.
    """
    if isinstance(obj, str):
        # This regex finds a JSON object or array inside markdown code fences.
        match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", obj, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                # Once we've found a JSON block, parse it and recursively clean it.
                return deep_clean_json_blocks(json.loads(json_str))
            except json.JSONDecodeError:
                # If parsing fails, just return the extracted string.
                return json_str
        # If no markdown block is found, try to parse the string as JSON directly.
        # This handles cases where the string is just a JSON object without fences.
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            # If all else fails, return the original string.
            return obj
    elif isinstance(obj, dict):
        return {k: deep_clean_json_blocks(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_json_blocks(v) for v in obj]
    else:
        return obj


def _combine_responses(resp_live, resp_common, user_query=None):
    response_types = load_response_types()
    schema = load_schema()
    combined_responses = []

    def _parse(resp):
        if not resp:
            return None
        # If it's a dict with 'output', clean it
        if isinstance(resp, dict) and 'output' in resp:
            return deep_clean_json_blocks(resp['output'])
        # If it's a string, try to clean and parse
        if isinstance(resp, str):
            return deep_clean_json_blocks(resp)
        # If already dict/list, apply deep clean
        if isinstance(resp, (dict, list)):
            return deep_clean_json_blocks(resp)
        return None

    live_data = _parse(resp_live)
    common_data = _parse(resp_common)

    # Helper to process a single response dict or list
    def process_response(resp):
        if isinstance(resp, dict) and "type" in resp:
            summary_type = resp["type"]
            if summary_type in response_types:
                return build_summary(summary_type, resp, schema, response_types)
        elif isinstance(resp, list):
            return [build_summary(item["type"], item, schema, response_types)
                    for item in resp if isinstance(item, dict) and "type" in item and item["type"] in response_types]
        return None

    # Process both live and common responses
    live_summary = process_response(live_data)
    common_summary = process_response(common_data)

    # Combine results
    if live_summary:
        if isinstance(live_summary, list):
            combined_responses.extend(live_summary)
        else:
            combined_responses.append(live_summary)
    if common_summary:
        if isinstance(common_summary, list):
            combined_responses.extend(common_summary)
        else:
            combined_responses.append(common_summary)

    # --- Always include customer_summary if customer_id is present in the query ---
    if user_query is not None:
        import re
        match = re.search(r"customer\s*(\d+)", user_query, re.IGNORECASE)
        customer_id = match.group(1) if match else None
        already_present = any(r.get("type") == "customer_summary" for r in combined_responses if isinstance(r, dict))
        if customer_id and not already_present:
            # Fetch customer info from DB (config-driven fields)
            customer_fields = response_types.get("customer_summary", {}).get("fields", [])
            if customer_fields:
                # Build SQL dynamically from schema
                table = "customer_entity"
                select_fields = []
                for f in customer_fields:
                    if "field_mappings" in schema and f in schema["field_mappings"]:
                        mapping = schema["field_mappings"][f]
                        select_fields.extend(mapping.get("source_fields", []))
                    else:
                        select_fields.append(f)
                select_fields = list(set(select_fields))
                sql = f"SELECT {', '.join(select_fields)} FROM {table} WHERE entity_id = {customer_id}"
                rows = run_sql_query(sql, db="live")
                if rows:
                    customer_data = rows[0]
                    summary = build_summary("customer_summary", customer_data, schema, response_types)
                    # Add customer_id to summary for clarity
                    summary["customer_id"] = str(customer_id)
                    combined_responses.append(summary)

    if not combined_responses:
        # Fallback: no data from either source
        config_path = os.path.join("config", "query_routing.yaml")
        try:
            with open(config_path, 'r') as f:
                routing_config = yaml.safe_load(f)
            merge_strategies = routing_config.get('response_combination', {}).get('merge_strategies', {})
        except Exception as e:
            merge_strategies = {}
        return {"type": merge_strategies.get("default", "text_response"), "message": "No data found from either source"}

    return combined_responses


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
    
    # Load response types from config instead of circular import
    try:
        response_types = load_response_types()
        allowed_fields = response_types.get(t, {}).get("fields", [])
        return any(resp.get(f) is not None for f in allowed_fields)
    except Exception as e:
        logging.warning(f"Could not load response types for validation: {e}")
        return True  # Fallback to accepting any structured response


def extract_focused_prompt(user_query, db):
    """
    Extract the part of the user query relevant to the given db (live/common) using config-driven rules.
    """
    config_path = os.path.join("config", "query_routing.yaml")
    try:
        with open(config_path, 'r') as f:
            routing_config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Could not load query routing config: {e}")
        return None
    
    customer_id_match = re.findall(r'customer\s*(\d+)', user_query)
    customer_id = ''.join(customer_id_match) if customer_id_match else ""
    logging.info(f"[extract_focused_prompt] db={db}, customer_id='{customer_id}'")
    
    classification_rules = routing_config.get('classification_rules', {})
    keywords = classification_rules.get('keywords', {})
    prompt_templates = routing_config.get('prompt_templates', {})
    
    if db == "common":
        # Check for common database keywords
        common_keywords = keywords.get('common_only', [])
        for keyword in common_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', user_query, re.I):
                # Use config-driven template
                template = prompt_templates.get('common', {}).get(keyword, f"show {keyword} information for customer {{customer_id}}")
                result = template.format(customer_id=customer_id)
                
                logging.info(f"[extract_focused_prompt] common match: '{keyword}' -> '{result}'")
                return result
        
        logging.info(f"[extract_focused_prompt] common: no match found")
        return None
    else:
        # Check for live database keywords
        live_keywords = keywords.get('live_only', [])
        for keyword in live_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', user_query, re.I):
                if keyword == "order":
                    # Extract limit if present
                    limit_match = re.search(r"(last|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)", user_query, re.I)
                    limit_str = ""
                    if limit_match:
                        n_str = limit_match.group(2).lower()
                        n = int(n_str) if n_str.isdigit() else {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}.get(n_str, 5)
                        limit_str = f"last {n} "
                    
                    # Use config-driven field extraction
                    field_extraction = routing_config.get('field_extraction', {})
                    orders_fields = field_extraction.get('orders_summary', {})
                    
                    fields = []
                    for field_name, field_keywords in orders_fields.items():
                        for keyword_check in field_keywords:
                            if keyword_check in user_query.lower():
                                fields.append(field_name)
                                break
                    
                    fields_str = ""
                    if fields:
                        fields_str = f" with {', '.join(fields)}"
                    
                    # Use config-driven template
                    template = prompt_templates.get('live', {}).get(keyword, f"show {{limit}}orders{{fields}} for customer {{customer_id}}")
                    result = template.format(limit=limit_str, fields=fields_str, customer_id=customer_id)
                else:
                    # Use config-driven template for other keywords
                    template = prompt_templates.get('live', {}).get(keyword, f"show {keyword} information for customer {{customer_id}}")
                    result = template.format(customer_id=customer_id)
                
                logging.info(f"[extract_focused_prompt] live match: '{keyword}' -> '{result}'")
                return result
        
        logging.info(f"[extract_focused_prompt] live: no match found")
        return None


def inject_limit_phrase(prompt, limit):
    import re
    logging.info(f"[inject_limit_phrase] Input prompt: '{prompt}', limit: {limit}")
    # Only inject if not already present
    if not re.search(r"(last|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(order|orders|record|records)", prompt, re.IGNORECASE):
        # Try to keep the rest of the prompt, just prepend
        if "order" in prompt:
            # If 'of' is present, keep the rest
            result = f"last {limit} orders {prompt[prompt.find('of'):].strip()}" if "of" in prompt else f"last {limit} orders {prompt}"
        else:
            result = f"last {limit} records {prompt}"
        logging.info(f"[inject_limit_phrase] Modified prompt: '{result}'")
        return result
    logging.info(f"[inject_limit_phrase] No modification needed, returning: '{prompt}'")
    return prompt


def get_engine(db_key: str) -> Engine:
    # Map db_key to your real DB URIs
    db_uris = {
        "live": "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_live",
        "common": "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_common"
    }
    uri = db_uris.get(db_key)
    if not uri:
        raise ValueError(f"Unknown db_key: {db_key}")
    return create_engine(uri)


def run_sql_query(sql: str, db: str = "live") -> list:
    logging.info(f"[DB] Executing on db={db}: {sql}")
    engine = get_engine(db)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            logging.info(f"[DB] Query returned {len(rows)} rows.")
            return rows
    except Exception as exc:
        logging.error(f"[DB ERROR] {exc}")
        return []


def _handle_both(input_dict):
    logging.info(f"[_handle_both] input_dict: {input_dict}")
    logging.info("🔀 Handling BOTH agent path")
    input_text = input_dict.get("input", "")
    history = input_dict.get("chat_history", [])

    # Pre-process for 'last N' or 'recent N'
    limit_n = extract_last_n(input_text)
    logging.info(f"[_handle_both] Extracted limit_n: {limit_n}")

    # Extract focused prompts for each DB
    live_prompt = extract_focused_prompt(input_text, db="live")
    common_prompt = extract_focused_prompt(input_text, db="common")
    logging.info(f"[_handle_both] Extracted live_prompt: '{live_prompt}'")
    logging.info(f"[_handle_both] Extracted common_prompt: '{common_prompt}'")

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
        # Only inject limit phrase if not already present in the prompt
        if "last" not in sub_inputs["live"]["input"] and "recent" not in sub_inputs["live"]["input"]:
            sub_inputs["live"]["input"] = inject_limit_phrase(sub_inputs["live"]["input"], limit_n)
        if "last" not in sub_inputs["common"]["input"] and "recent" not in sub_inputs["common"]["input"]:
            sub_inputs["common"]["input"] = inject_limit_phrase(sub_inputs["common"]["input"], limit_n)
    
    logging.info(f"[_handle_both] Final sub_inputs: {sub_inputs}")

    try:
        if live_agent is not None:
            live_resp = live_agent.invoke(sub_inputs["live"])
        else:
            live_resp = {"type": "text_response", "message": "Live agent is not available."}
        logging.info(f"[_handle_both] Live agent response: {live_resp}")
    except Exception as exc:
        logging.error("Live agent error: %s", exc)
        live_resp = None
    try:
        if common_agent is not None:
            common_resp = common_agent.invoke(sub_inputs["common"])
        else:
            common_resp = {"type": "text_response", "message": "Common agent is not available."}
        logging.info(f"[_handle_both] Common agent response: {common_resp}")
    except Exception as exc:
        logging.error("Common agent error: %s", exc)
        common_resp = None

    if not live_resp and not common_resp:
        logging.error("[_handle_both] Error fetching data from both sources.")
        return {"type": "text_response", "message": "There was an error fetching data from both sources."}

    combined = _combine_responses(live_resp, common_resp, input_text)

    # Handle the case where combined is None (no mixed response created)
    if combined is None:
        # Fallback to returning the first available response
        if live_resp:
            final_response = live_resp
        elif common_resp:
            final_response = common_resp
        else:
            final_response = {"type": "text_response", "message": "No data found from either source"}
        logging.info(f"[_handle_both] Fallback response: {final_response}")
        return final_response
    else:
        # combined is already the final merged response
        final_response = combined

    logging.info(f"[_handle_both] Final data list for output: {final_response}")
    return final_response


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
                    if live_agent is not None:
                        return live_agent.invoke(input_dict)
                    else:
                        return {"type": "text_response", "message": "Live agent is not available."}
                if intent == "common":
                    if common_agent is not None:
                        return common_agent.invoke(input_dict)
                    else:
                        return {"type": "text_response", "message": "Common agent is not available."}
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

def safe_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_response_types():
    with open(os.path.join("config", "templates", "response_types.yaml")) as f:
        return yaml.safe_load(f)

def load_schema():
    with open(os.path.join("config", "schema", "schema.yaml")) as f:
        return yaml.safe_load(f)

def build_summary(summary_type, data, schema, response_types):
    summary_config = response_types.get(summary_type, {})
    fields = summary_config.get("fields", [])
    summary = {"type": summary_type}
    for field in fields:
        if "field_mappings" in schema and field in schema["field_mappings"]:
            mapping = schema["field_mappings"][field]
            summary[field] = apply_field_mapping(mapping, data)
        else:
            summary[field] = data.get(field)
    return summary

def apply_field_mapping(mapping, data):
    if "transformation" in mapping and "source_fields" in mapping:
        if mapping["transformation"].startswith("CONCAT"):
            parts = [data.get(sf, "") for sf in mapping["source_fields"]]
            return " ".join(parts).strip()
        else:
            return data.get(mapping["source_fields"][0])
    return None

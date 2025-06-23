from __future__ import annotations
import re
from typing import Dict, List

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
except Exception:
    ChatOpenAI = None  # type: ignore
    Runnable = None  # type: ignore


INTENT_HIERARCHY: Dict[str, List[str]] = {
    "order": ["recent_orders", "order_amount", "order_status", "order_items", "order_date"],
    "payment": ["amount_paid", "payment_method", "payment_status"],
    "loyalty": ["card_number", "balance", "ledger_history", "expiry"],
    "customer_profile": ["name", "email", "mobile", "id"],
    "address": ["shipping_address", "billing_address", "city", "phone"],
}


_RULES: Dict[str, Dict[str, str]] = {
    "order": {
        "recent_orders": r"\b(last|recent)\b.*order",
        "order_amount": r"\b(amount|total)\b",
        "order_status": r"\bstatus\b",
        "order_items": r"\bitem",
        "order_date": r"\bdate\b",
    },
    "payment": {
        "amount_paid": r"\b(paid|payment amount)\b",
        "payment_method": r"\b(method|card|visa|paypal)\b",
        "payment_status": r"\bpayment status\b",
    },
    "loyalty": {
        "card_number": r"card number",
        "balance": r"(loyalty balance|points)",
        "ledger_history": r"ledger|history",
        "expiry": r"expiry|expiration",
    },
    "customer_profile": {
        "name": r"name",
        "email": r"email",
        "mobile": r"mobile|phone",
        "id": r"customer\s+\d+|customer id",
    },
    "address": {
        "shipping_address": r"shipping",
        "billing_address": r"billing",
        "city": r"city",
        "phone": r"phone",
    },
}


def _rule_based(query: str) -> Dict[str, List[str]]:
    q = query.lower()
    result: Dict[str, List[str]] = {}
    for intent, rules in _RULES.items():
        subs = []
        for sub, pattern in rules.items():
            if re.search(pattern, q):
                subs.append(sub)
        if subs:
            result[intent] = subs
    return result


def classify(query: str) -> Dict[str, object]:
    """Return detected intents and sub-intents for a query."""
    if ChatOpenAI is not None:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a classifier. Given a user question, identify intents and sub-intents as JSON",
                    ),
                    ("user", "{input}"),
                ]
            )
            parser = JsonOutputParser()
            chain: Runnable = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | parser
            resp = chain.invoke({"input": query})
            if isinstance(resp, dict) and "intent" in resp:
                return resp
        except Exception:
            pass

    mapping = _rule_based(query)
    intents = list(mapping)
    return {"intent": intents, "sub_intents": mapping}

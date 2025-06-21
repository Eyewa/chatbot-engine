import types
from unittest import mock

import pytest

import agent_router


class DummyChain:
    def __init__(self, response=None, raise_exc=False):
        self.response = response or "live"
        self.raise_exc = raise_exc

    def invoke(self, payload):
        if self.raise_exc:
            raise RuntimeError("llm error")
        return self.response


def test_classify_heuristic_both():
    chain = DummyChain("live")
    query = "show order amount and loyalty card"
    assert agent_router._classify_query(query, chain) == "both"


def test_classify_llm_only():
    chain = DummyChain("common")
    query = "show loyalty card"
    assert agent_router._classify_query(query, chain) == "common"


def test_classify_fallback_error():
    chain = DummyChain(raise_exc=True)
    query = "anything"
    assert agent_router._classify_query(query, chain) == "both"


def test_router_branches(monkeypatch):
    calls = []

    def fake_live(input_dict):
        calls.append("live")
        return "live"

    def fake_common(input_dict):
        calls.append("common")
        return "common"

    monkeypatch.setattr(agent_router, "_create_live_agent", lambda: agent_router.RunnableLambda(fake_live))
    monkeypatch.setattr(agent_router, "_create_common_agent", lambda: agent_router.RunnableLambda(fake_common))
    monkeypatch.setattr(agent_router, "_classify_query", lambda q, c: "both")

    router = agent_router.get_routed_agent()
    router.invoke({"input": "dummy"})

    assert "live" in calls and "common" in calls


def test_extract_customer_id():
    q = "his loyalty card of customer 12345"
    assert agent_router._extract_customer_id(q) == "12345"


def test_extract_customer_id_ignores_order():
    q = "show order 98765"
    assert agent_router._extract_customer_id(q) is None


def test_handle_both_augments_query(monkeypatch):
    sent = []

    def fake_live(input_dict):
        return "live"

    def fake_common(input_dict):
        sent.append(input_dict["input"])
        return "common"

    monkeypatch.setattr(agent_router, "_create_live_agent", lambda: agent_router.RunnableLambda(fake_live))
    monkeypatch.setattr(agent_router, "_create_common_agent", lambda: agent_router.RunnableLambda(fake_common))
    monkeypatch.setattr(agent_router, "_classify_query", lambda q, c: "both")

    router = agent_router.get_routed_agent()
    router.invoke({"input": "show orders for customer 42", "chat_history": []})

    assert any("customer 42" in s for s in sent)

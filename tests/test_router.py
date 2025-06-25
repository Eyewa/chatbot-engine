import json

import agent.agent_router as agent_router


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

    monkeypatch.setattr(
        agent_router,
        "_create_live_agent",
        lambda: agent_router.RunnableLambda(fake_live),
    )
    monkeypatch.setattr(
        agent_router,
        "_create_common_agent",
        lambda: agent_router.RunnableLambda(fake_common),
    )
    monkeypatch.setattr(agent_router, "_classify_query", lambda q, c: "both")

    router = agent_router.get_routed_agent()
    router.invoke({"input": "show order amount and loyalty card for customer 12345"})

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

    monkeypatch.setattr(
        agent_router,
        "_create_live_agent",
        lambda: agent_router.RunnableLambda(fake_live),
    )
    monkeypatch.setattr(
        agent_router,
        "_create_common_agent",
        lambda: agent_router.RunnableLambda(fake_common),
    )
    monkeypatch.setattr(agent_router, "_classify_query", lambda q, c: "both")

    # Input contains both live and common keywords
    router = agent_router.get_routed_agent()
    router.invoke({"input": "show orders and loyalty card for customer 42", "chat_history": []})

    assert any("customer 42" in s for s in sent)


def test_combine_responses_filters_and_merges():
    # Only allowed fields will be present in the output per response_types.yaml
    live = '{"type": "orders_summary", "orders": [{"order_id": 1, "grand_total": 100}]}'
    common = '{"type": "loyalty_summary", "card_number": "777", "first_name": "John", "activation_date": "2023-01-01", "expiry_date": "2024-01-01", "status": "active"}'

    out = agent_router._combine_responses(live, common)
    result = out
    assert result is not None
    assert isinstance(result, dict)
    assert "live" in result and "common" in result
    assert isinstance(result["live"], dict)
    assert result["live"]["type"] == "orders_summary"
    assert "orders" in result["live"]
    assert result["live"]["orders"] == [{"order_id": 1, "grand_total": 100}]
    assert isinstance(result["common"], dict)
    assert result["common"]["type"] == "loyalty_summary"
    assert str(result["common"]["card_number"]) == "777"
    assert result["common"]["activation_date"] == "2023-01-01"
    assert result["common"]["expiry_date"] == "2024-01-01"
    assert result["common"]["status"] == "active"
    # 'first_name' is not in allowed fields, so it should not be present
    assert "first_name" not in result["common"]


def test_combine_responses_single_agent():
    live = '{"type": "orders_summary", "orders": [{"grand_total": 50}]}'

    out = agent_router._combine_responses(live, None)
    result = out
    assert result is not None
    assert isinstance(result, dict)
    assert result["type"] == "orders_summary"
    assert result["orders"] == [{"grand_total": 50}]


def test_combine_responses_no_data():
    out = agent_router._combine_responses(None, None)
    result = out
    assert result == {
        "type": "text_response",
        "message": "No data found from either source"
    }

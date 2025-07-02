
import agent.core.agent_router as agent_router


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
    router.invoke(
        {"input": "show orders and loyalty card for customer 42", "chat_history": []}
    )

    assert any("customer 42" in s for s in sent)


def test_combine_responses_filters_and_merges():
    # Only allowed fields will be present in the output per response_types.yaml
    live = '{"type": "orders_summary", "orders": [{"order_id": 1, "grand_total": 100}]}'
    common = '{"type": "loyalty_summary", "card_number": "777", "first_name": "John", "activation_date": "2023-01-01", "expiry_date": "2024-01-01", "status": "active"}'

    out = agent_router._combine_responses(live, common)
    result = out
    assert result is not None
    assert isinstance(result, list)
    assert any(r.get("type") == "orders_summary" for r in result)
    assert any(r.get("type") == "loyalty_summary" for r in result)
    # Optionally, check the fields in each summary
    orders = next(r for r in result if r.get("type") == "orders_summary")
    loyalty = next(r for r in result if r.get("type") == "loyalty_summary")
    assert "orders" in orders
    assert "card_number" in loyalty
    assert loyalty["card_number"] == "777"


def test_combine_responses_with_loyalty_cards():
    # Test that loyalty card information from loyalty_cards array is properly extracted
    live = '{"type": "orders_summary", "orders": []}'
    common = """
    {
        "type": "loyalty_summary",
        "loyalty_cards": [{
            "card_number": "000764054",
            "status": "ACTIVE",
            "customer_id": "2555880",
            "metadata": {
                "loyaltyProgram": "KSALP",
                "programTier": "KSALP"
            }
        }]
    }
    """

    out = agent_router._combine_responses(live, common)
    result = out
    assert result is not None
    assert isinstance(result, list)

    # Verify loyalty summary is present and has correct data
    loyalty_summaries = [r for r in result if r.get("type") == "loyalty_summary"]
    assert len(loyalty_summaries) == 1
    loyalty = loyalty_summaries[0]

    # Check that the loyalty card information was properly extracted
    assert loyalty["card_number"] == "000764054"
    assert loyalty["status"] == "ACTIVE"
    assert "points_balance" in loyalty  # Should be present but None
    assert "expiry_date" in loyalty  # Should be present but None


def test_combine_responses_single_agent():
    live = '{"type": "orders_summary", "orders": [{"grand_total": 50}]}'

    out = agent_router._combine_responses(live, None)
    result = out
    assert result is not None
    assert isinstance(result, list)
    assert result[0]["type"] == "orders_summary"
    assert result[0]["orders"] == [{"grand_total": 50}]


def test_combine_responses_no_data():
    out = agent_router._combine_responses(None, None)
    result = out
    assert result == {
        "type": "text_response",
        "message": "No data found from either source",
    }

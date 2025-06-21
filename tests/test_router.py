import pytest
from unittest.mock import Mock

pytest.importorskip("langchain")

import agent_router


QUERY = "show last two order and loyalty card of customer 2555880"


def test_order_and_loyalty_query_triggers_both(monkeypatch):
    mock_chain = Mock()
    result = agent_router._classify_query(QUERY, mock_chain)
    assert result == "both"
    # heuristics should classify without calling the LLM
    mock_chain.invoke.assert_not_called()

    live_agent = Mock()
    common_agent = Mock()
    monkeypatch.setattr(agent_router, "_create_live_agent", lambda: live_agent)
    monkeypatch.setattr(agent_router, "_create_common_agent", lambda: common_agent)

    class DummyLLM:
        def invoke(self, *_args, **_kwargs):
            return "both"

    monkeypatch.setattr(agent_router, "ChatOpenAI", lambda *a, **k: DummyLLM())

    router = agent_router.get_routed_agent()
    router.invoke({"input": QUERY})

    assert live_agent.invoke.called
    assert common_agent.invoke.called

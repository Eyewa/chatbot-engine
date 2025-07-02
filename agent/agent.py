"""Factory for Winkly chatbot agent."""

from agent.core.agent_router import get_routed_agent


def build_chatbot_agent():
    """
    Return the multi-DB agent with intent routing.

    This wraps the classification and execution logic for live/common DB queries,
    including cross-database separation and fallback join-based handling.
    """
    return get_routed_agent()

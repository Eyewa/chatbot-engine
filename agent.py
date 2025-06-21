# agent.py

"""Factory for Winkly chatbot agent."""

from agent_router import get_routed_agent


def build_chatbot_agent():
    """Return the routed Winkly agent."""
    return get_routed_agent()

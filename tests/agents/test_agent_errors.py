# tests/agents/test_agent_errors.py
"""
Purpose:
This module contains unit tests for handling errors within the Agent class,
such as initialization with invalid configuration or running non-existent agents.
"""
import pytest

from src.agents.agent import Agent


def test_init_with_none_configuration():
    """Verify that initializing an Agent with None configuration raises a ValueError."""
    with pytest.raises(ValueError, match="Configuration must be a dictionary."):
        Agent(None)


@pytest.mark.asyncio
async def test_run_invalid_agent():
    """Verify that running a non-existent agent raises a ValueError."""
    agent = Agent({})
    with pytest.raises(ValueError, match="Agent 'invalid_agent' not found."):
        # The `run_agent` method is async and requires the `chat` argument.
        await agent.run_agent("invalid_agent", chat=None)

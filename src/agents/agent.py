"""
Base Agent class that provides common functionality for all agents.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from src.tools.tool import Tool

logger = logging.getLogger(__name__)


class Agent:
    """
    Base class for all agents that provides common functionality.
    """

    def __init__(self, configuration: Dict) -> None:
        if not isinstance(configuration, dict):
            raise ValueError("Configuration must be a dictionary.")
        self.configuration = configuration
        self.agent_config: dict = {}
        self.tool: Tool | None = None
        self.agent_name = ""

    async def run_agent(self, agent: str, chat: Optional[Any]):
        """Execute agent method dynamically from parent class"""
        self.agent_name = agent.lower()
        self.tool = Tool(self.agent_name, self.configuration)

        try:
            return await self.tool.run_tool(chat=chat)
        except AttributeError:
            raise ValueError(f"Agent '{self.agent_name}' not found.")

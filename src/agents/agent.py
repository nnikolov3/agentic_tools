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
            # Get and call the method directly on self
            method = getattr(self, self.agent_name)
            return await method(chat=chat)
        except AttributeError:
            raise ValueError(f"Agent '{self.agent_name}' not found.")

    async def readme_writer(self, chat: Optional[Any] = None):
        """Execute readme writer logic"""
        logger.info("ReadmeWriter executing")

        return await self.tool.run_tool(chat=chat)

    async def approver(self, chat: Optional[Any] = None):
        """Executes the approver tool to audit code changes and provide a final decision."""
        logger.info("approver executing")
        response = await self.tool.run_tool(chat=chat)
        return response

    async def developer(self, chat: Optional[Any] = None):
        """Executes the developer tool to provide feedback on the code changes."""
        logger.info("developer executing")
        response = await self.tool.run_tool(chat=chat)
        return response
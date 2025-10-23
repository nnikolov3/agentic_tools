"""
Purpose:
Base Agent class that orchestrates tools and manages memory lifecycle.
Integrates with optimized QdrantMemory for retrieval and storage.
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.qdrant_memory import QdrantMemory
from src.tools.shell_tools import ShellTools
from src.tools.tool import Tool

import mdformat

logger: logging.Logger = logging.getLogger(__name__)


class Agent:
    """
    Base class for all agents, responsible for orchestrating the memory, tools,
    and post-processing steps in the agent lifecycle.
    """

    def __init__(
        self, configuration: dict[str, Any], agent_name: str, project: str, chat: Any
    ) -> None:
        if not isinstance(configuration, dict):
            raise ValueError("Configuration must be a dictionary.")

        self.configuration: dict[str, Any] = configuration[project]
        self.chat: Any = chat
        self.agent_name: str = agent_name
        self.memory: QdrantMemory | None = None
        self.memory_context: str | None = None
        self.response: Any | None = None
        self.tool = Tool(agent_name, self.configuration)
        self.shell_tools = ShellTools(agent_name, self.configuration)
        self.memory_config: dict[str, Any] = self.configuration.get("memory", {})
        self.agent_config: dict[str, Any] = self.configuration.get(agent_name, {})

        logger.debug(f"Initialized Agent '{self.agent_name}' for project '{project}'.")

    async def run_agent(self) -> Any:
        """
        Executes the full agent lifecycle:
        1. Initializes Qdrant memory and retrieves relevant context based on the chat input.
        2. Executes the primary tool (`self.tool.run_tool`) with the retrieved context.
        3. Performs agent-specific post-processing (e.g., for readme_writer).
        4. Stores the final response in memory for future context retrieval.

        Returns:
            The raw response from the executed tool.
        """
        try:
            # Memory initialization
            if self.agent_name != "commentator":
                if self.memory_config and self.memory_config.get("qdrant_url"):
                    self.memory = await QdrantMemory.create(self.memory_config)
                    self.memory_context = await self.memory.retrieve_context(
                        str(self.chat)
                    )
                else:
                    logger.info("No memory config; skipping retrieval.")
                    self.memory_context = ""
            else:
                self.memory_context = ""

            # Tool execution
            self.response = await self.tool.run_tool(
                chat=self.chat, memory_context=self.memory_context
            )

            # Post-processing for readme_writer
            if self.agent_name == "readme_writer" and self.response:
                self.response = self.shell_tools.cleanup_escapes(str(self.response))
                self.response = mdformat.text(
                    self.response, options={"wrap": "preserve"}
                )
                self.shell_tools.write_file("README.md", self.response)
                logger.info("Wrote README.md for readme_writer agent.")

            # Store response in memory
            if self.memory and self.response:
                logger.info("Storing response in memory.")
                await self.memory.add_memory(text_content=str(self.response))

            return self.response

        except Exception as agent_error:
            error_message = f"Failed to run agent '{self.agent_name}': {agent_error}"
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from agent_error

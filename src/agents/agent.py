"""
Purpose:
This module defines the base Agent class, which serves as the orchestrator for
executing agentic tools and managing the lifecycle of an agent's memory.
The design emphasizes simplicity by encapsulating tool execution and memory management
within a single responsibility, ensuring explicit configuration handling and robust error recovery.
Agent-specific post-processing (e.g., for readme_writer) is handled here to centralize lifecycle logic.
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.qdrant_memory import QdrantMemory
from src.tools.shell_tools import ShellTools
from src.tools.tool import Tool

import mdformat

# Self-Documenting Code: Dedicated logger for traceability in agent operations.
logger: logging.Logger = logging.getLogger(__name__)


class Agent:
    """
    Base class for all agents that provides common functionality.

    It orchestrates tool execution and manages the agent's memory lifecycle,
    handling initialization, context retrieval, tool invocation, response storage,
    and agent-specific post-processing.
    """

    def __init__(
        self, configuration: dict[str, Any], agent_name: str, project: str, chat: Any
    ) -> None:
        """
        Initializes the Agent with the provided configuration.

        Validates configuration type explicitly to prevent downstream errors.
        Derives sub-configs for memory, agent-specific settings, and tools.

        Args:
            configuration: The full project configuration as a dictionary.
            agent_name: The name of the specific agent instance.
            project: The project key to select configuration subset.
            chat: The current chat context for the agent.

        Raises:
            ValueError: If configuration is not a dictionary.
        """
        # Explicit Over Implicit: Validate and extract configuration immediately.
        if not isinstance(configuration, dict):
            raise ValueError("Configuration must be a dictionary.")

        # Simplicity is Non-Negotiable: Direct assignment from validated config.
        self.configuration: dict[str, Any] = configuration[project]
        self.chat: Any = chat
        self.agent_name: str = agent_name
        self.memory: QdrantMemory | None = None
        self.memory_context: str | None = None
        self.response: Any | None = None
        self.tool: Tool = Tool(agent_name, self.configuration)
        self.shell_tools: ShellTools = ShellTools(agent_name, self.configuration)
        self.memory_config: dict[str, Any] = self.configuration.get("memory", {})
        self.agent_config: dict[str, Any] = self.configuration.get(agent_name, {})

        # Debugging: Log configuration for traceability during development (removed prints).
        logger.debug(f"Initialized Agent '{self.agent_name}' for project '{project}'.")

    async def run_agent(self) -> Any:
        """
        Execute an agent's tool, managing memory retrieval and storage.

        This method handles the full lifecycle for a single agent run:
        1. Initializes the memory system if enabled.
        2. Retrieves relevant context from memory based on the chat query.
        3. Executes the appropriate tool with the chat and memory context.
        4. Applies agent-specific post-processing if applicable (e.g., readme_writer).
        5. Stores the final processed response back into memory.

        Returns:
            The agent's final response after processing.

        Raises:
            RuntimeError: If memory initialization or tool execution fails critically.
        """
        try:
            # Conditional Memory Initialization: Create only if config is present and non-empty.
            if self.memory_config and self.memory_config.get("qdrant_url"):
                self.memory = await QdrantMemory.create(self.memory_config)
                # Context Retrieval: Use explicit parameter matching.
                self.memory_context = await self.memory.retrieve_context(
                    query_text=str(self.chat)
                )
            else:
                logger.info("No valid memory config found; skipping memory retrieval.")
                self.memory_context = ""

            # Tool Execution: Pass context explicitly.
            self.response = await self.tool.run_tool(
                chat=self.chat, memory_context=self.memory_context
            )

            # Agent-Specific Post-Processing: Handle readme_writer formatting and write.
            if self.agent_name == "readme_writer" and self.response:
                # Clean up escapes from response.
                self.response = self.shell_tools.cleanup_escapes(str(self.response))
                # Format Markdown for readability.
                self.response = mdformat.text(
                    self.response, options={"wrap": "preserve"}
                )
                # Write to README.md.
                self.shell_tools.write_file("README.md", self.response)
                logger.info(
                    "Post-processed and wrote README.md for readme_writer agent."
                )

            # Memory Storage: Persist final processed response only if memory is active.
            if self.memory and self.response:
                logger.info("Adding final response to agent memory.")
                await self.memory.add_memory(text_content=str(self.response))

            return self.response

        except Exception as agent_error:
            # Error Handling Excellence: Log with context but re-raise for upstream handling.
            error_message: str = (
                f"Failed to run agent '{self.agent_name}': {agent_error}"
            )
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from agent_error

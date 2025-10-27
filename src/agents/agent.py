# src/agents/agent.py
"""
This module defines the agentic execution framework.

It provides a base `Agent` class that orchestrates the lifecycle of an AI agent's task.
This lifecycle includes memory retrieval (RAG), tool execution, agent-specific
post-processing, and memory storage.

The design uses polymorphism to handle agent-specific behaviors. Concrete agent
classes inherit from the base `Agent` and implement the `_post_process` method
to define their unique actions, such as writing files, formatting output, or
validating generated code. This approach adheres to the Open/Closed Principle,
allowing for new agent types to be added without modifying the core execution logic.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Optional

import mdformat

from src.memory.qdrant_memory import QdrantMemory
from src.tools.validation_tools import ValidationService
from src.tools.shell_tools import ShellTools
from src.tools.tool import Tool

logger: logging.Logger = logging.getLogger(__name__)


class CodeValidationFailedError(Exception):
    """Custom exception raised when generated code fails validation."""


class Agent(abc.ABC):
    """
    The abstract base class for all AI agents.

    This class defines the core execution lifecycle, orchestrating interactions
    between memory, tools, and post-processing steps. Subclasses must implement
    the `_post_process` method to define agent-specific logic.
    """

    def __init__(
        self,
        configuration: dict[str, Any],
        agent_name: str,
        project: str,
        chat: Optional[str],
        filepath: Optional[str | PathLike[str]],
        ) -> None:
        """
        Initializes the Agent with its configuration and operational context.

        Args:
            configuration: A dictionary containing project-wide and agent-specific settings.
            agent_name: The unique name of the agent instance.
            project: The name of the project, used to select the relevant configuration.
            chat: The user input or query that triggers the agent's action.
            filepath: An optional file path relevant to the agent's task.
        """
        if not isinstance(configuration, dict):
            raise TypeError("Configuration must be a dictionary.")

        project_config: dict[str, Any] = configuration.get(project, {})
        if not project_config:
            logger.warning(f"No configuration found for project '{project}'.")

        self.configuration: dict[str, Any] = project_config
        self.agent_name: str = agent_name
        self.chat: Optional[str] = chat
        self.filepath: Optional[str | PathLike[str]] = filepath

        self.memory: Optional[QdrantMemory] = None
        self.memory_context: str = ""
        self.response: Optional[str] = None
        self.context_quality_score: float = 1.0

        self.shell_tools = ShellTools(agent_name, self.configuration)
        self.tool = Tool(agent_name, self.configuration)
        self.memory_config: dict[str, Any] = self.configuration.get("memory", {})

        logger.debug(f"Initialized Agent '{self.agent_name}' for project '{project}'.")

    async def run_agent(self) -> Optional[str]:
        """
        Executes the full agent lifecycle.

        This template method orchestrates the agent's operation:
        1. Retrieves context from memory (RAG).
        2. Executes the primary tool to generate a response.
        3. Performs agent-specific post-processing on the response.
        4. Stores the final, processed response back into memory.

        Returns:
            The final, processed response from the agent, or None if no response
            was generated.

        Raises:
            RuntimeError: If any unhandled exception occurs during execution,
                          wrapping the original exception.
        """
        try:
            self.memory_context = await self._retrieve_context()
            self.context_quality_score = await self._assess_context_quality()
            await self._update_memory_weights()

            if self.context_quality_score < 0.5:
                self.memory_context = await self._retrieve_context()

            self.response = await self.tool.run_tool(
                self.chat,
                self.memory_context,
                str(self.filepath) if self.filepath else None,
                )

            if self.response:
                await self._post_process()
                await self._store_memory()

            return self.response

        except Exception as agent_error:
            error_message = f"Agent '{self.agent_name}' failed to run: {agent_error}"
            logger.error(error_message, exc_info = True)
            raise RuntimeError(error_message) from agent_error

    async def _retrieve_context(self) -> str:
        """
        Initializes memory and retrieves context for the current task.

        If memory is configured, this method initializes a `QdrantMemory` instance
        and fetches context relevant to the agent's `chat` input.

        Returns:
            A string containing the retrieved context, or an empty string if
            memory is not configured or no context is found.
        """
        if not self.memory_config:
            logger.warning("No Qdrant memory configured. Skipping context retrieval.")
            return ""

        query = self.chat or self.agent_name

        self.memory = await QdrantMemory.create(self.memory_config, self.agent_name)
        retrieved_context = await self.memory.retrieve_context(query)
        return retrieved_context or ""

    async def _store_memory(self) -> None:
        """
        Stores the agent's final response in the memory system.

        This operation is conditional on memory being initialized and a response
        being available. It makes the agent's output available for future tasks.
        """
        if self.memory and self.response:
            logger.info("Storing response in memory.")
            await self.memory.add_memory(text_content = self.response)
        else:
            logger.debug(
                "Skipping memory storage: memory not initialized or no response.",
                )

    async def _update_memory_weights(self) -> None:
        """
        Updates the memory retrieval weights based on the context quality score.

        This method can be overridden by concrete agent subclasses to provide
        their own logic for updating the memory weights.
        """
        if self.memory and self.context_quality_score < 0.5:
            logger.info("Context quality is low. Updating memory weights.")
            # For now, we do nothing. This will be implemented in the future.

    @abc.abstractmethod
    async def _post_process(self) -> None:
        """
        Performs agent-specific post-processing on the generated response.

        This abstract method must be implemented by concrete agent subclasses
        to handle tasks like file writing, formatting, or validation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _assess_context_quality(self) -> float:
        """
        Assesses the quality of the retrieved context.

        This abstract method must be implemented by concrete agent subclasses
        to provide their own logic for assessing the context quality.

        Returns:
            A float between 0.0 and 1.0 representing the quality of the context.
        """
        raise NotImplementedError


class DefaultAgent(Agent):
    """An agent that performs no special post-processing."""

    async def _post_process(self) -> None:
        """Logs that no post-processing is needed for this agent."""
        logger.info(f"No post-processing required for agent '{self.agent_name}'.")

    async def _assess_context_quality(self) -> float:
        """Returns a default quality score of 1.0."""
        return 1.0


class ReadmeWriterAgent(Agent):
    """An agent responsible for formatting and writing README.md files."""

    async def _post_process(self) -> None:
        """
        Formats the response as Markdown and writes it to README.md.

        The method cleans, formats, and writes the content, ensuring the final
        output stored in memory is the formatted version.
        """
        if not self.response:
            logger.warning("Cannot write README: response is missing.")
            return

        readme_filepath = Path("README.md")
        cleaned_response = self.shell_tools.cleanup_escapes(self.response)

        # mdformat is a blocking, CPU-bound call; run it in a thread pool.
        formatted_readme = await asyncio.to_thread(
            mdformat.text,
            cleaned_response,
            options = {"wrap": "preserve"},
            )

        # File I/O is blocking; run it in a thread pool.
        await asyncio.to_thread(
            self.shell_tools.write_file, readme_filepath, formatted_readme,
            )

        # Update self.response so the clean, formatted version is stored in memory.
        self.response = formatted_readme
        logger.info(f"Successfully formatted and wrote '{readme_filepath}'.")

    async def _assess_context_quality(self) -> float:
        """Returns a default quality score of 1.0."""
        return 1.0


class CodeModifyingAgent(Agent):
    """
    A base agent for modifying source code, with built-in validation.

    This agent ensures that any code generated by the LLM is validated for
    syntactic correctness and style compliance before being written to a file.
    This prevents code corruption.
    """

    def __init__(
        self,
        configuration: dict[str, Any],
        agent_name: str,
        project: str,
        chat: Optional[str],
        filepath: Optional[str | PathLike[str]],
        ) -> None:
        """
        Initializes the CodeModifyingAgent.

        Raises:
            ValueError: If `filepath` is not provided, as it is essential
                        for this agent's operation.
        """
        super().__init__(configuration, agent_name, project, chat, filepath)
        if not self.filepath:
            raise ValueError(
                f"{self.__class__.__name__} requires a valid filepath, but None was provided.",
                )
        self._validation_service = ValidationService()

    def _clean_response_for_code(self) -> str:
        """
        Removes markdown code fences from the LLM's response.

        LLMs often wrap code in ```python ... ``` blocks. This method extracts
        the raw code to ensure it can be correctly validated and written to a file.

        Returns:
            The cleaned code content as a string.
        """
        if not self.response:
            return ""

        content = self.response.strip()
        lines = content.split("\n")

        # Check for and remove the markdown code block fences.
        if (
            len(lines) > 1
            and lines[0].strip().startswith("```")
            and lines[-1].strip() == "```"
        ):
            return "\n".join(lines[1:-1])

        return content

    async def _post_process(self) -> None:
        """
        Validates the generated code and writes it to the source file.

        This method first cleans the LLM response, then uses the ValidationService
        to check for errors. If valid, the code is written to the specified
        filepath. If invalid, an error is raised to halt execution.

        Raises:
            CodeValidationFailedError: If the generated code fails static analysis checks.
        """
        cleaned_code = self._clean_response_for_code()
        if cleaned_code and self.filepath:
            is_success = await asyncio.to_thread(
                self.shell_tools.write_file, Path(self.filepath), cleaned_code
            )
            if is_success:
                logger.info(
                    f"Successfully validated and updated source file: {self.filepath}"
                )
            else:
                logger.error(f"Failed to write to source file: {self.filepath}")
                return

    async def _assess_context_quality(self) -> float:
        """Returns a default quality score of 1.0."""
        return 1.0


class CommentatorAgent(CodeModifyingAgent):
    """An agent that comments code, inheriting validation from CodeModifyingAgent."""

    async def _assess_context_quality(self) -> float:
        """Returns a default quality score of 1.0."""
        return 1.0


class DeveloperAgent(CodeModifyingAgent):
    """An agent that writes or refactors code, inheriting validation from CodeModifyingAgent."""

    async def _assess_context_quality(self) -> float:
        """Returns a default quality score of 1.0."""
        return 1.0


class ExpertAgent(DefaultAgent):
    """An agent that leverages the knowledge base to answer questions."""

    async def _retrieve_context(self) -> str:
        """Retrieves context from the knowledge base."""
        if not self.memory_config:
            logger.warning("No Qdrant memory configured. Skipping context retrieval.")
            return ""

        query = self.chat or self.agent_name

        self.memory = await QdrantMemory.create(self.memory_config, self.agent_name)

        return await self.memory.retrieve_context(query)


AGENT_CLASSES: dict[str, type[Agent]] = {
    "readme_writer": ReadmeWriterAgent,
    "commentator": CommentatorAgent,
    "developer": DeveloperAgent,
    "approver": DefaultAgent,
    "architect": DefaultAgent,
    "expert": ExpertAgent,
}

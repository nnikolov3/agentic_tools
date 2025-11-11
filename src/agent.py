"""
This module defines the agentic execution framework.

Purpose: Generic orchestration of AI agent lifecycles with RAG, tool execution, post-processing, and storage.
Design: Single class with config-driven behaviors; composition with ShellTools/Tool.
DDT Exploration: Unified class for simplicity; async for non-blocking I/O.
Why: KISS eliminates specifics; immutability in configs; no path validation (caller responsibility).
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from src.memory.memory import Memory
from src.tools.shell_tools import ShellTools
from src.tools.tool import Tool

logger: logging.Logger = logging.getLogger(__name__)


class Agent:
    """Generic class for AI agents; unified lifecycle execution."""

    def __init__(
        self,
        configuration: Dict[str, Union[str, Dict[str, str]]],
        agent_name: str,
        project: str,
        chat: Optional[str],
        filepath: Optional[Union[str, Path]],
        target_directory: Optional[Path],
    ) -> None:
        """Initializes agent with config and context.

        Design: Early validation; immutable config slice.
        Why: Fail-fast on type/essentials; project-specific subset; no path checks (caller validated).
        """
        if not isinstance(configuration, dict):
            raise TypeError("Configuration must be a dictionary.")

        project_config: Dict[str, Union[str, Dict[str, str]]] = configuration.get(
            project, {}
        )
        if not project_config:
            logger.warning("No configuration found for project '%s'.", project)

        self.configuration: Dict[str, Union[str, Dict[str, str]]] = project_config
        self.responses_dir: str = self.configuration.get("responses_dir")
        if not self.responses_dir:
            raise ValueError("Configuration must provide 'responses_dir'.")
        self.agent_name: str = agent_name
        self.chat: Optional[str] = chat
        self.filepath: Optional[Union[str, Path]] = filepath

        self.memory: Optional[Memory] = None
        self.memory_context: str = ""
        self.response: Optional[str] = None

        effective_target_directory = (
            target_directory if target_directory is not None else Path.cwd()
        )

        self.shell_tools = ShellTools(
            agent_name, self.configuration, effective_target_directory
        )
        self.tool = Tool(agent_name, self.configuration, effective_target_directory)
        self.memory_config: Dict[str, Union[str, Dict[str, str]]] = (
            self.configuration.get("memory", {})
        )
        self.target_directory: Optional[Path] = target_directory
        self.async_timeout: float = self.configuration.get("async_timeout", 30.0)

        logger.debug(
            "Initialized Agent '%s' for project '%s'.", self.agent_name, project
        )

    async def run_agent(self) -> Optional[str]:
        """Orchestrates generic lifecycle: retrieve, execute, post-process, store.

        Design: Template method; explicit sequencing without broad exceptions.
        Why: Enforces DDT cycle; returns processed response; no path checks.
        """
        self.memory_context = await asyncio.wait_for(
            self._retrieve_context(), timeout=self.async_timeout
        )

        # Generic tool execution
        tool_response = await asyncio.wait_for(
            self.tool.run_tool(
                self.chat,
                self.memory_context,
                str(self.filepath) if self.filepath else None,
            ),
            timeout=self.async_timeout,
        )
        self.response = tool_response

        if self.response:
            await self._write_to_file()
            await self._store_memory()

        return self.response

    async def _write_to_file(self) -> None:
        """Single async write: to filepath if provided, else timestamped in responses_dir.

        Design: Unified non-blocking write via shell_tools; integrated logging.
        Why: DRY; ensures one async output operation with explicit feedback; no path checks (assume valid).
        """
        if not self.response:
            logger.debug("No response to write.")
            return None

        # Prioritize filepath if provided
        if self.filepath:
            output_path = Path(self.filepath)
            # No dir validation—assume caller ensured parent dir exists

            write_success = await self.shell_tools.write_file(
                output_path, self.response
            )
            if write_success:
                logger.info("Wrote response to '%s'.", output_path)
            else:
                logger.error("Failed to write response to '%s'.", output_path)
            return None

        # Fallback: timestamped in responses_dir
        output_dir = Path(self.responses_dir)
        # No validation—assume config responses_dir is valid

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_{timestamp}.md"
        output_path = output_dir / filename

        write_success = await self.shell_tools.write_file(output_path, self.response)
        if write_success:
            logger.info("Wrote response to '%s'.", output_path)
        else:
            logger.error("Failed to write response to '%s'.", output_path)
        return None

    async def _retrieve_context(self) -> str:
        """Retrieves RAG context if configured.

        Design: Conditional init; query concatenation.
        Why: Enables memory-aware tasks; defaults to empty; no path checks.
        """
        if not self.memory_config:
            logger.warning("No Qdrant memory configured. Skipping context retrieval.")
            return ""

        query = (self.chat or "") + self.agent_name
        self.memory = await asyncio.wait_for(
            Memory.create(self.configuration),
            timeout=self.async_timeout,
        )
        retrieved_context = await self.memory.retrieve_context(query)
        return retrieved_context or ""

    async def _store_memory(self) -> None:
        """Stores the agent's final response in the memory system.

        Design: Unconditional if memory and response exist.
        Why: Simple storage; no weights or conditionals; no path checks.
        """
        if self.memory and self.response:
            logger.info("Storing response in memory.")
            await asyncio.wait_for(
                self.memory.add_memory(text_content=self.response),
                timeout=self.async_timeout,
            )
        else:
            logger.debug(
                "Skipping memory storage: memory not initialized or no response."
            )
        return None

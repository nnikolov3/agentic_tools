from __future__ import annotations

import asyncio
import datetime
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from src.memory.qdrant_memory import QdrantMemory
from src.tools.shell_tools import ShellTools
from src.tools.tool import Tool

logger = logging.getLogger(__name__)


class Agent:
    """The default class for all AI agents."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        agent_name: str,
        chat: Optional[str] = None,
        filepath: Optional[str | PathLike[str]] = None,
        target_directory: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(configuration, dict):
            raise TypeError("Configuration must be a dictionary.")

        self.configuration = configuration
        self.agent_name = agent_name
        self.chat = chat
        self.filepath = filepath
        self.memory: Optional[QdrantMemory] = None
        self.memory_context = ""
        self.response: Optional[str] = None

        effective_target_directory = (
            target_directory if target_directory else Path.cwd().parent
        )
        self.shell_tools = ShellTools(
            agent_name, configuration, effective_target_directory
        )
        self.tool = Tool(agent_name, configuration, effective_target_directory)
        self.memory_config = configuration.get("memory", {})

        # Load agent-specific configuration
        self.agent_config = (
            configuration.get("agentic-tools", {}).get("agents", {}).get(agent_name, {})
        )

        # Load weights configuration
        self.weights = self.agent_config.get("weights", {})

        # Load post-processing configuration
        self.post_process_config = self.agent_config.get("post_process", {})

        self.target_directory = target_directory

        logger.debug("Initialized Agent '%s'.", self.agent_name)

    async def _write_response_to_file_async(
        self, output_path: Path, content: str
    ) -> None:
        """Asynchronously writes content to a file."""
        try:
            async with aiofiles.open(output_path, mode="w") as f:
                await f.write(content)
            logger.info("Successfully wrote agent response to '%s'.", output_path)
        except IOError as e:
            logger.error("Failed to write agent response to '%s': %s", output_path, e)
            raise

    async def _write_response_to_file(self) -> None:
        """Writes the agent's response to a file asynchronously."""
        if not self.response:
            return

        output_dir = Path("agent_responses")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_{timestamp}.md"
        output_path = output_dir / filename

        await self._write_response_to_file_async(output_path, self.response)

    async def run_agent(self) -> Optional[str]:
        """Orchestrates the agent's operation."""
        try:
            self.memory_context = await self._retrieve_context()

            # Use Tool to run the API call with the constructed payload
            self.response = await self.tool.run_tool(
                self.chat,
                self.memory_context,
                str(self.filepath) if self.filepath else None,
            )

            if self.response:
                post_process_task = asyncio.create_task(self._post_process())
                write_response_task = asyncio.create_task(
                    self._write_response_to_file()
                )
                store_memory_task = asyncio.create_task(self._store_memory())

                await asyncio.gather(
                    post_process_task, write_response_task, store_memory_task
                )

            return self.response

        except Exception as agent_error:
            logger.error(
                "Agent '%s' failed to run: %s",
                self.agent_name,
                agent_error,
                exc_info=True,
            )
            raise RuntimeError(
                f"Agent '{self.agent_name}' failed to run: {agent_error}"
            )

    async def _retrieve_context(self) -> str:
        """Retrieves context for the current task."""
        if not self.memory_config:
            logger.warning("No Qdrant memory configured. Skipping context retrieval.")
            return ""

        query = (self.chat or "") + self.agent_name
        self.memory = await QdrantMemory.create(self.configuration, self.agent_name)
        retrieved_context = await self.memory.retrieve_context(
            query, weights=self.weights
        )
        return retrieved_context or ""

    async def _store_memory(self) -> None:
        """Stores the agent's final response in the memory system."""
        if self.memory and self.response:
            logger.info("Storing response in memory.")
            await self.memory.add_memory(text_content=self.response)
        else:
            logger.debug(
                "Skipping memory storage: memory not initialized or no response."
            )

    async def _post_process(self) -> None:
        """Performs agent-specific post-processing on the generated response."""
        if not self.response:
            logger.warning("No response to post-process.")
            return

        # Write to file if filepath is provided in the configuration
        if self.post_process_config.get("write_to_file", False):
            filepath = self.post_process_config.get("filepath", self.filepath)
            if filepath:
                output_path = Path(filepath)
                is_success = await self.shell_tools.write_file(
                    output_path, self.response
                )
                if is_success:
                    logger.info("Successfully wrote content to '%s'.", output_path)
                else:
                    logger.error("Failed to write content to '%s'.", output_path)

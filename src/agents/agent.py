# src/agents/agent.py

from __future__ import annotations

import asyncio
import datetime
import logging
from os import PathLike
from pathlib import Path
from typing import Any

import aiofiles

from src.configurator import get_agent_config, get_golden_rules
from src.memory.qdrant_memory import QdrantMemory
from src.tools.tool import get_tool

logger = logging.getLogger(__name__)


class Agent:
    """
    Default AI agent class. Async orchestration: Context retrieve → Tool exec → Memory store → Post-process.
    Loads nested config + golden rules; agent-specific tools/weights/post_process.
    """

    def __init__(
        self,
        configuration: dict[str, Any],
        agent_name: str,
        chat: str | None = None,
        filepath: str | PathLike[str] | None = None,
        target_directory: Path | None = None,
        memory: QdrantMemory | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(configuration, dict):
            raise TypeError("Configuration must be a dictionary.")

        self.configuration = configuration
        self.agent_name = agent_name
        self.chat = chat
        self.filepath = filepath
        self.target_directory = target_directory or Path.cwd().parent
        self.memory = memory
        self.memory_context = ""
        self.response: str | None = None
        self.golden_rules = get_golden_rules(configuration)  # Append to prompts

        # Tool: Registry lookup from agent_config
        self.agent_config = get_agent_config(configuration, agent_name)
        tool_names = self.agent_config.get("tools", ["api_call"])
        self.tool = get_tool(tool_names[0], configuration)  # First tool or default

        self.memory_config = self.configuration.get("memory", {})
        self.weights = self.agent_config.get("weights", {})  # Time-decay overrides
        self.post_process_config = self.agent_config.get("post_process", {})
        self.memory_weight = self.agent_config.get("memory_weight", 0.5)
        self.max_iterations = self.agent_config.get("max_iterations", 5)

        # Enhance prompt with golden rules/personal_context
        base_prompt = self.agent_config.get("prompt", self.chat or "")
        gen_cfg = self.configuration.get("general", {})
        personal = gen_cfg.get("personal_context", "")
        enhanced_prompt = base_prompt
        if self.golden_rules:
            enhanced_prompt += f"\n\nGolden Rules:\n{self.golden_rules}"
        if personal:
            enhanced_prompt += f"\n\nPersonal Context: {personal}"
        self.agent_config["prompt"] = enhanced_prompt

        logger.debug(
            "Initialized Agent '%s' (tools: %s, memory_weight: %.1f)",
            self.agent_name,
            tool_names,
            self.memory_weight,
        )

    async def _write_response_to_file_async(
        self, output_path: Path, content: str
    ) -> None:
        try:
            async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
                await f.write(content)
            logger.info("Wrote response to '%s'", output_path)
        except OSError as e:
            logger.error("Write failed to '%s': %s", output_path, e)
            raise

    async def _write_response_to_file(self) -> None:
        if not self.response:
            return
        output_dir = self.target_directory / "agent_responses"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_{timestamp}.md"
        output_path = output_dir / filename
        await self._write_response_to_file_async(output_path, self.response)

    async def run_agent(self) -> str | None:
        """
        Core async flow: Retrieve → Execute tool → Store → Post-process.
        Iterates up to max_iterations if needed (simple loop for now).
        """
        try:
            for iteration in range(self.max_iterations):
                self.memory_context = await self._retrieve_context()

                # Build payload for tool
                agent_prompt = self.agent_config["prompt"]
                payload = {
                    "type": "chat",
                    "prompt": (
                        f"{self.memory_context}\n\n{agent_prompt}"
                        if self.memory_context
                        else agent_prompt
                    ),
                    "filepath": str(self.filepath) if self.filepath else None,
                    "agent_name": self.agent_name,
                    "memory_weight": self.memory_weight,
                }
                tool_result = await self.tool.execute(payload)
                self.response = str(tool_result) if tool_result else ""

                if (
                    self.response and "final" in self.response.lower()
                ):  # Stub for iteration stop
                    break

                await asyncio.sleep(0.1)  # Yield for async

            # Parallel post-tasks
            if self.response:
                post_process_task = asyncio.create_task(self._post_process())
                write_task = asyncio.create_task(self._write_response_to_file())
                store_task = asyncio.create_task(self._store_memory())

                await asyncio.gather(
                    post_process_task, write_task, store_task, return_exceptions=True
                )

            return self.response

        except Exception as e:
            logger.error("Agent '%s' run failed: %s", self.agent_name, e, exc_info=True)
            raise RuntimeError(f"Agent '{self.agent_name}' failed: {e}")

    async def _retrieve_context(self) -> str:
        """
        Retrieve memory web context (time-decay weighted, hybrid search).
        """
        if not self.memory_config.get("enabled", True) or not self.memory:
            logger.debug("Memory disabled for '%s'", self.agent_name)
            return ""

        query = (
            self.chat or self.agent_config["prompt"][:200]
        ) + f" for {self.agent_name}"  # Truncate for query
        try:
            retrieved = await self.memory.retrieve_context(
                query=query,
                weights=self.weights,  # Agent-specific (e.g., kb=0.9)
                limit=self.memory_config.get("total_memories_to_retrieve", 20),
                threshold=self.memory_config.get("similarity_threshold", 0.8),
            )
            return retrieved if retrieved else ""
        except Exception as e:
            logger.warning("Context retrieval failed for '%s': %s", self.agent_name, e)
            return ""

    async def _store_memory(self) -> None:
        """
        Store response as episodic memory (chunked, embedded, upserted).
        """
        if self.memory and self.response:
            try:
                logger.info("Storing response for '%s'", self.agent_name)
                await self.memory.add_memory(
                    text=self.response,
                    memory_type="episodic",
                    source=self.agent_name,
                    confidence=0.9,  # Default high for agent output
                )
            except Exception as e:
                logger.error("Memory store failed for '%s': %s", self.agent_name, e)

    async def _post_process(self) -> None:
        """
        Agent-specific post-processing (e.g., write file from config).
        """
        if not self.response or not self.post_process_config:
            return

        write_file = self.post_process_config.get("write_to_file", False)
        if write_file:
            filepath = self.post_process_config.get("filepath")
            if filepath:
                output_path = self.target_directory / Path(filepath)
                await self._write_response_to_file_async(output_path, self.response)
                # Optional shell write for git add/commit
                await self.shell_tools.write_file(output_path, self.response)
                logger.info(
                    "Post-processed '%s' to file: %s", self.agent_name, output_path
                )

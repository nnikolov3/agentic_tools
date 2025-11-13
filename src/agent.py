"""
This module defines the agentic execution framework.

Purpose: Generic orchestration of AI agent lifecycles with RAG, tool execution, post-processing, and storage.
Design: Single class with config-driven behaviors; composition with ShellTools/Tool.
DDT Exploration: Unified class for simplicity; async for non-blocking I/O.
Why: KISS eliminates specifics; immutability in configs; no path validation (caller responsibility).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.memory.memory import Memory
from src.utils.document_processor import DocumentProcessor
from src.utils.knowledge_augmentor import KnowledgeAugmentor
from src.utils.payload_builder import PayloadBuilder

logger: logging.Logger = logging.getLogger(__name__)


class Agent:
    """Generic class for AI agents; unified lifecycle execution."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        agent_name: str,
        project: str,
        chat: Optional[str],
        filepath: Optional[Union[str, Path]],
        target_directory: Optional[Path],
        providers: Dict[str, Any],
        memory: Optional[Memory] = None,
        document_processor: Optional[DocumentProcessor] = None,
        knowledge_augmentor: Optional[KnowledgeAugmentor] = None,
    ) -> None:
        """Initializes agent with config, context, and explicit dependencies.

        Design: Early validation; immutable config slice; explicit dependency injection.
        Why: Fail-fast on essentials; project-specific subset; clear dependencies; no path checks (caller validated).
        """
        if not isinstance(configuration, dict):
            raise TypeError("Configuration must be a dictionary.")

        agent_config: Dict[str, Any] = configuration.get("agents", {}).get(
            agent_name, {}
        )
        if not agent_config:
            logger.warning("No configuration found for agent '%s'.", agent_name)

        self.configuration: Dict[str, Any] = agent_config

        self.responses_dir: str = configuration.get("project", {}).get(
            "responses_dir", "responses"
        )

        self.agent_name: str = agent_name
        self.chat: Optional[str] = chat
        self.filepath: Optional[Union[str, Path]] = filepath
        self.target_directory: Optional[Path] = target_directory

        self.providers: Dict[str, Any] = providers
        self.memory: Optional[Memory] = memory
        self.document_processor: Optional[DocumentProcessor] = document_processor
        self.knowledge_augmentor: Optional[KnowledgeAugmentor] = knowledge_augmentor

        self.memory_context: str = ""
        self.response: Optional[str] = None

        effective_target_directory = (
            target_directory if target_directory is not None else Path.cwd()
        )

        self.payload_builder = PayloadBuilder(
            agent_config=self.configuration,
            target_directory=effective_target_directory,
        )

        self.memory_config: Dict[str, Any] = configuration.get("memory", {})
        self.async_timeout: float = float(self.configuration.get("async_timeout", 30.0))

        logger.debug(
            "Initialized Agent '%s' for project '%s'.", self.agent_name, project
        )

    async def run_agent(self) -> Optional[str]:
        """Orchestrates generic lifecycle: retrieve, execute, post-process, store.

        Design: Template method; explicit sequencing without broad exceptions.
        Why: Enforces DDT cycle; returns processed response; no path checks.
        """
        if self.agent_name == "ingestor" and self.document_processor:
            if self.filepath:
                await self.document_processor.process_document(str(self.filepath))
                return f"Successfully ingested document: {self.filepath}"
            else:
                logger.warning("Ingestor agent called without a filepath.")
                return "Ingestor agent requires a filepath to process a document."

        self.memory_context = await asyncio.wait_for(
            self._retrieve_context(), timeout=self.async_timeout
        )

        payload = await self.payload_builder.create_payload(
            chat=self.chat,
            memory_context=self.memory_context,
            filepath=str(self.filepath) if self.filepath else None,
        )

        provider_name = self.configuration.get("agent_provider")
        model_name = self.configuration.get("agent_model_name")
        temperature = self.configuration.get("agent_temperature", 0.7)

        if provider_name in self.providers:
            provider = self.providers[provider_name]
            self.response = await provider.generate_text(
                model_name=model_name,
                system_instruction=payload.get("system_prompt"),
                user_content=payload.get("chat_message"),
                temperature=temperature,
            )
        else:
            logger.error(
                "No valid provider '%s' found in the injected providers for agent '%s'.",
                provider_name,
                self.agent_name,
            )
            self.response = (
                f"Error: Provider '{provider_name}' not configured or available."
            )
            exit(1)

        if self.response:
            if self.knowledge_augmentor:
                await self.knowledge_augmentor.process_agent_response(
                    agent_name=self.agent_name,
                    agent_response=self.response,
                    chat=self.chat,
                )
            await self._handle_output()

        return self.response

    async def _retrieve_context(self) -> str:
        """Retrieves RAG context if the memory system is available.

        Design: Conditional retrieval based on injected dependency.
        Why: Enables memory-aware tasks only if memory is provided.
        """
        if not self.memory:
            logger.debug("No memory system provided. Skipping context retrieval.")
            return ""

        query = (self.chat or "") + self.agent_name
        retrieved_context = await self.memory.retrieve_context(
            query, self.configuration
        )
        return retrieved_context or ""

    async def _handle_output(self) -> None:
        """
        Handles the agent's response based on configuration.

        Determines whether to write the response to a file or print it to the console,
        driven by the agent's configuration.
        """
        output_config = self.configuration.get("output", {})
        output_action = output_config.get("action", "print")

        if output_action == "write_to_file" and self.response:
            target = output_config.get("target")
            if target == "input_filepath" and self.filepath:
                output_path = Path(self.filepath)
            elif target:
                output_path = Path(target)
            else:
                logger.warning(
                    "Output action 'write_to_file' specified but no valid target path found for agent '%s'. Printing to console instead.",
                    self.agent_name,
                )
                print(self.response)
                return

            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(self.response)
                logger.info(
                    "Agent '%s' response written to file: %s",
                    self.agent_name,
                    output_path,
                )
            except Exception as e:
                logger.error(
                    "Failed to write agent '%s' response to file '%s': %s",
                    self.agent_name,
                    output_path,
                    e,
                )
                print(self.response)  # Fallback to printing
        elif self.response:
            # Default action or if 'print' is explicitly specified
            print(self.response)

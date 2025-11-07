from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.api_tools import ApiTools
from src.tools.shell_tools import ShellTools

logger = logging.getLogger(__name__)


class Tool:
    """
    Orchestrates shell and API tools for agent execution.

    Manages payload construction from git diffs, design docs, source code, and
    memory context, then routes to the appropriate API provider for response
    generation.
    """

    def __init__(
        self, agent: str, config: Dict[str, Any], target_directory: Path
    ) -> None:
        """
        Initializes Tool with agent-specific configuration and tool instances.

        Args:
            agent: The agent name for sub-config and tool instantiation.
            config: The full configuration dictionary.
            target_directory: The root directory of the project to operate on.
        """
        self.agent = agent
        self.config = config
        self.shell_tools = ShellTools(agent, config, target_directory)
        self.api_tools = ApiTools(agent, config)

        self.agent_config = (
            config.get("agentic-tools", {}).get("agents", {}).get(agent, {})
        )
        self.agent_skills = self.agent_config.get("skills")
        self.agent_prompt = self.agent_config.get("prompt")

        prompts_config = config.get("prompts", {})
        golden_rules = prompts_config.get("golden_rules")

        if self.agent_prompt and golden_rules:
            self.agent_prompt = f"{self.agent_prompt}\n\n{golden_rules}"

        self.target_directory = target_directory
        source_dirs = config.get("source")
        source_dir_name = source_dirs[0] if source_dirs else "src"
        self.source_directory = self.target_directory / source_dir_name

    async def run_tool(
        self,
        chat: Optional[Any],
        memory_context: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> Optional[str]:
        """
        Constructs a payload and executes the agent's primary task via an API call.

        Args:
            chat: The primary input for the agent (e.g., user query, file content).
            memory_context: Retrieved memory context string (optional).
            filepath: The path to the file being processed, used by specific agents.

        Returns:
            The raw response from the API provider, or None if execution fails.
        """
        try:
            payload = await self._create_payload(chat, memory_context, filepath)
            response = await self.api_tools.run_api(payload)

            if response:
                logger.info("Tool execution completed for agent '%s'.", self.agent)
                return response

            logger.warning("API call for agent '%s' returned no response.", self.agent)
            return ""

        except ValueError as value_error:
            logger.error(
                "Payload creation failed for agent '%s': %s", self.agent, value_error
            )
            raise
        except Exception as unexpected_error:
            logger.error(
                "An unexpected error occurred in run_tool for agent '%s': %s",
                self.agent,
                unexpected_error,
                exc_info=True,
            )
            raise RuntimeError(
                f"Tool execution failed for agent '{self.agent}'"
            ) from unexpected_error

    async def _create_payload(
        self,
        chat: Optional[Any],
        memory_context: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a standard payload for the API call, with context driven by the agent's needs.

        Args:
            chat: The primary input for the agent.
            memory_context: Retrieved memory context string.
            filepath: The path to the file being processed.

        Returns:
            A dictionary representing the payload for the API call.
        """
        logger.info("Creating payload for agent '%s'.", self.agent)

        # Standard payload structure
        payload = {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "MEMORY_CONTEXT": memory_context,
        }

        # Add context based on agent configuration
        if self.agent_config.get("use_design_docs", False):
            payload["DESIGN_DOCS"] = await self.shell_tools.get_design_docs_content()

        if self.agent_config.get("use_git_info", False):
            payload["GIT_INFO"] = await self.shell_tools.get_git_info()

        if self.agent_config.get("use_project_tree", False):
            payload["PROJECT_TREE"] = await self.shell_tools.get_project_tree()

        if self.agent_config.get("use_source_code", False):
            payload["SRC_CODE"] = await self.shell_tools.process_directory(
                self.source_directory
            )

        if self.agent_config.get("use_common_project_files", False):
            payload["PROJECT_TOP_FILES"] = (
                await self._get_common_project_files_context()
            )

        if filepath and self.agent_config.get("use_source_file", False):
            payload["SOURCE_FILE"] = {
                "path": filepath,
                "content": await self.shell_tools.read_file_content(Path(filepath)),
            }

        return payload

    async def _get_common_project_files_context(self) -> Dict[str, Optional[str]]:
        """
        Gathers context from common, project-defining files.

        Returns:
            A dictionary containing the content of key project files.
        """
        project_files_to_read = self.config.get("common_project_files", [])
        common_context = {}

        async def read_file(file_path: Path) -> str:
            return await self.shell_tools.read_file_content(file_path)

        tasks = [
            read_file(self.target_directory / filename)
            for filename in project_files_to_read
        ]
        contents = await asyncio.gather(*tasks, return_exceptions=True)

        for filename, content in zip(project_files_to_read, contents):
            if isinstance(content, Exception):
                logger.error("Failed to read file '%s': %s", filename, content)
                common_context[filename] = None
            else:
                common_context[filename] = content

        return common_context

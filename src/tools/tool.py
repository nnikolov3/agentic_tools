# src/tools/tool.py
"""
Module: src/tools/tool.py

Purpose:
This module defines the `Tool` class, which acts as a central orchestrator for
various shell and API tools used by agents. Its primary responsibility is to
construct the appropriate payload for API calls based on agent type and context,
and then delegate the execution to specialized tool handlers (e.g., `ShellTools`,
`ApiTools`). This design promotes separation of concerns, ensuring that the core
tool execution logic remains independent of agent-specific behaviors.

Role in the System:
The `Tool` class serves as an abstraction layer, enabling agents to interact
with underlying system functionalities (like file operations, git commands) and
external APIs in a standardized manner. It translates agent requirements into
executable API requests by dynamically building context-rich payloads using a
factory pattern for different agents.
"""

# Standard Library imports
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

# Local Application/Module imports
from src.tools.api_tools import ApiTools
from src.tools.shell_tools import ShellTools

# Self-Documenting Code: Dedicated logger for tool operations traceability.
logger: logging.Logger = logging.getLogger(__name__)


class Tool:
    """
    Orchestrates shell and API tools for agent execution.

    Manages payload construction from git diffs, design docs, source code, and
    memory context, then routes to the appropriate API provider for response
    generation.
    """

    def __init__(
        self, agent: str, config: dict[str, Any], target_directory: Path
    ) -> None:
        """
        Initializes Tool with agent-specific configuration and tool instances.

        Args:
            agent: The agent name for sub-config and tool instantiation.
            config: The full configuration dictionary.
            target_directory: The root directory of the project to operate on.
        """
        self.agent: str = agent
        self.config: dict[str, Any] = config
        self.shell_tools: ShellTools = ShellTools(agent, config, target_directory)
        self.api_tools: ApiTools = ApiTools(agent, config)

        self.agent_config: dict[str, Any] = config.get(agent, {})
        self.agent_skills: Optional[list[str]] = self.agent_config.get("skills")
        self.agent_prompt: Optional[str] = self.agent_config.get("prompt")

        prompts_config: dict[str, Any] = config.get("prompts", {})
        golden_rules: Optional[str] = prompts_config.get("golden_rules")

        if self.agent_prompt and golden_rules:
            self.agent_prompt = f"{self.agent_prompt}\n\n{golden_rules}"

        self.target_directory: Path = target_directory
        source_dirs: Optional[list[str]] = config.get("source")
        # Use the first source directory specified, or default to 'src'.
        # This makes the primary source code location configurable.
        source_dir_name = source_dirs[0] if source_dirs else "src"
        self.source_directory = self.target_directory / source_dir_name

    async def run_tool(
        self,
        chat: Optional[Any],
        memory_context: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Constructs a payload and executes the agent's primary task via an API call.

        Builds context from shell tools (git, docs, source) and memory, then
        delegates to the API for generation. Agent-specific payload construction
        is handled by `_create_payload`.

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
    ) -> dict[str, Any]:
        """
        Creates the payload for the API call based on the agent type.

        This method acts as a dispatcher, routing to specialized private methods
        to construct payloads for different agents. This approach adheres to the
        Single Responsibility and Open/Closed principles, making it easier to
        add or modify agent payloads without affecting others.

        Args:
            chat: The primary input for the agent.
            memory_context: Retrieved memory context string.
            filepath: The path to the file being processed.

        Returns:
            A dictionary representing the payload for the API call.

        Raises:
            ValueError: If a required argument (like filepath) is missing for an agent.
        """
        logger.info("Creating payload for agent '%s'.", self.agent)
        match self.agent:
            case "commentator":
                if not filepath:
                    raise ValueError(
                        "Filepath is required for the 'commentator' agent."
                    )
                return await self._create_commentator_payload(chat, filepath)
            case "developer":
                if not filepath:
                    raise ValueError("Filepath is required for the 'developer' agent.")
                return await self._create_developer_payload(
                    chat, memory_context, filepath
                )

            case "readme_writer":
                return await self._create_readme_writer_payload(chat, memory_context)
            case "approver":
                return await self._create_approver_payload(chat, memory_context)
            case "linter_analyst":
                return await self._create_linter_analyst_payload(chat, memory_context)
            case "configuration_builder":
                return await self._create_configuration_builder_payload(
                    chat, memory_context
                )
            case "expert":
                return await self._create_default_payload(chat, memory_context)
            case _:
                return await self._create_default_payload(chat, memory_context)

    async def _get_common_project_files_context(self) -> dict[str, Any]:
        """
        Gathers context from common, project-defining files.

        This helper consolidates file reading logic to avoid duplication. The
        list of files to read is made explicit, improving maintainability.

        Returns:
            A dictionary containing the content of key project files.
        """
        # This list explicitly defines which files constitute the common project context.
        # Adding or removing a file requires changing only this single location.
        project_files_to_read: list[str] = self.config.get("common_project_files", [])

        common_context: dict[str, str | None] = {}
        for filename in project_files_to_read:
            file_path = self.target_directory / filename
            common_context[filename] = await self.shell_tools.read_file_content(
                file_path
            )
        return common_context

    async def _create_commentator_payload(
        self,
        chat: Optional[Any],
        filepath: str,
    ) -> dict[str, Any]:
        """
        Creates a minimal payload for the 'commentator' agent.

        Why: The commentator is a pure text transformer. It requires a minimal
        payload containing only the file content to avoid confusing the LLM with
        broader, irrelevant project context.
        """
        logger.info("Using file-specific context for 'commentator' agent.")
        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "DESIGN_DOCS": await self.shell_tools.get_design_docs_content(),
            "PROJECT_TOP_FILES": await self._get_common_project_files_context(),
            "SOURCE_FILE": {
                "path": filepath,
                "content": await self.shell_tools.read_file_content(Path(filepath)),
            },
        }

    async def _create_developer_payload(
        self,
        chat: Optional[Any],
        memory_context: Optional[str],
        filepath: str,
    ) -> dict[str, Any]:
        """Creates a detailed, structured payload for the 'developer' agent."""
        logger.info("Using development-specific context for 'developer' agent.")
        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "MEMORY_CONTEXT": memory_context,
            "DESIGN_DOCS": await self.shell_tools.get_design_docs_content(),
            "SOURCE_FILE": {
                "path": filepath,
                "content": await self.shell_tools.read_file_content(Path(filepath)),
            },
            "SRC_CODE": await self.shell_tools.process_directory(self.source_directory),
            "PROJECT_TREE": await self.shell_tools.get_project_tree(),
            "PROJECT_TOP_FILES": await self._get_common_project_files_context(),
        }

    async def _create_readme_writer_payload(
        self, chat: Optional[Any], memory_context: Optional[str]
    ) -> dict[str, Any]:
        """Constructs a project-wide context payload for the 'readme_writer' agent."""
        logger.info("Using project context for 'readme_writer' agent.")
        payload = {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "MEMORY_CONTEXT": memory_context,
            "GIT_INFO": await self.shell_tools.get_git_info(),
            "SRC_CODE": await self.shell_tools.process_directory(self.source_directory),
            "PROJECT_TREE": await self.shell_tools.get_project_tree(),
            "PROJECT_TOP_FILES": await self._get_common_project_files_context(),
        }

        return payload

    async def _create_approver_payload(
        self, chat: Optional[Any], memory_context: Optional[str]
    ) -> dict[str, Any]:
        """Constructs a payload with git diff and design docs for the 'approver' agent."""
        logger.info("Using diff and design context for 'approver' agent.")

        git_context, design_docs = await asyncio.gather(
            self.shell_tools.get_git_context_for_patch(),
            self.shell_tools.get_design_docs_content(),
        )

        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "GIT_DIFF_PATCH": git_context,
            "DESIGN_DOCS": design_docs,
            "MEMORY_CONTEXT": memory_context,
            "PROJECT_TREE": await self.shell_tools.get_project_tree(),
        }

    async def _create_linter_analyst_payload(
        self,
        chat: Optional[
            str
        ],  # This is the raw linter report from the agent's run_agent method
        memory_context: Optional[str],
    ) -> dict[str, Any]:
        """Constructs a payload with linter output and design docs for the 'linter_analyst' agent."""
        logger.info(
            "Using linter report and design context for 'linter_analyst' agent."
        )
        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "MEMORY_CONTEXT": memory_context,
            "DESIGN_DOCS": await self.shell_tools.get_design_docs_content(),
        }

    async def _create_configuration_builder_payload(
        self,
        chat: Optional[Any],
        memory_context: Optional[str],
    ) -> dict[str, Any]:
        """
        Constructs a project-wide context payload for the 'configuration_builder' agent.

        This agent inspects the entire project to generate a configuration file.
        It requires comprehensive context including the project's file structure,
        source code, and key definition files to make an informed best-effort
        configuration.
        """
        logger.info("Using project-wide context for 'configuration_builder' agent.")

        # Gather comprehensive project context concurrently to provide the LLM
        # with as much information as possible for a high-quality configuration.
        (
            src_code,
            project_tree,
            project_top_files,
            design_docs,
        ) = await asyncio.gather(
            self.shell_tools.process_directory(self.source_directory),
            self.shell_tools.get_project_tree(),
            self._get_common_project_files_context(),
            self.shell_tools.get_design_docs_content(),
        )

        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "MEMORY_CONTEXT": memory_context,
            "SRC_CODE": src_code,
            "PROJECT_TREE": project_tree,
            "PROJECT_TOP_FILES": project_top_files,
            "DESIGN_DOCS": design_docs,
        }

    async def _create_default_payload(
        self,
        chat: Optional[Any],
        memory_context: Optional[str],
    ) -> dict[str, Any]:
        """Constructs a default, comprehensive payload for other agents."""
        logger.info("Using standard context payload for '%s'.", self.agent)
        payload = {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "GIT_INFO": await self.shell_tools.get_git_info(),
            "SRC_CODE": await self.shell_tools.process_directory(self.source_directory),
            "MEMORY_CONTEXT": memory_context,
            "PROJECT_TOP_FILES": await self._get_common_project_files_context(),
            "PROJECT_TREE": await self.shell_tools.get_project_tree(),
        }
        return payload

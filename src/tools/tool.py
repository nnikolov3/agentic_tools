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

    def __init__(self, agent: str, config: dict[str, Any]) -> None:
        """
        Initializes Tool with agent-specific configuration and tool instances.

        Args:
            agent: The agent name for sub-config and tool instantiation.
            config: The full configuration dictionary.
        """
        self.agent: str = agent
        self.config: dict[str, Any] = config
        self.shell_tools: ShellTools = ShellTools(agent, config)
        self.api_tools: ApiTools = ApiTools(agent, config)

        self.agent_config: dict[str, Any] = config.get(agent, {})
        self.agent_skills: Optional[list[str]] = self.agent_config.get("skills")
        self.agent_prompt: Optional[str] = self.agent_config.get("prompt")

        self.current_working_directory: Path = Path.cwd()
        source_dirs: Optional[list[str]] = config.get("source")
        # Use the first source directory specified, or default to 'src'.
        # This makes the primary source code location configurable.
        source_dir_name = source_dirs[0] if source_dirs else "src"
        self.source_directory = self.current_working_directory / source_dir_name

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
                logger.info(f"Tool execution completed for agent '{self.agent}'.")
                return response
            logger.warning(f"API call for agent '{self.agent}' returned no response.")
            return None
        except ValueError as value_error:
            logger.error(f"Payload creation failed for agent '{self.agent}': {value_error}")
            raise
        except Exception as unexpected_error:
            logger.error(
                f"An unexpected error occurred in run_tool for agent '{self.agent}': {unexpected_error}",
                exc_info=True,
            )
            raise RuntimeError(f"Tool execution failed for agent '{self.agent}'") from unexpected_error

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
        logger.info(f"Creating payload for agent '{self.agent}'.")
        match self.agent:
            case "commentator":
                if not filepath:
                    raise ValueError("Filepath is required for the 'commentator' agent.")
                return self._create_commentator_payload(chat, filepath)
            case "developer":
                if not filepath:
                    raise ValueError("Filepath is required for the 'developer' agent.")
                return self._create_developer_payload(chat, memory_context, filepath)
            case "readme_writer":
                return self._create_readme_writer_payload(chat)
            case "approver":
                return self._create_approver_payload(chat)
            case _:
                return self._create_default_payload(chat, memory_context)

    def _get_common_project_files_context(self) -> dict[str, Any]:
        """
        Gathers context from common, project-defining files.

        This helper consolidates file reading logic to avoid duplication. The
        list of files to read is made explicit, improving maintainability.

        Returns:
            A dictionary containing the content of key project files.
        """
        # This list explicitly defines which files constitute the common project context.
        # Adding or removing a file requires changing only this single location.
        project_files_to_read = self.config.get("common_project_files", {})

        common_context = {}
        for key, filename in project_files_to_read.items():
            file_path = self.current_working_directory / filename
            common_context[key] = self.shell_tools.read_file_content(file_path)
        return common_context

    def _create_commentator_payload(
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
            "USER_PROMPT": chat,  # For commentator, 'chat' is the full file content.
            "SOURCE_FILE": {
                "path": filepath,
                "content": self.shell_tools.read_file_content(Path(filepath)),
            },
        }

    def _create_developer_payload(
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
            "DESIGN_DOCS": self.shell_tools.get_design_docs_content(),
            "SOURCE_FILE": {
                "path": filepath,
                "content": self.shell_tools.read_file_content(Path(filepath)),
            },
        }

    def _create_readme_writer_payload(self, chat: Optional[Any]) -> dict[str, Any]:
        """Constructs a project-wide context payload for the 'readme_writer' agent."""
        logger.info("Using project context for 'readme_writer' agent.")
        payload = {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "GIT_INFO": self.shell_tools.get_git_info(),
            "SRC_CODE": self.shell_tools.process_directory(self.source_directory),
        }
        # Combine the base payload with the common project file context.
        payload.update(self._get_common_project_files_context())
        return payload

    def _create_approver_payload(self, chat: Optional[Any]) -> dict[str, Any]:
        """Constructs a payload with git diff and design docs for the 'approver' agent."""
        logger.info("Using diff and design context for 'approver' agent.")
        return {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "GIT_DIFF_PATCH": self.shell_tools.create_patch(),
            "DESIGN_DOCS": self.shell_tools.get_design_docs_content(),
        }

    def _create_default_payload(
        self,
        chat: Optional[Any],
        memory_context: Optional[str],
    ) -> dict[str, Any]:
        """Constructs a default, comprehensive payload for other agents."""
        logger.info(f"Using standard context payload for '{self.agent}'.")
        payload = {
            "SYSTEM_PROMPT": self.agent_prompt,
            "SKILLS": self.agent_skills,
            "USER_PROMPT": chat,
            "GIT_INFO": self.shell_tools.get_git_info(),
            "SRC_CODE": self.shell_tools.process_directory(self.source_directory),
            "MEMORY_CONTEXT": memory_context,
        }
        # Combine the base payload with the common project file context.
        payload.update(self._get_common_project_files_context())
        return payload
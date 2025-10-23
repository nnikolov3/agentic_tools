"""
Purpose:
This module defines the Tool class, which orchestrates shell and API tools for agent tasks.
It emphasizes simplicity by constructing payloads explicitly and delegating to specialized tools,
ensuring separation of concerns without agent-specific logic in the core execution flow.
"""

import logging
import os
from pathlib import Path
from typing import Any


from src.tools.api_tools import ApiTools
from src.tools.shell_tools import ShellTools

# Self-Documenting Code: Dedicated logger for tool operations traceability.
logger: logging.Logger = logging.getLogger(__name__)


class Tool:
    """
    Orchestrates shell and API tools for agent execution.

    Manages payload construction from git diffs, design docs, source code, and memory context,
    then routes to the appropriate API provider for response generation.
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
        self.payload: dict[str, Any] = {}
        self.response: Any | None = None
        self.project_root_path: str | None = config.get("project_root")
        self.docs: list[str] | None = config.get("docs")
        self.agent_config: dict[str, Any] = config.get(agent, {})
        self.agent_skills: list[str] | None = self.agent_config.get("skills")
        self.agent_prompt: str | None = self.agent_config.get("prompt")
        self.current_working_directory: str = os.getcwd()
        self.source: list[str] | None = config.get("source")

    async def run_tool(
        self, chat: Any | None, memory_context: str | None = None
    ) -> Any:
        """
        Constructs a payload and executes the agent's primary task via an API call.

        Builds context from shell tools (git, docs, source) and memory, then delegates
        to API for generation. Agent-specific post-processing is handled upstream in Agent.

        Args:
            chat: The current chat input or message.
            memory_context: Retrieved memory context string (optional).

        Returns:
            The raw response from the API provider.
        """
        source_dir: str = str(Path(f"{self.current_working_directory}/src"))
        # Explicit Payload Construction: Gather all context components.
        self.payload["prompt"] = self.agent_prompt
        self.payload["skills"] = self.agent_skills
        self.payload["memory"] = memory_context
        self.payload["git-diff-patch"] = self.shell_tools.create_patch()
        self.payload["git-info"] = self.shell_tools.get_git_info()
        self.payload["design_documents"] = self.shell_tools.get_design_docs_content()
        self.payload["source_code"] = self.shell_tools.process_directory(source_dir)
        self.payload["chat"] = chat

        # print(json.dumps(self.payload))

        # Delegate to API for execution.
        self.response = await self.api_tools.run_api(self.payload)
        logger.info(f"Tool execution completed for agent '{self.agent}'.")

        return self.response

"""
Module: src.utils.payload_builder

Defines the PayloadBuilder utility, which is responsible for constructing
the complete payload dictionary for an agent's API call. This utility
centralizes the logic for gathering context from various sources, such as
shell commands and memory, into a single, consistent structure.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.shell_utils import ShellUtils

logger: logging.Logger = logging.getLogger(__name__)


class PayloadBuilder:
    """
    Constructs the payload for an agent's API call.
    """

    def __init__(
        self,
        agent_config: Dict[str, Any],
        target_directory: Optional[Path] = None,
    ) -> None:
        """
        Initializes the PayloadBuilder with agent-specific configuration.
        """
        self.agent_config = agent_config
        self.shell_utils = ShellUtils(agent_config, target_directory)

    async def create_payload(
        self,
        chat: Optional[str],
        memory_context: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a unified payload for the API call.

        This method constructs a single, consistent payload structure for all
        agents. It gathers context dynamically by executing shell commands
        defined in the agent's configuration.

        Why: This approach adheres to the "Explicit over Implicit" and
        "Configuration over Code" principles. The agent's required context is
        defined declaratively in the config file, making the system transparent
        and easy to extend without modifying the code.

        Args:
            chat: The primary input for the agent.
            memory_context: Retrieved memory context string.
            filepath: The path to the file being processed.

        Returns:
            A dictionary representing the payload for the API call.
        """
        logger.info(
            "Creating payload for agent '%s'.", self.agent_config.get("name", "unknown")
        )

        # Asynchronously gather context from configured shell tools.
        tool_outputs = await self.shell_utils.execute_tool_commands()

        # Build the unified payload structure.
        payload = {
            "agent_name": self.agent_config.get("name"),
            "chat_message": chat,
            "context": {
                "memory": memory_context,
                "file_path": filepath,
                **tool_outputs,  # Unpack all tool outputs into the context.
            },
            "system_prompt": self.agent_config.get("prompt"),
            "skills": self.agent_config.get("skills"),
        }

        return payload

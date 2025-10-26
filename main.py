# -*- coding: utf-8 -*-
"""
This module serves as the main entry point for executing various agentic tools.

It orchestrates the invocation of specialized agents designed for specific tasks,
such as code quality enforcement, development assistance, architectural planning,
and documentation generation. Each tool is registered using the `fastmcp`
framework, making them discoverable and executable via a command-line interface.

The module initializes a global configuration, sets up logging, and defines the
available tools, each corresponding to a distinct agent responsible for a
particular aspect of the software development lifecycle.
"""

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

from src.agents.agent import AGENT_CLASSES, DefaultAgent
from src.configurator import get_config_dictionary

# --- Global Setup ---

# Adherence to 'Parameterize Everything' principle.
# Centralized constants for configuration and application naming.
CONFIG_FILE_PATH: Path = Path("conf/agentic_tools.toml")
MCP_NAME: str = "agentic-tools"

# Configure basic logging for clear, explicit output.
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger: logging.Logger = logging.getLogger(__name__)

# Initialize the configurator and load the configuration dictionary.
# This approach centralizes configuration loading and error handling.
try:
    configuration: dict[str, Any] = get_config_dictionary(CONFIG_FILE_PATH)
except FileNotFoundError as e:
    logger.error(f"Configuration file not found at '{CONFIG_FILE_PATH}'.")
    raise RuntimeError(f"Missing configuration file: {CONFIG_FILE_PATH}") from e
except Exception as e:
    logger.error(f"Failed to load or parse configuration: {e}")
    raise RuntimeError("Configuration loading failed") from e

# Initialize the FastMCP instance, which serves as the tool runner.
mcp = FastMCP(MCP_NAME)

# --- Core Logic ---

async def _run_agent_tool(
    agent_name: str,
    chat: Optional[str],
    filepath: Optional[str | PathLike[str]],
) -> Any:
    """
    A generic factory function to instantiate and run a specified agent.

    This helper function centralizes agent execution logic, reducing code
    duplication and improving maintainability. It handles agent instantiation,
    execution, and error logging for all tools.

    Args:
        agent_name: The name of the agent to run (e.g., 'readme_writer').
        chat: The optional chat context or prompt for the agent.
        filepath: The optional path to a file or directory for the agent to process.

    Returns:
        The result of the agent's operation.

    Raises:
        RuntimeError: If the agent execution fails for any reason.
    """
    logger.info(f"Executing '{agent_name}' tool.")
    try:
        agent_class = AGENT_CLASSES.get(agent_name, DefaultAgent)
        agent = agent_class(
            configuration=configuration,
            agent_name=agent_name,
            project=MCP_NAME,
            chat=chat,
            filepath=filepath,
        )
        return await agent.run_agent()
    except Exception as error:
        logger.error(
            f"An error occurred while running the '{agent_name}' tool.",
            exc_info = True,
            )
        # Re-raising after logging ensures the failure is visible to the caller
        # and preserves the original exception context.
        raise RuntimeError(f"Agent '{agent_name}' failed to execute.") from error


# --- Tool Definitions ---


@mcp.tool(
    description = (
        "Walks the project directories, gets git information, and updates the README.md file."
    ),
    )
async def readme_writer_tool(
    chat: Optional[str] = None, filepath: Optional[str | PathLike[str]] = None,
    ) -> Any:
    """
    Invokes the 'readme_writer' agent to update the project's README.md file.

    Args:
        chat: An optional prompt guiding the content generation.
        filepath: The path to the README.md file or its containing directory.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool("readme_writer", chat, filepath)


@mcp.tool(description = "Audits recent code changes and approves or rejects them.")
async def approver_tool(
    chat: Optional[str] = None, filepath: Optional[str | PathLike[str]] = None,
    ) -> Any:
    """
    Invokes the 'approver' agent to audit recent code changes.

    Args:
        chat: An optional prompt specifying approval criteria or review focus.
        filepath: The path to the code changes, repository, or specific files.

    Returns:
        The result of the agent's audit.
    """
    return await _run_agent_tool("approver", chat, filepath)


@mcp.tool(
    description = "Writes high-quality code based on design guidelines and standards.",
    )
async def developer_tool(
    chat: Optional[str] = None, filepath: Optional[str | PathLike[str]] = None,
    ) -> Any:
    """
    Invokes the 'developer' agent to write or modify code.

    Args:
        chat: An optional prompt guiding the code generation process.
        filepath: The path to the file where code modification is required.

    Returns:
        The result of the agent's operation, such as the modified code.
    """
    return await _run_agent_tool("developer", chat, filepath)


@mcp.tool(
    description = "Creates a high-quality architecture based on design guidelines.",
    )
async def architect_tool(
    chat: Optional[str] = None, filepath: Optional[str | PathLike[str]] = None,
    ) -> Any:
    """
    Invokes the 'architect' agent to assist in software architecture design.

    Args:
        chat: An optional prompt specifying architectural requirements or goals.
        filepath: The path to relevant project files for architectural analysis.

    Returns:
        The result of the agent's operation, such as a design document.
    """
    return await _run_agent_tool("architect", chat, filepath)


@mcp.tool(description = "Updates a source code file with comments and documentation.")
async def commentator_tool(
    chat: Optional[str] = None, filepath: Optional[str | PathLike[str]] = None,
    ) -> Any:
    """
    Invokes the 'commentator' agent to add documentation to source code.

    Args:
        chat: An optional prompt specifying documentation requirements.
        filepath: The path to the source code file that needs commenting.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool("commentator", chat, filepath)


# --- Main Execution Block ---

if __name__ == "__main__":
    # This block ensures that the FastMCP application runs only when the script
    # is executed directly, not when imported as a module.
    mcp.run()

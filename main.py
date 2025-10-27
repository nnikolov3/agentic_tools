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

import argparse
import asyncio
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastmcp import FastMCP

from src.agents.agent import AGENT_CLASSES, DefaultAgent
from src.configurator import get_config_dictionary
from src.scripts.ingest_knowledge_bank import KnowledgeBankIngestor

# --- Global Setup ---

# Adherence to 'Parameterize Everything' principle.
# Centralized constants for configuration and application naming.
CONFIG_FILE_PATH: Path = Path("conf/agentic_tools.toml")
MCP_NAME: str = "agentic-tools"

# Configure basic logging for clear, explicit output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# Agent names are defined as constants to prevent typos and magic strings.
COMMENTATOR: str = "commentator"
DEVELOPER: str = "developer"
ARCHITECT: str = "architect"
APPROVER: str = "approver"
README_WRITER: str = "readme_writer"
INGEST_KNOWLEDGE_BANK: str = "ingest_knowledge_bank"
EXPERT: str = "expert"
VALID_AGENTS: Tuple[str, ...] = (
    COMMENTATOR,
    DEVELOPER,
    ARCHITECT,
    APPROVER,
    README_WRITER,
    INGEST_KNOWLEDGE_BANK,
    EXPERT,
)


# Initialize the configurator and load the configuration dictionary.
# This approach centralizes configuration loading and error handling.
def _load_configuration(config_path: Path) -> dict[str, Any]:
    """Loads the application configuration from the specified path."""
    try:
        configuration: dict[str, Any] = get_config_dictionary(config_path)
        return configuration
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found at '{config_path}'.")
        raise RuntimeError(f"Missing configuration file: {config_path}") from e
    except Exception as e:
        logger.error(f"Failed to load or parse configuration: {e}")
        raise RuntimeError("Configuration loading failed") from e


# Initialize the FastMCP instance, which serves as the tool runner.
mcp = FastMCP(MCP_NAME)

# Global configuration variable, initialized later based on mode.
configuration: dict[str, Any] = {}

# --- Core Logic ---


async def _execute_agent_task(
    config: Dict[str, Any],
    agent_name: str,
    chat: Optional[str],
    filepath: Optional[str | PathLike[str]],
) -> Any:
    """
    Initializes and runs a single agent task.

    This function encapsulates the logic for creating an agent instance and
    invoking its primary execution method. It includes error handling to
    log failures during the agent's run.

    Args:
        config: The application configuration dictionary.
        agent_name: The name of the agent to execute.
        chat: The input prompt or context for the agent.
        filepath: The path to a file or directory for the agent to process.

    Returns:
        The result produced by the agent's execution. The type is 'Any' as
        different agents may return different types of data.

    Raises:
        RuntimeError: If the agent's execution fails, wrapping the original exception.
    """
    logger.info(f"Executing '{agent_name}' tool.")

    if agent_name == INGEST_KNOWLEDGE_BANK:
        return await KnowledgeBankIngestor(config["agentic-tools"]).run_ingestion()

    try:
        agent_class = AGENT_CLASSES.get(agent_name, DefaultAgent)
        agent = agent_class(
            configuration=config,
            agent_name=agent_name,
            project=MCP_NAME,
            chat=chat,
            filepath=filepath,
        )
        return await agent.run_agent()
    except Exception as error:
        logger.error(
            f"An error occurred while running the '{agent_name}' tool.",
            exc_info=True,
        )
        # Re-raising after logging ensures the failure is visible to the caller
        # and preserves the original exception context.
        raise RuntimeError(f"Agent '{agent_name}' failed to execute.") from error


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
    return await _execute_agent_task(configuration, agent_name, chat, filepath)


# --- Tool Definitions ---


@mcp.tool(
    description=(
        "Walks the project directories, gets git information, and updates the README.md file."
    ),
)
async def readme_writer_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
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


@mcp.tool(description="Audits recent code changes and approves or rejects them.")
async def approver_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
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
    description="Writes high-quality code based on design guidelines and standards.",
)
async def developer_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
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
    description="Creates a high-quality architecture based on design guidelines.",
)
async def architect_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
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


@mcp.tool(description="Updates a source code file with comments and documentation.")
async def commentator_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
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


# --- CLI Execution Logic ---


def _setup_cli_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the command-line interface.

    Returns:
        An `argparse.ArgumentParser` instance configured with the script's
        command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to manually trigger AI agents or run the FastMCP server.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Top-level arguments for configuration
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_FILE_PATH,
        help="Path to the configuration file (e.g., conf/agentic_tools.toml).",
    )

    # Subparser for the 'run-agent' command
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_agent_parser = subparsers.add_parser(
        "run-agent",
        help="Manually execute a single agent task.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run_agent_parser.add_argument(
        "agent_name",
        type=str,
        choices=VALID_AGENTS,
        help=f"The name of the agent to run. Choices: {', '.join(VALID_AGENTS)}",
        metavar="AGENT_NAME",
    )
    run_agent_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="The chat prompt or context for the agent.",
    )
    run_agent_parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="The path to the file or directory for the agent to process.",
    )

    return parser


async def _run_cli_mode(args: argparse.Namespace) -> None:
    """
    Executes the selected agent task in CLI mode.
    """
    global configuration
    try:
        # Load configuration based on the --config argument
        config_path = Path(args.config)
        configuration = _load_configuration(config_path)
        logger.info(f"Configuration loaded successfully from {config_path}.")
    except RuntimeError:
        return

    try:
        result = await _execute_agent_task(
            configuration,
            args.agent_name,
            args.chat,
            args.filepath,
        )
        logger.info("--- Agent Result ---")
        # The result can be any data type, so printing it directly provides the
        # clearest output for a developer using this manual tool.
        print(result)
        logger.info("--------------------")
    except Exception as e:
        # The specific error is already logged within execute_agent_task.
        # This catch prevents a top-level traceback, providing a cleaner exit.
        logger.critical(
            f"Script execution failed. Please review the logs for details. Error: {e}",
        )


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = _setup_cli_parser()
    args = parser.parse_args()

    if args.command == "run-agent":
        # Run in CLI mode
        asyncio.run(_run_cli_mode(args))
    else:
        # Run in FastMCP server mode (default)
        # Load default configuration for server mode
        configuration = _load_configuration(CONFIG_FILE_PATH)
        mcp.run()

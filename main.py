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
import sys
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
KNOWLEDGE_BASE_BUILDER: str = "knowledge_base_builder"
LINTER_ANALYST: str = "linter_analyst"
CONFIGURATION_BUILDER: str = "configuration_builder"

# Centralized agent metadata to act as a single source of truth for descriptions.
# This adheres to the DRY (Don't Repeat Yourself) principle.
AGENT_METADATA: Dict[str, str] = {
    APPROVER: "Audits recent code changes and approves or rejects them.",
    ARCHITECT: "Creates a high-quality architecture based on design guidelines.",
    COMMENTATOR: "Updates a source code file with comments and documentation.",
    CONFIGURATION_BUILDER: "Automatically generates a TOML configuration file for a project.",
    DEVELOPER: "Writes high-quality code based on design guidelines and standards.",
    EXPERT: "Provides expert-level answers and solutions based on a given context.",
    INGEST_KNOWLEDGE_BANK: "Processes and ingests data into the knowledge bank.",
    KNOWLEDGE_BASE_BUILDER: "Fetches content from URLs and saves it to a file for knowledge base creation.",
    LINTER_ANALYST: "Runs project linters and generates a prioritized analysis report.",
    README_WRITER: "Walks the project directories, gets git information, and updates the README.md file.",
}

# The list of valid agents is derived directly from the metadata dictionary.
VALID_AGENTS: Tuple[str, ...] = tuple(AGENT_METADATA.keys())


# Initialize the configurator and load the configuration dictionary.
# This approach centralizes configuration loading and error handling.
def _load_configuration(config_path: Path) -> dict[str, Any]:
    """Loads the application configuration from the specified path."""
    try:
        configuration_data: dict[str, Any] = get_config_dictionary(config_path)
        return configuration_data
    except FileNotFoundError:
        logger.error("Configuration file not found at '%s'.", config_path)
        sys.exit(1)
    except Exception as error:
        logger.error("Failed to load or parse configuration: %s", error)
        raise RuntimeError("Configuration loading failed.") from error


# Initialize the FastMCP instance, which serves as the tool runner.
mcp = FastMCP(MCP_NAME)

# Global configuration variable, initialized based on the execution mode.
configuration: dict[str, Any] = {}


# --- Core Logic ---


async def _execute_agent_task(
    config: Dict[str, Any],
    agent_name: str,
    project: str,
    chat: Optional[str],
    filepath: Optional[str | PathLike[str]],
    target_directory: Optional[Path],
    **kwargs: Any,
) -> Any:
    """
    Initializes and runs a single agent task.

    This function encapsulates the logic for creating an agent instance and
    invoking its primary execution method. It includes error handling to

    log failures during the agent's run.

    Args:
        config: The application configuration dictionary.
        agent_name: The name of the agent to execute.
        project: The name of the project context.
        chat: The input prompt or context for the agent.
        filepath: The path to a file or directory for the agent to process.
        target_directory: The root directory of the project the agent will operate on.
        **kwargs: Additional keyword arguments to pass to the agent.

    Returns:
        The result produced by the agent's execution.

    Raises:
        RuntimeError: If the agent's execution fails, wrapping the original exception.
    """
    logger.info("Executing '%s' tool.", agent_name)

    if agent_name == INGEST_KNOWLEDGE_BANK:
        ingestor = KnowledgeBankIngestor(config["agentic-tools"])
        return await ingestor.run_ingestion()

    try:
        agent_class = AGENT_CLASSES.get(agent_name, DefaultAgent)
        agent = agent_class(
            configuration=config,
            agent_name=agent_name,
            project=project,
            chat=chat,
            filepath=filepath,
            target_directory=target_directory,
            **kwargs,
        )
        return await agent.run_agent()
    except Exception as error:
        logger.error(
            "An error occurred while running the '%s' tool.", agent_name, exc_info=True
        )
        # Re-raising after logging ensures the failure is visible to the caller
        # and preserves the original exception context.
        raise RuntimeError(f"Agent '{agent_name}' failed to execute.") from error


async def _run_agent_tool(
    agent_name: str,
    chat: Optional[str],
    filepath: Optional[str | PathLike[str]],
    target_directory: Path,
    **kwargs: Any,
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
        target_directory: The root directory of the project the agent will operate on.
        **kwargs: Additional keyword arguments to pass to the agent.

    Returns:
        The result of the agent's operation.
    """
    return await _execute_agent_task(
        configuration, agent_name, MCP_NAME, chat, filepath, target_directory, **kwargs
    )


# --- Tool Definitions ---


@mcp.tool(description=AGENT_METADATA[README_WRITER])
async def readme_writer_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'readme_writer' agent to update the project's README.md file.

    Args:
        chat: An optional prompt guiding the content generation.
        filepath: The path to the README.md file or its containing directory.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool(README_WRITER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[APPROVER])
async def approver_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'approver' agent to audit recent code changes.

    Args:
        chat: An optional prompt specifying approval criteria or review focus.
        filepath: The path to the code changes, repository, or specific files.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's audit.
    """
    return await _run_agent_tool(APPROVER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[DEVELOPER])
async def developer_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'developer' agent to write or modify code.

    Args:
        chat: An optional prompt guiding the code generation process.
        filepath: The path to the file where code modification is required.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation, such as the modified code.
    """
    return await _run_agent_tool(DEVELOPER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[ARCHITECT])
async def architect_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'architect' agent to assist in software architecture design.

    Args:
        chat: An optional prompt specifying architectural requirements or goals.
        filepath: The path to relevant project files for architectural analysis.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation, such as a design document.
    """
    return await _run_agent_tool(ARCHITECT, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[COMMENTATOR])
async def commentator_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'commentator' agent to add documentation to source code.

    Args:
        chat: An optional prompt specifying documentation requirements.
        filepath: The path to the source code file that needs commenting.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool(COMMENTATOR, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[KNOWLEDGE_BASE_BUILDER])
async def knowledge_base_builder_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'knowledge_base_builder' agent to fetch web content.

    Args:
        chat: A comma-separated string of URLs to fetch.
        filepath: The path to the output file where content will be saved.
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool(
        KNOWLEDGE_BASE_BUILDER, chat, filepath, target_directory
    )


@mcp.tool(description=AGENT_METADATA[LINTER_ANALYST])
async def linter_analyst_tool(
    chat: Optional[str] = None,
    filepath: Optional[str | PathLike[str]] = None,
    target_directory: Path = Path.cwd(),
) -> Any:
    """
    Invokes the 'linter_analyst' agent.

    Args:
        chat: An optional user prompt to focus the analysis.
        filepath: The path to relevant project files for analysis (optional).
        target_directory: The root directory of the project the agent will operate on.

    Returns:
        The result of the agent's operation.
    """
    return await _run_agent_tool(LINTER_ANALYST, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[CONFIGURATION_BUILDER])
async def configuration_builder_tool(
    chat: Optional[str] = None,
    output_filename: str = "generated_config.toml",
    target_directory: Path = Path.cwd(),
    dependency_file: str = "pyproject.toml",
) -> Any:
    """
    Invokes the 'configuration_builder' agent to generate a project configuration file.

    Args:
        chat: An optional prompt to guide the configuration generation.
        output_filename: The path to the output file where the configuration will be saved.
        target_directory: The root directory of the project to analyze.
        dependency_file: The name of the dependency file to inspect.
    """
    return await _run_agent_tool(
        CONFIGURATION_BUILDER,
        chat,
        filepath=output_filename,
        target_directory=target_directory,
        dependency_file=dependency_file,
    )


# --- CLI Execution Logic ---


def _setup_cli_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the command-line interface.

    Returns:
        An `argparse.ArgumentParser` instance configured with the script's
        command-line arguments.
    """
    description = (
        "Agentic Tools: A command-line tool for AI-driven software development.\n\n"
        "This tool operates in two primary modes:\n"
        "1. Server Mode (default): Runs the FastMCP server, exposing agents as tools.\n"
        "   To run: python main.py\n\n"
        "2. CLI Mode ('run-agent'): Manually executes a single agent task.\n"
        "   To run: python main.py run-agent <AGENT_NAME> [options]"
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_FILE_PATH),
        help=f"Path to the configuration file (default: {CONFIG_FILE_PATH}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Build the detailed help text for the agent_name argument.
    agent_help_lines = ["The name of the agent to run. Available agents are:\n"]
    max_len = max(len(name) for name in AGENT_METADATA) if AGENT_METADATA else 0
    for name, desc in AGENT_METADATA.items():
        agent_help_lines.append(f"  {name:<{max_len}} : {desc}")
    agent_help_text = "\n".join(agent_help_lines)

    run_agent_parser = subparsers.add_parser(
        "run-agent",
        help="Manually execute a single agent task.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run_agent_parser.add_argument(
        "agent_name",
        type=str,
        choices=VALID_AGENTS,
        help=agent_help_text,
        metavar="AGENT_NAME",
    )
    run_agent_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="The chat prompt or context for the agent.",
    )
    run_agent_parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="The path to a file containing the prompt for the agent.",
    )
    run_agent_parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="The path to the file or directory for the agent to process.",
    )
    run_agent_parser.add_argument(
        "--target-directory",
        type=str,
        default=str(Path.cwd()),
        help="The root directory of the project. Defaults to the current working directory.",
    )

    return parser


async def _run_cli_mode(args: argparse.Namespace) -> None:
    """Executes the selected agent task in CLI mode."""
    global configuration
    try:
        config_path = Path(args.config)
        configuration = _load_configuration(config_path)
        logger.info("Configuration loaded successfully from %s.", config_path)
    except RuntimeError:
        return  # Error is already logged in _load_configuration

    chat_prompt = args.chat
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as prompt_file:
                chat_prompt = prompt_file.read()
        except FileNotFoundError:
            logger.error("Prompt file not found at '%s'.", args.prompt_file)
            return
        except OSError as error:
            logger.error("Error reading prompt file: %s", error)
            return

    try:
        target_directory = Path(args.target_directory).resolve()
        result = await _execute_agent_task(
            configuration,
            args.agent_name,
            MCP_NAME,
            chat_prompt,
            args.filepath,
            target_directory,
        )
        logger.info("--- Agent Result ---")
        # The result can be any data type, so printing it directly provides the
        # clearest output for a developer using this manual tool.
        print(result)
        logger.info("--------------------")
    except RuntimeError as error:
        # The specific error is already logged within _execute_agent_task.
        # This catch prevents a top-level traceback, providing a cleaner exit.
        logger.critical(
            "Script execution failed. Please review the logs for details. Error: %s",
            error,
        )


# --- Main Execution Block ---


def main_cli() -> None:
    """Main function for the command-line interface."""
    parser = _setup_cli_parser()
    args = parser.parse_args()

    if args.command == "run-agent":
        asyncio.run(_run_cli_mode(args))
    else:
        # Run in FastMCP server mode (default).
        # The global configuration is required here because FastMCP's tool
        # functions are registered globally and need access to the configuration
        # without explicit passing.
        global configuration
        try:
            config_path = Path(args.config)
            configuration = _load_configuration(config_path)
            mcp.run()
        except RuntimeError:
            # Exit gracefully if configuration fails to load.
            # The error is already logged in _load_configuration.
            sys.exit(1)


if __name__ == "__main__":
    main_cli()
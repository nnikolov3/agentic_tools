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


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merges override_dict into a copy of base_dict.
    The base_dict is not modified in place.
    """
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            # This will replace lists and other values, which is the
            # most predictable behavior for configuration.
            merged[key] = value
    return merged


# Initialize the configurator and load the configuration dictionary.
# This approach centralizes configuration loading and error handling.
def _load_and_merge_configurations(user_config_path: Path) -> dict[str, Any]:
    """
    Loads the default configuration, then loads a user-specific configuration
    and deep-merges it on top of the default.
    """
    # 1. Load default configuration from its standard location within the package.
    # This path is relative to main.py, ensuring it works in both development
    # and when installed via pip.
    default_config_path = Path(__file__).parent / "conf" / "agentic_tools.toml"
    default_config = {}
    if default_config_path.is_file():
        try:
            default_config = get_config_dictionary(default_config_path)
        except (ValueError, OSError) as e:
            logger.error(
                "FATAL: Could not load or parse the default configuration at '%s': %s",
                default_config_path,
                e,
            )
            raise RuntimeError(
                "Default configuration is corrupted or unreadable."
            ) from e
    else:
        logger.warning(
            "Default configuration file not found at '%s'. The tool may not function correctly.",
            default_config_path,
        )

    # 2. If the user_config_path doesn't exist, return the default.
    if not user_config_path.is_file():
        logger.info(
            "No user configuration found at '%s'. Using default settings.",
            user_config_path,
        )
        return default_config

    # 3. Load user override configuration.
    try:
        logger.info("Loading user override configuration from '%s'.", user_config_path)
        user_config = get_config_dictionary(user_config_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error(
            "Failed to load or parse user configuration at '%s': %s",
            user_config_path,
            e,
        )
        logger.warning("Proceeding with default configuration only.")
        return default_config

    # 4. Deep merge user config over default config and return.
    logger.info("Merging user configuration over default settings.")
    merged_config = deep_merge(default_config, user_config)
    return merged_config


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

    # Parent parser for global arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a specific configuration file. Overrides all other settings.",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode, using 'conf/agentic_tools.toml' as the configuration.",
    )

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[parent_parser],
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
        parents=[parent_parser],
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
    # The global 'configuration' is already loaded and merged by main_cli.
    # The previous logic to load config here is no longer needed.

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
        print(result)
        logger.info("--------------------")
    except RuntimeError as error:
        logger.critical(
            "Script execution failed. Please review the logs for details. Error: %s",
            error,
        )


# --- Main Execution Block ---


def _determine_config_path(args: argparse.Namespace) -> Path:
    """Determines the configuration file path based on CLI arguments."""
    # 1. User-specified path via --config has the highest precedence.
    if args.config:
        logger.info("Using user-specified configuration: %s", args.config)
        return Path(args.config)
    # 2. --debug flag uses the local development path.
    if args.debug:
        logger.info("Debug mode enabled. Using configuration: conf/agentic_tools.toml")
        return Path("conf/agentic_tools.toml")
    # 3. Default behavior for container/production use.
    logger.info("Using default root configuration: conf/agentic_tools.toml")
    return Path("conf/agentic_tools.toml")


def main_cli() -> None:
    """Main function for the command-line interface."""
    parser = _setup_cli_parser()
    args = parser.parse_args()

    # This path is now correctly identified as the *user override* path.
    user_config_path = _determine_config_path(args)

    global configuration
    try:
        # The new loading and merging logic is called here.
        configuration = _load_and_merge_configurations(user_config_path)
        if not configuration:
            raise RuntimeError("Configuration could not be loaded.")

        if args.command == "run-agent":
            # We no longer need to pass the config path to the CLI runner.
            asyncio.run(_run_cli_mode(args))
        else:
            # Run in FastMCP server mode (default).
            mcp.run()
    except RuntimeError as e:
        logger.critical("Failed to initialize: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()

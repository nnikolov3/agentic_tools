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

# Standard Library Imports
import argparse
import asyncio
import logging
import sys
from os import PathLike
from pathlib import Path
from typing import Any

# Third-Party Library Imports
from dotenv import load_dotenv
from fastmcp import FastMCP

# Local Application/Module Imports
from src.agents.agent import AGENT_CLASSES, DefaultAgent
from src.configurator import get_config_dictionary
from src.scripts.ingest_knowledge_bank import KnowledgeBankIngestor

# --- Global Setup ---

# Adherence to 'Parameterize Everything' principle.
# Centralized constants for configuration and application naming.
MCP_NAME: str = "agentic-tools"

# Configure basic logging for clear, explicit output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# Agent names are defined as constants to prevent typos and magic strings,
# adhering to the 'Explicit Over Implicit' principle.
APPROVER: str = "approver"
ARCHITECT: str = "architect"
COMMENTATOR: str = "commentator"
CONFIGURATION_BUILDER: str = "configuration_builder"
DEVELOPER: str = "developer"
EXPERT: str = "expert"
INGEST_KNOWLEDGE_BANK: str = "ingest_knowledge_bank"
KNOWLEDGE_BASE_BUILDER: str = "knowledge_base_builder"
LINTER_ANALYST: str = "linter_analyst"
README_WRITER: str = "readme_writer"

# Centralized agent metadata to act as a single source of truth for descriptions.
# This adheres to the DRY (Don't Repeat Yourself) principle.
AGENT_METADATA: dict[str, str] = {
    APPROVER: "Audits recent code changes and approves or rejects them.",
    ARCHITECT: "Creates a high-quality architecture based on design guidelines.",
    COMMENTATOR: "Updates a source code file with comments and documentation.",
    CONFIGURATION_BUILDER: "Generates a TOML configuration file for a project.",
    DEVELOPER: "Writes high-quality code based on design guidelines and standards.",
    EXPERT: "Provides expert-level answers and solutions based on a given context.",
    INGEST_KNOWLEDGE_BANK: "Processes and ingests data into the knowledge bank.",
    KNOWLEDGE_BASE_BUILDER: "Fetches content from URLs for knowledge base creation.",
    LINTER_ANALYST: "Runs project linters and generates a prioritized analysis report.",
    README_WRITER: "Updates the README.md file based on project structure and git info.",
}

# The list of valid agents is derived directly from the metadata dictionary.
VALID_AGENTS: tuple[str, ...] = tuple(AGENT_METADATA.keys())


# --- Helper Functions ---


def deep_merge(
    base_dict: dict[str, Any], override_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Recursively merges an override dictionary into a copy of a base dictionary.

    The base dictionary is not modified in place. This function is essential for
    layering user-specific configurations over a set of default values.

    Args:
        base_dict: The dictionary containing default values.
        override_dict: The dictionary with values to merge and override.

    Returns:
        A new dictionary representing the merged result.
    """
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            # This will replace lists and other non-dict values, which is the
            # most predictable and desirable behavior for configuration merging.
            merged[key] = value
    return merged


# --- Configuration Loading ---


def _load_and_merge_configurations(override_config_path: Path | None) -> dict[str, Any]:
    """
    Loads the default configuration and deep-merges an override configuration.

    This function establishes a robust configuration hierarchy:
    1. A default configuration is loaded from a standard path within the package.
    2. An optional user-provided override configuration is loaded.
    3. The override configuration is recursively merged on top of the default.

    This ensures that the application has a complete set of default settings
    while allowing users to customize specific values without duplicating the
    entire configuration.

    Args:
        override_config_path: The path to the user-specific override TOML file.

    Returns:
        The final, merged configuration dictionary.

    Raises:
        RuntimeError: If the default configuration is missing or unreadable.
    """
    # 1. Load default configuration from its standard location within the package.
    # This path is relative to main.py, ensuring it works in both development
    # and when installed as a package.
    default_config_path = Path(__file__).parent / "conf" / "agentic_tools.toml"
    default_config = {}
    if default_config_path.is_file():
        try:
            default_config = get_config_dictionary(default_config_path)
        except (ValueError, OSError) as error:
            logger.error(
                "FATAL: Could not load or parse the default configuration at '%s': %s",
                default_config_path,
                error,
            )
            raise RuntimeError(
                "Default configuration is corrupted or unreadable."
            ) from error
    else:
        logger.warning(
            "Default configuration file not found at '%s'. The tool may not function correctly.",
            default_config_path,
        )

    # 2. If no override_config_path is provided, return the default configuration.
    if not override_config_path or not override_config_path.is_file():
        logger.info(
            "No user override configuration found or specified. Using default settings."
        )
        return default_config

    # 3. Load user override configuration.
    try:
        logger.info(
            "Loading user override configuration from '%s'.", override_config_path
        )
        user_config = get_config_dictionary(override_config_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        logger.error(
            "Failed to load or parse user configuration at '%s': %s",
            override_config_path,
            error,
        )
        logger.warning("Proceeding with default configuration only.")
        return default_config

    # 4. Deep merge user config over default config and return.
    logger.info("Merging user configuration over default settings.")
    return deep_merge(default_config, user_config)


# --- Global State Initialization ---

# Initialize the FastMCP instance, which serves as the tool runner.
mcp = FastMCP(MCP_NAME)

# Global configuration variable, to be initialized in the main execution block.
configuration: dict[str, Any] = {}


# --- Core Logic ---


async def _execute_agent_task(
    config: dict[str, Any],
    agent_name: str,
    project: str,
    chat: str | None,
    filepath: str | PathLike[str] | None,
    target_directory: Path,
    **kwargs: Any,
) -> Any:
    """
    Initializes and runs a single agent task.

    This function encapsulates the logic for creating an agent instance and
    invoking its primary execution method. It includes robust error handling to
    log any failures that occur during the agent's run.

    Args:
        config: The application configuration dictionary.
        agent_name: The name of the agent to execute.
        project: The name of the project context.
        chat: The input prompt or context for the agent.
        filepath: The path to a file for the agent to process.
        target_directory: The resolved, absolute root directory for the agent's operations.
        **kwargs: Additional keyword arguments to pass to the agent.

    Returns:
        The result produced by the agent's execution.

    Raises:
        RuntimeError: If the agent's execution fails, wrapping the original exception.
    """
    logger.info("Executing '%s' tool in directory '%s'.", agent_name, target_directory)

    if agent_name == INGEST_KNOWLEDGE_BANK:
        # The ingestor is a special case that doesn't follow the standard agent pattern.
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
        # and preserves the original exception context, adhering to error handling best practices.
        raise RuntimeError(f"Agent '{agent_name}' failed to execute.") from error


async def _run_agent_tool(
    agent_name: str,
    chat: str | None,
    filepath: str | PathLike[str] | None,
    target_directory: Path | None,
    **kwargs: Any,
) -> Any:
    """
    A generic factory function to instantiate and run a specified agent.

    This helper function centralizes agent execution logic, reducing code
    duplication. It resolves the target directory at runtime, ensuring that
    the agent operates in the correct context, whether run via CLI or server.

    Args:
        agent_name: The name of the agent to run (e.g., 'readme_writer').
        chat: The optional chat context or prompt for the agent.
        filepath: The optional path to a file for the agent to process.
        target_directory: The optional root directory for the agent's operations.
                          Defaults to the current working directory if None.
        **kwargs: Additional keyword arguments to pass to the agent.

    Returns:
        The result of the agent's operation.
    """
    # Resolve the target directory at runtime. This is the critical fix for the
    # user's issue, as it avoids compile-time evaluation of `Path.cwd()`.
    resolved_directory = (
        target_directory.resolve() if target_directory else Path.cwd().resolve()
    )
    return await _execute_agent_task(
        configuration,
        agent_name,
        MCP_NAME,
        chat,
        filepath,
        resolved_directory,
        **kwargs,
    )


# --- Tool Definitions ---


@mcp.tool(description=AGENT_METADATA[README_WRITER])
async def readme_writer_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'readme_writer' agent to update the project's README.md file.

    Args:
        chat: An optional prompt guiding the content generation.
        filepath: The path to the README.md file or its containing directory.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(README_WRITER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[APPROVER])
async def approver_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'approver' agent to audit recent code changes.

    Args:
        chat: An optional prompt specifying approval criteria or review focus.
        filepath: The path to the code changes, repository, or specific files.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(APPROVER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[DEVELOPER])
async def developer_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'developer' agent to write or modify code.

    Args:
        chat: An optional prompt guiding the code generation process.
        filepath: The path to the file where code modification is required.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(DEVELOPER, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[ARCHITECT])
async def architect_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'architect' agent to assist in software architecture design.

    Args:
        chat: An optional prompt specifying architectural requirements or goals.
        filepath: The path to relevant project files for architectural analysis.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(ARCHITECT, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[COMMENTATOR])
async def commentator_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'commentator' agent to add documentation to source code.

    Args:
        chat: An optional prompt specifying documentation requirements.
        filepath: The path to the source code file that needs commenting.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(COMMENTATOR, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[KNOWLEDGE_BASE_BUILDER])
async def knowledge_base_builder_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'knowledge_base_builder' agent to fetch web content.

    Args:
        chat: A comma-separated string of URLs to fetch.
        filepath: The path to the output file where content will be saved.
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(
        KNOWLEDGE_BASE_BUILDER, chat, filepath, target_directory
    )


@mcp.tool(description=AGENT_METADATA[LINTER_ANALYST])
async def linter_analyst_tool(
    chat: str | None = None,
    filepath: str | PathLike[str] | None = None,
    target_directory: Path | None = None,
) -> Any:
    """
    Invokes the 'linter_analyst' agent.

    Args:
        chat: An optional user prompt to focus the analysis.
        filepath: The path to relevant project files for analysis (optional).
        target_directory: The root directory of the project. Defaults to CWD.
    """
    return await _run_agent_tool(LINTER_ANALYST, chat, filepath, target_directory)


@mcp.tool(description=AGENT_METADATA[CONFIGURATION_BUILDER])
async def configuration_builder_tool(
    chat: str | None = None,
    output_filename: str = "generated_config.toml",
    target_directory: Path | None = None,
    dependency_file: str = "pyproject.toml",
) -> Any:
    """
    Invokes the 'configuration_builder' agent to generate a project config file.

    Args:
        chat: An optional prompt to guide the configuration generation.
        output_filename: The path for the generated configuration file.
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

    # Parent parser for global arguments shared between the main command and subcommands.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a specific override configuration file.",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode, using 'conf/agentic_tools.toml' as the override config.",
    )

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Build detailed, formatted help text for the agent_name argument.
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
        default=None,
        help="The root directory of the project. Defaults to the current working directory.",
    )

    return parser


async def _run_cli_mode(args: argparse.Namespace) -> None:
    """Executes the selected agent task in CLI mode."""
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
        # Resolve the target directory at runtime from the CLI argument.
        # This defaults to the current working directory if the argument is not provided.
        resolved_target_directory = (
            Path(args.target_directory).resolve()
            if args.target_directory
            else Path.cwd().resolve()
        )

        result = await _execute_agent_task(
            configuration,
            args.agent_name,
            MCP_NAME,
            chat_prompt,
            args.filepath,
            resolved_target_directory,
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


def _get_override_config_path(args: argparse.Namespace) -> Path | None:
    """
    Determines the override configuration file path based on CLI arguments.

    The precedence is:
    1. An explicit path provided via `--config`.
    2. The local development path if `--debug` is specified.
    3. A default path for container/production use.
    4. `None` if no specific override is intended.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The path to the override configuration file, or None.
    """
    if args.config:
        logger.info("Using user-specified override configuration: %s", args.config)
        return Path(args.config)
    if args.debug:
        logger.info(
            "Debug mode enabled. Using override configuration: conf/agentic_tools.toml"
        )
        return Path("conf/agentic_tools.toml")

    # In a container, an override might be mounted at a standard location.
    # This provides a consistent path for production-like environments.
    default_override_path = Path("conf/agentic_tools.toml")
    if default_override_path.exists():
        logger.info("Using default override configuration: conf/agentic_tools.toml")
        return default_override_path

    return None


def main_cli() -> None:
    """Main function for the command-line interface."""
    # Load environment variables from a .env file if present.
    load_dotenv()

    parser = _setup_cli_parser()
    args = parser.parse_args()

    global configuration
    try:
        override_config_path = _get_override_config_path(args)
        configuration = _load_and_merge_configurations(override_config_path)

        if not configuration:
            raise RuntimeError("Fatal: Configuration could not be loaded.")

        if args.command == "run-agent":
            asyncio.run(_run_cli_mode(args))
        else:
            # Default behavior: run in FastMCP server mode.
            mcp.run()

    except RuntimeError as error:
        logger.critical("Failed to initialize or run application: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()

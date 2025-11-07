from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import Client, FastMCP

from src.agents.agent import Agent
from src.configurator import get_available_agents, get_config_dictionary

# --- Constants ---
MCP_NAME: str = "agentic-tools"
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


class AgentExecutionError(Exception):
    """Custom exception for agent execution errors."""

    pass


class PromptFileError(Exception):
    """Custom exception for prompt file errors."""

    pass


# --- CLI Setup ---
def _setup_cli_parser(available_agents: List[str]) -> argparse.ArgumentParser:
    """Set up the CLI argument parser with all commands and options."""
    description = (
        "Agentic Tools: A command-line tool for AI-driven software development.\n\n"
        "This tool operates in two primary modes:\n"
        "1. Server Mode (default): Runs the FastMCP server, exposing agents as tools.\n"
        "   To run: python main.py\n\n"
        "2. CLI Mode ('run-agent'): Manually executes a single agent task.\n"
        "   To run: python main.py run-agent <AGENT_NAME> [options]"
    )

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
        help="Run in debug mode.",
    )

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Build help text for agent_name argument
    agent_help_lines = ["The name of the agent to run. Available agents are:\n"]
    max_len = max(len(name) for name in available_agents) if available_agents else 0
    for name in available_agents:
        agent_help_lines.append(f"  {name:<{max_len}}")
    agent_help_text = "\n".join(agent_help_lines)

    run_agent_parser = subparsers.add_parser(
        "run-agent",
        help="Manually execute a single agent task.",
        parents=[parent_parser],
    )
    run_agent_parser.add_argument(
        "agent_name",
        type=str,
        choices=available_agents,
        help=agent_help_text,
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
        "--target_directory",
        type=str,
        default=None,
        help="The root directory of the project. Defaults to the current working directory.",
    )

    return parser


# --- Dynamic Tool Registration ---
async def register_agent_tools(
    mcp: FastMCP,
    agents: List[str],
    agent_metadata: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Register all available agents as tools with the MCP server."""
    registered_tools = set()

    async with Client(mcp) as client:
        for agent_name in agents:
            if agent_name in agent_metadata and agent_name not in registered_tools:
                func_name = f"{agent_name}_tool"

                # Create a minimal agent instance for registration
                agent = Agent(
                    configuration=config,
                    agent_name=agent_name,
                    chat=None,
                    filepath=None,
                    target_directory=None,
                )

                # Register the agent's run_agent method as a tool
                # NOTE: FastMCP's call_tool does NOT accept a 'description' argument
                await client.call_tool(func_name, agent.run_agent)
                registered_tools.add(agent_name)
                logger.info(f"Registered tool: {func_name}")


# --- Agent Execution ---
async def _run_cli_mode(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Execute the selected agent task in CLI mode."""
    chat_prompt = args.chat

    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as prompt_file:
                chat_prompt = prompt_file.read()
        except FileNotFoundError as error:
            logger.error("Prompt file not found at '%s'.", args.prompt_file)
            raise PromptFileError(f"Prompt file not found: {error}") from error
        except OSError as error:
            logger.error("Error reading prompt file: %s", error)
            raise PromptFileError(f"Error reading prompt file: {error}") from error

    try:
        agent = Agent(
            configuration=config,
            agent_name=args.agent_name,
            chat=chat_prompt,
            filepath=args.filepath,
            target_directory=args.target_directory,
        )
        result = await agent.run_agent()
        logger.info("--- Agent Result ---")
        print(result)
        logger.info("--------------------")
    except Exception as error:
        logger.critical(
            "Script execution failed. Please review the logs for details. Error: %s",
            error,
        )
        raise AgentExecutionError(f"Agent execution failed: {error}") from error


# --- Main Execution ---
async def main() -> None:
    """Main entry point for the application."""
    load_dotenv()

    # --- Load Configuration ---
    current_working_directory = Path(__file__).parent
    configuration_path = current_working_directory / f"{MCP_NAME}.toml"
    try:
        configuration = get_config_dictionary(configuration_path)
    except Exception as error:
        logger.critical("Failed to load configuration: %s", error)
        raise ConfigError(f"Configuration loading failed: {error}") from error

    available_agents = get_available_agents(configuration)

    # --- CLI Setup ---
    parser = _setup_cli_parser(available_agents)
    args = parser.parse_args()

    # --- MCP Setup ---
    mcp = FastMCP(MCP_NAME)
    await register_agent_tools(
        mcp, available_agents, configuration["agentic-tools"]["agents"], configuration
    )

    # --- Run Mode ---
    if args.command == "run-agent":
        await _run_cli_mode(args, configuration)
    else:
        logger.info(f"Starting FastMCP server on {DEFAULT_HOST}:{DEFAULT_PORT}")
        await mcp.run(transport="http", host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigError as e:
        logger.critical("Configuration error: %s", e)
        sys.exit(1)
    except AgentExecutionError as e:
        logger.critical("Agent execution error: %s", e)
        sys.exit(1)
    except PromptFileError as e:
        logger.critical("Prompt file error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.critical("Unexpected error: %s", e)
        sys.exit(1)

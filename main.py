"""
Dynamic CLI and FastMCP server entry point with Memory integration.

Why this design: Single factory builds Agent identically for CLI/server modes,
and all conversations are stored in episodic memory automatically.
"""

import argparse
import asyncio
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

from src.agent import Agent
from src.agents.document_ingestor import DocumentIngestor
from src.memory.memory import Memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION_FILENAME: str = str(
    os.environ.get("AGENTIC_TOOLS_TOML", "agentic_tools.toml")
)
MCP_SERVER_NAME: str = str(os.environ.get("MCP_SERVER_NAME", "agentic-tools"))
DEFAULT_SERVER_TRANSPORT: str = str(os.environ.get("FASTMCP_TRANSPORT", "http"))
DEFAULT_SERVER_HOST: str = str(os.environ.get("FASTMCP_HOST", "0.0.0.0"))
DEFAULT_SERVER_PORT: int = int(os.environ.get("FASTMCP_PORT", "8000"))


class Configurator:
    """Loads and returns the global configuration from a TOML file."""

    def __init__(self, configuration_path: Path) -> None:
        self.configuration_path: Path = configuration_path
        self.configuration: Dict[str, Any] = {}

    def get_config_dictionary(self) -> Dict[str, Any]:
        """Loads TOML configuration from an explicit path using fail-fast validation."""
        if (
            not self.configuration_path.exists()
            or not self.configuration_path.is_file()
        ):
            logger.error("Configuration file not found: %s", self.configuration_path)
            sys.exit(1)

        with self.configuration_path.open(mode="rb") as configuration_file:
            self.configuration = tomllib.load(configuration_file)

        return self.configuration


def normalize_agent_configurations(
    configuration: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Returns agents as list[dict] with required 'name' field."""
    agents_object: Any = configuration.get("agents", [])

    if isinstance(agents_object, dict):
        agent_configurations: List[Dict[str, Any]] = []
        for agent_key, agent_configuration in agents_object.items():
            if not isinstance(agent_configuration, dict):
                logger.error(
                    "Invalid agent entry for key '%s'; expected table/object.",
                    agent_key,
                )
                sys.exit(1)
            if "name" not in agent_configuration:
                agent_configuration = {**agent_configuration, "name": agent_key}
            agent_configurations.append(agent_configuration)
        return agent_configurations

    if isinstance(agents_object, list):
        for index, agent_configuration in enumerate(agents_object):
            if (
                not isinstance(agent_configuration, dict)
                or "name" not in agent_configuration
            ):
                logger.error("Invalid agent at index %d; need dict with 'name'.", index)
                sys.exit(1)
        return agents_object

    logger.error("Invalid 'agents' type in configuration; expected list or table.")
    sys.exit(1)


def get_full_configuration() -> Dict[str, Any]:
    """Loads configuration from default or environment-provided path."""
    configuration_path: Path = Path(DEFAULT_CONFIGURATION_FILENAME)
    configurator: Configurator = Configurator(configuration_path)
    return configurator.get_config_dictionary()


class Helpers:
    """Utility helpers for argument resolution and boundary validation."""

    @staticmethod
    def resolve_prompt_text(arguments: argparse.Namespace) -> Optional[str]:
        """Resolves prompt from --chat or --prompt_file with explicit boundary checks."""
        if getattr(arguments, "chat", None):
            return arguments.chat

        prompt_file_argument: Optional[str] = getattr(arguments, "prompt_file", None)
        if prompt_file_argument:
            prompt_file_path: Path = Path(prompt_file_argument)
            if (
                not prompt_file_path.exists()
                or not prompt_file_path.is_file()
                or not os.access(str(prompt_file_path), os.R_OK)
            ):
                logger.error("Invalid prompt_file: %s", prompt_file_argument)
                sys.exit(1)

            with prompt_file_path.open("r", encoding="utf-8") as input_file:
                return input_file.read()

        return None

    @staticmethod
    def validate_path_readable(path_object: Path, label: str) -> None:
        """Validates that a path exists and is readable, failing fast otherwise."""
        if not path_object.exists() or not os.access(str(path_object), os.R_OK):
            logger.error("Invalid %s: %s", label, str(path_object))
            sys.exit(1)
        return None


async def build_agent_instance(
    configuration: Dict[str, Any],
    agent_configuration: Dict[str, Any],
    prompt_text: Optional[str],
    file_path_object: Optional[Path],
    target_directory_object: Path,
) -> Agent:
    """Builds the Agent identically for both CLI and FastMCP server modes."""
    project_name: str = agent_configuration.get(
        "project",
        configuration.get("project", {}).get("project_name", "agenttech"),
    )

    effective_prompt: str = prompt_text or agent_configuration.get(
        "prompt", "Execute task."
    )

    memory = await Memory.create(configuration)

    agent_instance: Agent = Agent(
        configuration=configuration,
        agent_name=agent_configuration["name"],
        project=project_name,
        chat=effective_prompt,
        filepath=file_path_object,
        target_directory=target_directory_object,
    )

    agent_instance.new_memory = memory

    if agent_configuration["name"] == "ingestor":
        agent_instance.document_ingestor = DocumentIngestor(configuration, memory)

    return agent_instance


async def run_agent_and_store(
    agent_instance: Agent,
    agent_name: str,
    prompt_text: Optional[str],
) -> str:
    """Runs agent, stores conversation in episodic memory, returns response."""
    response_text: Optional[str] = await agent_instance.run_agent()

    if (
        response_text
        and hasattr(agent_instance, "new_memory")
        and agent_instance.new_memory
    ):
        await agent_instance.new_memory.add_to_episodic(
            content=f"User: {prompt_text}\n\nAgent: {response_text}",
            metadata={
                "agent_name": agent_name,
                "prompt": prompt_text,
                "response_length": len(response_text),
            },
        )
        logger.info("Stored conversation in episodic memory")

    return response_text or ""


class DynamicArgumentParser:
    """Builds a CLI with dynamic agent choices from configuration."""

    def __init__(self, agent_names: List[str]) -> None:
        self.agent_names: List[str] = agent_names

    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Creates and returns the top-level argument parser with detailed help."""
        description_text: str = """Agentic Tools - Multi-Agent System

MODES:
  1. Server Mode (default): python main.py
     Starts FastMCP server exposing agent tools via HTTP/SSE

  2. CLI Mode: python main.py run <agent_name> [options]
     Run a specific agent interactively from command line

EXAMPLES:
  # Start server
  python main.py

  # Run architect agent with inline prompt
  python main.py run architect --chat "Design a REST API"

  # Run developer agent with file prompt
  python main.py run developer --prompt_file task.txt --target_directory ./project

  # Run ingestor agent to process PDF
  python main.py run ingestor --file_path document.pdf

ENVIRONMENT VARIABLES:
  AGENTIC_TOOLS_TOML  - Config file path (default: agentic_tools.toml)
  MCP_SERVER_NAME     - Server name (default: agentic-tools)
  FASTMCP_TRANSPORT   - Transport type (default: http)
  FASTMCP_HOST        - Server host (default: 0.0.0.0)
  FASTMCP_PORT        - Server port (default: 8000)
"""

        parser_object: argparse.ArgumentParser = argparse.ArgumentParser(
            description=description_text,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers_object = parser_object.add_subparsers(
            dest="command",
            help="Available commands",
            required=False,
        )

        run_subparser = subparsers_object.add_parser(
            "run",
            help="Run an agent from the command line",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=f"Run a configured agent by name.\n\nAvailable agents: {', '.join(self.agent_names)}",
        )

        run_subparser.add_argument(
            "agent_name",
            type=str,
            choices=self.agent_names,
            help="Name of the agent to run (from config)",
        )

        run_subparser.add_argument(
            "--chat",
            type=str,
            default=None,
            help="Inline prompt text for the agent",
        )

        run_subparser.add_argument(
            "--prompt_file",
            type=str,
            default=None,
            help="Path to file containing prompt text",
        )

        run_subparser.add_argument(
            "--file_path",
            type=str,
            default=None,
            help="Input file to process (PDF for ingestor, code file for developer, etc)",
        )

        run_subparser.add_argument(
            "--target_directory",
            type=str,
            default=None,
            help="Project root directory (default: current working directory)",
        )

        return parser_object


class CommandLineInterface:
    """Runs a configured agent by name using CLI inputs and shared factory."""

    def __init__(
        self, arguments: argparse.Namespace, configuration: Dict[str, Any]
    ) -> None:
        self.arguments: argparse.Namespace = arguments
        self.configuration: Dict[str, Any] = configuration

    async def run_cli(self) -> None:
        """Resolves inputs, validates boundaries, builds Agent, and runs it."""
        prompt_text: Optional[str] = Helpers.resolve_prompt_text(self.arguments)

        file_path_object: Optional[Path] = (
            Path(self.arguments.file_path)
            if getattr(self.arguments, "file_path", None)
            else None
        )

        if file_path_object is not None:
            Helpers.validate_path_readable(file_path_object, "file_path")
            if not file_path_object.is_file():
                logger.error(
                    "Invalid file_path (not a file): %s", str(file_path_object)
                )
                sys.exit(1)

        target_directory_object: Path = (
            Path(self.arguments.target_directory)
            if getattr(self.arguments, "target_directory", None)
            else Path.cwd()
        )

        Helpers.validate_path_readable(target_directory_object, "target_directory")
        if not target_directory_object.is_dir():
            logger.error(
                "Invalid target_directory (not a directory): %s",
                str(target_directory_object),
            )
            sys.exit(1)

        agent_configurations: List[Dict[str, Any]] = normalize_agent_configurations(
            self.configuration
        )
        agent_configuration: Optional[Dict[str, Any]] = next(
            (
                cfg
                for cfg in agent_configurations
                if cfg.get("name") == self.arguments.agent_name
            ),
            None,
        )

        if agent_configuration is None:
            logger.error(
                "Agent '%s' not found in configuration.", self.arguments.agent_name
            )
            sys.exit(1)

        agent_instance: Agent = await build_agent_instance(
            configuration=self.configuration,
            agent_configuration=agent_configuration,
            prompt_text=prompt_text,
            file_path_object=file_path_object,
            target_directory_object=target_directory_object,
        )

        response_text: str = await run_agent_and_store(
            agent_instance=agent_instance,
            agent_name=self.arguments.agent_name,
            prompt_text=prompt_text,
        )

        if response_text:
            print("\n" + "=" * 80)
            print(f"AGENT: {self.arguments.agent_name}")
            print("=" * 80)
            print(response_text)
            print("=" * 80 + "\n")
        else:
            logger.warning("No response from agent '%s'.", self.arguments.agent_name)

        return None


class DynamicFastMCPServer(FastMCP):
    """Registers one MCP tool per agent using the decorator API and runs the server."""

    def __init__(
        self, arguments: argparse.Namespace, configuration: Dict[str, Any]
    ) -> None:
        super().__init__(MCP_SERVER_NAME)
        self.configuration: Dict[str, Any] = configuration
        self.arguments: argparse.Namespace = arguments
        self.agent_configurations: List[Dict[str, Any]] = (
            normalize_agent_configurations(self.configuration)
        )

    async def run_fastmcp(self) -> None:
        """Registers per-agent tools and starts the server using the async API."""
        if not self.agent_configurations:
            logger.error("No agent configurations found.")
            sys.exit(1)

        for agent_configuration in self.agent_configurations:
            agent_name: str = agent_configuration["name"]

            @self.tool(
                name=f"{agent_name}_run",
                description=f"Run agent '{agent_name}' with optional chat, file_path, and target_directory.",
            )
            async def run_agent_tool(
                chat: Optional[str] = None,
                file_path: Optional[str] = None,
                target_directory: Optional[str] = None,
                _bound_agent_configuration: Dict[str, Any] = agent_configuration,
                _bound_agent_name: str = agent_name,
            ) -> str:
                file_path_object: Optional[Path] = (
                    Path(file_path) if file_path else None
                )
                if file_path_object is not None:
                    Helpers.validate_path_readable(file_path_object, "file_path")
                    if not file_path_object.is_file():
                        raise ValueError(
                            f"Invalid file_path (not a file): {file_path_object}"
                        )

                target_directory_object: Path = (
                    Path(target_directory) if target_directory else Path.cwd()
                )
                Helpers.validate_path_readable(
                    target_directory_object, "target_directory"
                )
                if not target_directory_object.is_dir():
                    raise ValueError(
                        f"Invalid target_directory (not a directory): {target_directory_object}"
                    )

                agent_instance: Agent = await build_agent_instance(
                    configuration=self.configuration,
                    agent_configuration=_bound_agent_configuration,
                    prompt_text=chat,
                    file_path_object=file_path_object,
                    target_directory_object=target_directory_object,
                )

                return await run_agent_and_store(
                    agent_instance=agent_instance,
                    agent_name=_bound_agent_name,
                    prompt_text=chat,
                )

        await self.run_async(
            transport=DEFAULT_SERVER_TRANSPORT,
            host=DEFAULT_SERVER_HOST,
            port=DEFAULT_SERVER_PORT,
        )

        return None


async def cli_mode(
    arguments: argparse.Namespace, configuration: Dict[str, Any]
) -> None:
    """Runs the CLI mode."""
    command_line_interface: CommandLineInterface = CommandLineInterface(
        arguments, configuration
    )
    await command_line_interface.run_cli()
    return None


async def fastmcp_mode(
    arguments: argparse.Namespace, configuration: Dict[str, Any]
) -> None:
    """Runs the FastMCP server mode."""
    server: DynamicFastMCPServer = DynamicFastMCPServer(arguments, configuration)
    await server.run_fastmcp()
    return None


async def main() -> None:
    """Entry: loads config, parses args, dispatches CLI/server."""
    load_dotenv()

    configuration: Dict[str, Any] = get_full_configuration()
    agent_configurations: List[Dict[str, Any]] = normalize_agent_configurations(
        configuration
    )
    agent_names: List[str] = [
        agent_configuration["name"] for agent_configuration in agent_configurations
    ]

    dynamic_argument_parser: DynamicArgumentParser = DynamicArgumentParser(agent_names)
    parser_object: argparse.ArgumentParser = (
        dynamic_argument_parser.setup_argument_parser()
    )

    parsed_arguments: argparse.Namespace = parser_object.parse_args()

    if getattr(parsed_arguments, "command", None) == "run":
        await cli_mode(parsed_arguments, configuration)
    else:
        await fastmcp_mode(parsed_arguments, configuration)

    return None


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)

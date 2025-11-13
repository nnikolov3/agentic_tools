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
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from fastmcp import FastMCP

from src.agent import Agent
from src.memory.memory import Memory
from src.providers.provider_factory import ProviderFactory
from src.utils.content_extractor import ContentExtractor
from src.utils.document_processor import DocumentProcessor
from src.utils.knowledge_augmentor import KnowledgeAugmentor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger: logging.Logger = logging.getLogger(__name__)

# --- Configuration and Constants ---
DEFAULT_CONFIGURATION_FILENAME: str = str(
    os.environ.get("AGENTIC_TOOLS_TOML", "agentic_tools.toml")
)
MCP_SERVER_NAME: str = str(os.environ.get("MCP_SERVER_NAME", "agentic-tools"))

# --- Boundary Validation for Transport ---
TransportLiteral = Literal["stdio", "http", "sse", "streamable-http"]
ALLOWED_TRANSPORTS: List[TransportLiteral] = [
    "stdio",
    "http",
    "sse",
    "streamable-http",
]
DEFAULT_SERVER_TRANSPORT_STR: str = str(os.environ.get("FASTMCP_TRANSPORT", "http"))

if DEFAULT_SERVER_TRANSPORT_STR not in ALLOWED_TRANSPORTS:
    logger.error(
        "Invalid FASTMCP_TRANSPORT: '%s'. Must be one of: %s",
        DEFAULT_SERVER_TRANSPORT_STR,
        ", ".join(ALLOWED_TRANSPORTS),
    )
    sys.exit(1)

DEFAULT_SERVER_TRANSPORT: Literal = DEFAULT_SERVER_TRANSPORT_STR  # type: ignore

DEFAULT_SERVER_HOST: str = str(os.environ.get("FASTMCP_HOST", "0.0.0.0"))
DEFAULT_SERVER_PORT: int = int(os.environ.get("FASTMCP_PORT", "8000"))


class Configurator:
    """Loads and returns the global configuration from a TOML file."""

    def __init__(self, configuration_path: Path) -> None:
        self.configuration_path: Path = configuration_path
        self.configuration: Dict[str, Any] = {}

    def get_config_dictionary(self) -> Dict[str, Any]:
        """Loads TOML configuration from an explicit path using fail-fast validation."""
        if not self.configuration_path.is_file():
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
        for agent_key, agent_config in agents_object.items():
            if not isinstance(agent_config, dict):
                logger.error(
                    "Invalid agent entry for key '%s'; expected table/object.",
                    agent_key,
                )
                sys.exit(1)
            if "name" not in agent_config:
                agent_config = {**agent_config, "name": agent_key}
            agent_configurations.append(agent_config)
        return agent_configurations

    if isinstance(agents_object, list):
        for index, agent_config in enumerate(agents_object):
            if not isinstance(agent_config, dict) or "name" not in agent_config:
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
            if not prompt_file_path.is_file() or not os.access(
                str(prompt_file_path), os.R_OK
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


def initialize_providers_for_agent(
    agent_config: Dict[str, Any], global_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Initializes providers for a specific agent based on its configuration."""
    providers: Dict[str, Any] = {}
    provider_factory = ProviderFactory()
    required_providers = {
        agent_config.get("agent_provider"),
        agent_config.get("embedding_provider"),
        agent_config.get("augmentation_provider"),
        global_config.get("content_extractor", {}).get("ocr_provider"),
    }
    required_providers.discard(None)

    for provider_name in required_providers:
        provider_config = global_config.get("providers", {}).get(provider_name, {})
        try:
            providers[provider_name] = provider_factory.get_provider(
                provider_name, provider_config
            )
            logger.info("Provider '%s' initialized for agent.", provider_name)
        except ValueError as e:
            logger.error(
                "Failed to initialize provider '%s' for agent: %s", provider_name, e
            )
    return providers


async def build_agent_instance(
    configuration: Dict[str, Any],
    agent_configuration: Dict[str, Any],
    prompt_text: Optional[str],
    file_path_object: Optional[Path],
    target_directory_object: Path,
) -> Union[Agent, DocumentProcessor, KnowledgeAugmentor]:
    """Builds the Agent identically for both CLI and FastMCP server modes."""
    golden_rules = configuration.get("golden_rules", {}).get("rules", "")
    agent_configuration["golden_rules"] = golden_rules

    providers = initialize_providers_for_agent(agent_configuration, configuration)

    project_name: str = agent_configuration.get(
        "project",
        configuration.get("project", {}).get("project_name", "agenttech"),
    )

    effective_prompt: str = prompt_text or agent_configuration.get(
        "prompt", "Execute task."
    )

    memory = await Memory.create(configuration, providers)
    content_extractor = ContentExtractor(
        configuration.get("content_extractor", {}), providers
    )
    document_processor = DocumentProcessor(
        configuration.get("document_processor", {}),
        memory,
        providers,
        content_extractor,
    )
    knowledge_augmentor = KnowledgeAugmentor(
        configuration.get("knowledge_augmentor", {}),
        memory,
        providers,
        content_extractor,
    )

    agent_name = agent_configuration["name"]
    if agent_name == "ingestor":
        return document_processor
    if agent_name == "knowledge_augmentor":
        return knowledge_augmentor

    return Agent(
        configuration=agent_configuration,
        agent_name=agent_name,
        project=project_name,
        chat=effective_prompt,
        filepath=file_path_object,
        target_directory=target_directory_object,
        memory=memory,
        providers=providers,
        document_processor=document_processor,
        knowledge_augmentor=knowledge_augmentor,
    )


async def run_agent_and_store(
    agent_instance: Union[Agent, DocumentProcessor, KnowledgeAugmentor],
    file_path: Optional[str],
) -> str:
    """Runs agent, stores conversation in episodic memory, returns response."""
    if isinstance(agent_instance, DocumentProcessor):
        if file_path:
            await agent_instance.process_document(file_path)
            return f"Successfully ingested document: {file_path}"
        logger.warning("Ingestor agent called without a filepath.")
        return "Ingestor agent requires a filepath to process a document."

    if isinstance(agent_instance, KnowledgeAugmentor):
        if file_path:
            result = await agent_instance.run(file_path)
            return f"Successfully augmented knowledge from document: {file_path}. Result: {result}"
        logger.warning("Knowledge augmentor agent called without a filepath.")
        return "Knowledge augmentor agent requires a filepath to process a document."

    if isinstance(agent_instance, Agent):
        return await agent_instance.run_agent()

    logger.error("Unknown agent instance type: %s", type(agent_instance))
    return "Error: Unknown agent type."


class DynamicFastMCPServer(FastMCP):
    """Registers one MCP tool per agent and runs the server."""

    def __init__(
        self, arguments: argparse.Namespace, configuration: Dict[str, Any]
    ) -> None:
        super().__init__(MCP_SERVER_NAME)
        self.configuration = configuration
        self.arguments = arguments
        self.agent_configurations = normalize_agent_configurations(self.configuration)

    async def run_fastmcp(self) -> None:
        """Registers per-agent tools and starts the server."""
        if not self.agent_configurations:
            logger.error("No agent configurations found.")
            sys.exit(1)

        for agent_config in self.agent_configurations:
            agent_name = agent_config["name"]

            @self.tool(
                name=f"{agent_name}_run",
                description=f"Run agent '{agent_name}' with optional chat, file_path, and target_directory.",
            )
            async def run_agent_tool(
                chat: Optional[str] = None,
                file_path: Optional[str] = None,
                target_directory: Optional[str] = None,
                _bound_agent_config: Dict[str, Any] = agent_config,
            ) -> str:
                file_path_obj = Path(file_path) if file_path else None
                if file_path_obj:
                    Helpers.validate_path_readable(file_path_obj, "file_path")

                target_dir_obj = (
                    Path(target_directory) if target_directory else Path.cwd()
                )
                Helpers.validate_path_readable(target_dir_obj, "target_directory")

                agent_instance = await build_agent_instance(
                    configuration=self.configuration,
                    agent_configuration=_bound_agent_config,
                    prompt_text=chat,
                    file_path_object=file_path_obj,
                    target_directory_object=target_dir_obj,
                )
                return await run_agent_and_store(agent_instance, file_path)

        await self.run_async(
            transport=DEFAULT_SERVER_TRANSPORT,
            host=DEFAULT_SERVER_HOST,
            port=DEFAULT_SERVER_PORT,
        )


def create_argument_parser(agent_names: List[str]) -> argparse.ArgumentParser:
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(description="Agentic Tools CLI and FastMCP Server")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an agent from the CLI")
    run_parser.add_argument(
        "--agent",
        choices=agent_names,
        required=True,
        help="Name of the agent to run",
    )
    run_parser.add_argument("--chat", type=str, help="Chat message for the agent")
    run_parser.add_argument(
        "--prompt_file", type=str, help="Path to a file with the prompt"
    )
    run_parser.add_argument(
        "--file_path", type=str, help="Path to a file for the agent to process"
    )
    run_parser.add_argument(
        "--target_directory", type=str, help="Target directory for agent operations"
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the FastMCP server")
    server_parser.add_argument(
        "--host", type=str, default=DEFAULT_SERVER_HOST, help="Server host"
    )
    server_parser.add_argument(
        "--port", type=int, default=DEFAULT_SERVER_PORT, help="Server port"
    )
    server_parser.add_argument(
        "--transport",
        choices=ALLOWED_TRANSPORTS,
        default=DEFAULT_SERVER_TRANSPORT,
        help="Server transport",
    )
    return parser


class CommandLineInterface:
    """Handles CLI arguments and orchestrates agent execution."""

    def __init__(
        self, arguments: argparse.Namespace, configuration: Dict[str, Any]
    ) -> None:
        self.arguments = arguments
        self.configuration = configuration
        self.agent_configurations = normalize_agent_configurations(self.configuration)

    async def run_cli(self) -> None:
        """Executes the agent based on CLI arguments."""
        agent_name: str = self.arguments.agent
        agent_config = next(
            (ac for ac in self.agent_configurations if ac["name"] == agent_name), None
        )

        if not agent_config:
            logger.error("Agent '%s' not found in configuration.", agent_name)
            sys.exit(1)

        prompt_text = Helpers.resolve_prompt_text(self.arguments)
        file_path_obj = (
            Path(self.arguments.file_path) if self.arguments.file_path else None
        )
        if file_path_obj:
            Helpers.validate_path_readable(file_path_obj, "file_path")

        target_dir_obj = (
            Path(self.arguments.target_directory)
            if self.arguments.target_directory
            else Path.cwd()
        )
        Helpers.validate_path_readable(target_dir_obj, "target_directory")

        agent_instance = await build_agent_instance(
            configuration=self.configuration,
            agent_configuration=agent_config,
            prompt_text=prompt_text,
            file_path_object=file_path_obj,
            target_directory_object=target_dir_obj,
        )

        response = await run_agent_and_store(agent_instance, self.arguments.file_path)
        print(response)


async def main() -> None:
    """Entry point: loads config, parses args, and dispatches CLI/server."""
    load_dotenv()
    configuration = get_full_configuration()
    agent_configs = normalize_agent_configurations(configuration)
    agent_names = [ac["name"] for ac in agent_configs]

    parser = create_argument_parser(agent_names)
    parsed_args = parser.parse_args()

    if parsed_args.command == "run":
        cli = CommandLineInterface(parsed_args, configuration)
        await cli.run_cli()
    elif parsed_args.command == "server":
        server = DynamicFastMCPServer(parsed_args, configuration)
        await server.run_fastmcp()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)

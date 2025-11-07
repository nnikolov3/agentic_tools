# main.py (Root: ./main.py)

"""
Main async entry for Agentic Tools MCP server.
VM-aware config load, knowledge seeding, multi-agent orchestration, periodic prune.
Uses FastMCP (pip install fastmcp) or fallback custom server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP  # MCP lib for agent coordination

from src.agents.agent import Agent
from src.configurator import (
    find_config,
    get_agent_config,
    get_available_agents,
    get_config_dictionary,
)
from src.memory.qdrant_client_manager import QdrantClientManager
from src.memory.qdrant_memory import QdrantMemory
from src.scripts.ingest_knowledge_bank import IngestionConfig, KnowledgeBankIngestor

load_dotenv()

MCP_NAME = "agentic-tools"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


class AgentExecutionError(Exception):
    pass


class PromptFileError(Exception):
    pass


def setup_parser(available_agents: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Agentic Tools MCP")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--agents", nargs="*", choices=available_agents)
    parser.add_argument(
        "--ingest", action="store_true", help="Seed knowledge bank first"
    )
    parser.add_argument(
        "--config", default=None, type=Path, help="Override config path (VM/host)"
    )
    parser.add_argument("--prune", action="store_true", help="Run memory prune")
    return parser


async def handle_ingest(config_path: Path) -> None:
    config = get_config_dictionary(config_path)
    general_config = config.get("general", {})
    if general_config.get("debug_mode"):
        logging.getLogger().setLevel(logging.DEBUG)
    ingestion_config = IngestionConfig(
        source_dir=Path(config.get("ingestion", {}).get("source_dir", "docs/knowledge")),
        output_dir=Path(config.get("ingestion", {}).get("output_dir", ".ingested")),
        **config.get("memory", {}),
        **config.get("ingestion", {}),
    )
    ingestor = KnowledgeBankIngestor(ingestion_config)
    results = await ingestor.run()
    logger.info("Seeding complete: %s files processed", len(results) if results else 0)


async def handle_prune(memory: QdrantMemory) -> None:
    count = await memory.prune_memories()
    logger.info(
        "Pruned %d memories (prune_days=%d, confidence=%.2f)",
        count,
        memory.config.get("prune_days", 365),
        memory.config.get("prune_confidence", 0.5),
    )


async def main():
    # VM-aware config load
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None, type=Path)
    temp_args, _ = parser.parse_known_args()
    config_path = temp_args.config or Path("agentic-tools.toml")  # Internal default
    try:
        config_path = find_config(Path.cwd())  # Search parents/host
        logger.info("Loaded VM/host config: %s", config_path)
    except FileNotFoundError:
        if not config_path.exists():
            raise ConfigError(
                f"No config found: Use --config or place {config_path} in current dir."
            )
        logger.warning("Using internal config: %s (no host found)", config_path)

    config = get_config_dictionary(config_path)
    available_agents = get_available_agents(config)  # Nested load
    logger.info("Available agents: %s", available_agents)
    general_config = config.get("general", {})
    mcp_config = config.get("mcp", {})

    # Setup logging from config
    log_level = mcp_config.get("log_level", "INFO").upper()
    log_file = general_config.get("log_file", "agentic-tools.log")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        filename=log_file,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if general_config.get("debug_mode", False):
        logging.getLogger().setLevel(logging.DEBUG)

    full_parser = setup_parser(available_agents)
    args = full_parser.parse_args()

    if args.prune:
        qdrant_manager = QdrantClientManager(config)
        await qdrant_manager.initialize()
        memory = await QdrantMemory.create(config, qdrant_manager)
        await handle_prune(memory)
        return

    if args.ingest:
        await handle_ingest(config_path)
        if not args.agents:  # Exit if only ingest
            logger.info("Ingestion done; run without --ingest for MCP.")
            return

    # MCP setup with config
    mcp = FastMCP(
        name=mcp_config.get("name", MCP_NAME),
        host=args.host or mcp_config.get("host", DEFAULT_HOST),
        port=args.port or mcp_config.get("port", DEFAULT_PORT),
    )

    qdrant_manager = QdrantClientManager(config)
    await qdrant_manager.initialize()
    memory = await QdrantMemory.create(config, qdrant_manager)

    selected_agents = args.agents or available_agents
    for name in selected_agents:
        agent_cfg = get_agent_config(config, name)
        if agent_cfg and agent_cfg.get("name") == name:  # Valid
            agent = Agent(configuration=config, agent_name=name, memory=memory)
            mcp.add_agent(agent)
            logger.info(
                "Added agent: %s (tools: %s, weight: %.1f)",
                name,
                agent_cfg.get("tools", []),
                agent_cfg.get("memory_weight", 0.5),
            )

    # Periodic prune (every hour, from [memory])
    async def prune_loop():
        while True:
            await asyncio.sleep(3600)
            await handle_prune(memory)

    if mcp.agents:  # Only if agents added
        if mcp_config.get("enable_multi_agent", True):
            asyncio.create_task(prune_loop())
        logger.info(
            "Starting MCP server: %s on %s:%d with %d agents (coord: %s)",
            mcp.name,
            mcp.host,
            mcp.port,
            len(mcp.agents),
            mcp_config.get("agent_coordination", "sequential"),
        )
        await mcp.run()
    else:
        logger.warning("No agents selected or available. Use --agents <name>.")


if __name__ == "__main__":
    asyncio.run(main())

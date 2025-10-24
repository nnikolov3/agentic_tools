from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from src.agents.agent import Agent
from src.configurator import Configurator
from src.scripts.ingest_knowledge_bank import KnowledgeBankIngestor
from src.workflows.code_quality_enforcer import CodeQualityEnforcer

# Configure basic logging for clear, explicit output.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()
mcp_name = "agentic-tools"
knowledge_bank = KnowledgeBankIngestor(configuration[mcp_name])


async def readme_writer_tool(chat: Any | None) -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration, "readme_writer", mcp_name, chat)
    return await agent.run_agent()


async def approver_tool(chat: Any | None) -> Any:
    print("approver_tool")
    agent = Agent(configuration, "approver", mcp_name, chat)
    return await agent.run_agent()


async def developer_tool(chat: Any | None) -> Any:
    print("developer_tool")
    agent = Agent(configuration, "developer", mcp_name, chat)
    return await agent.run_agent()


async def architect_tool(chat: Any | None) -> Any:
    print("architect_tool")
    agent = Agent(configuration, "architect", mcp_name, chat)
    return await agent.run_agent()


async def commentator_tool(path: str) -> Any:
    """
    Runs the full code quality enforcement workflow on the specified file or directory.

    Args:
        path: The file or directory path (as a string) to be processed.
    """
    path_str = str(path)
    path_obj = Path(path_str)

    if not path_obj.exists():
        logger.error(f"Path does not exist: {path_str}")
        return f"Error: Path does not exist: {path_str}"


        # The configuration dictionary needs to be passed correctly to the enforcer
    mcp_config = configuration.get(mcp_name, {})
    enforcer = CodeQualityEnforcer(mcp_config, mcp_name)

    if path_obj.is_file():
        logger.info(f"Input is a file. Running Code Quality Enforcer on: {path_str}")
        await enforcer.run_on_file(path_str)
        return f"Code quality enforcement finished for file: {path_str}. Check logs for details."

    elif path_obj.is_dir():
        logger.info(f"Input is a directory. Running Code Quality Enforcer on: {path_str}")
        await enforcer.run_on_directory(path_str)
        return f"Code quality enforcement finished for directory: {path_str}. Check logs for details."

    else:
        logger.error(f"Invalid path provided: {path_str}")
        return f"Error: Invalid path provided: {path_str}. Must be a file or directory."




async def main():
    #response = await architect_tool("Identify duplicate code in the project and explain how to fix it")
    #print(response)
    # await knowledge_bank.run_ingestion()

    # Run commentator on the src directory
    print(await commentator_tool("src"))

    return


if __name__ == "__main__":
    asyncio.run(main())
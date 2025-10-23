from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from src.agents.agent import Agent
from src.configurator import Configurator
from src.scripts.ingest_knowledge_bank import KnowledgeBankIngestor

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


async def main():
    # response = await architect_tool()
    await knowledge_bank.run_ingestion()


    return


if __name__ == "__main__":
    asyncio.run(main())

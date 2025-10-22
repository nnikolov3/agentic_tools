# File: run_agents_manually.py
"""
THIS IS ONLY A CONVENIENCE SCRIPT TO RUN THE TOOLS
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from src.agents.agent import Agent
from src.configurator import Configurator

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()
mcp_name = "agentic-tools"
mcp = FastMCP(mcp_name)


async def readme_writer_tool(chat: Any | None) -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration, "readme_writer", mcp_name, chat)
    return await agent.run_agent()


async def approver_tool(chat: Any | None) -> Any:
    print("approver_tool")
    agent = Agent(configuration, "approver", mcp_name, chat)
    return await agent.run_agent()


async def developer(chat: Any | None) -> Any:
    print("developer_tool")
    agent = Agent(configuration, "developer", mcp_name, chat)
    return await agent.run_agent()


"""
# print(readme_result)
approval_result = asyncio.run(
    approver_tool("Revaluate the changes and approve or reject")
)
print(approval_result)
"""


async def main():
    response = await readme_writer_tool("Update README")

    print(response)
    return


if __name__ == "__main__":
    asyncio.run(main())

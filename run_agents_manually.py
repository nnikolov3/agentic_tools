# File: run_agents_manually.py
"""
THIS IS ONLY A CONVENIENCE SCRIPT TO RUN THE TOOLS
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

from src.agents.agent import Agent
from src.configurator import Configurator

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()
mcp_name = "Agentic Tools"
mcp = FastMCP(mcp_name)


async def readme_writer_tool(chat: Optional[Any]) -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("readme_writer", chat=chat)


async def approver_tool(chat: Optional[Any]) -> Any:
    print("approver_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("approver", chat=chat)


async def developer_tool(chat: Optional[Any]) -> Any:
    print("developer_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("developer", chat=chat)


readme_result = asyncio.run(
    readme_writer_tool("Provide an updated README file based on the recent changes.")
)

"""
# print(readme_result)
approval_result = asyncio.run(
    approver_tool("Revaluate the changes and approve or reject")
)
print(approval_result)
"""

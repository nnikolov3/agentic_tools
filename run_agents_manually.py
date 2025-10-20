# File: main.py
"""
THIS IS ONLY A CONVENIENCE SCRIPT TO RUN THE TOOLS
"""

from __future__ import annotations

import asyncio

from src.configurator import Configurator
from src.agents.agent import Agent
from fastmcp import FastMCP
import os
from typing import Any
from pathlib import Path

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()
mcp_name = "Agentic Tools"
mcp = FastMCP(mcp_name)


async def readme_writer_tool() -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("readme_writer")


async def approver_tool() -> Any:
    print("approver_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("approver")

    # print(readme_writer_tool())


res = asyncio.run(readme_writer_tool())
# res = asyncio.run(approver_tool())
print(res)

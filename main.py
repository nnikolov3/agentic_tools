# File: main.py
"""
MCP entrypoint registering the Approver and ReadmeWriter tools.
"""

from __future__ import annotations
from src.configurator import Configurator
from src.agents.agent import Agent
from fastmcp import FastMCP
import os
from typing import Any, Optional
from pathlib import Path

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()
mcp_name = "Agentic Tools"
mcp = FastMCP(mcp_name)


@mcp.tool(
    description="Walks the project directories, gets github information, and updates directly the README.md file"
)
async def readme_writer_tool(chat: Optional[Any]) -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("readme_writer", chat)


@mcp.tool(description="Audit recent code changes, and approve or reject changes")
async def approver_tool(chat: Optional[Any]) -> Any:
    print("approver_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("approver", chat=chat)


@mcp.tool(
    description="Writes high quality code based on the design guidelines and coding standards"
)
async def developer(chat: Optional[Any]) -> Any:
    print("developer_tool")
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("developer", chat=chat)


if __name__ == "__main__":
    mcp.run()

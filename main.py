# File: main.py
"""
MCP entrypoint registering the Approver and ReadmeWriter tools.
"""

from __future__ import annotations

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


@mcp.tool(
    description=(
        "Walks the project directories, gets github information, and updates directly the README.md file"
    )
)
async def readme_writer_tool(chat: Any | None) -> Any:
    print("readme_writer_tool")
    agent = Agent(configuration, "readme_writer", mcp_name, chat)
    return await agent.run_agent()


@mcp.tool(description="Audit recent code changes, and approve or reject changes")
async def approver_tool(chat: Any | None) -> Any:
    print("approver_tool")
    agent = Agent(configuration, "approver", mcp_name, chat)
    return await agent.run_agent()


@mcp.tool(
    description=(
        "Writes high quality code based on the design guidelines and coding standards"
    )
)
async def developer(chat: Any | None) -> Any:
    print("developer_tool")
    agent = Agent(configuration, "developer", mcp_name, chat)
    return await agent.run_agent()


if __name__ == "__main__":
    mcp.run()

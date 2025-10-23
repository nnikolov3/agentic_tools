"""
Purpose:
Main entry point for executing agentic tools.
This file provides functions to invoke specialized agents and workflows,
such as the 'Code Quality Enforcer', for targeted tasks on the codebase.
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from src.agents.agent import Agent
from src.configurator import Configurator
from src.workflows.code_quality_enforcer import CodeQualityEnforcer

# Configure basic logging for clear, explicit output.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

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


@mcp.tool(
    description=(
        "Creates a high quality architecture based on the design guidelines and coding standards"
    )
)
async def architect_tool(chat: Any | None) -> Any:
    print("architect_tool")
    agent = Agent(configuration, "architect", mcp_name, chat)
    return await agent.run_agent()


@mcp.tool(
    description=(
        "Runs a multi-step code quality enforcement workflow on a file or directory, updating comments, docstrings, and organizing imports. It uses a developer agent to fix any linting errors introduced by the commentator."
    )
)
async def commentator_tool(chat: Any | None) -> Any:
    """
    Runs the full code quality enforcement workflow on the specified file or directory.

    Args:
        chat: The file or directory path (as a string) to be processed.
    """
    path_str = str(chat)
    path_obj = Path(path_str)

    if not path_obj.exists():
        logger.error(f"Path does not exist: {path_str}")
        return f"Error: Path does not exist: {path_str}"

    try:
        enforcer = CodeQualityEnforcer(configuration, mcp_name)
        
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

    except Exception as e:
        logger.error(f"The code quality enforcer failed: {e}", exc_info=True)
        return f"Error: The code quality enforcer failed: {e}"


if __name__ == "__main__":
    mcp.run()

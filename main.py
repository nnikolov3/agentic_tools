# File: main.py
"""
MCP entrypoint registering the Approver and ReadmeWriter tools.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from src.agents.agent import Agent
from src.configurator import Configurator
from src.tools.shell_tools import ShellTools

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
        "Reviews a single source file, updates comments, docstrings, and organizes imports."
    )
)
async def commentator_tool(chat: Any | None) -> Any:
    """
    Applies the 'commentator' agent to a single file to improve its documentation.

    Args:
        chat: The file path (as a string) to the file to be processed.
    """
    file_path = str(chat)
    logger.info(f"Running 'commentator' agent on: {file_path}")

    try:
        # Use ShellTools for all file I/O as per constraints.
        shell_tools = ShellTools("commentator", configuration["agentic-tools"])

        # 1. Read the source file content.
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found at {file_path}"

        file_content: str = shell_tools.read_file_content_for_path(path_obj)
        if not file_content:
            logger.error(f"Could not read file or file is empty: {file_path}")
            return f"Error: Could not read file or file is empty at {file_path}"

        # 2. Instantiate and run the agent. The file content is passed as 'chat'.
        commentator_agent = Agent(
            configuration,
            "commentator",
            mcp_name,
            file_content,
        )
        updated_content: str | None = await commentator_agent.run_agent()

        # 3. Write the updated content back to the original file.
        if updated_content and isinstance(updated_content, str):
            # The agent is instructed to return only the file content, but this
            # handles the common case of an LLM wrapping its response in a code block.
            if updated_content.strip().startswith(
                "```"
            ) and updated_content.strip().endswith("```"):
                lines = updated_content.strip().split("\n")
                updated_content = "\n".join(lines[1:-1])

            shell_tools.write_file(file_path, updated_content)
            logger.info(f"Successfully updated comments in: {file_path}")
            return f"Successfully updated comments and docstrings in: {file_path}"
        else:
            logger.warning(
                f"Commentator agent returned no content for {file_path}. File not modified."
            )
            return f"Warning: Commentator agent returned no content for {file_path}. File not modified."

    except Exception as e:
        logger.error(
            f"An error occurred while running the commentator tool: {e}", exc_info=True
        )
        return f"Error: An unexpected error occurred: {e}"


if __name__ == "__main__":
    mcp.run()

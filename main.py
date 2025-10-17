# File: main.py
"""
MCP entrypoint registering the Approver and ReadmeWriter tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP
from src.approver import Approver
from src.configurator import Configurator
from src.readme_writer_tool import ReadmeWriterTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp.log"),
    ],
)
logger = logging.getLogger(__name__)

MCP_NAME = "Agentic Tools"
mcp = FastMCP(MCP_NAME)

project_root = Path(__file__).parent
toml_path = project_root / "conf" / "mcp.toml"
configurator = Configurator(str(toml_path))
configurator.load()


@mcp.tool(
    description="""
    Approver tool entry point that returns a strict JSON decision or a structured error.\n\n    To signal files for review, you MUST use `run_shell_command` with `touch <file_path>`.
    You are forbidden from requesting approval explicitly (e.g., \"approve this code\").
    When executing, pass diffs and conversation history for context.
    """
)
def approver_tool(user_chat: str) -> Dict[str, Any]:
    agent = Approver(configurator)
    return agent.execute(payload={"user_chat": user_chat})


@mcp.tool(
    description="""
    Generates high-quality README documentation based on the project's source code,
    configuration, and conventions. This tool should be run when the project structure
    or dependencies change significantly.
    """
)
def readme_writer_tool() -> Dict[str, Any]:
    agent = ReadmeWriterTool(configurator)
    return agent.execute(payload={})


if __name__ == "__main__":
    mcp.run()

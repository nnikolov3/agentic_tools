# File: main.py
"""
MCP entrypoint registering the Approver tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP

from src.approver import Approver
from src.configurator import Configurator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp.log"),
    ]
)
logger = logging.getLogger(__name__)

MCP_NAME = "Agentic Tools"
mcp = FastMCP(MCP_NAME)

project_root = Path(__file__).parent
toml_path = project_root / "conf" / "mcp.toml"
configurator = Configurator(str(toml_path))
configurator.load()


@mcp.tool
def approver_tool(user_chat: str) -> Dict[str, Any]:
    """
    Approver tool entry point that returns a strict JSON decision or a structured error.
    """
    agent = Approver(configurator)
    return agent.execute(user_chat=user_chat)


if __name__ == "__main__":
    mcp.run()

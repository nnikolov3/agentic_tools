"""
File: main.py
Author: Niko Nikolov
Scope: main entry point
"""

import logging
from pathlib import Path

# imports
from fastmcp import FastMCP

from src.approver import Approver
from src.configurator import Configurator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MCP_NAME = "Agentic Tools"
mcp = FastMCP(MCP_NAME)

# Use the project root directory instead of current working directory
project_root = Path(__file__).parent
toml_path = project_root / "conf" / "mcp.toml"
configurator = Configurator(str(toml_path))
configurator.get_toml_configuration()


@mcp.tool
def approver_tool(user_chat: str):
    """
    The main tool entry point for the Approver agent.

    This function instantiates the Approver class and calls its main execute
    method, passing along the chat history. The agent itself
    handles gathering all other necessary context.

    Parameters:
        user_chat (str): The recent user chat history.

    Returns:
        Dict[str, Any]: A dictionary containing the structured verdict from the agent.
    """
    approver_config = configurator.construct_configuration_dict("approver")
    logger.debug(f"Approver config: {approver_config}")
    approver_instance = Approver(approver_config)

    return approver_instance.execute(user_chat=user_chat)


if __name__ == "__main__":
    main()
    mcp.run()

"""
This script provides a command-line interface for the manual orchestration of AI agents.

It allows a user to select a specific agent (e.g., 'commentator', 'developer'),
provide it with a text prompt and an optional file path, and execute its task.
The script handles configuration loading, argument parsing, and graceful error
handling, making it a robust tool for testing and interacting with individual
agent workflows.
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.agents.agent import (
    Agent,
    DefaultAgent,
    ReadmeWriterAgent,
    CommentatorAgent,
    DeveloperAgent,
    ExpertAgent,
)

from src.configurator import get_config_dictionary
from src.scripts.ingest_knowledge_bank import KnowledgeBankIngestor

# --- Constants ---
# Using constants for file paths and names makes the code more explicit and easier to maintain.
# The project root is determined relative to this script's location for robustness.
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CONFIG_FILE_PATH: Path = PROJECT_ROOT / "conf" / "agentic_tools.toml"
PROJECT_NAME: str = "agentic-tools"

# Agent names are defined as constants to prevent typos and magic strings.
COMMENTATOR: str = "commentator"
DEVELOPER: str = "developer"
ARCHITECT: str = "architect"
APPROVER: str = "approver"
README_WRITER: str = "readme_writer"
INGEST_KNOWLEDGE_BANK: str = "ingest_knowledge_bank"
EXPERT: str = "expert"
VALID_AGENTS: Tuple[str, ...] = (
    COMMENTATOR,
    DEVELOPER,
    ARCHITECT,
    APPROVER,
    README_WRITER,
    INGEST_KNOWLEDGE_BANK,
    EXPERT,
)

# --- Logging Configuration ---
# Configure basic logging to provide clear, explicit output during execution.
# INFO level is suitable for user-facing status updates.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


async def execute_agent_task(
    config: Dict[str, Any],
    agent_name: str,
    chat: Optional[str],
    filepath: Optional[str],
) -> Any:
    """
    Initializes and runs a single agent task.

    This function encapsulates the logic for creating an agent instance and
    invoking its primary execution method. It includes error handling to
    log failures during the agent's run.

    Args:
        config: The application configuration dictionary.
        agent_name: The name of the agent to execute.
        chat: The input prompt or context for the agent.
        filepath: The path to a file or directory for the agent to process.

    Returns:
        The result produced by the agent's execution. The type is 'Any' as
        different agents may return different types of data.

    Raises:
        RuntimeError: If the agent's execution fails, wrapping the original exception.
    """
    logger.info(f"Initializing '{agent_name}' agent task.")
    agent: Agent
    match agent_name:
        case "readme_writer":
            agent = ReadmeWriterAgent(config, agent_name, PROJECT_NAME, chat, filepath)
        case "commentator":
            agent = CommentatorAgent(config, agent_name, PROJECT_NAME, chat, filepath)
        case "developer":
            agent = DeveloperAgent(config, agent_name, PROJECT_NAME, chat, filepath)
        case "expert":
            agent = ExpertAgent(config, agent_name, PROJECT_NAME, chat, filepath)
        case "ingest_knowledge_bank":
            return await KnowledgeBankIngestor(config["agentic-tools"]).run_ingestion()
        case _:
            agent = DefaultAgent(config, agent_name, PROJECT_NAME, chat, filepath)
    try:
        result = await agent.run_agent()
        logger.info(f"Agent '{agent_name}' task completed successfully.")
        return result
    except Exception as error:
        logger.error(
            f"An error occurred during '{agent_name}' agent execution.",
            exc_info=True,
        )
        raise RuntimeError(f"Agent '{agent_name}' failed to execute.") from error


def _setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the command-line interface.

    Returns:
        An `argparse.ArgumentParser` instance configured with the script's
        command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to manually trigger AI agents.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "agent_name",
        type=str,
        choices=VALID_AGENTS,
        help=f"The name of the agent to run. Choices: {', '.join(VALID_AGENTS)}",
        metavar="AGENT_NAME",
    )
    parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="The chat prompt or context for the agent.",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="The path to the file or directory for the agent to process.",
    )

    return parser


async def main() -> None:
    """
    The main entry point for the script.

    This function orchestrates the entire process:
    1. Loads the application configuration.
    2. Parses command-line arguments.
    3. Executes the selected agent task.
    4. Prints the result to the console.
    It includes top-level error handling to ensure a clean exit.
    """
    try:
        configuration = get_config_dictionary(CONFIG_FILE_PATH)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {CONFIG_FILE_PATH}")
        return
    except Exception as config_error:
        logger.critical("Failed to load or parse configuration.", exc_info=True)
        raise RuntimeError("Configuration error.") from config_error

    parser = _setup_argument_parser()
    args = parser.parse_args()

    try:
        result = await execute_agent_task(
            configuration,
            args.agent_name,
            args.chat,
            args.filepath,
        )
        logger.info("--- Agent Result ---")
        # The result can be any data type, so printing it directly provides the
        # clearest output for a developer using this manual tool.
        print(result)
        logger.info("--------------------")
    except Exception as e:
        # The specific error is already logged within execute_agent_task.
        # This catch prevents a top-level traceback, providing a cleaner exit.
        logger.critical(
            f"Script execution failed. Please review the logs for details. Error: {e}",
        )


if __name__ == "__main__":
    asyncio.run(main())

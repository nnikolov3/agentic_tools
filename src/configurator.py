# src/configurator.py

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_config_dictionary(config_path: str | Path) -> dict[str, Any]:
    """
    Loads the TOML configuration file and returns it as a dictionary.
    """
    try:
        resolved_path = Path(config_path).resolve()
        with resolved_path.open(mode="rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Configuration file not found at path: '{config_path}'"
        ) from error
    except IsADirectoryError as error:
        raise IsADirectoryError(
            f"Configuration path is a directory, not a file: '{config_path}'"
        ) from error
    except PermissionError as error:
        raise PermissionError(
            f"Permission denied when reading configuration file: '{config_path}'"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"Invalid TOML syntax in configuration file '{config_path}': {error}"
        ) from error
    except OSError as error:
        raise OSError(
            f"An OS error occurred while reading configuration file '{config_path}': {error}"
        ) from error

    return config


def find_config(
    start_dir: Path | None = None,
    config_name: str = "agentic-tools.toml",
) -> Path:
    """
    Finds the configuration file with logging.
    1. Checks for CONFIG_PATH environment variable.
    2. Looks for config_name in the current working directory.
    """
    # Env override has the highest priority
    if env_path := os.getenv("CONFIG_PATH"):
        logger.debug("Checking for config file in CONFIG_PATH: %s", env_path)
        if Path(env_path).exists():
            logger.info("Found config file at CONFIG_PATH: %s", env_path)
            return Path(env_path).resolve()

    # Check current working directory
    cwd = start_dir or Path.cwd()
    candidate = cwd / config_name
    logger.debug("Checking for config file in current directory: %s", candidate)
    if candidate.exists():
        logger.info("Found config file in current directory: %s", candidate)
        return candidate.resolve()

    raise FileNotFoundError(
        f"Config '{config_name}' not found in CONFIG_PATH or current directory."
    )


def get_available_agents(config: dict[str, Any]) -> list[str]:
    """
    Extracts available agent names from the [agents] section.
    """
    agents_section = config.get("agents", {})
    return [
        agent_name
        for agent_name, agent_config in agents_section.items()
        if isinstance(agent_config, dict)
    ]


def get_agent_config(config: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """
    Retrieves a specific agent's configuration from the [agents] section.
    """
    return config.get("agents", {}).get(agent_name, {})


def get_golden_rules(config: dict[str, Any]) -> str:
    """
    Extracts golden rules string for agent prompts from the [golden_rules] section.
    """
    return config.get("golden_rules", {}).get("rules", "")

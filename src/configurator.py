import tomllib
from pathlib import Path
from typing import Any, Dict, List


def get_config_dictionary(config_path: str | Path) -> Dict[str, Any]:
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


def get_available_agents(config: Dict[str, Any]) -> List[str]:
    agents_section = config.get("agentic-tools", {}).get("agents", {})
    return [
        agent_name
        for agent_name in agents_section
        if isinstance(agents_section[agent_name], dict)
    ]

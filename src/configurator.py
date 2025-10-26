# src/configurator.py
"""
Provides a robust mechanism for loading and parsing TOML configuration files.

This module offers a simple function, `get_config_dictionary`, which encapsulates
the logic for reading a configuration file from a specified path. It performs
essential validation and provides clear, specific error handling for common
issues such as missing files, permission errors, or invalid TOML syntax.
Its primary goal is to ensure a reliable and straightforward way to manage
application settings.
"""

import tomllib
from pathlib import Path
from typing import Any


def get_config_dictionary(config_path: str | Path) -> dict[str, Any]:
    """
    Loads, validates, and parses a TOML configuration file.

    This function simplifies configuration loading by directly attempting to open
    and parse the file, relying on exceptions for flow control in error cases.
    It handles potential errors related to file existence, path type (i.e., a
    directory), permissions, and TOML syntax, raising specific, informative
    exceptions to the caller.

    Args:
        config_path: The file path to the TOML configuration file.

    Returns:
        A dictionary representing the parsed configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist at the
                           specified path.
        IsADirectoryError: If the specified path points to a directory
                           instead of a file.
        PermissionError: If the application lacks the necessary permissions
                         to read the configuration file.
        ValueError: If the configuration file contains invalid TOML syntax.
        OSError: For other general operating system errors encountered
                 during the file reading process.
    """
    try:
        # Using Path.open() is a clean way to handle file operations and will
        # raise appropriate, specific exceptions if the path is invalid or
        # permissions are incorrect.
        resolved_path = Path(config_path)
        with resolved_path.open(mode="rb") as config_file:
            return tomllib.load(config_file)
    except FileNotFoundError as error:
        # Chain the original exception to provide full context.
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

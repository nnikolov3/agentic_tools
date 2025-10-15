"""
File: src/configurator.py
Author: Niko Nikolov
Scope: Gets the configuration from mcp.toml
Implementation:
"""

from __future__ import annotations

import logging
import tomllib
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Configurator:
    """
    Loads conf/mcp.toml and exposes agent configuration dictionaries.
    """

    def __init__(self, toml_path: str) -> None:
        if not isinstance(toml_path, str) or not toml_path.strip():
            raise ValueError("toml_path must be a non-empty string")
        self.toml_path: str = toml_path
        self.toml_config: Optional[Dict[str, Any]] = None

    def get_toml_configuration(self) -> int:
        """
        Load TOML configuration; returns 0 on success, 1 on failure.
        """
        try:
            with open(self.toml_path, "rb") as f:
                self.toml_config = tomllib.load(f)
            logger.info(f"Successfully loaded TOML configuration from {self.toml_path}")
            return 0
        except FileNotFoundError:
            logger.error(f"TOML file not found at {self.toml_path}!")
            return 1
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Invalid TOML at {self.toml_path}: {e}")
            return 1

    def construct_configuration_dict(self, agent: str) -> Dict[str, Any]:
        """
        Return the dictionary for [multi-agent-mcp.<agent>] or raise on shape errors.
        """
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError("agent must be a non-empty string")
        if not isinstance(self.toml_config, dict):
            raise RuntimeError("TOML not loaded; call get_toml_configuration() first")

        root = self.toml_config.get("multi-agent-mcp")
        if not isinstance(root, dict):
            raise KeyError("Missing [multi-agent-mcp] section in configuration")

        section = root.get(agent)
        if not isinstance(section, dict):
            raise KeyError(
                f"Missing [multi-agent-mcp.{agent}] section in configuration"
            )
        return section

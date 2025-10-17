# File: src/configurator.py
"""
Loads conf/mcp.toml and exposes validated, typed configuration accessors.

Single responsibility: configuration schema parsing and validation.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Optional

logger = logging.getLogger(__name__)

TOML_ROOT = "multi-agent-mcp"

# Configuration values - these are defined in mcp.toml, no defaults here


@dataclass(frozen=True)
class ContextPolicy:
    recent_minutes: int
    include_extensions: Tuple[str, ...]
    exclude_dirs: Tuple[str, ...]
    max_file_bytes: int
    max_total_bytes: int
    design_docs: Tuple[str, ...]
    source_code_directory: Tuple[str, ...]
    tests_directory: Tuple[str, ...]
    project_directories: Tuple[str, ...]
    embedding_model_sizes: Dict[str, int]


class Configurator:
    """
    Usage:
        cfg = Configurator("conf/mcp.toml")
        cfg.load()
        approver_cfg = cfg.get_agent_config("approver")
    """

    def __init__(self, toml_path: str) -> None:
        if not isinstance(toml_path, str) or not toml_path.strip():
            raise ValueError("toml_path must be a non-empty string")
        self._toml_path = toml_path
        self._toml: Optional[Mapping[str, Any]] = None

    def load(self) -> None:
        path = Path(self._toml_path)
        if not path.exists():
            raise FileNotFoundError(f"TOML not found: {path}")
        with path.open("rb") as config_file:
            self._toml = tomllib.load(config_file)
        logger.info("Loaded configuration: %s", path)

    def _root(self) -> Mapping[str, Any]:
        if self._toml is None:
            raise RuntimeError("Configuration not loaded; call load() first")
        root = self._toml.get(TOML_ROOT)
        if not isinstance(root, dict):
            raise KeyError(f"Missing root section [{TOML_ROOT}]")
        return root

    def get_agent_config(self, agent_key: str) -> Mapping[str, Any]:
        root = self._root()
        section = root.get(agent_key)
        if not isinstance(section, dict):
            raise KeyError(f"Missing section [{TOML_ROOT}.{agent_key}]")
        required_keys = (
            "prompt",
            "model_name",
            "model_providers",
            "temperature",
            "description",
            "skills",
        )
        for key in required_keys:
            if key not in section:
                raise KeyError(f"Missing '{key}' in [{TOML_ROOT}.{agent_key}]")

        # Add Qdrant configuration if available for this agent
        agent_section_with_additional = dict(section)
        qdrant_section = root.get(f"{agent_key}.qdrant")
        if isinstance(qdrant_section, dict):
            agent_section_with_additional["qdrant"] = qdrant_section

        return agent_section_with_additional

    def get_context_policy(self) -> ContextPolicy:
        root = self._root()

        # Helper function to get values from root configuration
        def _get_tuple_of_strings(key: str) -> Tuple[str, ...]:
            value = root.get(key)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"'{key}' must be a list of strings in [{TOML_ROOT}]")
            return tuple(value)

        def _get_int_value(key: str) -> int:
            value = root.get(key)
            if not isinstance(value, int):
                raise ValueError(f"'{key}' must be an integer in [{TOML_ROOT}]")
            return value

        def _get_embedding_model_sizes() -> Dict[str, int]:
            sizes_config = root.get("embedding_model_sizes", {})
            # Validate that all values are integers
            validated_sizes = {}
            for model_name, size in sizes_config.items():
                if isinstance(size, int):
                    validated_sizes[model_name] = size
                else:
                    logger.warning(
                        f"Invalid size for model {model_name}: {size} (must be integer)"
                    )
            return validated_sizes

        return ContextPolicy(
            recent_minutes=_get_int_value("recent_minutes"),
            include_extensions=_get_tuple_of_strings("include_extensions"),
            exclude_dirs=_get_tuple_of_strings("exclude_dirs"),
            max_file_bytes=_get_int_value("max_file_bytes"),
            max_total_bytes=_get_int_value("max_total_bytes"),
            design_docs=_get_tuple_of_strings("design_docs"),
            source_code_directory=_get_tuple_of_strings("source_code_directory"),
            tests_directory=_get_tuple_of_strings("tests_directory"),
            project_directories=_get_tuple_of_strings("project_directories"),
            embedding_model_sizes=_get_embedding_model_sizes(),
        )

    def combine_prompt_with_skills(
        self, base_prompt: str, skills: tuple[str, ...]
    ) -> str:
        """
        Combine a base prompt with skills for tagging purposes.

        Args:
            base_prompt: The original system prompt
            skills: Tuple of skill strings to include as tags

        Returns:
            Combined prompt with skills as tags
        """
        if skills:
            skills_str = ", ".join(skills)
            return f"# Tags: {skills_str}\n{base_prompt}"
        else:
            return base_prompt

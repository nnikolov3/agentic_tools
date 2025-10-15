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
from typing import Any, List, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

TOML_ROOT = "multi-agent-mcp"

# Default configuration values
DEFAULT_MAX_DOCS = 2
DEFAULT_MAX_DOC_BYTES = 262_144
DEFAULT_MAX_FILE_BYTES = 262_144
DEFAULT_MAX_TOTAL_BYTES = 1_048_576
DEFAULT_RECENT_MINUTES = 10


@dataclass(frozen=True)
class SignalGroup:
    name: str
    keywords: Tuple[str, ...]


@dataclass(frozen=True)
class DocDiscovery:
    enabled: bool
    patterns: Tuple[str, ...]
    signal_groups: Tuple[SignalGroup, ...]
    max_docs: int
    max_doc_bytes: int


@dataclass(frozen=True)
class ContextPolicy:
    recent_minutes: int
    src_dir: str
    include_extensions: Tuple[str, ...]
    exclude_dirs: Tuple[str, ...]
    max_file_bytes: int
    max_total_bytes: int
    docs_paths: Tuple[str, ...]
    discovery: DocDiscovery


class Configurator:
    """
    Usage:
        cfg = Configurator("conf/mcp.toml")
        cfg.load()
        approver_cfg = cfg.get_agent_config("approver")
        policy = cfg.get_context_policy()
    """

    def __init__(self, toml_path: str) -> None:
        if not isinstance(toml_path, str) or not toml_path.strip():
            raise ValueError("toml_path must be a non-empty string")
        self._toml_path = toml_path

    def load(self) -> None:
        path = Path(self._toml_path)
        if not path.exists():
            raise FileNotFoundError(f"TOML not found: {path}")
        with path.open("rb") as f:
            self._toml = tomllib.load(f)
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
        )
        for key in required_keys:
            if key not in section:
                raise KeyError(f"Missing '{key}' in [{TOML_ROOT}.{agent_key}]")
        return section

    def get_context_policy(self) -> ContextPolicy:
        root = self._root()
        approver = root.get("approver", {})
        if not isinstance(approver, dict):
            raise KeyError(f"Missing [{TOML_ROOT}.approver]")

        def _get_tuple_of_strings(key: str, default: Sequence[str]) -> Tuple[str, ...]:
            value = approver.get(key, default)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(
                    f"'{key}' must be a list of strings in [{TOML_ROOT}.approver]"
                )
            return tuple(value)

        def _get_int_value(key: str, default: int) -> int:
            value = approver.get(key, default)
            if not isinstance(value, int):
                raise ValueError(
                    f"'{key}' must be an integer in [{TOML_ROOT}.approver]"
                )
            return value

        docs_paths = _get_tuple_of_strings("docs_paths", ())

        doc_discovery_config = approver.get("doc_discovery", {})
        if not isinstance(doc_discovery_config, dict):
            doc_discovery_config = {}

        enabled = bool(doc_discovery_config.get("enabled", True))
        patterns = tuple(map(str, doc_discovery_config.get("patterns", [])))

        raw_groups = doc_discovery_config.get("signal_groups", [])
        groups: List[SignalGroup] = []
        if isinstance(raw_groups, list):
            for item in raw_groups:
                if (
                    isinstance(item, dict)
                    and "name" in item
                    and "keywords" in item
                    and isinstance(item["keywords"], list)
                ):
                    groups.append(
                        SignalGroup(
                            name=str(item["name"]),
                            keywords=tuple(map(str.lower, item["keywords"])),
                        )
                    )

        max_docs_raw = doc_discovery_config.get("max_docs", DEFAULT_MAX_DOCS)
        max_docs = int(max_docs_raw) if isinstance(max_docs_raw, int) else DEFAULT_MAX_DOCS

        max_doc_bytes_raw = doc_discovery_config.get(
            "max_doc_bytes", approver.get("max_file_bytes", DEFAULT_MAX_DOC_BYTES)
        )
        max_doc_bytes = (
            int(max_doc_bytes_raw) if isinstance(max_doc_bytes_raw, int) else DEFAULT_MAX_DOC_BYTES
        )

        discovery = DocDiscovery(
            enabled=enabled,
            patterns=patterns,
            signal_groups=tuple(groups),
            max_docs=max_docs,
            max_doc_bytes=max_doc_bytes,
        )

        return ContextPolicy(
            recent_minutes=_get_int_value("recent_minutes", DEFAULT_RECENT_MINUTES),
            src_dir=str(approver.get("src_dir", "src")),
            include_extensions=_get_tuple_of_strings(
                "include_extensions",
                (
                    ".py",
                    ".rs",
                    ".go",
                    ".ts",
                    ".tsx",
                    ".js",
                    ".json",
                    ".md",
                    ".toml",
                    ".yml",
                    ".yaml",
                ),
            ),
            exclude_dirs=_get_tuple_of_strings(
                "exclude_dirs",
                (
                    ".git",
                    ".github",
                    ".gitlab",
                    "node_modules",
                    "venv",
                    ".venv",
                    "dist",
                    "build",
                    "target",
                    "__pycache__",
                ),
            ),
            max_file_bytes=_get_int_value("max_file_bytes", DEFAULT_MAX_FILE_BYTES),
            max_total_bytes=_get_int_value("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES),
            docs_paths=docs_paths,
            discovery=discovery,
        )

"""
Base Agent class that provides common functionality for all agents.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src._api import UnifiedResponse, api_caller
from src.configurator import Configurator, ContextPolicy
from src.prompt_utils import serialize_raw_response
from src.shell_tools import (
    collect_recent_sources,
    discover_docs_and_load,
    load_explicit_docs,
)

logger = logging.getLogger(__name__)

# Default value for project root when not specified
DEFAULT_PROJECT_ROOT = "PWD"


@dataclass(frozen=True)
class BaseInputs:
    prompt: str
    model_name: str
    model_providers: List[str]
    temperature: float
    project_root: str
    policy: ContextPolicy


class BaseAgent(ABC):
    """
    Base class for all agents that provides common functionality.
    """

    def __init__(self, configurator: Configurator) -> None:
        self._configurator = configurator

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the name of this agent for configuration lookup.
        """
        pass

    def _load_inputs(self) -> BaseInputs:
        """
        Load common inputs from configuration.
        """
        agent = self._configurator.get_agent_config(self.get_agent_name())
        policy = self._configurator.get_context_policy()
        project_root = str(agent.get("project_root", DEFAULT_PROJECT_ROOT))
        if project_root == DEFAULT_PROJECT_ROOT:
            project_root = os.getcwd()
        return BaseInputs(
            prompt=str(agent["prompt"]),
            model_name=str(agent["model_name"]),
            model_providers=list(agent["model_providers"]),
            temperature=float(agent["temperature"]),
            project_root=project_root,
            policy=policy,
        )

    def _assemble_docs(self, inputs: BaseInputs) -> str:
        """
        Assemble documentation files for context.
        """
        # Try explicit docs first
        explicit = load_explicit_docs(
            project_root=inputs.project_root,
            docs_paths=inputs.policy.docs_paths,
            max_doc_bytes=inputs.policy.discovery.max_doc_bytes,
        )
        docs: List[str] = []
        if explicit:
            for doc_path, content in explicit:
                docs.append(f"\n===== DOC: {doc_path} =====\n{content}\n")

        # Fallback to discovery if none found
        if not explicit and inputs.policy.discovery.enabled:
            groups = tuple(
                (group.name, group.keywords) for group in inputs.policy.discovery.signal_groups
            )
            discovered = discover_docs_and_load(
                project_root=inputs.project_root,
                exclude_dirs=inputs.policy.exclude_dirs,
                patterns=inputs.policy.discovery.patterns,
                signal_groups=groups,
                max_docs=inputs.policy.discovery.max_docs,
                max_doc_bytes=inputs.policy.discovery.max_doc_bytes,
            )
            for doc_path, content in discovered:
                docs.append(f"\n===== DOC: {doc_path} =====\n{content}\n")

        return "".join(docs)

    def _assemble_sources(self, inputs: BaseInputs) -> str:
        """
        Assemble recent source files for context.
        """
        return collect_recent_sources(
            project_root=inputs.project_root,
            include_extensions=inputs.policy.include_extensions,
            exclude_dirs=inputs.policy.exclude_dirs,
            recent_minutes=inputs.policy.recent_minutes,
            max_file_bytes=inputs.policy.max_file_bytes,
            max_total_bytes=inputs.policy.max_total_bytes,
        )

    def _make_api_call(self, inputs: BaseInputs, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make an API call using the provided inputs and messages.
        """
        response: Optional[UnifiedResponse] = api_caller(
            {
                "prompt": inputs.prompt,
                "model_name": inputs.model_name,
                "model_providers": inputs.model_providers,
                "temperature": inputs.temperature,
            },
            messages,
        )
        if response is None or not response.content:
            return {
                "status": "error",
                "data": {},
                "message": "No valid response received from any provider.",
            }

        serialized_raw = serialize_raw_response(response.raw_response)
        return {
            "status": "success",
            "data": {
                "provider": response.provider_name,
                "model": response.model_name,
                "raw_text": response.content,
                "raw_response": serialized_raw,
            },
            "message": "Analysis complete.",
        }

    @abstractmethod
    def _create_messages(self, inputs: BaseInputs, docs: str, sources: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create the system and user messages specific to this agent.
        """
        pass

    @abstractmethod
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        """
        pass
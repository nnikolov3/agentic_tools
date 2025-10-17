"""
Base Agent class that provides common functionality for all agents.
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src._api import UnifiedResponse, api_caller
from src.configurator import Configurator, ContextPolicy
from src.prompt_utils import serialize_raw_response
from src.shell_tools import (
    collect_recent_sources,
    load_explicit_docs,
)
from pathlib import Path

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
        project_root = str(agent.get("project_root", "PWD"))
        if project_root == "PWD":
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
        # Try explicit docs first using configured design documents
        explicit = load_explicit_docs(
            project_root=inputs.project_root,
            docs_paths=inputs.policy.design_docs,
            max_doc_bytes=inputs.policy.max_file_bytes,  # Use configured value instead of hardcoded
        )
        docs: List[str] = []
        if explicit:
            for doc_path, content in explicit:
                docs.append(f"\n===== DOC: {doc_path} =====\n{content}\n")

        return "".join(docs)

    def _assemble_sources(self, inputs: BaseInputs) -> Tuple[str, List[Path]]:
        """
        Assemble source files for context based on agent-specific needs.
        Each agent can override this method to use appropriate directories.
        Returns a tuple of (formatted_content, included_source_files).
        """
        # Default to original logic - this can be overridden by specific agents
        return collect_recent_sources(
            project_root=inputs.project_root,
            include_extensions=inputs.policy.include_extensions,
            exclude_dirs=inputs.policy.exclude_dirs,
            recent_minutes=inputs.policy.recent_minutes,
            max_file_bytes=inputs.policy.max_file_bytes,
            max_total_bytes=inputs.policy.max_total_bytes,
        )

    def _make_api_call(
        self, inputs: BaseInputs, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
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
    def _create_messages(
        self, inputs: BaseInputs, docs: str, sources: str, payload: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Create the system and user messages specific to this agent.
        """
        pass

    def _store_in_qdrant_if_enabled(
        self,
        agent_name: str,
        data: Dict[str, Any],
        store_func: Callable[..., bool],
        content_extractor_func: Callable[[Dict[str, Any]], str],
        store_func_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Helper method to store agent-generated data in Qdrant if enabled.

        Args:
            agent_name: Name of the agent for configuration lookup
            data: The data to store
            store_func: Function that performs the actual storage operation (e.g., store_approver_decision, store_patch)
            content_extractor_func: Function that takes the data dict and returns content for embedding
            store_func_kwargs: Optional keyword arguments to pass to the store_func.

        Returns:
            True if successfully stored or Qdrant disabled, False if storage failed
        """
        agent_config = self._configurator.get_agent_config(agent_name)
        qdrant_config = agent_config.get("qdrant")

        if not qdrant_config or not qdrant_config.get("enabled", False):
            return True  # Return True if Qdrant is not enabled (not a failure)

        try:
            from src.qdrant_integration import QdrantIntegration

            # Get the policy to access embedding_model_sizes
            policy = self._configurator.get_context_policy()

            qdrant_integration = QdrantIntegration(
                local_path=qdrant_config.get("local_path", "/qdrant"),
                collection_name=qdrant_config.get(
                    "collection_name", f"{agent_name}_storage"
                ),
                embedding_model=qdrant_config.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                model_sizes=policy.embedding_model_sizes,
            )

            # Extract content for embedding using provided function
            content_for_embedding = content_extractor_func(data)

            # Generate a unique ID for this storage
            storage_id = str(uuid.uuid4())

            # Call the specific storage function provided by the agent
            # Pass the extracted content and the data payload to the storage function
            kwargs = store_func_kwargs or {}
            success = store_func(
                qdrant_integration, storage_id, content_for_embedding, data, **kwargs
            )

            if success:
                logging.getLogger(__name__).info(
                    f"Successfully stored data in Qdrant with ID: {storage_id}"
                )
                return True
            else:
                logging.getLogger(__name__).error("Failed to store data in Qdrant")
                return False

        except ImportError as import_error:
            logging.getLogger(__name__).warning(
                f"Qdrant integration not available, skipping storage: {import_error}"
            )
            return True  # Don't treat import error as failure
        except Exception as general_error:
            logging.getLogger(__name__).error(
                f"Error initializing or using Qdrant integration: {general_error}"
            )
            return False

    @abstractmethod
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        """
        pass

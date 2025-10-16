# File: src/approver.py
"""
Approver agent: assembles context and requests a final decision as strict JSON.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default value for project root when not specified
DEFAULT_PROJECT_ROOT = "PWD"

from src._api import UnifiedResponse, api_caller
from src.configurator import Configurator, ContextPolicy
from src.prompt_utils import serialize_raw_response
from src.shell_tools import (
    collect_recent_sources,
    discover_docs_and_load,
    load_explicit_docs,
)

logger = logging.getLogger(__name__)





@dataclass(frozen=True)
class ApproverInputs:
    prompt: str
    model_name: str
    model_providers: List[str]
    temperature: float
    project_root: str
    policy: ContextPolicy


class Approver:
    """
    Final gatekeeper that reads design docs and recent code changes, then returns a JSON decision.
    """

    def __init__(self, configurator: Configurator) -> None:
        self._configurator = configurator

    def _load_inputs(self) -> ApproverInputs:
        agent = self._configurator.get_agent_config("approver")
        policy = self._configurator.get_context_policy()
        project_root = str(agent.get("project_root", DEFAULT_PROJECT_ROOT))
        if project_root == DEFAULT_PROJECT_ROOT:
            project_root = os.getcwd()
        return ApproverInputs(
            prompt=str(agent["prompt"]),
            model_name=str(agent["model_name"]),
            model_providers=list(agent["model_providers"]),
            temperature=float(agent["temperature"]),
            project_root=project_root,
            policy=policy,
        )

    def _assemble_docs(self, inputs: ApproverInputs) -> str:
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

    def _assemble_sources(self, inputs: ApproverInputs) -> str:
        return collect_recent_sources(
            project_root=inputs.project_root,
            include_extensions=inputs.policy.include_extensions,
            exclude_dirs=inputs.policy.exclude_dirs,
            recent_minutes=inputs.policy.recent_minutes,
            max_file_bytes=inputs.policy.max_file_bytes,
            max_total_bytes=inputs.policy.max_total_bytes,
        )

    def _messages(
        self, inputs: ApproverInputs, docs: str, sources: str, user_chat: str
    ) -> List[Dict[str, str]]:
        # Get skills from agent config if available
        agent_config = self._configurator.get_agent_config("approver")
        skills = agent_config.get("skills", [])
        
        # Use the configurator's method to combine prompt with skills
        system = self._configurator.combine_prompt_with_skills(inputs.prompt.strip(), tuple(skills))
        
        user = (
            "# Full Context for Final Review\n\n"
            "## 1. Design Documents & Principles:\n"
            f"{docs or '(no docs found)'}\n\n"
            "## 2. Recent Code Changes:\n"
            f"{sources or '(no recent source changes found)'}\n\n"
            "## 3. User Chat History:\n"
            f"{user_chat.strip()}\n\n"
            "## Your Task:\n"
            "Return ONLY the required JSON object with the specified schema.\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def execute(self, user_chat: str) -> Dict[str, Any]:
        inputs = self._load_inputs()
        docs = self._assemble_docs(inputs)
        sources = self._assemble_sources(inputs)

        if not sources:
            src_path = Path(inputs.project_root) / inputs.policy.src_dir
            src_files = []
            
            for extension in inputs.policy.include_extensions:
                for file_path in src_path.rglob(f"*{extension}"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(src_path)
                        src_files.append(str(relative_path))
            
            return {
                "status": "no_recent_files",
                "data": {"files": src_files},
                "message": "No recent source files were found to analyze. Please choose a file from the list below and re-run the tool with the file path as an argument.",
            }

        messages = self._messages(inputs, docs, sources, user_chat)

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

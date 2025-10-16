# File: src/readme_writer_tool.py
"""
Intelligent README writer tool that generates excellent documentation based on best practices,
technical writing standards, source code analysis, and local information.
"""

from __future__ import annotations

import logging
import os
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
    get_git_info,
    get_project_structure,
)

logger = logging.getLogger(__name__)

# Default value for project root when not specified
DEFAULT_PROJECT_ROOT = "PWD"





@dataclass(frozen=True)
class ReadmeWriterInputs:
    prompt: str
    model_name: str
    model_providers: List[str]
    temperature: float
    project_root: str
    policy: ContextPolicy


class ReadmeWriterTool:
    """
    Intelligent README writer that generates excellent documentation based on best practices,
    technical writing standards, source code analysis, and local information.
    """

    def __init__(self, configurator: Configurator) -> None:
        self._configurator = configurator

    def _load_inputs(self) -> ReadmeWriterInputs:
        agent = self._configurator.get_agent_config("readme_writer")
        policy = self._configurator.get_context_policy()
        project_root = str(agent.get("project_root", DEFAULT_PROJECT_ROOT))
        if project_root == DEFAULT_PROJECT_ROOT:
            project_root = os.getcwd()
        return ReadmeWriterInputs(
            prompt=str(agent["prompt"]),
            model_name=str(agent["model_name"]),
            model_providers=list(agent["model_providers"]),
            temperature=float(agent["temperature"]),
            project_root=project_root,
            policy=policy,
        )

    def _assemble_docs(self, inputs: ReadmeWriterInputs) -> str:
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

    def _assemble_sources(self, inputs: ReadmeWriterInputs) -> str:
        return collect_recent_sources(
            project_root=inputs.project_root,
            include_extensions=inputs.policy.include_extensions,
            exclude_dirs=inputs.policy.exclude_dirs,
            recent_minutes=inputs.policy.recent_minutes,
            max_file_bytes=inputs.policy.max_file_bytes,
            max_total_bytes=inputs.policy.max_total_bytes,
        )



    def _assemble_project_structure(self, inputs: ReadmeWriterInputs) -> str:
        """Generate project structure information."""
        project_structure = get_project_structure(inputs.project_root, max_depth=3)
        return f"# Project Structure\n{project_structure}"

    def _assemble_config_info(self, inputs: ReadmeWriterInputs) -> str:
        """Extract and summarize configuration information."""
        config_path = Path(inputs.project_root) / "conf" / "mcp.toml"
        if config_path.exists():
            config_content = config_path.read_text(encoding="utf-8")
            return f"\n===== CONFIG: conf/mcp.toml =====\n{config_content}\n"
        return "(no configuration file found)"

    def _format_git_info(self, git_info: dict[str, str | bool | None]) -> str:
        """Format git information for inclusion in the context."""
        git_lines = ["# Git Information"]
        if git_info.get("remote_url"):
            git_lines.append(f"Repository URL: {git_info['remote_url']}")
        if git_info.get("branch"):
            git_lines.append(f"Current Branch: {git_info['branch']}")
        if git_info.get("is_dirty") is not None:
            status = "dirty" if git_info["is_dirty"] else "clean"
            git_lines.append(f"Repository Status: {status}")
        
        return "\n".join(git_lines)

    def _messages(
        self, 
        inputs: ReadmeWriterInputs, 
        docs: str, 
        sources: str, 
        config_info: str, 
        project_structure: str,
        git_info: dict[str, str | bool | None]
    ) -> List[Dict[str, str]]:
        # Get skills from agent config if available
        agent_config = self._configurator.get_agent_config("readme_writer")
        skills = agent_config.get("skills", [])
        
        # Use the configurator's method to combine prompt with skills
        system = self._configurator.combine_prompt_with_skills(inputs.prompt.strip(), tuple(skills))
        
        git_info_formatted = self._format_git_info(git_info)
        user = (
            "# Project Information for README Generation\n\n"
            "## 1. Git Information:\n"
            f"{git_info_formatted}\n\n"
            "## 2. Design Documents & Principles:\n"
            f"{docs or '(no docs found)'}\n\n"
            "## 3. Source Code:\n"
            f"{sources or '(no source code found)'}\n\n"
            "## 4. Project Structure:\n"
            f"{project_structure}\n\n"
            "## 5. Configuration:\n"
            f"{config_info}\n\n"
            "## Your Task:\n"
            "Generate a comprehensive yet concise README.md file that follows best practices for technical documentation.\n"
            "Include relevant sections like:\n"
            "- Project title and description\n"
            "- Key features\n"
            "- Prerequisites and system requirements\n"
            "- Installation instructions (specific to this project's dependencies)\n"
            "- Usage examples\n"
            "- Configuration details\n"
            "- Project structure explanation\n"
            "- Contributing guidelines\n"
            "- License information\n\n"
            "Make sure to use specific information from this project, not generic placeholders.\n"
            f"Use the correct project URL: {git_info.get('remote_url', 'https://github.com/nnikolov3/multi-agent-mcp.git')}\n"
            "Focus on simplicity, clarity, and utility. Avoid generic examples like 'apt-get' if not applicable.\n"
            "Provide concrete, actionable examples based on the actual project structure and code.\n"
            "Do not include unnecessary information like virtual environment setup if not specifically relevant to the project.\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def execute(self) -> Dict[str, Any]:
        inputs = self._load_inputs()
        docs = self._assemble_docs(inputs)
        sources = self._assemble_sources(inputs)
        project_structure = self._assemble_project_structure(inputs)
        config_info = self._assemble_config_info(inputs)
        git_info = get_git_info(inputs.project_root)

        messages = self._messages(inputs, docs, sources, config_info, project_structure, git_info)

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
                "readme_content": response.content,
                "raw_response": serialized_raw,
            },
            "message": "README generation complete.",
        }
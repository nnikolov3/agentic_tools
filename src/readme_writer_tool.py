# File: src/readme_writer_tool.py
"""
Intelligent README writer tool that generates excellent documentation based on best practices,
technical writing standards, source code analysis, and local information.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.base_agent import BaseAgent, BaseInputs
from src.shell_tools import (
    get_git_info,
    get_project_structure,
    read_file_head,
    _find_source_files,
)

logger = logging.getLogger(__name__)


def store_readme_generation(
    qdrant_integration: Any,
    storage_id: str,
    content_for_embedding: str,
    data: Dict[str, Any],
    git_info: dict[str, str | bool | None],
    project_structure: str,
) -> bool:
    """
    Performs the actual storage of the generated README in Qdrant.
    """
    readme_content = data.get("readme_content", "")
    inputs = data.get("inputs")

    if inputs is None:
        logger.error("Inputs object missing from data payload for Qdrant storage.")
        return False

    patch_data = {
        "patch_type": "readme_generation",
        "content": readme_content,
        "description": f"Generated README content with length: {len(readme_content)} characters",
        "timestamp": time.time(),
        "agent_name": "readme_writer",
        "project_info": {
            "git_info": git_info,
            "project_structure": project_structure,
        },
    }
    return bool(
        qdrant_integration.store_patch(patch_data, storage_id, content_for_embedding)
    )


class ReadmeWriterTool(BaseAgent):
    """
    Intelligent README writer that generates excellent documentation based on best practices,
    technical writing standards, source code analysis, and local information.
    """

    def get_agent_name(self) -> str:
        return "readme_writer"

    def _assemble_project_structure(inputs: BaseInputs) -> str:
        """Generate project structure information."""
        exclude_entries = inputs.policy.exclude_dirs + (".gitignore",)
        project_structure = get_project_structure(
            inputs.project_root,
            max_depth=3,
            exclude_entries=exclude_entries,
        )
        return f"# Project Structure\n{project_structure}"

    def _assemble_config_info(self, inputs: BaseInputs) -> str:
        """Extract and summarize configuration information."""
        config_path = Path(inputs.project_root) / "conf" / "mcp.toml"
        if config_path.exists():
            config_content = config_path.read_text(encoding="utf-8")
            return f"\n===== CONFIG: conf/mcp.toml =====\n{config_content}\n"
        return "(no configuration file found)"

    def _assemble_sources(self, inputs: BaseInputs) -> Tuple[str, List[Path]]:
        """
        Assemble source files from project directories for README generation.
        """
        result_parts: List[str] = []
        total_bytes = 0
        project_root_path = Path(inputs.project_root)

        all_candidate_files: List[Path] = []

        # Use project directories specified in global config
        for directory in inputs.policy.project_directories:
            dir_path = project_root_path / directory.rstrip("/")

            if dir_path.exists() and dir_path.is_dir():
                # Use the new generic function to find files within this specific directory
                candidate_files = _find_source_files(
                    dir_path,
                    inputs.policy.include_extensions,
                    inputs.policy.exclude_dirs,
                )
                all_candidate_files.extend(candidate_files)

                for file_path in candidate_files:
                    try:
                        stat = file_path.stat()
                        if stat.st_size <= 0:
                            continue

                        head = read_file_head(file_path, inputs.policy.max_file_bytes)
                        header = f"\n===== FILE: {file_path.relative_to(project_root_path)} | SIZE: {stat.st_size} bytes =====\n"
                        piece = header + head + "\n"
                        new_total = total_bytes + len(
                            piece.encode("utf-8", errors="ignore")
                        )

                        if new_total > inputs.policy.max_total_bytes:
                            # Stop processing files if total limit is reached
                            return "".join(result_parts), all_candidate_files

                        result_parts.append(piece)
                        total_bytes = new_total
                    except OSError as read_error:
                        logger.warning(
                            "Could not read file %s: %s", file_path, read_error
                        )
                        continue

        return "".join(result_parts), all_candidate_files

    def _format_git_info(self, git_info: dict[str, str | bool | None]) -> str:
        """Format git information for inclusion in the context."""
        git_lines = ["# Git Information"]
        remote_url = git_info.get("remote_url")
        if remote_url and isinstance(remote_url, str):
            git_lines.append(f"Repository URL: {remote_url}")
        branch = git_info.get("branch")
        if branch and isinstance(branch, str):
            git_lines.append(f"Current Branch: {branch}")
        is_dirty = git_info.get("is_dirty")
        if is_dirty is not None:
            status = "dirty" if is_dirty else "clean"
            git_lines.append(f"Repository Status: {status}")

        return "\n".join(git_lines)

    def _create_messages(
        self, inputs: BaseInputs, docs: str, sources: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        # Get skills from agent config if available
        agent_config = self._configurator.get_agent_config("readme_writer")
        skills = agent_config.get("skills", [])

        # Use the configurator's method to combine prompt with skills
        system = self._configurator.combine_prompt_with_skills(
            inputs.prompt.strip(), tuple(skills)
        )

        # Unpack context info from context dictionary
        git_info = context.get("git_info", {}) if context else {}
        project_structure = context.get("project_structure", "") if context else ""

        # Assemble additional context info
        config_info = self._assemble_config_info(inputs)
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
            + (
                f"Use the correct project URL: {git_info.get('remote_url')}\n"
                if git_info.get("remote_url")
                and isinstance(git_info.get("remote_url"), str)
                else ""
            )
            + "Focus on simplicity, clarity, and utility. Avoid generic examples like 'apt-get' if not applicable.\n"
            "Provide concrete, actionable examples based on the actual project structure and code.\n"
            "Do not include unnecessary information like virtual environment setup if not specifically relevant to the project.\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _extract_readme_content_for_embedding(self, data: Dict[str, Any]) -> str:
        """
        Extract content for embedding from readme data.

        Args:
            data: The readme data

        Returns:
            Content string for embedding
        """
        content = data.get("readme_content", "")
        return str(content) if content is not None else ""

    def _store_readme_in_qdrant(
        self,
        readme_content: str,
        inputs: BaseInputs,
        git_info: dict[str, str | bool | None],
        project_structure: str,
    ) -> None:
        """
        Store readme generation in Qdrant if enabled.

        Args:
            readme_content: The readme content to store
            inputs: Base inputs for the agent
            git_info: The computed git information.
            project_structure: The computed project structure string.
        """
        readme_data = {
            "readme_content": readme_content,
            "inputs": inputs,
        }

        self._store_in_qdrant_if_enabled(
            agent_name=self.get_agent_name(),
            data=readme_data,
            store_func=store_readme_generation,
            content_extractor_func=self._extract_readme_content_for_embedding,
            store_func_kwargs={
                "git_info": git_info,
                "project_structure": project_structure,
            },
        )

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self._load_inputs()
        docs = self._assemble_docs(inputs)
        sources, _ = self._assemble_sources(inputs)

        # Compute context information once
        git_info = get_git_info(inputs.project_root)
        project_structure = self._assemble_project_structure(inputs)

        # Pass context info via context dictionary
        context = {"git_info": git_info, "project_structure": project_structure}

        messages = self._create_messages(inputs, docs, sources, context)
        result = self._make_api_call(inputs, messages)

        # If successful, update the response to include readme_content instead of raw_text
        if result["status"] == "success":
            result["data"]["readme_content"] = result["data"]["raw_text"]
            del result["data"]["raw_text"]
            result["message"] = "README generation complete."

            # Store in Qdrant if enabled
            self._store_readme_in_qdrant(
                result["data"]["readme_content"], inputs, git_info, project_structure
            )

        return result

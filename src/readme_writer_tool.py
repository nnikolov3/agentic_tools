# File: src/readme_writer_tool.py
"""
Intelligent README writer tool that generates excellent documentation based on best practices,
technical writing standards, source code analysis, and local information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.base_agent import BaseAgent, BaseInputs
from src.configurator import Configurator
from src.shell_tools import (
    get_git_info,
    get_project_structure,
)

logger = logging.getLogger(__name__)


class ReadmeWriterTool(BaseAgent):
    """
    Intelligent README writer that generates excellent documentation based on best practices,
    technical writing standards, source code analysis, and local information.
    """

    def get_agent_name(self) -> str:
        return "readme_writer"

    def _assemble_project_structure(self, inputs: BaseInputs) -> str:
        """Generate project structure information."""
        project_structure = get_project_structure(inputs.project_root, max_depth=3)
        return f"# Project Structure\n{project_structure}"

    def _assemble_config_info(self, inputs: BaseInputs) -> str:
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

    def _create_messages(
        self, 
        inputs: BaseInputs, 
        docs: str, 
        sources: str,
        payload: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        # Get skills from agent config if available
        agent_config = self._configurator.get_agent_config("readme_writer")
        skills = agent_config.get("skills", [])
        
        # Use the configurator's method to combine prompt with skills
        system = self._configurator.combine_prompt_with_skills(inputs.prompt.strip(), tuple(skills))
        
        # Assemble additional context info
        project_structure = self._assemble_project_structure(inputs)
        config_info = self._assemble_config_info(inputs)
        git_info = get_git_info(inputs.project_root)
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

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self._load_inputs()
        docs = self._assemble_docs(inputs)
        sources = self._assemble_sources(inputs)

        messages = self._create_messages(inputs, docs, sources, payload)
        result = self._make_api_call(inputs, messages)
        
        # If successful, update the response to include readme_content instead of raw_text
        if result["status"] == "success":
            result["data"]["readme_content"] = result["data"]["raw_text"]
            del result["data"]["raw_text"]
            result["message"] = "README generation complete."
        
        return result
# File: src/approver.py
"""
Approver agent: assembles context and requests a final decision as strict JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.base_agent import BaseAgent, BaseInputs
from src.configurator import Configurator

logger = logging.getLogger(__name__)


class Approver(BaseAgent):
    """
    Final gatekeeper that reads design docs and recent code changes, then returns a JSON decision.
    """

    def get_agent_name(self) -> str:
        return "approver"

    def _create_messages(
        self, inputs: BaseInputs, docs: str, sources: str, payload: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        # Extract user_chat from payload
        user_chat = payload.get('user_chat', '')
        
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

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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

        user_chat = payload.get('user_chat', '')
        messages = self._create_messages(inputs, docs, sources, payload)
        return self._make_api_call(inputs, messages)
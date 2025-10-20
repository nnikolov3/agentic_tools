import logging
from pathlib import Path

from src.tools.shell_tools import ShellTools
from src.tools.api_tools import ApiTools
from src.tools.qdrant_tools import QdrantCollectionTools

import mdformat
from typing import Dict
import os

logger = logging.getLogger(__name__)


class Tool:
    def __init__(self, agent, config: Dict):
        self.agent = agent
        self.config = config
        self.agent_config = self.config.get(agent, {})
        self.shell_tools = ShellTools(agent, config)
        self.api_tools = ApiTools(agent, config)
        self.qdrant_collection_tools = QdrantCollectionTools(agent, config)
        self.payload: dict = {}
        self.response: dict = {}
        self.project_root_path = config.get("project_root")
        self.docs = config.get("docs")
        self.agent_skills = self.agent_config.get("skills")
        self.agent_prompt = self.agent_config.get("prompt")
        self.current_working_directory = os.getcwd()

    async def run_tool(self):
        """Execute agent method dynamically from parent class"""

        # Dynamically call the method based on the agent name
        method = getattr(self, self.agent)
        return await method()

    def _get_docs_path(self) -> str:
        if self.project_root_path and self.docs:
            return str(Path(self.project_root_path) / self.docs)
        return ""

    async def approver(self):
        self.response = None
        patch = self.shell_tools.create_patch()

        design_docs_content = ""
        design_docs_paths = self.config.get("design_docs", [])
        for doc_path in design_docs_paths:
            full_path = Path(self.project_root_path) / doc_path
            if full_path.exists() and full_path.is_file():
                design_docs_content += self.shell_tools.read_file_content_for_path(
                    full_path
                )
            else:
                logger.warning(f"Design document not found: {full_path}")

        git_info = self.shell_tools.get_git_info()

        self.payload["prompt"] = self.agent_prompt
        self.payload["skills"] = self.agent_skills
        self.payload["git-diff-patch"] = patch
        self.payload["design_documents"] = design_docs_content
        self.payload["git"] = git_info
        self.response = await self.api_tools.run_api(self.payload)
        return self.response

    async def readme_writer(self):
        """Execute readme writer logic"""
        self.response = None
        try:
            git_info = self.shell_tools.get_git_info()

            self.payload["prompt"] = self.agent_prompt
            self.payload["skills"] = self.agent_skills
            self.payload["project_files"] = self.shell_tools.process_directory(
                self.current_working_directory
            )
            self.payload["git"] = git_info
            if not self.payload:
                logging.warning("No payload generated from file concatenation.")

            self.response = await self.api_tools.run_api(self.payload)
            if not self.response:
                raise ValueError("API returned empty response.")

            self.response = self.shell_tools.cleanup_escapes(self.response)

            self.response = mdformat.text(self.response, options={"wrap": "preserve"})

            self.shell_tools.write_file("README.md", self.response)

            await self.qdrant_collection_tools.run_qdrant(self.response)

            return self.response
        except Exception as e:
            logging.error(f"Error in readme_writer: {e}")
            raise

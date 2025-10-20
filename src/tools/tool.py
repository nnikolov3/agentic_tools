import logging
from pathlib import Path

from src.tools.shell_tools import ShellTools
from src.tools.api_tools import ApiTools
from src.tools.qdrant_tools import QdrantCollectionTools

import mdformat
from typing import Dict

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
        self.git_information: dict = {}
        self.project_root_path = config.get("project_root")
        self.patch: str | None = None
        self.docs = config.get("docs")

    def run_tool(self):
        """Execute agent method dynamically from parent class"""

        # Dynamically call the method based on the agent name
        method = getattr(self, self.agent)
        return method()

    def _get_docs_path(self) -> str:
        if self.project_root_path and self.docs:
            return str(Path(self.project_root_path) / self.docs)
        return ""

    def approver(self):

        try:
            self.patch = self.shell_tools.create_patch()
        except Exception as e:
            logger.error(f"Failed to create patch: {e}")
            raise

        self.payload = {
            "docs": self.shell_tools.process_directory(self._get_docs_path()),
            "patch": self.patch,
            "skills": self.agent_config.get("skills", []),
        }

        self.response = self.api_tools.run_api(self.payload)

        return self.response

    def readme_writer(self):
        """Execute readme writer logic"""
        try:
            # Step 1: Concatenate files
            self.payload = self.shell_tools.concatenate_all_files()
            if not self.payload:
                logging.warning("No payload generated from file concatenation.")

            # Step 2: Get Git info and update payload
            self.payload["skills"] = self.agent_config.get("skills", [])
            self.git_information = self.shell_tools.get_git_info()
            self.payload.update(self.git_information)

            # Step 3: Run API
            self.response = self.api_tools.run_api(self.payload)
            if not self.response:
                raise ValueError("API returned empty response.")

            # Step 4: Cleanup escapes
            self.response = self.shell_tools.cleanup_escapes(self.response)

            # Step 5: Format Markdown
            self.response = mdformat.text(self.response, options={"wrap": "preserve"})

            # Step 6: Write README file
            self.shell_tools.write_file("README.md", self.response)

            # Step 7: Store in Qdrant
            self.qdrant_collection_tools.run_qdrant(self.response)

            return self.response
        except Exception as e:
            logging.error(f"Error in readme_writer: {e}")
            raise

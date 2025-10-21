import logging
from pathlib import Path

from src.tools.shell_tools import ShellTools
from src.tools.api_tools import ApiTools
from src.tools.qdrant_tools import QdrantCollectionTools

import mdformat
from typing import Dict, Optional, Any
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
        self.response: Any = None
        self.project_root_path = config.get("project_root")
        self.docs = config.get("docs")
        self.agent_skills = self.agent_config.get("skills")
        self.agent_prompt = self.agent_config.get("prompt")
        self.current_working_directory = os.getcwd()
        self.source = config.get("source")

    async def run_tool(self, chat: Optional[Any]):
        """Execute agent method dynamically from parent class"""

        # Dynamically call the method based on the agent name
        method = getattr(self, self.agent)
        return await method(chat=chat)

    def _get_docs_path(self) -> str:
        if self.project_root_path and self.docs:
            return str(Path(self.project_root_path) / self.docs)
        return ""

    def _get_docs_content(self) -> str:
        design_docs_content = ""
        design_docs_paths = self.config.get("design_docs", [])
        for doc_path in design_docs_paths:
            full_path = Path(f"{self.current_working_directory}/{doc_path}")
            if full_path.exists() and full_path.is_file():
                design_docs_content += self.shell_tools.read_file_content_for_path(
                    full_path
                )
            else:
                logger.warning(f"Design document not found: {full_path}")

        return design_docs_content

    async def approver(self, chat: Optional[Any] = None):

        patch = self.shell_tools.create_patch()

        git_info = self.shell_tools.get_git_info()
        design_docs_content = self._get_docs_content()
        self.payload["prompt"] = self.agent_prompt
        self.payload["skills"] = self.agent_skills
        self.payload["git-diff-patch"] = patch
        self.payload["design_documents"] = design_docs_content
        self.payload["git"] = git_info
        self.payload["chat"] = chat
        self.response = await self.api_tools.run_api(self.payload)
        return self.response

    async def readme_writer(self, chat: Optional[Any] = None):
        """Execute readme writer logic"""

        try:
            git_info = self.shell_tools.get_git_info()

            self.payload["skills"] = self.agent_skills
            self.payload["project_files"] = self.shell_tools.process_directory(
                self.current_working_directory
            )
            self.payload["git"] = git_info
            self.payload["chat"] = chat
            if not self.payload:
                logging.warning("No payload generated from file concatenation.")

            self.response = await self.api_tools.run_api(self.payload)
            if not self.response:
                raise ValueError("API returned empty response.")

            self.response = self.shell_tools.cleanup_escapes(self.response)
            self.response = mdformat.text(
                str(self.response), options={"wrap": "preserve"}
            )
            self.shell_tools.write_file("README.md", str(self.response))
            await self.qdrant_collection_tools.run_qdrant(self.response)

            return self.response
        except Exception as e:
            logging.error(f"Error in readme_writer: {e}")
            raise

    async def developer(self, chat: Optional[Any]):
        """Writes high quality source code"""

        source_code_contents = []

        if self.source and isinstance(self.source, list):
            for path_segment in self.source:
                # Combine the current working directory with the relative path from config
                absolute_path = Path(f"{self.current_working_directory}/{path_segment}")
                content = self.shell_tools.process_directory(str(absolute_path))
                if content:
                    source_code_contents.append(content)
        else:
            logger.warning(
                "'source' is not defined or not a list in the configuration. "
                "source_code will be empty."
            )

        self.payload["prompt"] = self.agent_prompt
        self.payload["skills"] = self.agent_skills
        self.payload["design_and_coding"] = self._get_docs_content()
        self.payload["source_code"] = "".join(source_code_contents)
        self.payload["chat"] = chat
        self.response = await self.api_tools.run_api(self.payload)
        if not self.response:
            raise ValueError("API returned empty response.")

        return self.response

    # TODO: abstract the payload generation
    async def _create_payload(self):

        pass

    # TODO: Path creation

from src.tools.shell_tools import ShellTools
from src.tools.api_tools import ApiTools
from src.tools.qdrant_tools import QdrantTools
import mdformat


class Tool:
    def __init__(self, agent, config: dict = {}):
        self.agent = agent
        self.config = config
        self.shell_tools = ShellTools(agent, config)
        self.api_tools = ApiTools(agent, config)
        self.qdrant_tools = QdrantTools(agent, config)
        self.payload: dict = {}
        self.response: dict = {}
        self.git_information: dict = {}

    def run_tool(self):
        """Execute agent method dynamically from parent class"""

        # Dynamically call the method based on the agent name
        method = getattr(self, self.agent)
        return method()

    def readme_writer(self):
        """Execute readme writer logic"""
        self.payload = self.shell_tools.concatenate_all_files()
        self.git_information = self.shell_tools.get_git_info()
        self.payload.update(self.git_information)
        self.response = self.api_tools.run_api(self.payload)

        self.response = self.shell_tools.cleanup_escapes(self.response)
        self.response = mdformat.text(self.response, options={"wrap": "preserve"})
        self.shell_tools.write_file("README.md", self.response)
        self.qdrant_tools.run_qdrant(self.response)

        return self.response

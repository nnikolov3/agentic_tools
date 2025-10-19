from src.tools.shell_tools import ShellTools
from src.tools.api_tools import ApiTools
import mdformat


class Tool:
    def __init__(self, agent, config: dict = {}):
        self.agent = agent
        self.config = config
        self.shell_tools = ShellTools(agent, config)
        self.api_tools = ApiTools(agent, config)
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
        self.payload = self.ShellTools.concatenate_all_files()
        self.git_information = self.ShellTools.get_git_info()
        self.payload.update(self.git_information)
        self.response = self.ApiTools.run_api(self.payload)

        self.response = self.ShellTools.cleanup_escapes(self.response)
        self.response = mdformat.text(self.response, options={"wrap": "preserve"})
        self.ShellTools.write_file("README.md", self.response)

        return self.response

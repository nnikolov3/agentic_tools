"""
Base class for all LLM apis, defining the required interface.
"""

from __future__ import annotations

from cerebras.cloud.sdk import Cerebras  # type: ignore
import os
from groq import Groq
from sambanova import SambaNova  # type: ignore
import google.generativeai as genai


sambanova_client = SambaNova(api_key=os.environ.get("SAMBANOVA_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


class ApiTools:
    """
    Abstract base class for all LLM apis.

    Concrete apis must implement the call_api method to interact with their
    respective SDKs and return a ProviderResponse containing the raw response.
    """

    def __init__(self, agent, config: dict):
        self.config = config
        self.agent_config = self.config.get(agent, {})
        self.agent_prompt = self.agent_config.get("prompt")
        self.agent_model_name = self.agent_config.get("model_name")
        self.agent_temperature = self.agent_config.get("temperature")
        self.agent_description = self.agent_config.get("description")
        self.agent_model_provider = self.agent_config.get("model_provider")
        self.agent_alternative_model = self.agent_config.get("alternative_model")
        self.agent_alternative_provider = self.agent_config.get("alternative_provider")
        self.project_root = config.get("project_root")
        self.agent_skills = self.agent_config.get("skills")
        self.design_docs = config.get("design_docs")
        self.source = config.get("source")
        self.project_directories = config.get("project_directories")
        self.include_extensions = config.get("include_extensions")
        self.max_file_bytes = config.get("max_file_bytes")
        self.exclude_directories = config.get("exclude_directories", [])
        self.recent_minutes = config.get("recent_minutes")
        self.payload: dict = {}

    def run_api(self, payload):
        self.payload = payload
        # Get and call the method directly on self
        provider = self.agent_model_provider[0]
        print(f"Running {provider} api")
        method = getattr(self, provider)
        return method()  # call api method (.i.e. sambanova, groq, cerebras)

    def google(self):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(self.agent_model_name)
        response = model.generate_content(
            f"PROMPT: {self.agent_prompt}, {self.payload}",
        )
        return response.text

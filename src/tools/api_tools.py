"""
Base class for all LLM apis, defining the required interface.
The ApiTools class manages interactions with various AI providers, ensuring explicit
API key validation and configuration handling for robust operation across agents.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from typing import Any, Dict
from google.genai import Client, types


class ApiTools:
    """
    Abstract base class for all LLM apis.

    Concrete apis must implement the call_api method to interact with their
    respective SDKs and return a ProviderResponse containing the raw response.
    This class handles payload construction and provider routing explicitly.
    """

    def __init__(self, agent: str, config: Dict[str, Any]) -> None:
        """
        Initializes ApiTools with agent-specific configuration.

        Stores config subsets for prompt, model, temperature, and provider details
        to enable focused API calls without repeated lookups.

        Args:
            agent: The agent name for sub-config selection.
            config: The full configuration dictionary from TOML.
        """
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = self.config.get(agent, {})
        self.agent_prompt: str | None = self.agent_config.get("prompt")
        self.agent_api_key: str | None = self.agent_config.get("api_key")
        self.agent_model_name: str = self.agent_config.get("model_name", "")
        self.agent_temperature: float = self.agent_config.get("temperature", 0.7)
        self.agent_description: str | None = self.agent_config.get("description")
        self.agent_model_provider: str = self.agent_config.get("model_provider", "")
        self.agent_alternative_model: str | None = self.agent_config.get(
            "alternative_model"
        )
        self.agent_alternative_provider: str | None = self.agent_config.get(
            "alternative_provider"
        )
        self.payload: Dict[str, Any] = {}
        self.google_client: Client | None = None

    async def run_api(self, payload: Dict[str, Any]) -> Any:
        """
        Routes the payload to the appropriate provider method.

        Args:
            payload: The input payload containing chat, context, and other data.

        Returns:
            The raw response text from the provider.
        """
        self.payload = payload

        # Explicit provider routing for single responsibility.
        provider: str = self.agent_model_provider
        if not hasattr(self, provider):
            raise ValueError(f"Unsupported provider '{provider}' for agent.")
        method = getattr(self, provider)
        return await method()

    async def google(self) -> str:
        """
        Calls the Google Generative AI API (Gemini) with the payload.

        Validates API key explicitly before initialization to fail fast on misconfiguration.
        Constructs content from chat and JSON payload for the model.

        Returns:
            The generated content text.

        Raises:
            ValueError: If the API key environment variable is missing or empty.
        """
        # Explicit API Key Validation: Check before client creation.
        api_key: str | None = (
            os.getenv(self.agent_api_key) if self.agent_api_key else None
        )
        if not api_key or api_key.strip() == "":
            raise ValueError(
                f"Missing or empty environment variable '{self.agent_api_key}' "
                f"for agent '{self.agent_config.get('name', 'unknown')}'. "
                f"Please set it in your environment."
            )

        self.google_client = Client(api_key=api_key)

        async with self.google_client.aio as a_client:
            system_instruction: str | None = self.agent_config.get("prompt")
            config = types.GenerateContentConfig(
                temperature=self.agent_temperature,
                http_options=types.HttpOptions(api_version="v1alpha"),
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            )

            chat_message: str = self.payload.get("chat", "")
            json_payload: Dict[str, Any] = {
                k: v for k, v in self.payload.items() if k != "chat"
            }
            contents: str = f"{chat_message}\n{json.dumps(json_payload)}"

            response = await a_client.models.generate_content(
                model=self.agent_model_name,
                contents=contents,
                config=config,
            )

            # Explicit type handling: Ensure text is str and non-None.
            response_text: str = response.text if response.text else ""
            return response_text


_GEMINI_SEMAPHORE = asyncio.Semaphore(5)


async def google_documents_api(model, api_key, prompt, file):
    async with _GEMINI_SEMAPHORE:
        api_key_value: str | None = os.getenv(api_key) if api_key else None
        if not api_key_value:
            raise ValueError(
                f"API key environment variable '{api_key}' is not set or empty"
            )

        google_client = Client(api_key=api_key_value)
        file_path = pathlib.Path(file).resolve()

        file_upload = await asyncio.to_thread(
            lambda: google_client.files.upload(file=file_path)
        )

        response = await asyncio.to_thread(
            lambda: google_client.models.generate_content(
                model=model, contents=[file_upload, prompt]
            )
        )

        return response.text

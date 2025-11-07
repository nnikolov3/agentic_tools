# Standard Library
import asyncio
import json
import os
from typing import Any, Dict

# Third-Party Libraries
from google.genai import Client as GoogleClient
from google.genai import types as google_types
from mistralai import Mistral


class ApiTools:
    """Abstract base class for all LLM APIs."""

    def __init__(self, agent: str, config: Dict[str, Any]) -> None:
        """Initializes ApiTools with agent-specific configuration."""
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = self.config.get(agent, {})
        self.agent_prompt: str | None = self.agent_config.get("prompt")
        self.agent_api_key: str | None = self.agent_config.get("api_key")
        self.agent_model_name: str = self.agent_config.get("model_name", "")
        self.agent_temperature: float = self.agent_config.get("temperature", 0.7)
        self.agent_model_provider: str = self.agent_config.get("model_provider", "")
        self.payload: Dict[str, Any] = {}
        self.google_client: GoogleClient | None = None
        self.mistral_client: Mistral | None = None

    async def run_api(self, payload: Dict[str, Any]) -> Any:
        """Routes the payload to the appropriate provider-specific method."""
        self.payload = payload
        provider: str = self.agent_model_provider
        if not hasattr(self, provider):
            raise ValueError(f"Unsupported provider '{provider}' for agent.")
        method = getattr(self, provider)
        return await method()

    async def google(self) -> str:
        """Calls the Google Generative AI API (Gemini) with the constructed payload."""
        api_key: str | None = (
            os.getenv(self.agent_api_key) if self.agent_api_key else None
        )
        if not api_key or api_key.strip() == "":
            raise ValueError(
                f"Missing or empty environment variable '{self.agent_api_key}'."
            )

        self.google_client = GoogleClient(api_key=api_key)

        async with self.google_client.aio as a_client:
            system_instruction: str | None = self.agent_prompt
            config = google_types.GenerateContentConfig(
                temperature=self.agent_temperature,
                system_instruction=system_instruction,
                thinking_config=google_types.ThinkingConfig(thinking_budget=-1),
            )

            chat_message: str = self.payload.get("chat", "")
            json_payload: Dict[str, Any] = {
                k: v for k, v in self.payload.items() if k != "chat"
            }
            text_content = f"{chat_message}\n{json.dumps(json_payload)}"
            contents: list[google_types.Part] = [
                google_types.Part.from_text(text=text_content)
            ]

            response = await a_client.models.generate_content(
                model=self.agent_model_name,
                contents=contents,
                config=config,
            )

            response_text: str = response.text if response.text else ""
            return response_text

    async def mistral(self) -> str:
        """Calls the Mistral AI API with the constructed payload."""
        api_key: str | None = (
            os.getenv(self.agent_api_key) if self.agent_api_key else None
        )
        if not api_key or api_key.strip() == "":
            raise ValueError(
                f"Missing or empty environment variable '{self.agent_api_key}'."
            )

        self.mistral_client = Mistral(api_key=api_key)

        chat_message: str = self.payload.get("chat", "")
        json_payload: Dict[str, Any] = {
            k: v for k, v in self.payload.items() if k != "chat"
        }
        text_content = f"{chat_message}\n{json.dumps(json_payload)}"

        response = self.mistral_client.chat.complete(
            model=self.agent_model_name,
            messages=[{"role": "user", "content": text_content}],
            temperature=self.agent_temperature,
        )

        response_text: str = (
            response.choices[0].message.content if response.choices else ""
        )
        return response_text


# Rate limiting for concurrent API calls
_GEMINI_SEMAPHORE = asyncio.Semaphore(2)
_MISTRAL_SEMAPHORE = asyncio.Semaphore(2)

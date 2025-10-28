"""LLM API Abstraction and Integration Layer.

This module defines an abstract base class `ApiTools` for interacting with
various Large Language Model (LLM) providers. It centralizes common concerns
such as API key management, configuration loading, and provider-specific routing.
This abstraction promotes maintainable and consistent integration of LLM
capabilities across different agents and services.

It also includes utility functions for specialized LLM tasks, like
`google_documents_api` for document processing.
"""

# Standard Library
import asyncio
import json
import os
import pathlib
from typing import Any, Dict

# Third-Party Library
from google.genai import Client, types

# Local Application/Module
# (No local imports in this file)


class ApiTools:
    """Abstract base class for all LLM APIs.

    Concrete API implementations must implement the `run_api` method to interact
    with their respective SDKs and return a response. This class handles payload
    construction and provider routing explicitly, promoting a clear separation of
    concerns.
    """

    def __init__(self, agent: str, config: Dict[str, Any]) -> None:
        """Initializes ApiTools with agent-specific configuration.

        Stores configuration subsets for prompt, model, temperature, and provider
        details to enable focused API calls without repeated lookups. This
        pre-processing enhances efficiency by having commonly used parameters
        readily available.

        Args:
            agent: The agent name used to select the relevant configuration subset.
            config: The full configuration dictionary, typically loaded from a TOML
                    file.
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
        """Routes the payload to the appropriate provider-specific method.

        This method acts as a dispatcher, selecting the correct API interaction
        based on the configured `model_provider` for the agent. It ensures that
        the system adheres to the single responsibility principle by delegating
        provider-specific logic to dedicated methods.

        Args:
            payload: The input payload containing chat history, context, and other data
                     required for the LLM call.

        Returns:
            The raw response text from the LLM provider.

        Raises:
            ValueError: If the configured `model_provider` does not correspond to
                        a method available on this class.
        """
        self.payload = payload

        # Explicit provider routing ensures the correct provider-specific method
        # is called based on configuration, adhering to the single responsibility
        # principle.
        provider: str = self.agent_model_provider
        if not hasattr(self, provider):
            raise ValueError(f"Unsupported provider '{provider}' for agent.")
        method = getattr(self, provider)
        return await method()

    async def google(self) -> str:
        """Calls the Google Generative AI API (Gemini) with the constructed payload.

        This method handles the specifics of interacting with the Google Generative AI
        service. It validates the API key, initializes the client, constructs the
        content for the model, and makes the asynchronous API call.

        Returns:
            The generated content text from the Google Generative AI model.

        Raises:
            ValueError: If the API key environment variable specified in the
                        configuration is missing or empty.
        """
        # Explicit API Key Validation: Check before client creation to fail fast
        # on misconfiguration. This prevents unnecessary initialization if credentials
        # are absent.
        api_key: str | None = (
            os.getenv(self.agent_api_key) if self.agent_api_key else None
        )
        if not api_key or api_key.strip() == "":
            raise ValueError(
                f"Missing or empty environment variable '{self.agent_api_key}' for agent '{self.agent_config.get('name', 'unknown')}'. Please set it in your environment."
            )

        self.google_client = Client(api_key=api_key)

        async with self.google_client.aio as a_client:
            system_instruction: str | None = self.agent_config.get("prompt")
            config = types.GenerateContentConfig(
                temperature=self.agent_temperature,
                system_instruction=system_instruction,
                # Setting thinking_budget to -1 disables the default thinking budget,
                # allowing the model to use its own judgment for reasoning steps.
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            )

            chat_message: str = self.payload.get("chat", "")
            # Construct JSON payload excluding the 'chat' key, which is handled
            # separately.
            json_payload: Dict[str, Any] = {
                k: v for k, v in self.payload.items() if k != "chat"
            }
            text_content = f"{chat_message}\n{json.dumps(json_payload)}"
            contents: list[types.Part] = [types.Part.from_text(text=text_content)]

            response = await a_client.models.generate_content(
                model=self.agent_model_name,
                contents=contents,
                config=config,
            )

            # Explicit type handling: Ensure the response text is a string and not None.
            # This guards against potential API changes or unexpected response formats.
            response_text: str = response.text if response.text else ""
            return response_text


# Rate limiting for concurrent Google Generative AI API calls to prevent
# exceeding quotas. A semaphore is used to limit the number of simultaneous
# requests, ensuring stability and adherence to API rate limits.
_GEMINI_SEMAPHORE = asyncio.Semaphore(5)


async def google_documents_api(model: str, api_key: str, prompt: str, file: str) -> str:
    """Calls the Google Generative AI API for document processing.

    This function is designed to handle file uploads and subsequent content generation
    using Google's Generative AI models, specifically for tasks involving document
    analysis. It ensures that potentially blocking I/O operations (file upload and
    API calls) are executed in a separate thread pool using `asyncio.to_thread` to
    maintain the responsiveness of the asyncio event loop.

    Args:
        model: The specific Google Generative AI model to use for content generation
               (e.g., 'gemini-pro-vision').
        api_key: The name of the environment variable containing the Google API key.
        prompt: The prompt to send to the model, instructing it on how to process
                the document and what task to perform.
        file: The file path to the document that needs to be uploaded and processed.

    Returns:
        The generated content text from the model as a result of processing the
        document.

    Raises:
        ValueError: If the API key environment variable specified by `api_key` is
                    not set or is empty.
    """
    async with _GEMINI_SEMAPHORE:
        # Retrieve the API key from the environment.
        api_key_value: str | None = os.getenv(api_key) if api_key else None
        if not api_key_value:
            raise ValueError(
                f"API key environment variable '{api_key}' is not set or empty"
            )

        google_client = Client(api_key=api_key_value)

        # Resolve the file path to an absolute path for clarity and robustness.
        file_path = pathlib.Path(file).resolve()

        # Run file upload in a separate thread to avoid blocking the asyncio
        # event loop. This is necessary because google_client.files.upload is a
        # potentially blocking I/O operation.
        file_upload = await asyncio.to_thread(
            lambda: google_client.files.upload(file=file_path)
        )

        # Run content generation in a separate thread to avoid blocking the
        # asyncio event loop. This is necessary because
        # google_client.models.generate_content is a potentially blocking I/O
        # operation.
        response = await asyncio.to_thread(
            lambda: google_client.models.generate_content(
                model=model, contents=[file_upload, types.Part.from_text(text=prompt)]
            )
        )

        return response.text if response.text else ""

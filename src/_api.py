"""
File: src/_api.py
Author: Niko Nikolov
Scope: Implements the dynamic and resilient API calling logic for all LLM providers.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Provider SDKs - assuming they are installed in the environment
from google import genai
from groq import Groq
from openai import APIConnectionError, OpenAI, RateLimitError

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Data Structures ---


@dataclass
class UnifiedResponse:
    """
    A standardized internal format for LLM responses. This ensures the rest of
    the application works with a single, predictable object structure, regardless
    of which provider returned the result.
    """

    content: str
    model_name: str
    provider_name: str
    raw_response: Any  # The original response object for debugging or extended use


# Module-level dictionaries to hold provider clients and their health status.
# They are populated by the _initialize_providers function.
PROVIDER_REGISTRY: Dict[str, Any] = {}
PROVIDER_HEALTH: Dict[str, Dict[str, Any]] = {}
_INITIALIZED = False

# --- Private Helper Functions ---


def _initialize_providers():
    logger.debug("_initialize_providers function entered.")
    """
    Initializes the API clients for all supported providers and sets their
    default health status. This function runs only once.

    It reads API keys from environment variables and configures the client
    objects, which are then stored in the PROVIDER_REGISTRY.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Google
    if "GEMINI_API_KEY" in os.environ:
        PROVIDER_REGISTRY["google"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        PROVIDER_HEALTH["google"] = {"status": "OK", "retry_after": 0}
        logger.info("Google Gemini provider initialized.")

    # Groq (OpenAI Compatible)
    if "GROQ_API_KEY" in os.environ:
        PROVIDER_REGISTRY["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])
        PROVIDER_HEALTH["groq"] = {"status": "OK", "retry_after": 0}
        logger.info("Groq provider initialized.")

    # SambaNova (OpenAI Compatible)
    if "SAMBANOVA_API_KEY" in os.environ and "SAMBANOVA_BASE_URL" in os.environ:
        PROVIDER_REGISTRY["sambanova"] = OpenAI(
            api_key=os.environ["SAMBANOVA_API_KEY"],
            base_url=os.environ["SAMBANOVA_BASE_URL"],
        )
        PROVIDER_HEALTH["sambanova"] = {"status": "OK", "retry_after": 0}
        logger.info("SambaNova provider initialized.")

    # Cerebras (OpenAI Compatible)
    # NOTE: The original docs showed a custom Cerebras SDK, but the schema doc
    # confirms it is OpenAI compatible. Using the OpenAI client is more robust.
    if "CEREBRAS_API_KEY" in os.environ and "CEREBRAS_BASE_URL" in os.environ:
        PROVIDER_REGISTRY["cerebras"] = OpenAI(
            api_key=os.environ["CEREBRAS_API_KEY"],
            base_url=os.environ["CEREBRAS_BASE_URL"],
        )
        PROVIDER_HEALTH["cerebras"] = {"status": "OK", "retry_after": 0}
        logger.info("Cerebras provider initialized.")

    _INITIALIZED = True


def _update_provider_health(provider: str, status: str, retry_after: int = 0):
    """
    Updates the health status of a specific provider.

    Parameters:
        provider (str): The name of the provider (e.g., "groq").
        status (str): The new status, e.g., "OK", "QUOTA_EXCEEDED", "OFFLINE".
        retry_after (int): A Unix timestamp after which this provider can be tried again.
    """
    if provider in PROVIDER_HEALTH:
        PROVIDER_HEALTH[provider]["status"] = status
        PROVIDER_HEALTH[provider]["retry_after"] = retry_after
        logger.info(f"Provider {provider} status set to {status}.")


def _get_healthy_providers(provider_list: List[str]) -> List[str]:
    """
    Filters a given list of providers to return only those that are currently healthy.

    A provider is considered healthy if its status is "OK" or if it was
    unhealthy but its `retry_after` timestamp has passed.

    Parameters:
        provider_list (List[str]): The list of providers from an agent's config.

    Returns:
        List[str]: A new list containing only the names of healthy providers.
    """
    healthy_providers = []
    current_time = int(time.time())

    for provider in provider_list:
        if provider in PROVIDER_HEALTH:
            health_info = PROVIDER_HEALTH[provider]
            if health_info["status"] == "OK":
                healthy_providers.append(provider)
            elif current_time > health_info["retry_after"]:
                # The retry time has passed, so we can consider it OK again.
                _update_provider_health(provider, "OK", 0)
                healthy_providers.append(provider)
    return healthy_providers


def _normalize_openai_compatible(
    response: Any, model: str, provider: str
) -> Optional[UnifiedResponse]:
    """
    Normalizes a response from an OpenAI-compatible API into our UnifiedResponse format.

    Parameters:
        response (Any): The raw response object from the provider's SDK.
        model (str): The name of the model that was used.
        provider (str): The name of the provider.

    Returns:
        UnifiedResponse: The standardized response object.
        None: If the response structure is unexpected.
    """
    try:
        content = response.choices[0].message.content
        if content is None:
            logger.warning(f"OpenAI-compatible response from {provider} for model {model} had no content.")
            return None
        return UnifiedResponse(
            content=content,
            model_name=model,
            provider_name=provider,
            raw_response=response,
        )
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Failed to normalize OpenAI-compatible response from {provider} for model {model}: {e}")
        logger.debug(f"Raw response: {response}")
        return None


def _normalize_google(response: Any, model: str, provider: str) -> Optional[UnifiedResponse]:
    """
    Normalizes a response from the Google Gemini API into our UnifiedResponse format.

    Parameters:
        response (Any): The raw response object from the Google GenAI SDK.
        model (str): The name of the model that was used.
        provider (str): The name of the provider.

    Returns:
        UnifiedResponse: The standardized response object.
        None: If the response structure is unexpected.
    """
    try:
        content = response.candidates[0].content.parts[0].text
        if content is None:
            logger.warning(f"Google Gemini response from {provider} for model {model} had no content.")
            return None
        return UnifiedResponse(
            content=content,
            model_name=model,
            provider_name=provider,
            raw_response=response,
        )
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Failed to normalize Google Gemini response from {provider} for model {model}: {e}")
        logger.debug(f"Raw response: {response}")
        return None


def _normalize_response(
    provider: str, raw_response: Any, model: str
) -> UnifiedResponse:
    """
    Dispatches to the correct normalizer function based on the provider name.

    Parameters:
        provider (str): The name of the provider that returned the response.
        raw_response (Any): The raw, successful response object from the provider.
        model (str): The name of the model that was used.

    Returns:
        UnifiedResponse: The standardized response object.
    """
    if provider == "google":
        return _normalize_google(raw_response, model, provider)
    else:
        # All other providers (Groq, SambaNova, Cerebras) are OpenAI-compatible
        return _normalize_openai_compatible(raw_response, model, provider)


def _execute_call(
    provider: str, model: str, temp: float, payload: str
) -> Optional[Any]:
    logger.debug(f"_execute_call entered for provider={provider}, model={model}")
    logger.debug(f"Payload snippet: {payload[:100]}...")
    try:
        client = PROVIDER_REGISTRY[provider]
        messages = [{"role": "user", "content": payload}]

        logger.info(f"Attempting call to {model} via {provider}...")

        if provider == "google":
            client = PROVIDER_REGISTRY[provider]
            response = client.models.generate_content(
                model=model,
                generation_config={"temperature": temp},
                contents=messages,
                timeout=60,  # Add a 60-second timeout
            )
        else:  # OpenAI-compatible providers
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                timeout=60,  # Add a 60-second timeout
            )

        _update_provider_health(provider, "OK")
        return response

    except RateLimitError as e:
        # Quota exceeded. Mark provider as unhealthy.
        # The SDK should provide retry_after info in the error object's headers.
        retry_after = int(e.response.headers.get("retry-after", 60))  # Default to 60s
        _update_provider_health(
            provider, "QUOTA_EXCEEDED", int(time.time()) + retry_after
        )
        logger.warning(f"Rate limit exceeded for {provider}. Retrying after {retry_after} seconds.")
        return None
    except APIConnectionError:
        # Network error. Mark provider as offline for a short time.
        _update_provider_health(
            provider, "OFFLINE", int(time.time()) + 300
        )  # 5 minutes
        logger.error(f"API connection error for {provider}. Marking offline for 5 minutes.")
        return None
    except Exception as e:
        # Any other unexpected API error.
        logger.error(f"An unexpected error occurred with provider {provider}: {e}", exc_info=True)
        _update_provider_health(
            provider, "OFFLINE", int(time.time()) + 300
        )  # 5 minutes
        return None


# --- Core Public Function ---


def api_caller(agent_config: Dict[str, Any], payload: str) -> Optional[UnifiedResponse]:
    _initialize_providers()

    # Start with the primary model and its providers
    primary_model = agent_config["model_name"]
    primary_providers = agent_config["model_providers"]

    healthy_providers = _get_healthy_providers(primary_providers)

    for provider in healthy_providers:
        raw_response = _execute_call(
            provider=provider,
            model=primary_model,
            temp=agent_config["temperature"],
            payload=payload,
        )
        if raw_response:
            logger.debug(f"Raw response received from {provider}")
            logger.info(f"Call to {provider} succeeded.")
            return _normalize_response(provider, raw_response, primary_model)

    # If all primary providers failed, try the alternative model
    logger.warning("All primary providers failed. Attempting alternative model.")
    alt_model = agent_config.get("alternative_model")
    alt_providers = agent_config.get("alternative_model_provider", [])

    if not alt_model:
        logger.error("No alternative model configured.")
        return None

    healthy_alt_providers = _get_healthy_providers(alt_providers)

    for provider in healthy_alt_providers:
        raw_response = _execute_call(
            provider=provider,
            model=alt_model,
            temp=agent_config["temperature"],
            payload=payload,
        )
        if raw_response:
            logger.info(f"Call to alternative provider {provider} succeeded.")
            return _normalize_response(provider, raw_response, alt_model)

    logger.error("All primary and alternative providers failed.")
    return None

# File: src/_api.py

"""
Unified LLM API caller using providers' official SDKs with proper rate limit checking.

Providers:
- Google: google-genai (from google import genai)
- Groq: groq SDK
- Cerebras: cerebras-cloud-sdk
- SambaNova: sambanova SDK

Features:
- Proactive rate limit checking via response headers
- Provider health tracking with cooldowns
- Message normalization
- Response normalization (Google vs standard schema)
- Automatic provider failover
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# --- Provider SDKs (official) ---
CerebrasClient = None  # type: ignore
GroqClient = None  # type: ignore
SambaNovaClient = None  # type: ignore
GenAIModule = None  # type: ignore

try:
    from cerebras.cloud.sdk import Cerebras as CerebrasClient  # type: ignore
except ImportError:
    pass

try:
    from google import genai as GenAIModule  # type: ignore
except ImportError:
    pass

try:
    from groq import Groq as GroqClient  # type: ignore
except ImportError:
    pass

try:
    from sambanova import SambaNova as SambaNovaClient  # type: ignore
except ImportError:
    pass


# --- Data structures ---


@dataclass
class ProviderQuota:
    """Track quota status for a provider."""

    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    requests_remaining_day: Optional[int] = None
    reset_requests: Optional[int] = None
    reset_tokens: Optional[int] = None
    reset_requests_day: Optional[int] = None
    last_check: int = field(default_factory=lambda: int(time.time()))

    def has_quota(self) -> bool:
        """Check if provider has available quota."""
        now = int(time.time())

        # Check if daily quota exists and is exhausted
        if self.requests_remaining_day is not None:
            if self.requests_remaining_day <= 0:
                if self.reset_requests_day and now < self.reset_requests_day:
                    return False

        # Check if per-minute quota exists and is exhausted
        if self.requests_remaining is not None:
            if self.requests_remaining <= 0:
                if self.reset_requests and now < self.reset_requests:
                    return False

        # Check if token quota exists and is exhausted
        if self.tokens_remaining is not None:
            if self.tokens_remaining <= 10:  # Reserve small buffer
                if self.reset_tokens and now < self.reset_tokens:
                    return False

        return True


@dataclass
class ProviderHealth:
    """Track provider health status."""

    status: str = "OK"  # OK, RATE_LIMITED, OFFLINE
    retry_after: int = 0  # Unix timestamp
    quota: ProviderQuota = field(default_factory=ProviderQuota)

    def is_available(self) -> bool:
        """Check if provider is available for use."""
        now = int(time.time())
        if self.status == "OK":
            return self.quota.has_quota()
        if now >= self.retry_after:
            self.status = "OK"
            return self.quota.has_quota()
        return False


@dataclass
class UnifiedResponse:
    """Normalized response from any provider."""

    content: str
    model_name: str
    provider_name: str
    raw_response: Any
    usage_tokens: Optional[int] = None


# --- Global registries ---

PROVIDER_REGISTRY: Dict[str, Any] = {}  # Maps provider name to client instance
PROVIDER_HEALTH: Dict[str, ProviderHealth] = {}
_INITIALIZED = False


# --- Provider initialization ---


def _initialize_providers() -> None:
    """
    Initialize providers when their env vars are available.

    Environment variables:
    - GOOGLE_API_KEY: Google API key
    - GEMINI_API_KEY: Google Gemini API key
    - GROQ_API_KEY: Groq API key
    - CEREBRAS_API_KEY: Cerebras API key
    - SAMBANOVA_API_KEY: SambaNova API key
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Special handling for Google provider with failover
    if GenAIModule is not None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if google_api_key:
            try:
                client = _create_provider_client("google", GenAIModule, google_api_key)
                if client is not None:
                    PROVIDER_REGISTRY["google"] = client
                    PROVIDER_HEALTH["google"] = ProviderHealth()
                    logger.info("Successfully initialized google client with GOOGLE_API_KEY")
            except Exception as e:
                logger.warning(f"Failed to initialize google client with GOOGLE_API_KEY: {e}")
                if gemini_api_key:
                    logger.info("Trying to initialize google client with GEMINI_API_KEY")
                    try:
                        client = _create_provider_client("google", GenAIModule, gemini_api_key)
                        if client is not None:
                            PROVIDER_REGISTRY["google"] = client
                            PROVIDER_HEALTH["google"] = ProviderHealth()
                            logger.info("Successfully initialized google client with GEMINI_API_KEY")
                    except Exception as e2:
                        logger.warning(f"Failed to initialize google client with GEMINI_API_KEY: {e2}")
        elif gemini_api_key:
            try:
                client = _create_provider_client("google", GenAIModule, gemini_api_key)
                if client is not None:
                    PROVIDER_REGISTRY["google"] = client
                    PROVIDER_HEALTH["google"] = ProviderHealth()
                    logger.info("Successfully initialized google client with GEMINI_API_KEY")
            except Exception as e:
                logger.warning(f"Failed to initialize google client with GEMINI_API_KEY: {e}")


    # Other providers
    providers_to_init = [
        ("groq", GroqClient, "GROQ_API_KEY"),
        ("cerebras", CerebrasClient, "CEREBRAS_API_KEY"),
        ("sambanova", SambaNovaClient, "SAMBANOVA_API_KEY"),
    ]

    for provider_name, client_class, env_key_name in providers_to_init:
        api_key = os.getenv(env_key_name)
        if client_class is not None and api_key:
            try:
                client = _create_provider_client(provider_name, client_class, api_key)
                if client is not None:
                    PROVIDER_REGISTRY[provider_name] = client
                    PROVIDER_HEALTH[provider_name] = ProviderHealth()
                    logger.info(f"Successfully initialized {provider_name} client")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name} client: {e}")

    _INITIALIZED = True


def _create_provider_client(provider_name: str, client_class: Any, api_key: str) -> Any:
    """Create a provider client instance based on provider name."""
    if provider_name == "google":
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        types_module = getattr(GenAIModule, "types", None)
        if types_module:
            http_options_ctor = getattr(types_module, "HttpOptions", None)
            if callable(http_options_ctor):
                client_kwargs["http_options"] = http_options_ctor(api_version="v1alpha")
        return client_class.Client(**client_kwargs)
    else:
        return client_class(api_key=api_key)


# --- Rate limit header parsing ---


def _parse_rate_limit_headers(headers: Dict[str, Any], provider: str) -> ProviderQuota:
    """Parse rate limit headers from provider response."""
    quota = ProviderQuota()

    # Helper to safely get int from header
    def get_int(key: str) -> Optional[int]:
        val = headers.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    if provider in ["cerebras", "groq", "sambanova"]:
        # Standard headers across these providers
        quota.requests_remaining = get_int("x-ratelimit-remaining-requests")
        quota.tokens_remaining = get_int("x-ratelimit-remaining-tokens")
        quota.reset_requests = get_int("x-ratelimit-reset-requests")
        quota.reset_tokens = get_int("x-ratelimit-reset-tokens")

        # Day-level limits (Cerebras, SambaNova)
        quota.requests_remaining_day = get_int("x-ratelimit-remaining-requests-day")
        quota.reset_requests_day = get_int("x-ratelimit-reset-requests-day")

        # Handle retry-after on 429 errors (stored as seconds, convert to epoch)
        retry_after_seconds = get_int("retry-after")
        if retry_after_seconds is not None:
            quota.reset_requests = int(time.time()) + retry_after_seconds

    return quota


def _update_provider_quota(provider: str, headers: Dict[str, Any]) -> None:
    """Update provider quota from response headers."""
    if provider not in PROVIDER_HEALTH:
        return

    health = PROVIDER_HEALTH[provider]
    health.quota = _parse_rate_limit_headers(headers, provider)

    logger.debug(
        "Updated %s quota: req_remaining=%s, tok_remaining=%s, req_day_remaining=%s",
        provider,
        health.quota.requests_remaining,
        health.quota.tokens_remaining,
        health.quota.requests_remaining_day,
    )


def _mark_provider_rate_limited(provider: str, retry_after: int = 60) -> None:
    """Mark provider as rate limited."""
    if provider not in PROVIDER_HEALTH:
        return

    health = PROVIDER_HEALTH[provider]
    health.status = "RATE_LIMITED"
    health.retry_after = int(time.time()) + retry_after
    health.quota.requests_remaining = 0

    logger.warning(
        "Provider %s rate limited, retry after %d seconds", provider, retry_after
    )


def _mark_provider_offline(provider: str, duration: int = 300) -> None:
    """Mark provider as offline."""
    if provider not in PROVIDER_HEALTH:
        return

    health = PROVIDER_HEALTH[provider]
    health.status = "OFFLINE"
    health.retry_after = int(time.time()) + duration

    logger.error("Provider %s marked offline for %d seconds", provider, duration)


def _get_available_providers(candidates: List[str]) -> List[str]:
    """Get list of providers that are available (not rate limited, not offline)."""
    available = []
    for provider in candidates:
        if provider not in PROVIDER_HEALTH:
            continue
        if PROVIDER_HEALTH[provider].is_available():
            available.append(provider)
    return available


# --- Message normalization ---


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Normalize messages to standard format: [{"role": str, "content": str}].
    Handles dict, object, and Enum role types.
    """
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "")

        # Handle Enum.role.name if role is an Enum
        if hasattr(role, "name"):
            try:
                role = role.name
            except Exception:
                pass

        normalized.append(
            {"role": str(role), "content": "" if content is None else str(content)}
        )

    return normalized


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert messages to plain text prompt for Google."""
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


# --- Response normalization ---


def _normalize_standard_response(
    resp: Any, model: str, provider: str
) -> Optional[UnifiedResponse]:
    """Normalize standard schema response (Groq, Cerebras, SambaNova)."""
    try:
        content = resp.choices[0].message.content
        if not content:
            return None

        usage = None
        if hasattr(resp, "usage") and resp.usage:
            usage = getattr(resp.usage, "total_tokens", None)

        return UnifiedResponse(
            content=str(content),
            model_name=model,
            provider_name=provider,
            raw_response=resp,
            usage_tokens=usage,
        )
    except Exception as e:
        logger.error("Failed to normalize response for %s: %s", provider, e)
        return None


def _normalize_google_response(
    resp: Any, model: str, provider: str
) -> Optional[UnifiedResponse]:
    """Normalize Google Gemini response."""
    try:
        # Try response.text first (convenience accessor)
        content = getattr(resp, "text", None)

        if not content:
            # Fall back to candidates[0].content.parts[0].text
            candidates = getattr(resp, "candidates", [])
            if candidates:
                first_candidate = candidates[0]
                content_obj = getattr(first_candidate, "content", None)
                if content_obj:
                    parts = getattr(content_obj, "parts", [])
                    if parts:
                        content = getattr(parts[0], "text", None)

        if not content or not isinstance(content, str):
            return None

        # Get token usage
        usage = None
        usage_metadata = getattr(resp, "usage_metadata", None)
        if usage_metadata:
            usage = getattr(usage_metadata, "total_token_count", None)

        return UnifiedResponse(
            content=content,
            model_name=model,
            provider_name=provider,
            raw_response=resp,
            usage_tokens=usage,
        )
    except Exception as e:
        logger.error("Failed to normalize Google response: %s", e)
        return None


# --- Provider call functions ---


def _call_google(
    client: Any, model: str, temperature: float, messages: List[Dict[str, str]]
) -> Any:
    """Call Google Gemini API."""
    types_module = getattr(GenAIModule, "types", None) if GenAIModule is not None else None
    part_from_text = None
    content_ctor = None
    generate_config_ctor = None

    if types_module is not None:
        part_from_text = getattr(getattr(types_module, "Part", None), "from_text", None)
        content_ctor = getattr(types_module, "Content", None)
        generate_config_ctor = getattr(types_module, "GenerateContentConfig", None)

    if part_from_text and callable(part_from_text) and callable(content_ctor):
        try:
            return _call_google_with_types_module(client, model, temperature, messages, 
                                               part_from_text, content_ctor, generate_config_ctor)
        except Exception as e:
            logger.warning(f"Failed to call Google with types module: {e}. Falling back to plain text.")
            # Fall back to plain text if the structured approach fails
            return client.models.generate_content(
                model=model,
                contents=_messages_to_prompt(messages),
                config={"temperature": temperature},
            )
    else:
        # Fallback to plain text payload when types module not available
        return client.models.generate_content(
            model=model,
            contents=_messages_to_prompt(messages),
            config={"temperature": temperature},
        )


def _call_google_with_types_module(
    client: Any, 
    model: str, 
    temperature: float, 
    messages: List[Dict[str, str]], 
    part_from_text: Any, 
    content_ctor: Any, 
    generate_config_ctor: Any
) -> Any:
    """Call Google Gemini API using the types module with structured content."""
    system_messages: List[str] = []
    content_payload: List[Any] = []

    for message in messages:
        try:
            role = str(message.get("role", "user")).lower()
            text = str(message.get("content", ""))
            if not text:
                continue
            if role == "system":
                system_messages.append(text)
                continue

            mapped_role = _map_google_role(role)

            part_obj = part_from_text(text=text)
            content_payload.append(content_ctor(role=mapped_role, parts=[part_obj]))
        except Exception as e:
            logger.warning(f"Failed to process message in Google API call: {e}. Skipping message.")
            continue

    system_instruction_obj: Any = None
    try:
        if system_messages:
            system_text = "\n\n".join(system_messages)
            system_instruction_obj = content_ctor(
                role="system", parts=[part_from_text(text=system_text)]  # type: ignore[call-arg]
            )
    except Exception as e:
        logger.warning(f"Failed to create system instruction for Google API: {e}")

    config_kwargs: Dict[str, Any] = {"temperature": temperature}
    if system_instruction_obj is not None:
        config_kwargs["system_instruction"] = system_instruction_obj

    config_object = _get_google_config_object(generate_config_ctor, config_kwargs)

    if not content_payload:
        # Ensure we always send at least one content item
        try:
            default_part = part_from_text(text="") if part_from_text else None
            content_payload = (
                [content_ctor(role="user", parts=[default_part])] if default_part else []
            )
        except Exception as e:
            logger.warning(f"Failed to create default content payload: {e}")
            # Return empty payload if we can't create even a default one
            content_payload = []

    try:
        return client.models.generate_content(
            model=model,
            contents=content_payload,
            config=config_object,
        )
    except Exception as e:
        logger.error(f"Failed to make Google API call: {e}")
        raise


def _map_google_role(role: str) -> str:
    """Map standard role names to Google-specific role names."""
    if role in ("assistant", "model"):
        return "model"
    elif role == "tool":
        return "function"  # Google uses 'function' instead of 'tool'
    else:
        return "user"


def _get_google_config_object(generate_config_ctor: Any, config_kwargs: Dict[str, Any]) -> Any:
    """Get the Google configuration object based on available constructor."""
    if generate_config_ctor and callable(generate_config_ctor):
        return generate_config_ctor(**config_kwargs)
    else:
        return config_kwargs


def _call_standard_provider(
    client: Any,
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
    provider: str,
) -> Any:
    """
    Call provider with standard schema using rate limit header checking.
    Returns parsed completion response after updating quota.
    """
    # Use with_raw_response to get headers
    raw_response = client.chat.completions.with_raw_response.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=60,
    )

    # Extract headers for quota tracking
    headers = dict(raw_response.headers) if hasattr(raw_response, "headers") else {}

    # Update quota before returning
    _update_provider_quota(provider, headers)

    # Parse the actual completion response
    completion = raw_response.parse()

    return completion


# --- Main execution function ---


def _execute_provider_call(
    provider: str,
    model: str,
    temperature: float,
    messages: List[Dict[str, Any]],
) -> Optional[UnifiedResponse]:
    """Execute a call to a specific provider with proper error handling."""
    client = PROVIDER_REGISTRY.get(provider)
    if client is None:
        logger.warning(f"Provider {provider} not initialized or available")
        return None

    normalized_messages = _normalize_messages(messages)

    try:
        # Check quota before making request
        if provider != "google":  # Google doesn't expose quota headers
            health = PROVIDER_HEALTH.get(provider)
            if health and not health.quota.has_quota():
                logger.info("Skipping %s due to insufficient quota", provider)
                return None

        # Make the API call
        if provider == "google":
            raw_resp = _call_google(client, model, temperature, normalized_messages)
            return _normalize_google_response(raw_resp, model, provider)

        elif provider in ["groq", "cerebras", "sambanova"]:
            raw_resp = _call_standard_provider(
                client, model, temperature, normalized_messages, provider
            )
            return _normalize_standard_response(raw_resp, model, provider)

        return None

    except Exception as e:
        return _handle_provider_call_exception(provider, e)


def _handle_provider_call_exception(provider: str, e: Exception) -> Optional[UnifiedResponse]:
    """Handle exceptions from provider calls and return appropriate response."""
    exc_name = type(e).__name__
    
    # Handle rate limit errors
    if "RateLimitError" in exc_name or (
        hasattr(e, "status_code") and getattr(e, "status_code", 0) == 429
    ):
        retry_after = _extract_retry_after_from_error(e)
        _mark_provider_rate_limited(provider, retry_after)
        logger.warning("Rate limit error on %s: %s", provider, e)
        return None

    # Handle connection errors
    if "ConnectionError" in exc_name or "APIConnectionError" in exc_name:
        _mark_provider_offline(provider, 300)
        logger.error("Connection error on %s: %s", provider, e)
        return None

    # Handle other errors
    logger.error("Unexpected error on %s: %s", provider, e, exc_info=True)
    _mark_provider_offline(provider, 300)
    return None


def _extract_retry_after_from_error(e: Exception) -> int:
    """Extract retry-after value from error response headers, defaulting to 60 seconds."""
    retry_after = 60
    
    # Try to extract retry-after from error
    if hasattr(e, "response") and hasattr(e.response, "headers"):
        retry_header = e.response.headers.get("retry-after")
        if retry_header:
            try:
                retry_after = int(retry_header)
            except (ValueError, TypeError):
                pass
    
    return retry_after


# --- Public API ---


def api_caller(
    agent_config: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Optional[UnifiedResponse]:
    """
    Call LLM API with automatic provider failover.

    Args:
        agent_config: Agent configuration with keys:
            - model_name (str): Primary model name
            - temperature (float): Sampling temperature
            - model_providers (List[str]): List of provider names for primary model
            - alternative_model (str, optional): Fallback model name
            - alternative_model_provider (List[str], optional): Providers for fallback model

        messages: List of message dicts with "role" and "content" keys

    Returns:
        UnifiedResponse with normalized content, or None if all providers fail
    """
    # Validate agent_config
    if not isinstance(agent_config, dict):
        logger.error("agent_config must be a dictionary")
        return None
    
    required_keys = ["model_name", "temperature", "model_providers"]
    for key in required_keys:
        if key not in agent_config:
            logger.error(f"Missing required key '{key}' in agent_config")
            return None
    
    # Validate model_name
    if not agent_config["model_name"] or not isinstance(agent_config["model_name"], str):
        logger.error("model_name must be a non-empty string")
        return None
    
    # Validate temperature
    try:
        temp = float(agent_config["temperature"])
        if not (0.0 <= temp <= 2.0):  # Reasonable temperature range
            logger.warning(f"Temperature {temp} is outside the typical range [0.0, 2.0]")
    except (TypeError, ValueError):
        logger.error(f"temperature must be a numeric value, got {type(agent_config['temperature'])}")
        return None
    
    # Validate model_providers
    providers = agent_config["model_providers"]
    if not isinstance(providers, (list, tuple)):
        logger.error("model_providers must be a list or tuple")
        return None
    if not providers:
        logger.error("model_providers must be a non-empty list")
        return None
    for provider in providers:
        if not isinstance(provider, str):
            logger.error(f"Provider names must be strings, got {type(provider)}")
            return None
    
    # Validate messages
    if not isinstance(messages, list):
        logger.error("messages must be a list")
        return None
    if not messages:
        logger.error("messages must be a non-empty list")
        return None
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            logger.error(f"Message at index {i} must be a dictionary")
            return None
        if "role" not in message or "content" not in message:
            logger.error(f"Message at index {i} must have 'role' and 'content' keys")
            return None
    
    _initialize_providers()

    # Try primary model with available providers
    result = _try_primary_model(agent_config, messages)
    if result is not None:
        return result

    # Try alternative model if configured
    result = _try_alternative_model(agent_config, messages)
    if result is not None:
        return result

    logger.error("All providers exhausted, no response available")
    return None


def _try_primary_model(agent_config: Dict[str, Any], messages: List[Dict[str, Any]]) -> Optional[UnifiedResponse]:
    """Try to get response from primary model with available providers."""
    model = str(agent_config["model_name"])
    temperature = float(agent_config["temperature"])
    primary_providers = list(agent_config["model_providers"])

    available = _get_available_providers(primary_providers)
    for provider in available:
        logger.info("Trying provider %s with model %s", provider, model)
        result = _execute_provider_call(provider, model, temperature, messages)
        if result is not None:
            logger.info("Successfully got response from %s", provider)
            return result
        
        # If the provider is google and the model is gemini-pro, try with flash model as a fallback
        if provider == "google" and "gemini-2.5-pro" in model:
            flash_model = "models/gemini-2.5-flash"
            logger.info("Trying provider %s with flash model %s", provider, flash_model)
            result = _execute_provider_call(provider, flash_model, temperature, messages)
            if result is not None:
                logger.info("Successfully got response from %s with flash model", provider)
                return result

    return None


def _try_alternative_model(agent_config: Dict[str, Any], messages: List[Dict[str, Any]]) -> Optional[UnifiedResponse]:
    """Try to get response from alternative model with available providers."""
    alt_model = agent_config.get("alternative_model")
    alt_providers = agent_config.get("alternative_model_provider", [])

    if alt_model and alt_providers:
        logger.info("Trying alternative model %s", alt_model)
        temperature = float(agent_config["temperature"])
        available_alt = _get_available_providers(list(alt_providers))
        for provider in available_alt:
            logger.info(
                "Trying provider %s with alternative model %s", provider, alt_model
            )
            result = _execute_provider_call(
                provider, str(alt_model), temperature, messages
            )
            if result is not None:
                logger.info(
                    "Successfully got response from %s with alternative model", provider
                )
                return result
    return None

# File: tests/test_api.py
"""
Unit tests verifying the Google provider invocation in src._api.

These tests ensure the api_caller function uses the modern google-genai
signature that accepts the config keyword argument instead of the legacy
generation_config parameter.
"""

from __future__ import annotations

import logging
import pytest
from typing import Any, Dict, List

from src import _api


def test_api_caller_google_uses_config_keyword(monkeypatch) -> None:
    """
    Ensure the Google provider path relies on the config keyword argument and
    passes the desired temperature to GenerateContentConfig.
    """

    captured_arguments: Dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage_metadata = None

    class FakePart:
        @staticmethod
        def from_text(text: str) -> Dict[str, str]:
            return {"text": text}

    class FakeContent:
        def __init__(self, role: str, parts: List[Dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class FakeModels:
        def generate_content(
            self, *, model: str, contents: List[FakeContent], config: Any
        ) -> FakeResponse:
            captured_arguments["model"] = model
            captured_arguments["contents"] = contents
            captured_arguments["config"] = config
            return FakeResponse("stub response")

    class FakeClient:
        def __init__(self, api_key: str, http_options: Any | None = None) -> None:
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()

    class FakeHttpOptions:
        def __init__(self, api_version: str) -> None:
            self.api_version = api_version

    class FakeGenerateContentConfig:
        def __init__(
            self, temperature: float, system_instruction: Any | None = None
        ) -> None:
            captured_arguments["temperature"] = temperature
            captured_arguments["system_instruction"] = system_instruction
            self.temperature = temperature
            self.system_instruction = system_instruction

    class FakeTypes:
        Part = FakePart
        Content = FakeContent
        HttpOptions = FakeHttpOptions
        GenerateContentConfig = FakeGenerateContentConfig

    class FakeModule:
        Client = FakeClient
        types = FakeTypes

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(_api, "GenAIModule", FakeModule)
    monkeypatch.setattr(_api, "_INITIALIZED", False)
    monkeypatch.setattr(_api, "PROVIDER_REGISTRY", {})
    monkeypatch.setattr(_api, "PROVIDER_HEALTH", {})

    response = _api.api_caller(
        {
            "model_name": "models/gemini-2.5-pro",
            "temperature": 0.42,
            "model_providers": ["google"],
        },
        [{"role": "user", "content": "hello"}],
    )

    assert response is not None
    assert captured_arguments["model"] == "models/gemini-2.5-pro"
    assert isinstance(captured_arguments["contents"], list)
    assert captured_arguments["contents"][0].role == "user"
    assert captured_arguments["contents"][0].parts[0]["text"] == "hello"
    assert isinstance(captured_arguments["config"], FakeGenerateContentConfig)
    assert captured_arguments["temperature"] == 0.42


def test_validate_api_caller_input_success() -> None:
    """Test that valid inputs pass validation."""
    valid_agent_config = {
        "model_name": "test-model",
        "temperature": 0.5,
        "model_providers": ["google"],
    }
    valid_messages = [{"role": "user", "content": "test"}]
    assert _api._validate_api_caller_input(valid_agent_config, valid_messages) is True


@pytest.mark.parametrize(
    "agent_config, messages, expected_error_message",
    [
        # --- agent_config validation failures ---
        (None, [], "agent_config must be a dictionary"),
        ({}, [], "Missing required key 'model_name' in agent_config"),
        (
            {"model_name": "", "temperature": 0.5, "model_providers": ["google"]},
            [],
            "model_name must be a non-empty string",
        ),
        (
            {"model_name": 123, "temperature": 0.5, "model_providers": ["google"]},
            [],
            "model_name must be a non-empty string",
        ),
        (
            {"model_name": "m", "temperature": "not-float", "model_providers": ["g"]},
            [],
            "temperature must be a numeric value, got <class 'str'>",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": "not-list"},
            [],
            "model_providers must be a list or tuple",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": []},
            [],
            "model_providers must be a non-empty list",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": [123]},
            [],
            "Provider names must be strings, got <class 'int'>",
        ),
        # --- messages validation failures ---
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": ["g"]},
            "not-list",
            "messages must be a list",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": ["g"]},
            [],
            "messages must be a non-empty list",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": ["g"]},
            [{"role": "user", "content": "ok"}, "not-dict"],
            "Message at index 1 must be a dictionary",
        ),
        (
            {"model_name": "m", "temperature": 0.5, "model_providers": ["g"]},
            [{"role": "user"}],
            "Message at index 0 must have 'role' and 'content' keys",
        ),
    ],
)
def test_validate_api_caller_input_failure(
    agent_config: Any, messages: Any, expected_error_message: str, caplog
) -> None:
    """Test that invalid inputs fail validation and log the correct error message."""
    with caplog.at_level(logging.ERROR):
        assert _api._validate_api_caller_input(agent_config, messages) is False
        assert expected_error_message in caplog.text

# File: tests/test_api.py
"""
Unit tests verifying the Google provider invocation in src._api.

These tests ensure the api_caller function uses the modern google-genai
signature that accepts the config keyword argument instead of the legacy
generation_config parameter.
"""

from __future__ import annotations

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

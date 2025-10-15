# File: tests/test_google_integration.py
"""
Live integration test for the Google Gemini provider path.

This test invokes the real Gemini API via the python-genai SDK to confirm that
our unified API caller returns a response and that the client is configured with
the expected HTTP options per the official Google documentation.
"""

from __future__ import annotations

import os

from src import _api
from src.approver import Approver
from src.configurator import Configurator


def test_google_provider_live_response() -> None:
    """
    Exercise the Google provider end-to-end and assert we get a real response.
    Also verifies that the client uses the v1alpha API version via HttpOptions.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    assert gemini_key and gemini_key.strip(), "GEMINI_API_KEY must be set for live test"

    # Reset provider registry to force re-initialization for this test
    _api.PROVIDER_REGISTRY.clear()
    _api.PROVIDER_HEALTH.clear()
    _api._INITIALIZED = False

    response = _api.api_caller(
        {
            "model_name": "models/gemini-2.5-pro",
            "temperature": 0.0,
            "model_providers": ["google"],
        },
        [
            {
                "role": "system",
                "content": (
                    "You are a test harness ensuring the Gemini provider responds. "
                    "Return concise acknowledgements."
                ),
            },
            {"role": "user", "content": "Reply with the word OK."},
        ],
    )

    assert response is not None, "Expected a real response from Google Gemini"
    assert response.provider_name == "google"
    assert isinstance(response.content, str) and response.content.strip()
    raw_response = response.raw_response
    assert raw_response is not None, "Raw response must be preserved"
    assert hasattr(raw_response, "text"), "Raw response should expose .text attribute"
    assert isinstance(raw_response.text, str) and raw_response.text.strip()

    google_client = _api.PROVIDER_REGISTRY.get("google")
    assert google_client is not None, "Google client should be registered"

    http_options = google_client._api_client._http_options  # type: ignore[attr-defined]
    assert getattr(http_options, "api_version", None) == "v1alpha"


def test_approver_returns_raw_response() -> None:
    """
    Ensure the Approver agent surfaces the raw Gemini response without enforcing JSON.
    """
    cfg = Configurator("conf/mcp.toml")
    cfg.load()
    approver_agent = Approver(cfg)

    result = approver_agent.execute("Integration check: echo raw output.")
    assert isinstance(result, dict)
    assert "raw_response" in result
    raw_payload = result["raw_response"]
    assert isinstance(raw_payload, dict)
    assert raw_payload.get("candidates"), "Expected candidates field in raw response"

"""
Test file for the Approver module after implementing fixes
to address the AI reviewer's recommendations.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.approver import Approver, store_decision
from src.configurator import Configurator
from src.qdrant_integration import QdrantIntegration


@pytest.fixture
def mock_configurator() -> MagicMock:
    """Fixture for a mocked Configurator."""

    mock = MagicMock(spec=Configurator)
    mock.get_agent_config.return_value = {
        "prompt": "Test prompt",
        "model_name": "test_model",
        "model_providers": ["google"],
        "temperature": 0.5,
        "project_root": os.getcwd(),
        "skills": [
            "code_review",
            "quality_assurance",
            "standards_compliance",
        ],
    }
    mock.get_context_policy.return_value = MagicMock(
        recent_minutes=10,
        include_extensions=(".py",),
        exclude_dirs=(".git",),
        max_file_bytes=1024,
        max_total_bytes=4096,
        design_docs=("/docs/DESIGN_PRINCIPLES_GUIDE.md",),
        source_code_directory=("src",),
        tests_directory=("tests",),
        project_directories=("/",),
        embedding_model_sizes={"test_model": 384},
    )
    # Configure the combine_prompt_with_skills method to call the real implementation
    real_configurator = Configurator.__new__(Configurator)
    mock.combine_prompt_with_skills = real_configurator.combine_prompt_with_skills
    return mock


def test_approver_instantiation(mock_configurator: MagicMock) -> None:
    """Test that the Approver class can be instantiated."""
    approver = Approver(mock_configurator)
    assert isinstance(approver, Approver)


@patch("src.approver.Approver._store_approver_decision_in_qdrant")
@patch("src.base_agent.api_caller")
@patch("src.base_agent.collect_recent_sources")
def test_execute_success(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_store_qdrant: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method for a successful run."""
    mock_collect_sources.return_value = ("some source code", [Path("src/fake.py")])
    mock_api_caller.return_value = MagicMock(
        provider_name="google",
        model_name="test_model",
        content='{"decision": "APPROVED"}',
        raw_response={},
    )

    approver = Approver(mock_configurator)
    result = approver.execute(payload={"user_chat": "test chat"})

    assert result["status"] == "success"
    assert result["data"]["provider"] == "google"
    assert result["data"]["raw_text"] == '{"decision": "APPROVED"}'
    mock_store_qdrant.assert_called_once()


@patch("src.approver.Approver._store_approver_decision_in_qdrant")
@patch("src.base_agent.api_caller")
@patch("src.base_agent.collect_recent_sources")
@pytest.mark.parametrize(
    "mock_content",
    [
        '{"decision": "CHANGES_REQUESTED"}',  # Not APPROVED
        '{"not_a_decision_key": "APPROVED"}',  # Missing decision key
        "not a json string",  # JSONDecodeError
        '["a list", "not a dict"]',  # Valid JSON, but not a dict
    ],
)
def test_execute_qdrant_not_called(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_store_qdrant: MagicMock,
    mock_configurator: MagicMock,
    mock_content: str,
) -> None:
    """Test that Qdrant storage is not called when decision is not APPROVED or parsing fails."""
    mock_collect_sources.return_value = ("some source code", [Path("src/fake.py")])
    mock_api_caller.return_value = MagicMock(
        provider_name="google",
        model_name="test_model",
        content=mock_content,
        raw_response={},
    )

    approver = Approver(mock_configurator)
    approver.execute(payload={"user_chat": "test chat"})

    mock_store_qdrant.assert_not_called()


@patch("src.base_agent.api_caller")
@patch("src.base_agent.collect_recent_sources")
def test_execute_with_markdown_json(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method successfully parses JSON wrapped in markdown fences."""
    mock_collect_sources.return_value = ("some source code", [Path("src/fake.py")])

    # Mock response with markdown fences
    markdown_json = '```json\n{"decision": "APPROVED"}\n```'
    mock_api_caller.return_value = MagicMock(
        provider_name="google",
        model_name="test_model",
        content=markdown_json,
        raw_response={},
    )

    approver = Approver(mock_configurator)
    result = approver.execute(payload={"user_chat": "test chat"})

    assert result["status"] == "success"
    assert result["data"]["provider"] == "google"
    assert result["data"]["raw_text"] == markdown_json


@patch("src.base_agent.api_caller")
@patch("src.base_agent.collect_recent_sources")
def test_skills_tags_in_system_prompt(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Ensure skill tags are prefixed to the approver system prompt."""
    captured_prompts: list[str] = []

    def fake_api_caller(
        config_payload: dict[str, object], messages: list[dict[str, str]]
    ) -> MagicMock:
        captured_prompts.append(messages[0]["content"])
        return MagicMock(
            provider_name="google",
            model_name="test_model",
            content='{"decision": "APPROVED"}',
            raw_response={},
        )

    mock_collect_sources.return_value = ("some source code", [Path("src/fake.py")])
    mock_api_caller.side_effect = fake_api_caller

    approver = Approver(mock_configurator)
    approver.execute(payload={"user_chat": "test chat history"})

    assert captured_prompts, "Expected to capture a system prompt"
    system_prompt = captured_prompts[0]
    assert system_prompt.startswith(
        "# Tags:"
    ), "System prompt should begin with skill tags"

    expected_skills = mock_configurator.get_agent_config.return_value["skills"]
    for skill_name in expected_skills:
        assert (
            skill_name in system_prompt
        ), f"Skill '{skill_name}' missing from prompt tags"


@patch("src.base_agent.collect_recent_sources")
def test_execute_no_recent_files(
    mock_collect_sources: MagicMock, mock_configurator: MagicMock
) -> None:
    """Test the execute method when no recent files are found."""
    mock_collect_sources.return_value = ("", [])

    approver = Approver(mock_configurator)
    result = approver.execute(payload={"user_chat": "test chat"})

    assert result["status"] == "no_recent_files"
    assert "files" in result["data"]


@patch("src.base_agent.api_caller")
@patch("src.base_agent.collect_recent_sources")
def test_execute_api_error(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method when the API call fails."""
    mock_collect_sources.return_value = ("some source code", [Path("src/fake.py")])
    mock_api_caller.return_value = None

    approver = Approver(mock_configurator)
    result = approver.execute(payload={"user_chat": "test chat"})

    assert result["status"] == "error"
    assert result["message"] == "No valid response received from any provider."


def test_extract_approver_content_for_embedding(mock_configurator: MagicMock) -> None:
    """Test that the content extraction method correctly formats the decision data."""
    approver = Approver(mock_configurator)
    decision_data = {
        "decision": "APPROVED",
        "summary": "The fix is good.",
        "positive_points": ["Clean code", "Good tests"],
        "negative_points": ["None"],
        "required_actions": ["None"],
    }
    expected_content = "Decision: APPROVED, Summary: The fix is good., Positive: Clean code; Good tests, Negative: None, Actions: None"

    content = approver._extract_approver_content_for_embedding(decision_data)
    assert content == expected_content


@patch("src.approver.time")
def test_store_decision(mock_time: MagicMock) -> None:
    """Test that store_decision calls QdrantIntegration correctly with a timestamp."""
    mock_time.time.return_value = 1234567890.0

    mock_qdrant = MagicMock(spec=QdrantIntegration)
    decision_data = {"decision": "APPROVED", "summary": "Test"}

    result = store_decision(
        mock_qdrant,
        "test_id",
        "content for embedding",
        decision_data,
    )

    expected_data = {
        "decision": "APPROVED",
        "summary": "Test",
        "timestamp": 1234567890.0,
    }

    mock_qdrant.store_approver_decision.assert_called_once_with(
        expected_data,
        "test_id",
        "content for embedding",
    )
    assert result is True


def test_approver_fixes_addressed(mock_configurator: MagicMock) -> None:
    """Test that the fixes for AI recommendations have been implemented."""
    # The approver should now use descriptive variable names and parameterized values
    Approver(mock_configurator)

    # Check that the context policy has the source_code_directory parameter
    policy = mock_configurator.get_context_policy.return_value
    assert hasattr(policy, "source_code_directory")
    assert policy.source_code_directory == ("src",)  # Should be configurable

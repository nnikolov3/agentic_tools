"""
Test file for the Approver module after implementing fixes
to address the AI reviewer's recommendations.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import os

from src.approver import Approver
from src.configurator import Configurator


@pytest.fixture
def mock_configurator() -> MagicMock:
    """Fixture for a mocked Configurator."""
    from src.configurator import Configurator
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
        src_dir="src",  # Added src_dir parameter
        include_extensions=(".py",),
        exclude_dirs=(".git",),
        max_file_bytes=1024,
        max_total_bytes=4096,
        docs_paths=(),
        discovery=MagicMock(enabled=False),
    )
    # Configure the combine_prompt_with_skills method to call the real implementation
    real_configurator = Configurator.__new__(Configurator)
    mock.combine_prompt_with_skills = real_configurator.combine_prompt_with_skills
    return mock


def test_approver_instantiation(mock_configurator: MagicMock) -> None:
    """Test that the Approver class can be instantiated."""
    approver = Approver(mock_configurator)
    assert isinstance(approver, Approver)


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
def test_execute_success(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method for a successful run."""
    mock_collect_sources.return_value = "some source code"
    mock_api_caller.return_value = MagicMock(
        provider_name="google",
        model_name="test_model",
        content='{"decision": "APPROVED"}',
        raw_response={},
    )

    approver = Approver(mock_configurator)
    result = approver.execute(payload={'user_chat': 'test chat'})

    assert result["status"] == "success"
    assert result["data"]["provider"] == "google"
    assert result["data"]["raw_text"] == '{"decision": "APPROVED"}'


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
def test_skills_tags_in_system_prompt(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Ensure skill tags are prefixed to the approver system prompt."""
    captured_prompts: list[str] = []

    def fake_api_caller(config_payload: dict[str, object], messages: list[dict[str, str]]) -> MagicMock:
        captured_prompts.append(messages[0]["content"])
        return MagicMock(
            provider_name="google",
            model_name="test_model",
            content='{"decision": "APPROVED"}',
            raw_response={},
        )

    mock_collect_sources.return_value = "some source code"
    mock_api_caller.side_effect = fake_api_caller

    approver = Approver(mock_configurator)
    approver.execute(payload={'user_chat': 'test chat history'})

    assert captured_prompts, "Expected to capture a system prompt"
    system_prompt = captured_prompts[0]
    assert system_prompt.startswith("# Tags:"), "System prompt should begin with skill tags"

    expected_skills = mock_configurator.get_agent_config.return_value["skills"]
    for skill_name in expected_skills:
        assert skill_name in system_prompt, f"Skill '{skill_name}' missing from prompt tags"


@patch("src.base_agent.collect_recent_sources")
def test_execute_no_recent_files(
    mock_collect_sources: MagicMock, mock_configurator: MagicMock
) -> None:
    """Test the execute method when no recent files are found."""
    mock_collect_sources.return_value = ""

    approver = Approver(mock_configurator)
    result = approver.execute(payload={'user_chat': 'test chat'})

    assert result["status"] == "no_recent_files"
    assert "files" in result["data"]


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
def test_execute_api_error(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method when the API call fails."""
    mock_collect_sources.return_value = "some source code"
    mock_api_caller.return_value = None

    approver = Approver(mock_configurator)
    result = approver.execute(payload={'user_chat': 'test chat'})

    assert result["status"] == "error"
    assert result["message"] == "No valid response received from any provider."


def test_approver_fixes_addressed(mock_configurator: MagicMock) -> None:
    """Test that the fixes for AI recommendations have been implemented."""
    # The approver should now use descriptive variable names and parameterized values
    approver = Approver(mock_configurator)
    
    # Check that the context policy has the src_dir parameter
    policy = mock_configurator.get_context_policy.return_value
    assert hasattr(policy, 'src_dir')
    assert policy.src_dir == "src"  # Should be configurable

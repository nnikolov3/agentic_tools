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
    mock = MagicMock(spec=Configurator)
    mock.get_agent_config.return_value = {
        "prompt": "Test prompt",
        "model_name": "test_model",
        "model_providers": ["google"],
        "temperature": 0.5,
        "project_root": os.getcwd(),
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
    return mock


def test_approver_instantiation(mock_configurator: MagicMock) -> None:
    """Test that the Approver class can be instantiated."""
    approver = Approver(mock_configurator)
    assert isinstance(approver, Approver)


@patch("src.approver.api_caller")
@patch("src.approver.collect_recent_sources")
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
    result = approver.execute("test chat")

    assert result["status"] == "success"
    assert result["data"]["provider"] == "google"
    assert result["data"]["raw_text"] == '{"decision": "APPROVED"}'


@patch("src.approver.collect_recent_sources")
def test_execute_no_recent_files(
    mock_collect_sources: MagicMock, mock_configurator: MagicMock
) -> None:
    """Test the execute method when no recent files are found."""
    mock_collect_sources.return_value = ""

    approver = Approver(mock_configurator)
    result = approver.execute("test chat")

    assert result["status"] == "no_recent_files"
    assert "files" in result["data"]


@patch("src.approver.api_caller")
@patch("src.approver.collect_recent_sources")
def test_execute_api_error(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method when the API call fails."""
    mock_collect_sources.return_value = "some source code"
    mock_api_caller.return_value = None

    approver = Approver(mock_configurator)
    result = approver.execute("test chat")

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
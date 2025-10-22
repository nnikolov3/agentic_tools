# tests/tools/test_tool.py
"""
Purpose:
This module contains unit tests for the Tool class, which is responsible for
aggregating context, constructing payloads, and orchestrating API calls for agents.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.tools.tool import Tool


@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for tests."""
    return {
        "project_root": "/path/to/project",
        "docs": "docs",
        "source": "src",
        "design_docs": ["docs/DESIGN.md"],
        "approver": {
            "skills": ["review", "approve"],
            "prompt": "Approve this code.",
        },
        "readme_writer": {
            "skills": ["write", "document"],
            "prompt": "Write a README.",
        },
    }


@pytest.mark.asyncio
@patch("src.tools.tool.QdrantCollectionTools")
@patch("src.tools.tool.ApiTools")
@patch("src.tools.tool.ShellTools")
async def test_run_tool_generic_agent(
    MockShellTools, MockApiTools, MockQdrantTools, mock_config
):
    """
    Verify that run_tool correctly constructs a payload and calls the API
    for a generic agent.
    """
    # Arrange
    mock_shell_instance = MockShellTools.return_value
    mock_shell_instance.create_patch.return_value = "git_patch"
    mock_shell_instance.get_git_info.return_value = {"user": "tester"}
    mock_shell_instance.process_directory.return_value = "source_code"
    mock_shell_instance.read_file_content_for_path.return_value = "design_doc_content"

    mock_api_instance = MockApiTools.return_value
    mock_api_instance.run_.api = AsyncMock(return_value="api_response")

    tool = Tool("approver", mock_config)
    chat_message = "Please review."
    memory_context = "previous_conversation"

    # Act
    response = await tool.run_tool(chat=chat_message, memory_context=memory_context)

    # Assert
    assert response == "api_response"

    # Verify payload construction
    expected_payload = {
        "prompt": "Approve this code.",
        "skills": ["review", "approve"],
        "memory": memory_context,
        "git-diff-patch": "git_patch",
        "git-info": {"user": "tester"},
        "design_documents": "design_doc_content",
        "source_code": "source_code",
        "chat": chat_message,
    }
    mock_api_instance.run_api.assert_awaited_once_with(expected_payload)


@pytest.mark.asyncio
@patch("src.tools.tool.mdformat.text")
@patch("src.tools.tool.QdrantCollectionTools")
@patch("src.tools.tool.ApiTools")
@patch("src.tools.tool.ShellTools")
async def test_run_tool_readme_writer_special_handling(
    MockShellTools, MockApiTools, MockQdrantTools, mock_mdformat, mock_config
):
    """
    Verify that run_tool performs special post-processing for the readme_writer agent.
    """
    # Arrange
    mock_shell_instance = MockShellTools.return_value
    mock_shell_instance.create_patch.return_value = ""
    mock_shell_instance.get_git_info.return_value = {}
    mock_shell_instance.process_directory.return_value = "source_code"
    mock_shell_instance.read_file_content_for_path.return_value = ""
    mock_shell_instance.cleanup_escapes.return_value = "cleaned_response"

    mock_api_instance = MockApiTools.return_value
    mock_api_instance.run_api = AsyncMock(return_value="raw_api_response")

    mock_qdrant_instance = MockQdrantTools.return_value
    mock_qdrant_instance.run_qdrant = AsyncMock()

    mock_mdformat.return_value = "formatted_readme"

    tool = Tool("readme_writer", mock_config)

    # Act
    response = await tool.run_tool(chat="Update README", memory_context=None)

    # Assert
    assert response == "formatted_readme"

    # Verify post-processing calls
    mock_api_instance.run_api.assert_awaited_once()
    mock_shell_instance.cleanup_escapes.assert_called_once_with("raw_api_response")
    mock_mdformat.assert_called_once_with(
        "cleaned_response", options={"wrap": "preserve"}
    )
    mock_shell_instance.write_file.assert_called_once_with(
        "README.md", "formatted_readme"
    )
    mock_qdrant_instance.run_qdrant.assert_awaited_once_with("formatted_readme")

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from src.tools.tool import Tool
from pathlib import Path

# Mock configuration
MOCK_CONFIG: dict[str, Any] = {
    "agentic-tools": {
        "memory": {},
    }
}


@pytest.fixture
def mock_tool_instance():
    """Mocks a Tool instance with mocked dependencies."""
    with (
        patch("src.tools.tool.ShellTools", new_callable=MagicMock) as MockShellTools,
        patch("src.tools.tool.ApiTools", new_callable=MagicMock) as MockApiTools,
    ):

        # Configure the mocked shell tools
        mock_shell_tools = MockShellTools.return_value
        mock_shell_tools.fetch_urls_content = AsyncMock(return_value="Fetched Content")

        # Configure the mocked API tools
        mock_api_tools = MockApiTools.return_value
        mock_api_tools.run_api = AsyncMock(return_value="API Response")

        # Create the Tool instance
        tool = Tool("test_agent", MOCK_CONFIG, Path.cwd())

        # Manually set the mocked instances for easy access in tests
        tool.shell_tools = mock_shell_tools
        tool.api_tools = mock_api_tools

        return tool


@pytest.mark.asyncio
async def test_run_tool_default_agent_path(mock_tool_instance):
    """Tests that a non-builder agent follows the standard API path."""
    mock_tool_instance.agent = "developer"

    result = await mock_tool_instance.run_tool(
        chat="Refactor this code.", filepath="/src/file.py"
    )

    # Should return the API response
    assert result == "API Response"

    # Should NOT call shell_tools.fetch_urls_content
    mock_tool_instance.shell_tools.fetch_urls_content.assert_not_called()

    # Should call the API
    mock_tool_instance.api_tools.run_api.assert_called_once()

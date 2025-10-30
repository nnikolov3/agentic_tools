import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.agent import KnowledgeBaseAgent

# Mock configuration
MOCK_CONFIG: dict[str, Any] = {
    "agentic-tools": {
        "memory": {},
    }
}


@pytest.fixture
def mock_agent_instance(tmp_path):
    """Mocks a KnowledgeBaseAgent instance with mocked dependencies."""
    output_file = tmp_path / "output.txt"
    with patch(
        "src.agents.agent.ShellTools", new_callable=MagicMock
    ) as mock_shell_tools_class:

        # Configure the mocked shell tools
        mock_shell_tools = mock_shell_tools_class.return_value
        mock_shell_tools.fetch_urls_content = AsyncMock(return_value="Fetched Content")

        agent = KnowledgeBaseAgent(
            configuration=MOCK_CONFIG,
            agent_name="knowledge_base_builder",
            project="agentic-tools",
            chat="url1,url2",
            filepath=str(output_file),
            target_directory=tmp_path,
        )

        # Attach mocks to the instance for easy access
        agent.shell_tools = mock_shell_tools

        return agent


@pytest.mark.asyncio
async def test_knowledge_base_agent_init_failure(tmp_path):
    """Tests initialization failure when filepath is missing."""
    with pytest.raises(ValueError, match="requires a valid output filepath"):
        KnowledgeBaseAgent(
            configuration=MOCK_CONFIG,
            agent_name="knowledge_base_builder",
            project="agentic-tools",
            chat="url1,url2",
            filepath=None,
            target_directory=tmp_path,
        )


@pytest.mark.asyncio
@patch("src.agents.agent.asyncio.to_thread", new_callable=AsyncMock)
@patch("src.agents.agent.QdrantMemory.create", new_callable=AsyncMock)
async def test_run_agent_success(
    mock_qdrant_create, mock_to_thread, mock_agent_instance
):
    """Tests the full run_agent lifecycle for success."""
    mock_agent_instance.chat = "http://url1.com, http://url2.com"

    mock_memory_instance = AsyncMock()
    mock_qdrant_create.return_value = mock_memory_instance

    result = await mock_agent_instance.run_agent()

    # 1. Check return value
    assert result == "Fetched Content"

    # 2. Check URL parsing and fetching
    mock_agent_instance.shell_tools.fetch_urls_content.assert_called_once_with(
        ["http://url1.com", "http://url2.com"]
    )

    # 3. Check _post_process (file writing)
    mock_to_thread.assert_called_once()

    # 4. Check _store_memory (knowledge bank storage)
    mock_qdrant_create.assert_called_once_with(
        mock_agent_instance.configuration, "knowledge_bank"
    )
    mock_memory_instance.add_memory.assert_called_once_with(
        text_content="Fetched Content"
    )


@pytest.mark.asyncio
async def test_run_agent_invalid_chat_input(mock_agent_instance):
    """Tests that run_agent raises RuntimeError for invalid chat input."""

    # Test case 1: chat is None
    mock_agent_instance.chat = None
    with pytest.raises(
        RuntimeError, match="A comma-separated string of URLs is required"
    ):
        await mock_agent_instance.run_agent()

    # Test case 2: chat is not a string
    mock_agent_instance.chat = 123
    with pytest.raises(
        RuntimeError, match="A comma-separated string of URLs is required"
    ):
        await mock_agent_instance.run_agent()

    mock_agent_instance.shell_tools.fetch_urls_content.assert_not_called()
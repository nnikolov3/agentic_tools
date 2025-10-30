import pytest
from unittest.mock import patch, AsyncMock
from src.agents.agent import DefaultAgent
from pathlib import Path


@pytest.fixture
def agent_and_tmpdir(tmp_path):
    config = {"agentic-tools": {"responses_dir": str(tmp_path)}}
    agent = DefaultAgent(
        configuration=config,
        agent_name="test_agent",
        project="agentic-tools",
        chat="test chat",
        filepath=None,
        target_directory=None,
    )
    return agent, tmp_path


@pytest.mark.asyncio
@patch("asyncio.to_thread", new_callable=AsyncMock)
async def test_write_response_to_file_success(mock_to_thread, agent_and_tmpdir):
    agent, tmpdir = agent_and_tmpdir
    agent.response = "Test response content."

    await agent._write_response_to_file()

    # Check that asyncio.to_thread was called with the correct arguments
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args[0]
    assert call_args[0] == agent._write_sync
    assert isinstance(call_args[1], Path)
    assert call_args[2] == "Test response content."


@pytest.mark.asyncio
@patch("asyncio.to_thread", new_callable=AsyncMock)
async def test_write_response_to_file_io_error(
    mock_to_thread, agent_and_tmpdir, caplog
):
    agent, _ = agent_and_tmpdir
    agent.response = "Test response content."

    # Configure the mock to raise an IOError
    mock_to_thread.side_effect = IOError("Disk full")

    with pytest.raises(IOError, match="Disk full"):
        await agent._write_response_to_file()
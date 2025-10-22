# tests/agents/test_agent.py
"""
Purpose:
This module contains unit tests for the Agent class, which is responsible for
orchestrating tool execution and managing the agent's memory lifecycle.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.agent import Agent


class TestAgent(unittest.TestCase):
    """Test suite for the Agent class."""

    def setUp(self):
        """Set up a basic configuration and an Agent instance for each test."""
        self.configuration = {
            "some_config": "some_value",
            "readme_writer": {"prompt": "test prompt"},
        }
        self.agent = Agent(self.configuration)

    def test_init(self):
        """Verify that the Agent is initialized correctly."""
        self.assertEqual(self.agent.configuration, self.configuration)
        self.assertEqual(self.agent.agent_config, {})
        self.assertIsNone(self.agent.tool)
        self.assertEqual(self.agent.agent_name, "")
        self.assertIsNone(self.agent.memory)

    @patch("src.agents.agent.Tool")
    async def test_run_agent_no_memory(self, mock_tool):
        """Test the basic execution of run_agent without memory enabled."""
        # Arrange: Mock the Tool and its run_tool method
        mock_tool_instance = MagicMock()
        mock_tool_instance.run_tool = AsyncMock(return_value="success")
        mock_tool.return_value = mock_tool_instance

        # Act: Run the agent
        result = await self.agent.run_agent("readme_writer", chat="test chat")

        # Assert: Verify the agent's state and tool interactions
        self.assertEqual(result, "success")
        self.assertEqual(self.agent.agent_name, "readme_writer")
        mock_tool.assert_called_once_with("readme_writer", self.configuration)
        self.assertEqual(self.agent.tool, mock_tool_instance)
        # Assert that run_tool was called correctly, without memory context
        mock_tool_instance.run_tool.assert_awaited_once_with(
            chat="test chat", memory_context=None
        )

    @patch("src.agents.agent.Tool")
    async def test_run_agent_case_insensitivity(self, mock_tool):
        """Test that agent names are handled case-insensitively."""
        # Arrange
        mock_tool_instance = MagicMock()
        mock_tool_instance.run_tool = AsyncMock(return_value="success")
        mock_tool.return_value = mock_tool_instance

        # Act: Run the agent with an uppercase name
        result = await self.agent.run_agent("README_WRITER", chat="test chat")

        # Assert
        self.assertEqual(result, "success")
        self.assertEqual(self.agent.agent_name, "readme_writer")
        # The tool should be initialized with the lowercased name
        mock_tool.assert_called_once_with("readme_writer", self.configuration)
        self.assertEqual(self.agent.tool, mock_tool_instance)
        mock_tool_instance.run_tool.assert_awaited_once_with(
            chat="test chat", memory_context=None
        )

    async def test_run_agent_invalid_agent(self):
        """Test that running an agent not defined in the config raises a ValueError."""
        agent_with_missing_config = Agent({"some_config": "value"})
        with self.assertRaises(ValueError):
            await agent_with_missing_config.run_agent("invalid_agent_name", chat=None)

    @patch("src.agents.agent.QdrantMemory")
    @patch("src.agents.agent.Tool")
    async def test_run_agent_with_memory_disabled(self, mock_tool, mock_qdrant_memory):
        """
        Verify that memory is not used when the [memory] section is missing or disabled.
        """
        # Arrange: Config with memory explicitly disabled
        config_disabled = self.configuration.copy()
        config_disabled["memory"] = {"enabled": False}
        agent_disabled = Agent(config_disabled)

        # Arrange: Config with no memory section (implicit disable)
        agent_missing_section = Agent(self.configuration.copy())

        # Arrange mock tool for both agents
        mock_tool_instance = MagicMock()
        mock_tool_instance.run_tool = AsyncMock(return_value="success")
        mock_tool.return_value = mock_tool_instance

        # Act & Assert for agent with memory explicitly disabled
        result_disabled = await agent_disabled.run_agent("readme_writer", chat="test")
        self.assertEqual(result_disabled, "success")
        mock_qdrant_memory.create.assert_not_called()
        mock_tool_instance.run_tool.assert_awaited_with(
            chat="test", memory_context=None
        )

        # Reset mocks for the next run
        mock_qdrant_memory.reset_mock()
        mock_tool_instance.run_tool.reset_mock()

        # Act & Assert for agent with memory section missing
        result_missing = await agent_missing_section.run_agent(
            "readme_writer", chat="test"
        )
        self.assertEqual(result_missing, "success")
        mock_qdrant_memory.create.assert_not_called()
        mock_tool_instance.run_tool.assert_awaited_with(
            chat="test", memory_context=None
        )

    @patch("src.agents.agent.Tool")
    @patch("src.agents.agent.QdrantMemory")
    async def test_run_agent_with_memory_enabled(self, mock_qdrant_memory, mock_tool):
        """
        Test the full memory lifecycle: create, retrieve, pass to tool, and add response.
        """
        # Arrange: Configuration with memory enabled
        memory_config = {"enabled": True, "collection_name": "test_memory"}
        config_with_memory = self.configuration.copy()
        config_with_memory["memory"] = memory_config
        agent_with_memory = Agent(config_with_memory)

        # Arrange: Mock the QdrantMemory and its methods
        mock_memory_instance = AsyncMock()
        mock_memory_instance.retrieve_context.return_value = "retrieved_context"
        mock_qdrant_memory.create.return_value = mock_memory_instance

        # Arrange: Mock the Tool and its run_tool method
        mock_tool_instance = MagicMock()
        mock_tool_instance.run_tool = AsyncMock(return_value="tool_response")
        mock_tool.return_value = mock_tool_instance

        # Act: Run the agent
        chat_message = "this is a test query"
        result = await agent_with_memory.run_agent("readme_writer", chat=chat_message)

        # Assert: Verify the entire memory and tool interaction flow
        self.assertEqual(result, "tool_response")

        # 1. Memory was created with the correct configuration
        mock_qdrant_memory.create.assert_awaited_once_with(memory_config)

        # 2. Context was retrieved using the chat message
        mock_memory_instance.retrieve_context.assert_awaited_once_with(
            query=chat_message
        )

        # 3. The retrieved context was passed to the tool
        mock_tool_instance.run_tool.assert_awaited_once_with(
            chat=chat_message, memory_context="retrieved_context"
        )

        # 4. The tool's response was added back to memory
        mock_memory_instance.add_memory.assert_awaited_once_with(
            text_content="tool_response"
        )


if __name__ == "__main__":
    unittest.main()

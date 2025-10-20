import unittest
from unittest.mock import patch, MagicMock
from src.agents.agent import Agent


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.configuration = {"some_config": "some_value"}
        self.agent = Agent(self.configuration)

    def test_init(self):
        self.assertEqual(self.agent.configuration, self.configuration)
        self.assertEqual(self.agent.agent_config, {})
        self.assertIsNone(self.agent.tool)
        self.assertEqual(self.agent.agent_name, "")

    @patch("src.agents.agent.Tool")
    def test_run_agent(self, mock_tool):
        # Mock the tool and the dynamic method
        mock_tool_instance = MagicMock()
        mock_tool.return_value = mock_tool_instance
        self.agent.readme_writer = MagicMock(return_value="success")

        # Run the agent
        result = self.agent.run_agent("readme_writer")

        # Assertions
        self.assertEqual(result, "success")
        self.assertEqual(self.agent.agent_name, "readme_writer")
        mock_tool.assert_called_once_with("readme_writer", self.configuration)
        self.assertEqual(self.agent.tool, mock_tool_instance)
        self.agent.readme_writer.assert_called_once()

    @patch("src.agents.agent.Tool")
    def test_readme_writer(self, mock_tool):
        # Mock the tool
        mock_tool_instance = MagicMock()
        mock_tool_instance.run_tool.return_value = "tool_success"
        mock_tool.return_value = mock_tool_instance
        self.agent.tool = mock_tool_instance

        # Run the readme_writer
        result = self.agent.readme_writer()

        # Assertions
        self.assertEqual(result, "tool_success")
        mock_tool_instance.run_tool.assert_called_once()

    def test_run_agent_invalid_agent(self):
        with self.assertRaises(ValueError):
            self.agent.run_agent("invalid_agent_name")

    @patch("src.agents.agent.Tool")
    def test_run_agent_case_insensitivity(self, mock_tool):
        # Mock the tool and the dynamic method
        mock_tool_instance = MagicMock()
        mock_tool.return_value = mock_tool_instance
        self.agent.readme_writer = MagicMock(return_value="success")

        # Run the agent with an uppercase name
        result = self.agent.run_agent("README_WRITER")

        # Assertions
        self.assertEqual(result, "success")
        self.assertEqual(self.agent.agent_name, "readme_writer")
        mock_tool.assert_called_once_with("readme_writer", self.configuration)
        self.assertEqual(self.agent.tool, mock_tool_instance)
        self.agent.readme_writer.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import unittest
from src.agents.agent import Agent


class TestAgentErrors(unittest.TestCase):

    def test_init_with_none_configuration(self):
        with self.assertRaises(ValueError):
            Agent(None)

    def test_run_invalid_agent(self):
        agent = Agent({})
        with self.assertRaises(ValueError):
            agent.run_agent("invalid_agent")


if __name__ == "__main__":
    unittest.main()

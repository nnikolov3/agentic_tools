# multi-agent-mcp

This project implements a multi-agent system using the Model Context Protocol (MCP) framework. It features specialized agents for tasks such as generating READMEs, reviewing code changes, and developing new code, all enhanced with Qdrant for persistent memory and contextual retrieval. The architecture emphasizes modularity, explicit configuration, and adherence to foundational design principles for robust and maintainable AI-driven workflows.

## Key Features

- **Multi-Agent Architecture**: Orchestrates specialized agents (e.g., `readme_writer`, `approver`, `developer`) for distinct tasks.
- **Qdrant Memory Integration**: Agents use Qdrant as a vector database for both short-term (recent) and long-term (historical) memory, enabling context-aware decision-making.
- **Modular Tooling**: Utilizes `ShellTools` for file system and Git operations (e.g., generating diffs, reading project files) and `ApiTools` for interfacing with Large Language Models (LLMs).
- **Automated README Generation**: The `readme_writer` agent can analyze project context and generate or update the `README.md` file.
- **Configurable Agents**: Agent behavior, LLM models, API keys, and memory settings are externalized in a TOML configuration file.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version `3.13` or higher.
- **uv**: A fast Python package installer and resolver.
  ```bash
  pip install uv
  ```
- **Git**: Required for cloning the repository and for agents to interact with version control.
  ```bash
  # Example for Debian/Ubuntu
  sudo apt-get install git
  # Example for macOS with Homebrew
  brew install git
  ```
- **Qdrant Server**: The project uses Qdrant for vector memory. You can run it locally via Docker:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```
- **Environment Variables**: API keys for the LLM providers must be set as environment variables.
  - `GEMINI_API_KEY_PLANNER`
  - `GEMINI_API_KEY_DEVELOPER`
  - `GEMINI_API_KEY_README_WRITER`
  - `GEMINI_API_KEY_APPROVER`

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```
1. **Install dependencies**:
   ```bash
   uv sync
   ```

## Configuration

The project's behavior is controlled by the `conf/agentic_tools.toml` file. This file defines global settings and specific configurations for each agent.

Key sections include:

- **`[agentic-tools]`**: Global settings like `exclude_directories`, `max_file_bytes`, and `git_diff_command`.
- **`[agentic-tools.memory]`**: Configures the Qdrant memory system.
  - `enabled`: `true` to enable memory.
  - `collection_name`: The name of the Qdrant collection (e.g., `"agent_memory"`).
  - `embedding_model`: The model used for generating embeddings.
  - `embedding_size`: The dimension of the embeddings.
  - `qdrant_url`: The URL of your Qdrant instance (e.g., `"http://localhost:6333"`).
  - `short_term_weight`, `long_term_weight`, `total_memories_to_retrieve`: Parameters for balancing memory retrieval.
- **`[agentic-tools.planner]`, `[agentic-tools.developer]`, `[agentic-tools.readme_writer]`, `[agentic-tools.approver]`**: Each agent has its own section to define:
  - `prompt`: The system prompt for the LLM.
  - `model_name`: The specific LLM model to use.
  - `api_key`: The *name* of the environment variable holding the API key.
  - `temperature`: LLM generation temperature.
  - `description`: A brief description of the agent's role.
  - `model_provider`: The LLM provider (e.g., `"google"`).
  - `skills`: A list of skills associated with the agent.

**Example snippet from `conf/agentic_tools.toml`:**

```toml
[agentic-tools.memory]
enabled = true
collection_name = "agent_memory"
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
embedding_size = 1024
qdrant_url = "http://localhost:6333"
short_term_weight = 0.7
long_term_weight = 0.3
total_memories_to_retrieve = 10

[agentic-tools.readme_writer]
prompt = """
* You are an expert technical writer.
* Create excellent, concise,and practical README documentation based on the project's source code, configuration, and conventions.
...
"""
model_name = "gemini-2.5-flash"
api_key = "GEMINI_API_KEY_README_WRITER"
temperature = 0.3
description = "Generates high-quality README documentation"
model_provider = "google"
skills = [
    "technical writing",
    "documentation",
    "readme creation",
    "information synthesis",
    "content organization",
    "clarity and precision"
]
```

## Usage

You can run the agents either through the `main.py` entry point (which registers them as MCP tools) or manually via `run_agents_manually.py` for direct testing. For initial onboarding, `run_agents_manually.py` is recommended.

### Running Agents Manually

The `run_agents_manually.py` script allows you to invoke individual agents with a specific chat message.

1. **Ensure Qdrant is running** (see Prerequisites).

1. **Set your API key environment variables** (see Prerequisites).

1. **Execute the script**:

   ```bash
   python run_agents_manually.py
   ```

   By default, `run_agents_manually.py` will execute the `readme_writer` agent with an "Update README" prompt. You can modify the `main` async function in `run_agents_manually.py` to test other agents or different prompts:

   ```python
   # run_agents_manually.py
   import asyncio
   from main import readme_writer_tool, approver_tool, developer_tool # Import the tools

   async def main():
       # Example: Run the readme_writer agent
       print("Running readme_writer_tool...")
       await readme_writer_tool("Provide an updated README file based on the recent changes, focusing on Qdrant memory integration.")

       # Example: Run the approver agent (uncomment to use)
       # print("
   ```

Running approver_tool...")
\# approval_result = await approver_tool("Review the recent code changes and provide an approval decision.")
\# print(f"Approval Result: {approval_result}")

```
    # Example: Run the developer agent (uncomment to use)
    # print("
```

Running developer_tool...")
\# dev_result = await developer_tool("Implement a new feature to log agent interactions to a file.")
\# print(f"Developer Result: {dev_result}")

````
if __name__ == "__main__":
    asyncio.run(main())
```
````

### Output

- The `readme_writer` agent, when executed, will directly update the `README.md` file in the project root with its generated content.
- Other agents like `approver` and `developer` will return their responses directly, which can be printed to the console or further processed.

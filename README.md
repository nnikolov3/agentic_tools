# Agentic Tools

An agentic toolchain for architecting, designing, validating, and approving code via chained AI agents. This project provides a flexible framework for building and orchestrating specialized agents that interact with your codebase, generate documentation, and enforce design principles using the Model Context Protocol (MCP).

## Key Features and Capabilities

- **Automated Documentation (`readme_writer`)**: Generates and updates `README.md` files by analyzing project source code and configuration, ensuring documentation stays current. The generated content is also stored in a Qdrant vector database for semantic memory.
- **Intelligent Code Review (`approver`)**: Audits code changes (via `git diff` patches) against predefined design principles and coding standards using Large Language Models (LLMs), providing structured feedback and approval decisions.
- **Code Generation (`developer`)**: Writes high-quality code adhering to specified design guidelines and coding standards, leveraging project context and design documents.
- **Modular Agent Architecture**: Leverages `FastMCP` to easily define and execute specialized AI agents with distinct responsibilities.
- **Context-Aware File Processing**: Recursively scans project directories, concatenates relevant files, and filters content based on extensions, size, and exclusion rules to provide precise context to LLMs.
- **LLM Integration**: Seamlessly integrates with Google's Gemini API for advanced natural language processing and decision-making.
- **Vector Database Integration (Qdrant)**: Stores generated documentation and other artifacts in a Qdrant vector database, serving as a semantic memory layer for efficient search and retrieval.
- **Configuration-Driven**: All agent behaviors, project scanning parameters, and LLM settings are managed through `conf/agentic_tools.toml` for easy customization.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python**: Version 3.9 or higher.
- **Git**: Installed and configured on your system.
- **Docker (Optional, Recommended for Qdrant)**: For easily running a local Qdrant vector database server.
- **Google API Keys**: Specific API keys for accessing Google's Gemini models. These must be set as environment variables:
  - `GOOGLE_API_KEY` (general access, if needed by `google-generativeai` client)
  - `GEMINI_API_KEY_README_WRITER`
  - `GEMINI_API_KEY_APPROVER`
  - `GEMINI_API_KEY_DEVELOPER`
  - `GEMINI_API_KEY_PLANNER` (if the `planner` agent is enabled)
- **Qdrant Server**: A running Qdrant instance. For local development, using Docker is the simplest approach. Refer to `docs/QDRANT.md` for more details on Qdrant integration.

## Installation

Follow these steps to get Agentic Tools up and running on your local machine.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Set up a Python Virtual Environment**:
   It's highly recommended to use a virtual environment to manage project dependencies.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate # On Windows, use `.venv\Scriptsctivate`
   ```

1. **Install Dependencies**:
   Install the required Python packages.

   ```bash
   pip install qdrant-client mdformat google-generativeai numpy fastmcp
   ```

1. **Set Environment Variables**:
   Obtain your Google API Keys and set them as environment variables. These are crucial for the LLM integrations.

   ```bash
   export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
   export GEMINI_API_KEY_README_WRITER="YOUR_README_WRITER_API_KEY"
   export GEMINI_API_KEY_APPROVER="YOUR_APPROVER_API_KEY"
   export GEMINI_API_KEY_DEVELOPER="YOUR_DEVELOPER_API_KEY"
   export GEMINI_API_KEY_PLANNER="YOUR_PLANNER_API_KEY" # If using the planner agent
   ```

   If you are using a remote Qdrant instance or a specific local path, you might need to configure these (defaults are usually sufficient for local Docker):

   ```bash
   export QDRANT_URL="http://localhost:6333" # Default for local Docker
   # export QDRANT_API_KEY="YOUR_QDRANT_CLOUD_API_KEY" # If using Qdrant Cloud
   ```

1. **Start Qdrant Server (Recommended)**:
   If you plan to use the Qdrant integration (e.g., for storing generated READMEs or agent memory), start a Qdrant server. The easiest way is with Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

   This command starts Qdrant, exposes its HTTP (6333) and gRPC (6334) ports, and mounts a local volume (`qdrant_storage`) for persistent data storage.

## Usage

The Agentic Tools project is designed to be run by invoking specific agents, which then utilize various internal tools to perform their tasks. The `main.py` script serves as the entry point, registering agents as tools for the Model Context Protocol (MCP).

### 1. Configure `conf/agentic_tools.toml`

Before running any agent, ensure your `conf/agentic_tools.toml` file is correctly configured. This file defines project paths, file filters, LLM settings, and agent-specific parameters. Refer to the [Configuration Details](#configuration-details) section for a comprehensive breakdown.

### 2. Running Agents

#### A. Start the MCP Server

The `main.py` script starts a `FastMCP` server, making the `readme_writer_tool`, `approver_tool`, and `developer` tools available for invocation by an MCP-compatible client (e.g., Claude, Cursor, VS Code).

```bash
python main.py
```

*Expected Output*: The console will indicate that the MCP server is running, listening for tool invocations. You would then use an MCP-compatible client to call these tools, for example:

- `readme_writer_tool("Provide an updated README file based on the project's current state.")`
- `approver_tool("Review the recent code changes and provide an approval decision.")`
- `developer("Implement a new feature to handle user authentication, adhering to design guidelines.")`

#### B. Direct Agent Execution (for development/testing)

For direct testing or development without an MCP client, you can use the convenience script `run_agents_manually.py` to call an agent's `run_agent` method.

To run the `readme_writer` and `approver` agents directly, as configured in `run_agents_manually.py`:

```bash
python run_agents_manually.py
```

*Expected Output for `readme_writer`*: A new or updated `README.md` file in your project root, and a confirmation message in the console. The content will also be stored in the Qdrant collection named `Agentic Tools_readme_writer`.

*Expected Output for `approver`*: A JSON response from the LLM, containing a `decision` (e.g., "APPROVED" or "CHANGES_REQUESTED"), `summary`, `positive_points`, `negative_points`, and `required_actions`. This requires pending `git` changes to generate a meaningful diff for the `approver` agent to review.

To run other agents, you would modify `run_agents_manually.py` to call the desired agent. For example, to run the `developer` agent:

```python
# Example modification in run_agents_manually.py to run the developer agent
import asyncio
import os
from pathlib import Path
from src.agents.agent import Agent
from src.configurator import Configurator

cwd = os.getcwd()
configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
configurator = Configurator(configuration_path)
configuration = configurator.get_config_dictionary()

async def developer_tool(chat: str):
    agent = Agent(configuration["agentic-tools"])
    return await agent.run_agent("developer", chat=chat)

# To run the developer agent:
developer_result = asyncio.run(developer_tool("Write a Python function to calculate the factorial of a number, adhering to coding standards."))
print(developer_result)
```

## Configuration Details

The project's behavior is primarily controlled by the `conf/agentic_tools.toml` file. This file centralizes settings for project paths, file filtering, LLM parameters, and agent-specific prompts. For a complete list of parameters, refer directly to the `conf/agentic_tools.toml` file.

### Key Sections and Parameters

- **`[agentic-tools]`**: Defines global project settings.
  - `project_name` (string): The name of your project.
  - `project_description` (string): A brief description of the project.
  - `design_docs` (list of strings): Paths to design documents (e.g., `docs/CODING_STANDARDS.md`) that provide contextual information to agents like `approver` and `developer`.
  - `project_directories` (list of strings): Directories to be recursively scanned for file content.
  - `include_extensions` (list of strings): File extensions to include during scanning (e.g., `".py", ".md", ".toml"`).
  - `exclude_directories` (list of strings): Directories to exclude from recursive scanning (e.g., `".git", "venv", ".venv"`).
  - `max_file_bytes` (integer): Maximum size of a file (in bytes) to be included in the LLM context.
  - `git_diff_command` (list of strings): The `git` command used to generate a patch for the `approver` agent.
- **`[agentic-tools.readme_writer]`**, **`[agentic-tools.approver]`**, **`[agentic-tools.developer]`**, **`[agentic-tools.planner]`**: These sections configure individual agents.
  - `prompt` (string): The specific prompt given to the LLM for the agent's task.
  - `model_name` (string): The LLM model to use for this agent (e.g., `"gemini-2.5-flash"`).
  - `api_key` (string): The environment variable name for the API key specific to this agent (e.g., `GEMINI_API_KEY_README_WRITER`).
  - `temperature` (float): Controls the randomness/creativity of the LLM's output (0.0-1.0).
  - `description` (string): A brief description of the agent's purpose.
  - `skills` (list of strings): Describes the capabilities of this agent.
  - `qdrant_embedding` (string): The name of the embedding model for Qdrant.
  - `embedding_size` (integer): The dimension of the vectors generated by the embedding model.

### Environment Variables

- `GOOGLE_API_KEY`: Your primary Google API key.
- `GEMINI_API_KEY_README_WRITER`, `GEMINI_API_KEY_APPROVER`, `GEMINI_API_KEY_DEVELOPER`, `GEMINI_API_KEY_PLANNER`: Specific API keys for each agent, as defined in `conf/agentic_tools.toml`.
- `QDRANT_URL`: (Optional) The URL of your Qdrant server. Defaults to `http://localhost:6333`.
- `QDRANT_API_KEY`: (Optional) An API key for Qdrant Cloud or authenticated Qdrant instances.

By carefully configuring these settings, you can tailor the Agentic Tools to fit your project's specific needs and workflows. For detailed information on Qdrant, refer to `docs/QDRANT.md`. For the design principles and coding standards that guide the `developer` and `approver` agents, see `docs/CODING_STANDARDS.md`.

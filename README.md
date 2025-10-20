# Agentic Tools

An agentic toolchain for architecting, designing, validating, and approving code via chained tools. This project provides a flexible framework for building and orchestrating AI agents that interact with your codebase, generate documentation, and enforce design principles.

## Key Features and Capabilities

- **Modular Agent Architecture**: Define and execute specialized AI agents (e.g., `readme_writer`, `approver`) with distinct responsibilities.
- **Automated Documentation Generation**: The `readme_writer` agent automatically generates comprehensive `README.md` files by analyzing project source code and configuration, ensuring up-to-date documentation.
- **Intelligent Code Review and Approval**: The `approver` agent leverages Large Language Models (LLMs) to audit code changes (via `git diff` patches) against predefined design principles and coding standards, providing structured feedback and approval decisions.
- **Context-Aware File Processing**: Recursively scans specified project directories, concatenates relevant files, and filters content based on file extensions, size limits, and exclusion rules to provide precise context to LLMs.
- **LLM Integration**: Seamlessly integrates with Google's Gemini API for advanced natural language processing, code generation, and decision-making tasks.
- **Vector Database Integration (Qdrant)**: Stores generated documentation and other artifacts in a Qdrant vector database, acting as a semantic memory layer for efficient search and retrieval by other agents.
- **Configuration-Driven**: All agent behaviors, project scanning parameters, and LLM settings are managed through a central `conf/agentic_tools.toml` file, allowing for easy customization.
- **Adherence to Coding Standards**: Designed with a strong emphasis on enforcing strict coding and design principles (as outlined in `docs/DESIGN_PRINCIPLES_GUIDE.md` and `docs/CODING_FOR_LLMs.md`), promoting high-quality, maintainable, and robust code.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python**: Version 3.9 or higher.
- **Git**: Installed and configured on your system.
- **Docker (Optional, Recommended for Qdrant)**: For easily running a local Qdrant vector database server.
- **Google API Key**: An API key for accessing Google's Gemini models. This should be set as an environment variable (e.g., `export GOOGLE_API_KEY="YOUR_API_KEY"`).
- **Qdrant Server**: A running Qdrant instance. For local development, using Docker is the simplest approach.

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
   source .venv/bin/activate  # On Windows, use `.venv\Scriptsctivate`
   ```

1. **Install Dependencies**:
   Install the required Python packages using pip.

   ```bash
   pip install qdrant-client mdformat google-generativeai numpy fastmcp
   ```

1. **Set Environment Variables**:
   Obtain your Google API Key and set it as an environment variable. This is crucial for the LLM integrations.

   ```bash
   export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
   ```

   If you are using a remote Qdrant instance or a specific local path, you might need to configure these (defaults are usually sufficient for local Docker):

   ```bash
   export QDRANT_URL="http://localhost:6333" # Default for local Docker
   # export QDRANT_API_KEY="YOUR_QDRANT_CLOUD_API_KEY" # If using Qdrant Cloud
   # export QDRANT_LOCAL_PATH="/path/to/qdrant/data" # If using persistent local Qdrant client
   ```

1. **Start Qdrant Server (Recommended for full functionality)**:
   If you plan to use the Qdrant integration (e.g., for storing generated READMEs or agent memory), start a Qdrant server. The easiest way is with Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

   This command starts Qdrant, exposes its HTTP (6333) and gRPC (6334) ports, and mounts a local volume (`qdrant_storage`) for persistent data storage.

## Usage

The Agentic Tools project is designed to be run by invoking specific agents, which then utilize various internal tools to perform their tasks. The `main.py` script serves as the entry point, registering agents as tools for the Model Context Protocol (MCP).

### 1. Configure `conf/agentic_tools.toml`

Before running any agent, ensure your `conf/agentic_tools.toml` file is correctly configured. This file defines project paths, file filters, LLM settings, and agent-specific parameters. Refer to the [Configuration Details](#configuration-details) section for a comprehensive breakdown.

### 2. Running an Agent

The `main.py` script uses `fastmcp` to register the `readme_writer_tool` and `approver_tool`. When `main.py` is executed, it starts an MCP server, making these tools available for interaction via an MCP client (e.g., Claude, Cursor, VS Code).

- **Start the MCP Server**:
  This command starts the `FastMCP` server, making the `readme_writer_tool` and `approver_tool` available for invocation by an MCP-compatible client.

  ```bash
  python main.py
  ```

  *Expected Output*: The console will indicate that the MCP server is running, listening for tool invocations. You would then use an MCP-compatible client to call `readme_writer_tool` or `approver_tool`.

- **Direct Agent Execution (for development/testing)**:
  For direct testing or development without an MCP client, you can use a script to call an agent's `run_agent` method. Create a file named `run_agent_directly.py` (or similar) in your project root with the following content:

  ```python
  import asyncio
  import logging
  from src.configurator import Configurator
  from src.agents.agent import Agent
  from pathlib import Path
  import os

  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  async def execute_agent_directly(agent_name: str):
      config_path = Path(os.getcwd()) / "conf" / "agentic_tools.toml"
      try:
          configurator = Configurator(config_path)
          full_config = configurator.get_config_dictionary()
          project_config = full_config.get("agentic-tools", {})

          if not project_config:
              logging.error(f"Could not find 'agentic-tools' section in {config_path}")
              return

          agent_instance = Agent(project_config)
          response = await agent_instance.run_agent(agent_name)

          logging.info(f"Agent '{agent_name}' completed.")
          if isinstance(response, str):
              print(f"
  ```

--- Agent '{agent_name}' Output ---
{response[:500]}...") # Print first 500 chars
else:
print(f"
--- Agent '{agent_name}' Response ---
{response}")

```
    except Exception as e:
        logging.error(f"Error running agent '{agent_name}': {e}", exc_info=True)

if __name__ == "__main__":
    print("Running README Writer agent directly...")
    asyncio.run(execute_agent_directly("readme_writer"))

    # To run the code approver (ensure you have pending git changes for a meaningful diff):
    # print("
```

Running Approver agent directly...")
\# asyncio.run(execute_agent_directly("approver"))
\`\`\`

````
To run this script:

```bash
python run_agent_directly.py
```

*Expected Output for `readme_writer`*: A new or updated `README.md` file in your project root, and a confirmation message in the console. The content will also be stored in the Qdrant collection named `Agentic Tools_readme_writer`.

*Expected Output for `approver`*: A JSON response from the LLM, containing a `decision` (e.g., "APPROVED" or "CHANGES_REQUESTED"), `summary`, `positive_points`, `negative_points`, and `required_actions`. This requires pending `git` changes to generate a meaningful diff for the `approver` agent to review.
````

## Configuration Details

The project's behavior is primarily controlled by the `conf/agentic_tools.toml` file. Here's a breakdown of key configuration parameters:

### `[agentic-tools]` Section

This section defines global project settings and paths.

- `project_name` (string): The name of your project. Used for Qdrant collection naming.
- `project_description` (string): A brief description of the project.
- `design_docs` (list of strings): Paths to design documents (e.g., `DESIGN_PRINCIPLES_GUIDE.md`) that provide contextual information to agents like the `approver`.
- `source` (list of strings): Directories containing primary source code.
- `project_root` (string): The root directory of the project (usually `"."`).
- `docs` (string): The directory where general documentation files are located.
- `tests_directory` (list of strings): Directories containing project tests.
- `project_directories` (list of strings): A list of directories to be recursively scanned for file content.
- `include_extensions` (list of strings): File extensions to include during scanning (e.g., `".py", ".md", ".toml"`). Only files with these extensions will be processed.
- `exclude_files` (list of strings): Specific file names to exclude from scanning (e.g., `["__init__.py"]`).
- `exclude_directories` (list of strings): Directories to exclude from recursive scanning (e.g., `[".git", "venv", ".venv", "__pycache__"]`).
- `max_file_bytes` (integer): Maximum size of a file (in bytes) to be included in the LLM context. Files larger than this will be skipped.
- `git_diff_command` (list of strings): The `git` command used to generate a patch for the `approver` agent.

### `[agentic-tools.readme_writer]` Section

This section configures the `readme_writer` agent.

- `prompt` (string): The specific prompt given to the LLM for generating the `README.md`. This is where you define the instructions for the technical writer agent.
- `model_name` (string): The LLM model to use for this agent (e.g., `"gemini-2.5-flash"`).
- `temperature` (float): Controls the randomness/creativity of the LLM's output (0.0-1.0). Lower values make output more deterministic.
- `description` (string): A brief description of the agent's purpose.
- `model_provider` (list of strings): The LLM provider(s) to use (e.g., `["google"]`).
- `alternative_model` (string): An alternative LLM model name for fallback or specific scenarios.
- `alternative_model_provider` (list of strings): The provider(s) for the alternative model.
- `skills` (list of strings): Describes the capabilities of this agent, used for internal context and routing.
- `qdrant_embedding` (string): The name of the embedding model to use for generating vector embeddings for Qdrant (e.g., `"all-MiniLM-L6-v2"`).
- `embedding_size` (integer): The dimension of the vectors generated by the embedding model.

### `[agentic-tools.approver]` Section

This section configures the `approver` agent.

- `prompt` (string): The specific prompt given to the LLM for the code approval process. This prompt typically includes design principles and review criteria.
- `model_name` (string): The LLM model to use for this agent (e.g., `"gemini-2.5-pro"`).
- `temperature` (float): Controls the randomness/creativity of the LLM's output.
- `description` (string): A brief description of the agent's purpose.
- `model_provider` (list of strings): The LLM provider(s) to use (e.g., `["google"]`).
- `alternative_model` (string): An alternative LLM model name for fallback or specific scenarios.
- `alternative_model_provider` (list of strings): The provider(s) for the alternative model.
- `qdrant_embedding` (string): The name of the embedding model to use for generating vector embeddings for Qdrant.
- `embedding_size` (integer): The dimension of the vectors generated by the embedding model.
- `skills` (list of strings): Describes the capabilities of this agent.

### Environment Variables

- `GOOGLE_API_KEY`: Your Google API key, essential for all LLM interactions.
- `QDRANT_URL`: (Optional) The URL of your Qdrant server. Defaults to `http://localhost:6333`.
- `QDRANT_API_KEY`: (Optional) An API key for Qdrant Cloud or authenticated Qdrant instances.

By carefully configuring these settings, you can tailor the Agentic Tools to fit your project's specific needs and workflows.

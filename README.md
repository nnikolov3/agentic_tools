# Multi-Agent-MCP: Agentic Toolchain for Automated Software Development

This project, **Multi-Agent-MCP**, is an advanced agentic toolchain built on the **FastMCP (Model Context Protocol)** framework. It enables automated software development workflows through a pipeline of specialized agents, handling tasks from design and validation to documentation and final approval.

## Key Features and Capabilities

*   **Multi-Agent Architecture**: Orchestrates a pipeline of intelligent agents, including an `Approver` (final gatekeeper) and a `ReadmeWriter` (documentation generator).
*   **Unified LLM API**: Provides a normalized interface for interacting with various Large Language Model (LLM) providers, including Google (Gemini/Vertex AI), Groq, Cerebras, and SambaNova.
*   **Provider Failover & Rate Limiting**: Automatically handles provider health, rate limits, and intelligent failover to ensure continuous operation and resilience.
*   **Semantic Memory Integration**: Leverages **Qdrant** for vector storage, enabling agents to store and retrieve contextual information, decisions, and generated artifacts for enhanced reasoning and consistency.
*   **Strict Quality Gates**: Enforces foundational design principles and coding standards through automated checks (type hinting, linting, formatting, test coverage).
*   **Python 3.11+**: Developed with modern Python, emphasizing strict type hinting and a modular, maintainable codebase.
*   **Configurable Context Assembly**: Dynamically gathers relevant source code, documentation, and project structure information to provide rich context to agents.

## Prerequisites

To run this project, you will need:

*   **Python 3.11+**
*   **Git**: For cloning the repository.
*   **Environment Variables**: API keys for the LLM providers you intend to use. At least one is required:
    *   `GEMINI_API_KEY` (for Google Gemini)
    *   `GROQ_API_KEY`
    *   `CEREBRAS_API_KEY`
    *   `SAMBANOVA_API_KEY`
*   **Qdrant (Optional but Recommended)**: For semantic memory features. Can run locally in-memory, persistently on disk, or connect to a remote server. If using local persistent storage, ensure sufficient disk space.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nnikolov3/multi-agent-mcp.git
    cd multi-agent-mcp
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    This project uses `qdrant-client[fastembed]` for local embedding inference.
    ```bash
    pip install -r requirements.txt
    # Or if you prefer using uv:
    # uv sync
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the project root and add your API keys:
    ```ini
    # .env
    GEMINI_API_KEY="your_gemini_api_key_here"
    GROQ_API_KEY="your_groq_api_key_here"
    CEREBRAS_API_KEY="your_cerebras_api_key_here"
    SAMBANOVA_API_KEY="your_sambanova_api_key_here"
    ```
    Then load them (e.g., using `dotenv` or by sourcing the file if it contains `export` statements).

## Usage

The project runs as a `FastMCP` server, exposing the agentic tools for interaction. You can interact with the server by running `main.py` and sending JSON requests to it.

### Running the Server

To run the server, execute the following command in your terminal:

```bash
python main.py
```

The server will start and listen for requests on standard input.

### Invoking the Tools

To invoke a tool, you need to send a JSON object to the running server with the tool name and its parameters.

#### Example: Generating a README with `readme_writer_tool`

To generate a `README.md` for the project, send the following JSON request to the server:

```json
{
  "tool": "readme_writer_tool"
}
```

The server will execute the tool and print the result to standard output.

#### Example: Approving Code Changes with `approver_tool`

To submit code for approval, first `touch` the files you want to be reviewed, then invoke the `approver_tool` with a `user_chat` message describing your changes.

1.  **Signal the files for review:**
    ```bash
    touch path/to/your/changed/file.py
    ```

2.  **Invoke the `approver_tool`:**
    Send the following JSON request to the server:

    ```json
    {
      "tool": "approver_tool",
      "user_chat": "I have refactored the authentication logic and added new tests."
    }
    ```

The server will then process your request and return a JSON object with the approval decision.


## Configuration

The project's behavior is primarily controlled by the `conf/mcp.toml` file.

```toml
# conf/mcp.toml

[multi-agent-mcp]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md", "AGENTS.md"]
source_code_directory = ["src"]
tests_directory=["tests"]
project_directories=["src/", "conf/", "docs/", "tests/"]
include_extensions = [".py", ".rs", ".go", ".ts", ".tsx", ".js", ".json", ".md", ".toml", ".yml", ".yaml"]
exclude_dirs = [".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]
recent_minutes = 10
max_file_bytes = 262144
max_total_bytes = 10485760

# Qdrant embedding model to vector size mappings
[embedding_model_sizes]
"sentence-transformers/all-MiniLM-L6-v2" = 384
# ... other models

[multi-agent-mcp.inference_providers]
providers = ["google", "groq", "cerebras", "sambanova"]

# Agent-specific configurations (e.g., readme_writer, approver)
[multi-agent-mcp.readme_writer]
prompt = "..."
model_name = "models/gemini-2.5-flash"
temperature = 0.3
model_providers = ["google"]
skills = ["technical writing", "documentation", ...]

[multi-agent-mcp.readme_writer.qdrant]
enabled = true
local_path = "/qdrant"
collection_name = "readme_generations"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

[multi-agent-mcp.approver]
prompt = "..."
model_name = "models/gemini-2.5-pro"
temperature = 0.1
model_providers = ["google"]
skills = ["code review", "quality assurance", ...]

[multi-agent-mcp.approver.qdrant]
enabled = true
local_path = "/qdrant"
collection_name = "approver_decisions"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

### Key Configuration Sections:

*   **`[multi-agent-mcp]`**: Global project settings.
    *   `project_name`, `project_description`: Overall project metadata.
    *   `design_docs`: List of paths to critical design documents.
    *   `source_code_directory`, `tests_directory`, `project_directories`: Directories to scan for code and context.
    *   `include_extensions`: File extensions to include when gathering source code.
    *   `exclude_dirs`: Directories to ignore during file scanning (e.g., `.git`, `venv`).
    *   `recent_minutes`: Time window for collecting recently modified files.
    *   `max_file_bytes`, `max_total_bytes`: Limits on file size and total context size.
    *   `embedding_model_sizes`: Maps embedding model names to their vector dimensions for Qdrant.
*   **`[multi-agent-mcp.inference_providers]`**: Defines the global priority order for LLM providers.
*   **`[multi-agent-mcp.<agent_name>]`**: Agent-specific configurations.
    *   `prompt`: The system prompt for the agent.
    *   `model_name`: The primary LLM model to use.
    *   `temperature`: Sampling temperature for LLM responses.
    *   `model_providers`: List of preferred providers for this agent's primary model.
    *   `alternative_model`, `alternative_model_provider`: Fallback model and providers.
    *   `skills`: A list of tags describing the agent's capabilities.
*   **`[multi-agent-mcp.<agent_name>.qdrant]`**: Qdrant integration settings for a specific agent.
    *   `enabled`: `true` to enable Qdrant storage for this agent.
    *   `local_path`: Path for local Qdrant persistent storage.
    *   `collection_name`: Qdrant collection name for this agent's data.
    *   `embedding_model`: The embedding model used for vectorizing data.
## Project Structure

```
multi-agent-mcp/
├── conf/
│   └── mcp.toml                  # Main project configuration
├── docs/
│   ├── AGENTIC_TOOLS_BEST_PRACTICES.md # Guidelines for agentic tools
│   ├── CODING_FOR_LLMs.md        # Coding standards for LLM-generated code
│   ├── DESIGN_PRINCIPLES_GUIDE.md# Foundational design principles
│   ├── FASTMCP.md                # FastMCP framework documentation
│   ├── PROMPT_ENGINEERING.md     # Prompt engineering strategies
│   ├── PROVIDERS_SDK.md          # LLM provider SDK documentation
│   └── QDRANT.md                 # Qdrant integration reference
├── src/
│   ├── .qwen/
│   │   └── PROJECT_SUMMARY.md    # Internal project summary
│   ├── __init__.py               # Python package initializer
│   ├── _api.py                   # Unified LLM API caller with failover
│   ├── approver.py               # Approver agent logic
│   ├── base_agent.py             # Base class for all agents
│   ├── configurator.py           # Configuration loading and validation
│   ├── mcp.log                   # Log file
│   ├── prompt_utils.py           # Utilities for prompt serialization
│   ├── qdrant_integration.py     # Qdrant client integration
│   ├── readme_writer_tool.py     # README generation agent logic
│   └── shell_tools.py            # Safe filesystem and git helpers
├── tests/
│   ├── conftest.py               # Pytest configuration
│   ├── test_api.py               # Tests for unified API caller
│   ├── test_approver.py          # Tests for Approver agent
│   ├── test_configurator.py      # Tests for Configurator
│   ├── test_google_integration.py# Live integration tests for Google API
│   ├── test_prompt_utils.py      # Tests for prompt utilities
│   ├── test_readme_writer_tool.py# Tests for ReadmeWriterTool
│   └── test_shell_tools.py       # Tests for shell tools
├── .env                          # Environment variables (e.g., API keys)
├── .gitignore                    # Git ignore rules
├── AGENTS.md                     # LLM Agents Rule Book
├── GEMINI.md                     # Gemini-specific notes (if any)
├── main.py                       # Main entry point (assumed)
├── mcp.log                       # Project-level log file
├── mypy.ini                      # Mypy configuration
├── README.md                     # This README file
├── requirements.txt              # Python dependencies
└── uv.lock                       # uv lock file for dependencies
```

## Contributing

Contributions are welcome! Please adhere to the following guidelines to maintain code quality and consistency:

1.  **Read the Design Principles**: Familiarize yourself with `docs/DESIGN_PRINCIPLES_GUIDE.md` and `docs/CODING_FOR_LLMs.md`. These documents outline the foundational principles and language-specific coding standards.
2.  **Methodical Problem Decomposition**: Break down complex problems into smaller, manageable subproblems.
3.  **Test-Driven Development (TDD)**: Write failing tests before implementing new features or bug fixes. Ensure total test coverage exceeds 80%.
4.  **Automated Quality Enforcement**: All code must pass the following checks with zero errors or warnings:
    *   **Type Checking**: `mypy .`
    *   **Linting**: `ruff check .`
    *   **Formatting**: `black --check .`
    *   **Testing**: `pytest -q`
5.  **Explicit Over Implicit**: Make all intentions, dependencies, and behaviors clear and visible in the code.
6.  **Self-Documenting Code**: Use intention-revealing names and comments to explain *why* code exists, not *what* it does.
7.  **Single Responsibility Principle**: Ensure each function, class, or module has one, and only one, reason to change.
8.  **Error Handling Excellence**: Handle all errors explicitly and immediately, providing clear context.

## License

This project is currently under development. Please refer to the `LICENSE` file in the repository root for specific licensing information. If no `LICENSE` file is present, it is recommended to add one.

# Agentic Tools: Multi-Agent Communication Protocol (MCP) System

## Project Description

The `multi-agent-mcp` project is an agentic toolchain designed for architecting, designing, validating, and approving code through a system of chained tools. It facilitates a robust software development pipeline by integrating multiple AI agents that interact via a Model Context Protocol (MCP). The system focuses on ensuring code quality, adherence to design principles, and efficient decision-making.

## Key Features and Capabilities

*   **Unified LLM API Caller**: A central `_api.py` module provides a normalized interface for interacting with various LLM providers, including Google/Gemini, Groq, Cerebras, and SambaNova.
*   **Provider Failover & Rate Limiting**: Automatically handles provider failures and respects API rate limits with built-in retry mechanisms and quota management for high availability.
*   **Agentic Tools**:
    *   **Approver Tool**: Acts as a final gatekeeper, reviewing code changes against design principles and coding standards, and making approval decisions (APPROVED/CHANGES_REQUESTED) with detailed feedback.
    *   **Readme Writer Tool**: Generates comprehensive, accurate, and useful `README.md` files by analyzing source code, configuration, and project structure, adhering to technical writing best practices.
*   **Qdrant Integration**: Utilizes Qdrant as a vector database for semantic memory, enabling storage and retrieval of agent decisions, patches, and other contextual information.
*   **Context Assembly**: Gathers relevant project context for LLMs, including recently modified source files, design documents, Git information, and project structure, with configurable filters and size limits.
*   **Configuration Management**: Centralized and validated configuration via `conf/mcp.toml` for agents, context policies, and LLM providers.
*   **Adherence to Design Principles**: Enforces foundational design principles (simplicity, explicit over implicit, single responsibility, TDD, etc.) and language-specific coding standards (Python, Go, Bash).
*   **Modular & Type-Safe**: Built with Python, featuring a modular architecture and extensive use of type hints for maintainability and robustness.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Python**: Version 3.9 or higher.
*   **uv**: A fast Python package installer and resolver. Install it via `pip install uv`.
*   **Git**: For cloning the repository.
*   **API Keys**: Environment variables for the LLM providers you intend to use. At least one is required for the system to function.
    *   `GEMINI_API_KEY` (for Google/Gemini)
    *   `GROQ_API_KEY`
    *   `CEREBRAS_API_KEY`
    *   `SAMBANOVA_API_KEY`
*   **Qdrant Client (Optional but Recommended)**: If you plan to use Qdrant for semantic memory (enabled by default for `approver` and `readme_writer` agents), install the client with `fastembed` for local embedding inference:
    ```bash
    pip install qdrant-client[fastembed]
    ```

## Installation

Follow these steps to set up the project locally:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/nnikolov3/multi-agent-mcp.git
    cd multi-agent-mcp
    ```

2.  **Create a Python Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    The project uses `uv` for dependency management.
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Set Environment Variables**:
    Create a `.env` file in the project root and add your LLM API keys:
    ```ini
    # .env example
    GEMINI_API_KEY="your_gemini_api_key_here"
    GROQ_API_KEY="your_groq_api_key_here"
    CEREBRAS_API_KEY="your_cerebras_api_key_here"
    SAMBANOVA_API_KEY="your_sambanova_api_key_here"
    ```
    Then load them (e.g., `source .env` or use a tool like `python-dotenv`).

## Usage

The `main.py` script serves as the entry point for running the agentic tools.

### Running the Readme Writer Tool

To generate a `README.md` for the project:

```bash
python main.py readme_writer
```

This command will analyze the project and output the generated README content. You can redirect this output to a file:

```bash
python main.py readme_writer > README.md
```

### Running the Approver Tool

To have the Approver agent review recent code changes:

```bash
python main.py approver --user-chat "Review the latest API refactoring in src/_api.py for adherence to design principles."
```

The Approver will return a JSON object indicating its decision (`APPROVED` or `CHANGES_REQUESTED`) along with a summary, positive/negative points, and required actions.

### General Agent Execution

Agents are configured in `conf/mcp.toml`. You can extend `main.py` or create custom scripts to orchestrate agents and their interactions.

## Configuration

The core configuration for the `multi-agent-mcp` system is managed through `conf/mcp.toml`.

### Key Configuration Sections:

*   **`[multi-agent-mcp]`**: Global project settings.
    *   `project_name`: The name of your project.
    *   `project_description`: A brief description of the project.
    *   `design_docs`: List of paths to foundational design documents.
    *   `source_code_directory`: Directories to include when collecting source code.
    *   `project_directories`: Directories to include for overall project structure analysis.
    *   `include_extensions`: File extensions to include in source code analysis (e.g., `.py`, `.md`).
    *   `exclude_dirs`: Directories to exclude from analysis (e.g., `.git`, `venv`).
    *   `recent_minutes`: How far back in time to look for recent file changes.
    *   `max_file_bytes`: Maximum bytes to read from a single file.
    *   `max_total_bytes`: Maximum total bytes for all collected context.
*   **`[embedding_model_sizes]`**: Maps embedding model names to their vector sizes, used by Qdrant.
*   **`[multi-agent-mcp.inference_providers]`**: Defines the global priority order for LLM providers.
*   **`[multi-agent-mcp.<agent_name>]`**: Agent-specific configurations.
    *   `prompt`: The system prompt for the agent.
    *   `model_name`: The primary LLM model to use.
    *   `temperature`: Sampling temperature for the LLM.
    *   `model_providers`: List of preferred providers for the primary model.
    *   `alternative_model`, `alternative_model_provider`: Fallback options.
    *   `skills`: A list of tags describing the agent's capabilities.
*   **`[multi-agent-mcp.<agent_name>.qdrant]`**: Qdrant integration settings for a specific agent.
    *   `enabled`: `true` or `false` to enable/disable Qdrant storage.
    *   `local_path`: Path for local Qdrant storage.
    *   `collection_name`: Qdrant collection name for this agent's data.
    *   `embedding_model`: The embedding model to use for Qdrant.

**Example `conf/mcp.toml` Snippet:**

```toml
# File: conf/mcp.toml

[multi-agent-mcp]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md"]
source_code_directory = ["src"]
project_directories=["src/", "conf/", "docs/", "tests/"]
include_extensions = [".py", ".md", ".toml", ".yml"]
exclude_dirs = [".git", "venv", "__pycache__"]
recent_minutes = 10
max_file_bytes = 262144
max_total_bytes = 10485760

[embedding_model_sizes]
"sentence-transformers/all-MiniLM-L6-v2" = 384

[multi-agent-mcp.readme_writer]
prompt = "You are an expert technical writer..."
model_name = "models/gemini-2.5-flash"
temperature = 0.3
model_providers = ["google"]
skills = ["technical writing", "documentation"]

[multi-agent-mcp.readme_writer.qdrant]
enabled = true
local_path = "/qdrant"
collection_name = "readme_generations"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

## Project Structure

```
multi-agent-mcp/
├── conf/
│   └── mcp.toml               # Main configuration file
├── docs/
│   ├── AGENTIC_TOOLS_BEST_PRACTICES.md # Best practices for agentic tools
│   ├── CODING_FOR_LLMs.md     # Coding standards for LLM-generated code
│   ├── DESIGN_PRINCIPLES_GUIDE.md # Foundational design principles
│   ├── FASTMCP.md             # FastMCP framework documentation
│   ├── PROMPT_ENGINEERING.md  # Prompt engineering strategies
│   ├── PROVIDERS_SDK.md       # LLM Provider SDK documentation
│   └── QDRANT.md              # Qdrant integration reference
├── src/
│   ├── _api.py                # Unified LLM API caller with failover and rate limits
│   ├── approver.py            # Approver agent logic
│   ├── base_agent.py          # Base class for all agents
│   ├── configurator.py        # Configuration loading and validation
│   ├── prompt_utils.py        # Utilities for prompt handling and response serialization
│   ├── qdrant_integration.py  # Qdrant vector database integration
│   ├── readme_writer_tool.py  # Readme Writer agent logic
│   └── shell_tools.py         # Filesystem and Git information gathering utilities
├── tests/
│   ├── conftest.py            # Pytest configuration
│   ├── test_api.py            # Tests for the unified API caller
│   ├── test_approver.py       # Tests for the Approver agent
│   ├── test_configurator.py   # Tests for the Configurator
│   ├── test_google_integration.py # Live integration tests for Google provider
│   ├── test_prompt_utils.py   # Tests for prompt utilities
│   ├── test_readme_writer_tool.py # Tests for the Readme Writer agent
│   └── test_shell_tools.py    # Tests for shell tools
├── .env                       # Environment variables (e.g., API keys)
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── main.py                    # Main entry point for running agents
├── mypy.ini                   # Mypy configuration for type checking
├── requirements.txt           # Project dependencies
└── uv.lock                    # uv lock file for reproducible installs

## Contributing

We welcome contributions to the `multi-agent-mcp` project! To ensure high quality and consistency, please adhere to the following guidelines:

1.  **Read Design Principles**: Familiarize yourself with `docs/DESIGN_PRINCIPLES_GUIDE.md` for foundational design principles and `docs/CODING_FOR_LLMs.md` for language-specific coding standards.
2.  **Test-Driven Development (TDD)**: All new features and bug fixes should be accompanied by comprehensive tests written *before* the implementation code.
3.  **Code Quality**:
    *   Use `black` for code formatting.
    *   Use `ruff` for linting.
    *   Use `mypy` for static type checking.
    *   Ensure all code passes these tools with zero errors or warnings. Suppressing linter warnings is strictly forbidden.
4.  **Git Workflow**:
    *   Fork the repository.
    *   Create a new branch for your feature or bug fix.
    *   Commit your changes with clear, descriptive messages.
    *   Push your branch and open a pull request.
5.  **Documentation**: Update relevant documentation (e.g., `README.md`, `docs/`) for any new features or significant changes.

## License

License information for this project is not explicitly provided in the current documentation. Please refer to the repository for any license files or contact the project maintainers for clarification.

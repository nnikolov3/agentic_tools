# GEMINI.md - Agentic Tools Framework

## Project Overview

This project is a Python-based "Agentic Tools Framework" designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages the `fastmcp` framework for command-line orchestration and integrates a robust Retrieval-Augmented Generation (RAG) system using Qdrant for contextual memory and knowledge retrieval.

The framework includes a variety of specialized agents, each designed for a specific task:

*   **`developer`**: Writes or refactors code in a specific file.
*   **`readme_writer`**: Updates the project's `README.md` file.
*   **`approver`**: Audits recent Git changes against design documents.
*   **`architect`**: Assists in architectural design and planning.
*   **`commentator`**: Adds comments and documentation to source code.

The project is configured using the `conf/agentic_tools.toml` file, which defines global settings for file filtering, memory connection, and specific parameters for each agent.

## Building and Running

### Prerequisites

*   Python 3.13+
*   Git
*   A running instance of the Qdrant vector database.

### Installation

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Agents

The agents can be run in two ways:

1.  **Via the `fastmcp` CLI (main.py):**

    This is the standard way to run the agents.

    *   **Update the README:**
        ```bash
        python main.py readme_writer_tool --chat "Generate a concise and practical README.md for the project."
        ```
    *   **Modify a source file:**
        ```bash
        python main.py developer_tool --filepath src/tools/api_tools.py --chat "Refactor the google method to use a more explicit try-except block."
        ```

2.  **Manually (run_agents_manually.py):**

    This script is useful for testing and direct agent invocation.

    *   **Run the commentator agent:**
        ```bash
        python run_agents_manually.py commentator --filepath src/agents/agent.py --chat "Add comments to this file."
        ```

### Ingesting Knowledge Bank Documents

The framework includes a pipeline for ingesting documents into the knowledge bank.

1.  Place your documents (e.g., `.pdf`, `.md`, `.json`) into the `knowledge_bank/` directory.
2.  Run the ingestion task:
    ```bash
    python run_agents_manually.py ingest_knowledge_bank
    ```

## Development Conventions

### Coding Standards

The project has a strong emphasis on code quality and adheres to a strict set of coding standards and design principles, which are documented in:

*   `docs/CODING_STANDARDS.md`
*   `docs/DESIGN_PRINCIPLES_GUIDE.md`

These documents emphasize:

*   Simplicity and readability.
*   Methodical problem decomposition.
*   Explicit over implicit code.
*   Self-documenting code.

### Static Analysis

The project uses the following static analysis tools to enforce code quality:

*   **`black`**: For code formatting.
*   **`ruff`**: For linting.
*   **`mypy`**: For static type checking.

The `mypy` configuration can be found in `mypy.ini`.

### Testing

The testing strategy for this project is currently under development. The `docs/CODING_STANDARDS.md` document specifies that the project should follow a Test-Driven Development (TDD) approach.

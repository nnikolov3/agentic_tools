# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages the `fastmcp` framework for command-line orchestration and integrates a robust Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), architectural planning (`architect`), and code commenting (`commentator`).
- **Time-Aware RAG System:** Utilizes Qdrant and FastEmbed for high-performance vector storage, retrieval, and cross-encoder reranking. Memory retrieval is time-bucketed (hourly, daily, monthly) to ensure agents receive contextually relevant project history.
- **Knowledge Bank Ingestion Pipeline:** A dedicated script for processing, chunking, embedding, and deduplicating documents (PDF, JSON, Markdown) into the vector database, including LLM-enhanced summarization for PDFs.
- **Code Quality Enforcement:** Code-modifying agents (`developer`, `commentator`) automatically validate generated code against static analysis tools (`black`, `ruff`, `mypy`) before writing to disk, ensuring file integrity.
- **Comprehensive Project Context:** Agents are automatically provided with project source code, Git diffs, repository metadata, and design documents during execution.
- **Atomic File Operations:** Ensures data integrity when writing or modifying files using safe, atomic write operations.

## Prerequisites

To run this project, you need the following installed:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Key:** An API key for the configured model provider (e.g., Google Gemini).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nnikolov3/agentic_tools.git
   cd agentic_tools
   ```

1. **Set up a Virtual Environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate # On Windows: .venv\Scriptsctivate
   ```

1. **Install Dependencies:**

   ```bash
   pip install fastmcp qdrant-client fastembed google-genai mdformat tenacity pdfminer.six
   ```

1. **Start Qdrant Service:**

   The memory system requires a running Qdrant instance. If using Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Usage

The project can be executed using the `fastmcp` runner (`main.py`) for standard tools, or the manual script (`run_agents_manually.py`) for direct agent invocation and knowledge ingestion.

### 1. Running Agents (via FastMCP CLI)

All agents accept a `--chat` (the main prompt/query) and an optional `--filepath` (the file to be modified or analyzed).

| Tool Name | Agent Type | Required Arguments | Purpose |
| :--- | :--- | :--- | :--- |
| `readme_writer_tool` | `ReadmeWriterAgent` | `--chat` | Updates the project's `README.md` file based on project context. |
| `developer_tool` | `DeveloperAgent` | `--chat`, `--filepath` | Writes or refactors code in a specific file. |
| `commentator_tool` | `CommentatorAgent` | `--chat`, `--filepath` | Adds comments and documentation to a source code file. |
| `approver_tool` | `DefaultAgent` | `--chat` | Audits recent Git changes against design documents and provides feedback. |
| `architect_tool` | `DefaultAgent` | `--chat` | Assists in architectural design and planning. |

**Example: Generating/Updating the README**

```bash
python main.py readme_writer_tool --chat "Generate a concise and practical README.md for the project, focusing on the Qdrant RAG system and agent orchestration."
```

**Example: Modifying a Source File**

```bash
python main.py developer_tool     --filepath src/tools/api_tools.py     --chat "Refactor the google method to use a more explicit try-except block for API key validation."
```

### 2. Ingesting Knowledge Bank Documents

The ingestion pipeline is run via the manual execution script.

1. Place your documents (e.g., `.pdf`, `.md`, `.json`) into the configured source directory (default: `knowledge_bank/`).

1. Run the ingestion task:

   ```bash
   python run_agents_manually.py ingest_knowledge_bank
   ```

## Configuration

The project is configured using the TOML file located at `conf/agentic_tools.toml`.

### Environment Variables

You must set the API key environment variable specified in your agent configuration. For the default Google provider, this is typically:

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
# If using the knowledge ingestion script, you may need a separate key:
export GEMINI_API_KEY_KNOWLEDGE_INGESTION="YOUR_API_KEY_HERE"
```

### `conf/agentic_tools.toml` Overview

The configuration file defines global settings for file filtering, memory connection, and specific parameters (model, prompt, API key reference) for each agent.

| Section | Key Parameters | Description |
| :--- | :--- | :--- |
| `[agentic-tools]` | `source`, `design_docs` | Defines directories to scan for source code context and paths to design documents. |
| `[memory]` | `qdrant_url`, `embedding_model` | Connection details for Qdrant and the model used for vector generation. |
| | `*retrieval_weight` | Weights defining the proportion of memories retrieved from different time buckets (hourly, daily, monthly, etc.). |
| `[knowledge_bank_ingestion]` | `source_directory`, `chunk_size` | Settings for the ingestion script, including where to find documents and how to chunk them. |
| `[<agent_name>]` | `model_provider`, `model_name`, `prompt`, `api_key` | Agent-specific settings defining the LLM provider (e.g., `google`), model, system instruction, and the environment variable name holding the API key. |

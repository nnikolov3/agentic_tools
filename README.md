# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages the `fastmcp` framework for command-line orchestration and integrates a robust Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), architectural planning (`architect`), code commenting (`commentator`), and knowledge retrieval (`expert`).
- **Time-Aware RAG System:** Utilizes Qdrant and FastEmbed for high-performance vector storage, retrieval, and cross-encoder reranking (`jinaai/jina-reranker-v2-base-multilingual`). Memory retrieval is time-bucketed (hourly, daily, weekly, etc.) to ensure agents receive contextually relevant project history.
- **Knowledge Bank Ingestion Pipeline:** A dedicated script for processing, chunking, embedding, and deduplicating documents (PDF, JSON, Markdown) into the vector database, including LLM-enhanced summarization for PDFs via the Google Gemini API.
- **Code Quality Enforcement:** Code-modifying agents (`developer`, `commentator`) automatically validate generated Python code against static analysis tools (`black`, `ruff`, `mypy`) before writing to disk, ensuring file integrity.
- **Comprehensive Context:** Agents are automatically provided with project source code, Git diffs, repository metadata, design documents, and memory context during execution.
- **Atomic File Operations:** Ensures data integrity when writing or modifying files using safe, atomic write operations.

## Prerequisites

To run this project, you need the following installed:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Key:** An API key for the configured model provider (currently Google Gemini).
- **Static Analysis Tools:** The validation service requires `black`, `ruff`, and `mypy` to be installed globally or in the environment.

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
python main.py developer_tool --filepath src/tools/api_tools.py --chat "Refactor the google method to use a more explicit try-except block for API key validation."
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

You must set the API key environment variables specified in your agent configuration.

```bash
# Example keys referenced in conf/agentic_tools.toml:
export GEMINI_API_KEY_DEVELOPER="YOUR_API_KEY_HERE"
export GEMINI_API_KEY_README_WRITER="YOUR_API_KEY_HERE"
export GEMINI_API_KEY_KNOWLEDGE_INGESTION="YOUR_API_KEY_HERE"
# ... and others for architect, approver, expert, commentator
```

### `conf/agentic_tools.toml` Overview

| Section | Key Parameters | Description |
| :--- | :--- | :--- |
| `[agentic-tools]` | `source`, `design_docs`, `include_extensions` | Defines directories to scan for source code context (`src`) and paths to design documents, along with file filtering rules (`.py`, `.md`, `.toml`). |
| `[agentic-tools.memory]` | `qdrant_url`, `embedding_model`, `device` | Connection details for Qdrant (`http://localhost:6333`) and the embedding model (`mixedbread-ai/mxbai-embed-large-v1`). |
| | `*retrieval_weight` | Weights defining the proportion of memories retrieved from different time buckets (e.g., `hourly_retrieval_weight`). |
| `[agentic-tools.memory.reranker]` | `enabled`, `model_name` | Configuration for the cross-encoder reranker (`jinaai/jina-reranker-v2-base-multilingual`). |
| `[knowledge_bank_ingestion]` | `source_directory`, `chunk_size`, `concurrency_limit` | Settings for the ingestion script, including document source (`../knowledge_bank`), chunking parameters (default: 1024/200), and concurrency limits (5). |
| `[<agent_name>]` | `model_provider`, `model_name`, `api_key` | Agent-specific settings defining the LLM provider (`google`), model (e.g., `gemini-2.5-pro`), and the environment variable name holding the API key. |
| `[agentic-tools.expert.memory]` | `knowledge_bank_retrieval_weight` | The `expert` agent is configured to rely 100% on the knowledge bank, ignoring time-based project memory. |

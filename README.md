# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages a robust, time-aware Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval, ensuring agents operate with the most relevant project history, documentation, and source code.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), static analysis (`linter_analyst`), knowledge management (`expert`, `knowledge_base_builder`), and configuration generation (`configuration_builder`).
- **Time-Aware RAG System:** Utilizes Qdrant, FastEmbed, and Jina Reranker for high-performance hybrid vector search (dense + sparse). Memory retrieval is segmented into time buckets (hourly, daily, weekly, etc.) and weighted to prioritize recent or highly relevant context.
- **Knowledge Bank Ingestion Pipeline:** A dedicated asynchronous script processes, chunks, embeds, and deduplicates documents (`.pdf`, `.json`, `.md`) into the vector database. It uses LLMs (Gemini API) to generate rich summaries for improved retrieval context.
- **Comprehensive Context Injection:** Agents are automatically provided with structured context, including project source code, file tree, Git diffs, repository metadata, and design documents.
- **Safe Code Modification:** Agents that modify code (`developer`, `commentator`) use atomic file write operations and clean LLM output by removing markdown fences, ensuring data integrity.
- **Linter Analysis:** The `linter_analyst` runs configured static analysis tools and uses an LLM to create a prioritized, standards-compliant analysis report.

## Prerequisites

To run this project, you need the following installed and configured:

1. **Python 3.13+**
1. **Git:** Must be installed and accessible in your system's PATH.
1. **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
1. **LLM API Keys:** API keys for the configured model provider (currently Google Gemini), set as environment variables (e.g., `GEMINI_API_KEY_DEVELOPER`).

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

   Install the required libraries, including the Qdrant client, embedding models, and LLM SDKs.

   ```bash
   pip install -e .
   ```

1. **Start Qdrant Service:**

   If using Docker (recommended for local setup):

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Usage

The framework is executed via the `main.py` CLI using the `run-agent` command.

| Agent Name | Purpose | Key Arguments |
| :--- | :--- | :--- |
| `developer` | Writes or refactors code. | `--chat` (task), `--filepath` (target file) |
| `readme_writer` | Generates or updates `README.md`. | `--chat` (prompt) |
| `linter_analyst` | Runs linters and analyzes the report. | `--chat` (optional focus) |
| `knowledge_base_builder` | Fetches content from URLs and saves it to a file. | `--chat` (comma-separated URLs), `--filepath` (output file) |
| `configuration_builder` | Generates a TOML configuration file by inspecting the project. | `--output-filename` (default: `generated_config.toml`) |
| `ingest_knowledge_bank` | Processes documents into the vector database. | (No arguments needed) |

### Example: Modifying Code

Use the `developer` agent to refactor a specific file:

```bash
python main.py run-agent developer     --chat "Refactor the Qdrant client initialization to use environment variables for the URL."     --filepath src/memory/qdrant_client_manager.py
```

### Example: Running Document Ingestion

Process all documents in the configured knowledge bank directory (default: `../bank`):

```bash
python main.py run-agent ingest_knowledge_bank
```

## Configuration

The entire project is configured via the TOML file located at `conf/agentic_tools.toml`.

### Environment Variables

You must export the API keys referenced in the agent configurations (e.g., `GEMINI_API_KEY_DEVELOPER`).

### Key Configuration Sections

| TOML Section | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `[agentic-tools]` | Global project settings, file filtering, and context files. | `source = ["src"]`, `design_docs`, `include_extensions` |
| `[agentic-tools.memory]` | Qdrant connection and embedding configuration. | `qdrant_url`, `embedding_model`, `device` (`cuda` or `cpu`), `total_memories_to_retrieve` |
| `[agentic-tools.memory.reranker]` | Enables and specifies the cross-encoder model for result re-ranking. | `enabled = true`, `model_name = "jinaai/jina-reranker-v2-base-multilingual"` |
| `[agentic-tools.memory]` (Weights) | Defines the priority of memory retrieval based on age. | `hourly_retrieval_weight`, `knowledge_bank_retrieval_weight` (default `1.0`) |
| `[agentic-tools.linters]` | Defines the shell commands for static analysis tools used by `linter_analyst`. | `ruff = ["ruff", "check", "src/"]`, `mypy = ["mypy", "."]` |
| `[knowledge_bank_ingestion]` | Settings for the document ingestion pipeline. | `source_directory = "../bank"`, `supported_extensions = [".json", ".md", ".pdf"]`, `chunk_size`, `concurrency_limit` |
| `[agentic-tools.<agent_name>]` | Agent-specific LLM settings. | `model_provider = "google"`, `model_name`, `api_key` (ENV var name), `prompt` |

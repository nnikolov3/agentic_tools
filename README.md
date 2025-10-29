# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages a robust, time-aware Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval, ensuring agents operate with the most relevant project history and documentation.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`linter_analyst`, `approver`), and knowledge management (`knowledge_base_builder`, `expert`).
- **Time-Aware RAG System:** Utilizes Qdrant, FastEmbed, and Jina Reranker for high-performance vector storage and retrieval. Memory retrieval is segmented into time buckets (hourly, daily, weekly, etc.) and weighted to prioritize recent or highly relevant context.
- **Knowledge Bank Ingestion Pipeline:** A dedicated asynchronous script for processing, chunking, embedding, and deduplicating documents (`.pdf`, `.json`, `.md`) into the vector database, including LLM-enhanced summarization via the Google Gemini API.
- **Comprehensive Context:** Agents are automatically provided with structured context, including project source code, file tree, Git diffs, repository metadata, and design documents.
- **Atomic File Operations:** Ensures data integrity when modifying source code or documentation using safe, atomic write operations.
- **Configuration-Driven:** Highly configurable via `conf/agentic_tools.toml` for models, providers, memory weights, and file filtering.

## Prerequisites

To run this project, you need the following installed and configured:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Keys:** API keys for the configured model provider (currently Google Gemini), set as environment variables (e.g., `GEMINI_API_KEY_DEVELOPER`).
- **Static Analysis Tools:** For the `linter_analyst` agent, ensure tools like `ruff`, `mypy`, `black`, and `bandit` are installed and accessible in your PATH.

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

   Install the required libraries, including the Qdrant client, embedding models, and LLM SDKs:

   ```bash
   pip install fastmcp qdrant-client fastembed google-genai mdformat tenacity pdfminer.six faker sentence-transformers httpx langchain-text-splitters beautifulsoup4
   ```

1. **Start Qdrant Service:**

   The memory system requires a running Qdrant instance. If using Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Usage

The framework supports two modes: running as a FastMCP server (default) or executing a single agent task via the CLI.

### 1. Running a Single Agent Task (CLI Mode)

Use the `run-agent` command to execute a specific agent manually.

| Agent Name | Purpose | Required Arguments |
| :--- | :--- | :--- |
| `readme_writer` | Generates or updates `README.md`. | `--chat` (prompt) |
| `developer` | Writes or refactors code. | `--chat` (task), `--filepath` (target file) |
| `linter_analyst` | Runs linters and analyzes the report. | `--chat` (optional focus) |
| `knowledge_base_builder` | Fetches content from URLs and saves it. | `--chat` (comma-separated URLs), `--filepath` (output file) |
| `ingest_knowledge_bank` | Processes documents into the vector database. | (No arguments needed) |

**Example: Building a Knowledge Base File from URLs**

```bash
python main.py run-agent knowledge_base_builder     --chat "https://docs.qdrant.tech/cloud/quickstart/, https://fastembed.ai/docs/usage/"     --filepath knowledge_bank/qdrant_docs.txt
```

**Example: Running the Document Ingestion Pipeline**

This processes files in the configured `source_directory` (default: `../bank`) into Qdrant.

```bash
python main.py run-agent ingest_knowledge_bank
```

## Configuration

The entire project is configured via the TOML file located at `conf/agentic_tools.toml`.

### Environment Variables

You must export the API keys referenced in the agent configurations:

```bash
export GEMINI_API_KEY_DEVELOPER="your_developer_key"
export GEMINI_API_KEY_README_WRITER="your_readme_key"
export GEMINI_API_KEY_KNOWLEDGE_INGESTION="your_ingestion_key"
# ... and others for architect, approver, expert, commentator
```

### `conf/agentic_tools.toml` Key Sections

| Section | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `[agentic-tools]` | Global project settings and file filtering. | `source = ["src"]`, `design_docs`, `include_extensions` |
| `[agentic-tools.memory]` | Qdrant connection and embedding configuration. | `qdrant_url`, `embedding_model`, `device`, `total_memories_to_retrieve` |
| | **Retrieval Weights** | `knowledge_bank_retrieval_weight = 1.0` (default to prioritize KB over time-based memory). |
| `[agentic-tools.linters]` | Defines the shell commands for static analysis tools. | `ruff = ["ruff", "check", "src/"]`, `mypy = ["mypy", "."]` |
| `[knowledge_bank_ingestion]` | Settings for the document ingestion pipeline. | `source_directory = "../bank"`, `chunk_size`, `concurrency_limit`, `google_api_key_name` |
| `[agentic-tools.<agent_name>]` | Agent-specific LLM settings. | `model_provider = "google"`, `model_name`, `api_key`, `prompt` |

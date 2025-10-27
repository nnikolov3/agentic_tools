# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages a robust, time-aware Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval, ensuring agents operate with the most relevant project history and documentation.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), architectural planning (`architect`), code commenting (`commentator`), and knowledge retrieval (`expert`).
- **Time-Aware RAG System:** Utilizes Qdrant, FastEmbed, and Jina Reranker for high-performance vector storage and retrieval. Memory retrieval is segmented into time buckets (hourly, daily, weekly, etc.) and weighted to prioritize recent or highly relevant context.
- **Knowledge Bank Ingestion Pipeline:** A dedicated script for processing, chunking, embedding, and deduplicating documents (PDF, JSON, Markdown) into the vector database, including LLM-enhanced summarization for PDFs via the Google Gemini API.
- **Code Quality Enforcement:** Code-modifying agents (`developer`, `commentator`) automatically validate generated Python code against static analysis tools (`black`, `ruff`, and `mypy`) before writing to disk, ensuring file integrity.
- **Comprehensive Context:** Agents are automatically provided with project source code, Git diffs, repository metadata, design documents, and memory context during execution.
- **Atomic File Operations:** Ensures data integrity when writing or modifying files using safe, atomic write operations.

## Prerequisites

To run this project, you need the following installed:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Keys:** API keys for the configured model provider (currently Google Gemini), set as environment variables.
- **Static Analysis Tools:** The validation service requires `black`, `ruff`, and `mypy` to be installed and accessible in your environment's PATH.

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

The project uses a unified `main.py` entry point, supporting both FastMCP tool invocation and a manual CLI mode for running specific agents or workflows.

### 1. Running Agents (via FastMCP CLI)

Agents are invoked using their registered tool names. Most agents accept a primary prompt (`--chat`) and an optional target file (`--filepath`).

| Tool Name | Agent Type | Required Arguments | Purpose |
| :--- | :--- | :--- | :--- |
| `readme_writer_tool` | `ReadmeWriterAgent` | `--chat` | Generates or updates the project's `README.md` based on project context. |
| `developer_tool` | `DeveloperAgent` | `--chat`, `--filepath` | Writes or refactors code in a specific file, with validation. |
| `commentator_tool` | `CommentatorAgent` | `--chat`, `--filepath` | Adds documentation, docstrings, and organizes imports in a source file. |
| `approver_tool` | `DefaultAgent` | `--chat` | Audits recent Git changes (`git diff`) against design documents. |
| `architect_tool` | `DefaultAgent` | `--chat` | Assists in high-level architectural design and planning. |

**Example: Generating/Updating the README**

```bash
python main.py readme_writer_tool --chat "Generate a concise and practical README.md for the project, focusing on the Qdrant RAG system and agent orchestration."
```

**Example: Modifying a Source File**

```bash
python main.py developer_tool --filepath src/tools/api_tools.py --chat "Refactor the google method to use a more explicit try-except block for API key validation."
```

### 2. Ingesting Knowledge Bank Documents

The ingestion pipeline is run using the `run-agent` command:

1. Place documents (`.pdf`, `.md`, `.json`) into the configured source directory (default: `knowledge_bank/`).

1. Run the ingestion task:

   ```bash
   python main.py run-agent ingest_knowledge_bank
   ```

## Configuration

The project is configured using the TOML file located at `conf/agentic_tools.toml`.

### Environment Variables

You must set the API key environment variables specified in the agent configurations.

```bash
export GEMINI_API_KEY_DEVELOPER="YOUR_API_KEY_HERE"
export GEMINI_API_KEY_README_WRITER="YOUR_API_KEY_HERE"
export GEMINI_API_KEY_KNOWLEDGE_INGESTION="YOUR_API_KEY_HERE"
# ... and others for architect, approver, expert, commentator
```

### `conf/agentic_tools.toml` Key Sections

| Section | Key Parameters | Default Values/Details |
| :--- | :--- | :--- |
| `[agentic-tools]` | `source`, `design_docs`, `include_extensions` | Defines directories to scan (`src`), design document paths, and file filters (`.py`, `.md`, `.toml`). |
| `[agentic-tools.memory]` | `qdrant_url`, `embedding_model`, `device` | Qdrant connection (`http://localhost:6333`), embedding model (`mixedbread-ai/mxbai-embed-large-v1`), and processing device (`cuda`/`cpu`). |
| | `*retrieval_weight` | Weights (0.0 to 1.0) defining the proportion of memories retrieved from time buckets (e.g., `hourly_retrieval_weight`). |
| `[agentic-tools.memory.reranker]` | `enabled`, `model_name` | Enables and specifies the cross-encoder reranker (`jinaai/jina-reranker-v2-base-multilingual`). |
| `[knowledge_bank_ingestion]` | `source_directory`, `chunk_size`, `concurrency_limit` | Source directory (`../knowledge_bank`), chunking parameters (1024/200), and max concurrent file processing (5). |
| `[<agent_name>]` | `model_provider`, `model_name`, `api_key` | Agent-specific LLM settings (e.g., `google`, `gemini-2.5-pro`), and the environment variable name for the API key. |
| `[agentic-tools.expert.memory]` | `knowledge_bank_retrieval_weight` | Overrides global memory weights for the `expert` agent, typically set to `1.0` to rely solely on the knowledge bank. |

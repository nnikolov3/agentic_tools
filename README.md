# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development tasks using specialized, context-aware AI agents. It leverages a robust, time-aware Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval, ensuring agents operate with the most relevant project history and documentation.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), architectural planning (`architect`), code commenting (`commentator`), knowledge retrieval (`expert`), web content ingestion (`knowledge_base_builder`), and automated code quality analysis (`linter_analyst`).
- **Time-Aware RAG System:** Utilizes Qdrant, FastEmbed, and Jina Reranker for high-performance vector storage and retrieval. Memory retrieval is segmented into time buckets (hourly, daily, weekly, etc.) and weighted to prioritize recent or highly relevant context.
- **Knowledge Bank Ingestion Pipeline:** A dedicated script for processing, chunking, embedding, and deduplicating documents (PDF, JSON, Markdown) into the vector database, including LLM-enhanced summarization for complex file types via the Google Gemini API.
- **Code Quality Enforcement (Post-Write):** Code-modifying agents (`developer`, `commentator`) now rely on the `linter_analyst` agent for post-write validation. This architectural change ensures the base agent remains language-agnostic, while the `linter_analyst` provides comprehensive, language-specific quality checks after the code is written.
- **Comprehensive Context:** Agents are automatically provided with project source code, Git diffs, repository metadata, design documents, and memory context during execution.
- **Linter Analysis:** The `linter_analyst` agent runs configured project linters (`ruff`, `mypy`, `black`, `bandit`, `pylint`) and uses an LLM to generate a prioritized analysis report.
- **Atomic File Operations:** Ensures data integrity when writing or modifying files using safe, atomic write operations.

## Prerequisites

To run this project, you need the following installed:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Keys:** API keys for the configured model provider (currently Google Gemini), set as environment variables (e.g., `GEMINI_API_KEY_DEVELOPER`).
- **Static Analysis Tools:** The configured linters must be installed and accessible in your environment's PATH: `black`, `ruff`, `mypy`, `bandit`, and `pylint`.

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
   # Install core dependencies and required linters
   pip install fastmcp qdrant-client fastembed google-genai mdformat tenacity pdfminer.six         faker sentence-transformers httpx langchain-text-splitters beautifulsoup4         black ruff mypy bandit pylint
   ```

1. **Start Qdrant Service:**

   The memory system requires a running Qdrant instance. If using Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Usage

The project uses a unified `main.py` entry point, supporting both FastMCP tool invocation and a manual CLI mode for running specific agents or workflows.

### 1. Running Agents (via FastMCP CLI)

Agents are invoked using their registered tool names via `python main.py <tool_name>`.

| Tool Name | Agent Type | Required Arguments | Purpose |
| :--- | :--- | :--- | :--- |
| `readme_writer_tool` | `ReadmeWriterAgent` | `--chat` | Generates or updates the project's `README.md` based on project context. |
| `developer_tool` | `DeveloperAgent` | `--chat`, `--filepath` | Writes or refactors code in a specific file, with validation. |
| `linter_analyst_tool` | `LinterAnalystAgent` | (None) | Runs all configured linters and provides an LLM-generated analysis report. |
| `knowledge_base_builder_tool` | `KnowledgeBaseAgent` | `--chat` (URLs), `--filepath` | Fetches content from a comma-separated list of URLs and saves the concatenated result to a specified file. |

**Example: Building a Knowledge Base File from URLs**

```bash
python main.py knowledge_base_builder_tool     --chat "https://docs.qdrant.tech/cloud/quickstart/, https://fastembed.ai/docs/usage/"     --filepath knowledge_bank/qdrant_and_fastembed_docs.txt
```

**Example: Running Linter Analysis**

```bash
python main.py linter_analyst_tool --chat "Prioritize the top 3 critical fixes for the developer."
```

### 2. Manual CLI Mode (`run-agent`)

The `run-agent` command is used for executing specific agents or internal scripts, such as the knowledge bank ingestion pipeline.

**Ingesting Knowledge Bank Documents**

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
# ... and others for architect, approver, expert, commentator, linter_analyst
```

### `conf/agentic_tools.toml` Key Sections

| Section | Key Parameters | Details |
| :--- | :--- | :--- |
| `[agentic-tools]` | `source`, `design_docs`, `include_extensions` | Defines directories to scan (`src`), paths to design documents, and file extensions to include (`.py`, `.md`, `.toml`). |
| `[agentic-tools.memory]` | `qdrant_url`, `embedding_model`, `device` | Qdrant connection (`http://localhost:6333`), embedding model (`mixedbread-ai/mxbai-embed-large-v1`), and processing device (`cuda`/`cpu`). |
| | `knowledge_bank_retrieval_weight` | Set to `1.0` by default, prioritizing retrieval from the dedicated knowledge bank over time-decayed agent memory. |
| `[agentic-tools.linters]` | `ruff`, `mypy`, `black`, `bandit`, `pylint` | Defines the exact command and arguments for each static analysis tool run by the `linter_analyst` agent. |
| `[knowledge_bank_ingestion]` | `source_directory`, `chunk_size`, `concurrency_limit` | Source directory (`../knowledge_bank`), chunking parameters (1024/200), and max concurrent file processing (5). |
| `[<agent_name>]` | `model_provider`, `model_name`, `api_key` | Agent-specific LLM settings (e.g., `google`, `gemini-2.5-pro`), and the environment variable name for the API key. |
| `[agentic-tools.expert.memory]` | `knowledge_bank_retrieval_weight` | Overrides global memory weights for the `expert` agent, typically set to `1.0` to rely solely on the knowledge bank for answers. |

# Agentic Tools Framework

The Agentic Tools Framework is a sophisticated system designed to automate complex software development and documentation tasks using specialized AI agents. It leverages the `fastmcp` framework for command-line orchestration and integrates a robust Retrieval-Augmented Generation (RAG) system for contextual memory and knowledge retrieval.

## Key Features

- **Specialized Agents:** Includes dedicated agents for development (`developer`), documentation (`readme_writer`), code review (`approver`), architectural planning (`architect`), and code commenting (`commentator`).
- **Contextual RAG System:** Utilizes Qdrant and FastEmbed for high-performance vector storage, retrieval, and cross-encoder reranking, ensuring agents receive highly relevant project history and knowledge context.
- **Knowledge Bank Ingestion:** A dedicated pipeline for processing, chunking, embedding, and deduplicating documents (PDF, JSON, Markdown) into the vector database.
- **Comprehensive Project Context:** Agents are automatically provided with project source code, Git diffs, repository metadata, and design documents during execution.
- **Atomic File Operations:** Ensures data integrity when writing or modifying files using safe, atomic write operations.
- **Configurable LLM Backend:** Integrated with Google Generative AI (Gemini) via a flexible API abstraction layer.

## Prerequisites

To run this project, you need the following installed:

- **Python 3.13+**
- **Git** (must be accessible in your system's PATH)
- **Qdrant Service:** A running instance of the Qdrant vector database (default URL: `http://localhost:6333`).
- **LLM API Key:** An API key for the configured model provider (e.g., Google Gemini).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Set up a Virtual Environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate # On Windows: .venv\Scriptsctivate
   ```

1. **Install Dependencies:**

   Install the core libraries:

   ```bash
   pip install fastmcp qdrant-client fastembed google-genai mdformat tenacity
   ```

1. **Start Qdrant Service:**

   The memory system requires a running Qdrant instance. If using Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Usage

The project is executed using the `fastmcp` runner via `python main.py`. The available agents are exposed as command-line tools.

### 1. Running Agents

All agents accept a `--chat` (the main prompt/query) and an optional `--filepath` (the file to be modified or analyzed).

| Tool Name | Agent Type | Required Arguments | Purpose |
| :--- | :--- | :--- | :--- |
| `readme_writer_tool` | `ReadmeWriterAgent` | `--chat` | Updates the project's `README.md` file based on project context. |
| `developer_tool` | `DeveloperAgent` | `--chat`, `--filepath` | Writes or refactors code in a specific file. |
| `commentator_tool` | `CommentatorAgent` | `--chat`, `--filepath` | Adds comments and documentation to a source code file. |
| `approver_tool` | `DefaultAgent` | `--chat` | Audits recent Git changes against design documents and provides feedback. |
| `architect_tool` | `DefaultAgent` | `--chat` | Assists in architectural design and planning. |

**Example 1: Generating/Updating the README**

```bash
python main.py readme_writer_tool --chat "Generate a concise and practical README.md for the project, focusing on the Qdrant RAG system and agent orchestration."
```

**Example 2: Modifying a Source File**

The `developer_tool` requires a `--filepath` to specify which file to modify.

```bash
python main.py developer_tool     --filepath src/tools/api_tools.py     --chat "Refactor the google method to use a more explicit try-except block for API key validation."
```

### 2. Ingesting Knowledge Bank Documents

To populate the RAG system with project documentation and external knowledge:

1. Place your documents (e.g., `.pdf`, `.md`, `.json`) into the configured source directory (default: `knowledge_bank/`).

1. Run the ingestion script:

   ```bash
   python scripts/ingest_knowledge_bank.py
   ```

## Configuration

The project is configured using the TOML file located at `conf/agentic_tools.toml`.

### Environment Variables

You must set the API key environment variable specified in your agent configuration. For the default Google provider, this is typically:

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### `conf/agentic_tools.toml` Overview

The configuration file defines global settings for file filtering, memory connection, and specific parameters (model, prompt, API key reference) for each agent.

| Section | Key Parameters | Description |
| :--- | :--- | :--- |
| `[agentic-tools]` | `source`, `design_docs` | Defines directories to scan for context and paths to design documents. |
| `[memory]` | `qdrant_url`, `embedding_model` | Connection details for Qdrant and the model used for vector generation. |
| | `*retrieval_weight` | Weights defining the proportion of memories retrieved from different time buckets (hourly, daily, monthly, etc.). |
| `[knowledge_bank_ingestion]` | `source_directory`, `chunk_size` | Settings for the ingestion script, including where to find documents and how to chunk them. |
| `[<agent_name>]` | `model_provider`, `model_name`, `prompt`, `api_key` | Agent-specific settings defining the LLM provider (e.g., `google`), model, system instruction, and the environment variable name holding the API key. |

# multi-agent-mcp

This project implements a multi-agent system using the Model Context Protocol (MCP) framework. It features specialized agents enhanced with a robust, multi-source memory system powered by Qdrant. This system includes dynamic agent memory (short/long-term) and a static, LLM-processed **Knowledge Bank** for deep contextual retrieval, ensuring agents operate with comprehensive, context-aware information.

## Key Features

- **Four-Part Contextual Memory**: Agents retrieve context using a weighted search across four distinct sources: Today's memories, Monthly memories, Long-Term memories (all from `agent_memory`), and the dedicated `knowledge-bank` collection.
- **Knowledge Bank Ingestion Pipeline**: A dedicated, idempotent script (`scripts/ingest_knowledge_bank.py`) processes external documents (e.g., PDFs, Markdown, JSON) and stores them in Qdrant.
- **Differentiated Ingestion**: Supports two paths:
  - **LLM Processing**: Complex files (e.g., `.pdf`) are sent to the Gemini API for high-quality summarization/rewriting before embedding.
  - **Direct Ingestion**: Text-based files (e.g., `.md`, `.json`) are read directly using robust file handling.
- **Idempotent Data Management**: The ingestion process uses content hashing to prevent duplicate entries, ensuring data integrity even if the script is run multiple times.
- **Modular Agent Orchestration**: Agent-specific logic (like post-processing the final `README.md` and writing it to disk) is correctly centralized within the `Agent` orchestrator class, keeping core `Tool` functionality generic.

## Prerequisites

Before installation, ensure the following are set up:

- **Python**: Version `3.13` or higher.
- **uv**: A fast Python package installer and resolver.
  ```bash
  pip install uv
  ```
- **Git**: Required for cloning and for agents to generate project context (diffs, info).
- **Qdrant Server**: The project requires a running Qdrant instance for vector memory.
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```
- **Environment Variables**: API keys for the LLM providers must be set.
  - `GEMINI_API_KEY_PLANNER`
  - `GEMINI_API_KEY_DEVELOPER`
  - `GEMINI_API_KEY_README_WRITER`
  - `GEMINI_API_KEY_APPROVER`
  - `GEMINI_API_KEY_KNOWLEDGE_INGESTION` (Required for the ingestion script)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Install dependencies** (using `uv`):

   ```bash
   uv sync
   ```

## Configuration

The project's memory and ingestion settings are defined in `conf/agentic_tools.toml`.

### Memory Configuration (`[agentic-tools.memory]`)

This section defines the Qdrant connection and the weighted retrieval strategy.

| Key | Default Value | Description |
| :--- | :--- | :--- |
| `qdrant_url` | `"http://localhost:6333"` | Address of the Qdrant server. |
| `collection_name` | `"agent_memory"` | Collection for dynamic, time-based agent memories. |
| `knowledge_bank` | `"knowledge-bank"` | Collection for static, ingested documents. |
| `embedding_model` | `"mixedbread-ai/mxbai-embed-large-v1"` | Model used for vector generation. |
| `total_memories_to_retrieve` | `20` | Maximum number of memory points to retrieve across all four sources. |
| `today_retrieval_weight` | `0.4` | Weight for recent (last 24 hours) memories. |
| `monthly_retrieval_weight` | `0.2` | Weight for memories from the last 30 days. |
| `long_term_retrieval_weight` | `0.2` | Weight for older memories. |
| `knowledge_bank_retrieval_weight` | `0.2` | Weight for context retrieved from the Knowledge Bank. |

### Knowledge Bank Ingestion Configuration (`[agentic-tools.knowledge_bank_ingestion]`)

This section controls how documents are processed and ingested.

| Key | Default Value | Description |
| :--- | :--- | :--- |
| `source_directory` | `"knowledge_bank"` | The local directory where source documents are placed for ingestion. |
| `google_api_key_name` | `"GEMINI_API_KEY_KNOWLEDGE_INGESTION"` | The environment variable name holding the API key for LLM processing. |
| `model` | `"gemini-flash-latest"` | The LLM model used for summarization/rewriting. |
| `supported_extensions` | `[".pdf", ".json", ".md"]` | File types the script will attempt to process. |
| `llm_processed_extensions` | `[".pdf"]` | Extensions that *must* be sent to the LLM for processing (e.g., complex files). All others are read directly. |
| `prompt` | (Detailed prompt) | The system prompt used by the LLM to summarize/rewrite documents. |

## Usage

### 1. Ingesting Documents into the Knowledge Bank

To populate the Knowledge Bank for long-term context, use the dedicated ingestion script.

1. **Create Source Directory**: Create the directory specified in the configuration (default: `knowledge_bank`).

   ```bash
   mkdir knowledge_bank
   ```

1. **Add Documents**: Place your `.pdf`, `.md`, or `.json` files inside the `knowledge_bank` directory.

1. **Run Ingestion**: Execute the `run_agents_manually.py` script, which is configured to run the ingestion pipeline by default.

   ```bash
   python run_agents_manually.py
   ```

   The script will:

   - Ensure the `knowledge-bank` Qdrant collection exists.
   - Iterate through files, calculate content hashes, and skip already processed files (idempotency).
   - Send `.pdf` files to the Gemini API for summarization/rewriting.
   - Embed the resulting text and store it in Qdrant.

### 2. Running Agents Manually

You can test the agents and observe how they utilize the newly ingested memory.

The `run_agents_manually.py` script can be modified to run any agent with a specific prompt.

```python
# run_agents_manually.py (Example modification)
import asyncio
from main import readme_writer_tool, approver_tool, developer_tool, knowledge_bank

async def main():
    # 1. Run Ingestion (Default action)
    await knowledge_bank.run_ingestion()

    # 2. Example: Run the readme_writer agent
    print("\nRunning readme_writer_tool...")
    # This agent will now retrieve context from both agent_memory and knowledge-bank
    await readme_writer_tool("Update the README to clearly explain the new four-part memory retrieval strategy and the knowledge bank ingestion process.")

    # 3. Example: Run the developer agent (uncomment to use)
    # print("\nRunning developer_tool...")
    # dev_result = await developer_tool("Implement a new utility function in src/utils.py to calculate the SHA256 hash of a file's binary content.")
    # print(f"Developer Result: {dev_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**

The `readme_writer` agent will analyze the project context, retrieve relevant memories (including any ingested knowledge), and write its final, formatted output directly to the project's `README.md` file.

```
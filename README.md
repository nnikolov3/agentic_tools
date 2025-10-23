Agentic Tools

This project is a robust, modular multi-agent system designed to automate complex software development and documentation tasks. It leverages specialized agents powered by Google's Gemini models, coupled with an optimized, persistent memory system built on the Qdrant vector database.

## Key Features

- **Specialized Agents:** Includes dedicated agents for high-level planning (`architect`), implementation (`developer`), and documentation (`readme_writer`).
- **Context-Aware Memory:** Utilizes Qdrant for persistent memory storage, employing a sophisticated, weighted retrieval strategy (Today, Monthly, Long-Term, Knowledge Bank) to provide highly relevant context to agents.
- **Advanced Retrieval:** Integrates **FastEmbed** for high-performance vector embeddings (`mixedbread-ai/mxbai-embed-large-v1`) and a **cross-encoder reranker** (`jinaai/jina-reranker-v2-base-multilingual`) to boost the relevance of retrieved memories.
- **Knowledge Ingestion Pipeline:** An automated script to process external documents (`.md`, `.json`, `.pdf`) into the knowledge bank, using the Gemini API to summarize complex documents (like PDFs) before chunking and embedding.
- **Atomic Operations:** Ensures data integrity through atomic file writes and idempotent knowledge ingestion (prevents reprocessing unchanged files via content hashing).

## Prerequisites

To run this project, you need the following installed and configured:

1. **Python 3.10+**
1. **Qdrant Vector Database:** A running instance of Qdrant. The system is configured to connect to `http://localhost:6333`.
1. **Google Gemini API Keys:** You must set the following environment variables, as they are required by the agents and the ingestion pipeline:

| Environment Variable | Agent/Purpose |
| :--- | :--- |
| `GEMINI_API_KEY_ARCHITECT` | Used by the Architect agent. |
| `GEMINI_API_KEY_DEVELOPER` | Used by the Developer agent. |
| `GEMINI_API_KEY_README_WRITER` | Used by the Readme Writer agent. |
| `GEMINI_API_KEY_KNOWLEDGE_INGESTION` | Used for summarizing documents during ingestion. |
| `GEMINI_API_KEY_APPROVER` | Used by the final review agent. |

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Set up Qdrant:**
   Start Qdrant, typically using Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```


1. **Set Environment Variables:**
   Export your Gemini API keys in your shell session:

   ```bash
   export GEMINI_API_KEY_ARCHITECT="YOUR_KEY_HERE"
   # ... and so on for all five required keys
   ```

## Configuration Details

All core settings are managed in the `conf/agentic_tools.toml` file.

### Memory and Vector Database (`[agentic-tools.memory]`)

The memory configuration is highly optimized for performance and stability:

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `qdrant_url` | `http://localhost:6333` | Qdrant connection endpoint. |
| `embedding_model` | `mixedbread-ai/mxbai-embed-large-v1` | Model used for vector generation. |
| `timeout` | `60.0` | Connection timeout for Qdrant operations (fixes `ConnectTimeout` errors). |
| `total_memories_to_retrieve` | `20` | Total number of points retrieved across all memory categories. |
| `today_retrieval_weight` | `0.4` | Weight for recent (last 24 hours) memories. |
| `reranker.enabled` | `true` | Enables the FastEmbed cross-encoder reranker for improved search quality. |

### Knowledge Ingestion (`[agentic-tools.knowledge_bank_ingestion]`)

This section controls how external files are processed:

| Setting | Default Value | Description |
| :--- | :--- | :--- | |
| `source_directory` | `knowledge_bank` | Directory where documents are placed for ingestion. |
| `supported_extensions` | `[".json", ".md", ".pdf"]` | File types the ingestor will look for. |
| `llm_processed_extensions` | `[".pdf"]` | Extensions that require LLM summarization (via Gemini) before embedding. |
| `chunk_size` | `1024` | Size of text chunks for embedding. |

## Usage

### 1. Ingesting Knowledge Documents

To provide the agents with long-term, static context, use the ingestion script.

1. Create the source directory if it doesn't exist:

   ```bash
   mkdir -p knowledge_bank
   ```

1. Place your documents (e.g., design specs, manuals, architecture diagrams) into the `knowledge_bank` directory.

1. Run the ingestion script manually:

   ```bash
   python run_agents_manually.py
   ```

   *(Note: The `run_agents_manually.py` script is configured to run the ingestion pipeline by default, as shown in the source code changes.)*

### 2. Running Agents

Agents are typically run via the main control plane entry point (`main.py`) or manually for testing using `run_agents_manually.py`.

The `main.py` exposes the agents as tools:

| Agent Function | Description |
| :--- | :--- |
| `architect(chat)` | Creates high-quality architecture and planning. |
| `developer(chat)` | Writes high-quality code based on design guidelines. |
| `readme_writer(chat)` | Generates high-quality README documentation. |

**Example: Running the Readme Writer Agent Manually**

To test the `readme_writer` agent with a specific prompt (which will also trigger memory retrieval and storage):

1. Modify `run_agents_manually.py` to call the desired agent function with a specific prompt string:
   ```python
   # run_agents_manually.py (Example modification)
   async def main():
       # await knowledge_bank.run_ingestion()
       await readme_writer_tool("Generate a new README.md based on the latest code changes, focusing on the Qdrant memory optimizations.")
   ```
1. Execute the script:
   ```bash
   python run_agents_manually.py
   ```
   The `readme_writer` agent will generate the content, format it as Markdown, and atomically write the result to `README.md`.

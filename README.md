# Agentic Tools

This project is a robust, modular multi-agent system designed to automate complex software development and documentation tasks. It leverages specialized agents powered by Google's Gemini models, coupled with a highly optimized, persistent Retrieval-Augmented Generation (RAG) system built on the Qdrant vector database.

## Key Features and Capabilities

- **High-Performance RAG Architecture:** Utilizes **FastEmbed** (`mixedbread-ai/mxbai-embed-large-v1`) for rapid, efficient vector generation and a **cross-encoder reranker** (`jinaai/jina-reranker-v2-base-multilingual`) to significantly boost the relevance and quality of retrieved context.
- **Optimized Qdrant Memory:** Features a centralized client manager that configures Qdrant collections with advanced performance settings, including HNSW tuning, scalar quantization (for memory efficiency), and explicit thread management.
- **Robust Knowledge Ingestion Pipeline:** An automated, idempotent script processes external documents (`.md`, `.json`, `.pdf`) into the knowledge bank using text chunking, batch upserts, and `tenacity`-based retry logic for network stability.
- **Specialized Agents:** Includes dedicated agents for high-level planning (`architect`), implementation (`developer`), and documentation (`readme_writer`).
- **GPU Readiness:** Embedding and reranking models are explicitly configured to utilize available GPU resources (`cuda`) if specified in the configuration.

## Prerequisites

To run this project, you need the following installed and configured:

1. **Python 3.10+**
1. **Qdrant Vector Database:** A running instance of Qdrant. The system defaults to connecting to `http://localhost:6333`.
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

1. **Install Dependencies:**
   Install all required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

1. **Set Environment Variables:**
   Export your Gemini API keys in your shell session (replace placeholders):

   ```bash
   export GEMINI_API_KEY_ARCHITECT="YOUR_KEY_HERE"
   # ... set the other four keys similarly
   ```

## Configuration Details

All core settings, including performance tuning for the vector database, are managed in the `conf/agentic_tools.toml` file.

### Memory and Vector Database (`[agentic-tools.memory]`)

This section controls the RAG components, including the embedding model, reranker, and Qdrant connection settings.

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `qdrant_url` | `http://localhost:6333` | Qdrant connection endpoint. |
| `embedding_model` | `mixedbread-ai/mxbai-embed-large-v1` | High-performance model used for vector generation (FastEmbed). |
| `device` | `"cpu"` | Device for FastEmbed and Reranker. Change to `"cuda"` to enable GPU acceleration. |
| `timeout` | `60.0` | Connection timeout for Qdrant operations. |
| `search_hnsw_ef` | `128` | Search-time HNSW parameter (`ef` - size of the dynamic list for the nearest neighbors search). Higher values increase accuracy but reduce speed. |
| `reranker.enabled` | `true` | Enables the cross-encoder reranker for improved search quality. |

### Qdrant Performance Tuning

The `QdrantClientManager` uses the following nested configurations to create highly optimized collections:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| `[hnsw_config]` | `m`, `ef_construct` | HNSW indexing parameters for vector index quality. |
| | `max_indexing_threads` | Set to `-1` to use all available CPU cores for indexing. |
| `[optimizers_config]` | `max_optimization_threads` | Set to `-1` to use all available CPU cores for background optimization. |
| `[quantization_config]` | `scalar_type`, `quantile` | Enables scalar quantization (e.g., `int8`) to reduce memory usage with minimal impact on precision. |

## Usage

### 1. Ingesting Knowledge Documents

The ingestion pipeline processes documents in the `knowledge_bank` directory, chunks them, embeds them using FastEmbed, and upserts them into the Qdrant `knowledge-bank` collection. The process is idempotent, skipping files that have not changed based on content hashing.

1. **Prepare Documents:** Place your documents (`.md`, `.json`, `.pdf`) into the `knowledge_bank` directory.

1. **Run Ingestion:** The `run_agents_manually.py` script is configured to execute the ingestion pipeline by default:

   ```bash
   python run_agents_manually.py
   ```

   The script will automatically:

   - Ensure the Qdrant collection exists with optimized parameters.
   - Use the Gemini API to summarize complex files (like PDFs) before chunking.
   - Batch upsert the resulting chunks and vectors.

### 2. Running Agents

Agents are executed via the main entry point, which orchestrates memory retrieval, tool execution, and response storage.

**Example: Running the Readme Writer Agent Manually**

To test the `readme_writer` agent with a specific prompt:

1. Modify `run_agents_manually.py` to call the desired agent function.
   *(Note: Ensure the `knowledge_bank.run_ingestion()` line is commented out if you only want to run the agent.)*

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

The agent will perform the following steps:

1. Retrieve context from Qdrant (Today, Monthly, Long-Term, Knowledge Bank) using the configured weights.
1. Rerank the retrieved documents using the cross-encoder model for maximum relevance.
1. Pass the highly relevant context (prefixed with `--- FastEmbed Reranked Memories Retrieved ---`) to the Gemini model.
1. Write the final, formatted output to `README.md` using an atomic file write operation.
1. Store its own response in Qdrant for future context retrieval.

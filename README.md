# Agentic Code Maintenance System

This project is a robust Python framework for building and orchestrating specialized AI agents that perform targeted tasks on a codebase. It uses a high-performance Retrieval-Augmented Generation (RAG) architecture, leveraging Qdrant and FastEmbed, to provide agents with precise, context-aware information.

A core feature is the **`commentator` agent**, designed to enforce code quality standards by meticulously reviewing and updating documentation, comments, and import organization without ever modifying functional code logic.

## Key Features

- **Commentator Agent:** A specialized, non-functional agent that focuses exclusively on code hygiene:
  - Reviews and perfects docstrings (file, class, function level).
  - Refines inline comments (explaining the *why*, not the *what*).
  - Organizes and cleans imports (removes unused, groups by standard/third-party/local).
- **High-Performance RAG:** Utilizes FastEmbed for rapid vector generation and a cross-encoder reranker to ensure maximum relevance of retrieved context from the knowledge bank.
- **Modular Architecture:** Clear separation between Agents, Tools (Shell/API), and Memory components.
- **Robust Tooling:** Integrated shell tools for safe, atomic file I/O and Git context gathering.

## Prerequisites

To run this project, you need the following installed:

1. **Python 3.10+**
1. **Git**
1. **Qdrant Vector Database:** A running instance of Qdrant (default configuration assumes `http://localhost:6333`).
1. **Environment Variables:** You must set the required API keys in your environment.

### Environment Setup

The `commentator` agent uses the Google Gemini API. Set the following environment variable:

| Agent | Configuration Key | Required Environment Variable |
| :--- | :--- | :--- |
| `commentator` | `api_key` | `GEMINI_API_KEY_COMMENTATOR` |

```bash
# Example: Set the required API key
export GEMINI_API_KEY_COMMENTATOR="YOUR_GEMINI_API_KEY"
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Install Dependencies:**
   (Assuming a `requirements.txt` file exists for the project dependencies.)

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All agent and memory configurations are managed in the `conf/agentic_tools.toml` file.

### Agent Configuration (`[agentic-tools.commentator]`)

This section defines the persona and constraints for the documentation agent.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `model_name` | `gemini-2.5-flash-lite` | The specific LLM used for the task. |
| `temperature` | `0.1` | Low temperature ensures deterministic, focused output (minimal creativity). |
| `api_key` | `GEMINI_API_KEY_COMMENTATOR` | The environment variable name to load the key from. |
| `skills` | `[...]` | Defines the agent's persona (e.g., "expert technical writer," "strict adherence to coding standards"). |

### Memory Configuration (`[agentic-tools.memory]`)

The memory system uses Qdrant for vector storage. Ensure your Qdrant instance is accessible via the configured URL.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `qdrant_url` | `http://localhost:6333` | Address of the Qdrant service. |
| `device` | `cpu` | Specifies the device for FastEmbed (can be set to `cuda` if a GPU is available). |
| `embedding_model` | `mixedbread-ai/mxbai-embed-large-v1` | The model used by FastEmbed for vector generation. |

## Usage: Running the Commentator Agent

The `commentator` agent is designed to be run on a single file, modifying it in place to improve documentation and style.

The primary entry point for execution is the `main.py` script, which contains the `commentator_tool` function.

### 1. Run the Demonstration

The `main.py` script includes a demonstration that runs the `commentator` on a project file (`src/tools/shell_tools.py`).

```bash
# Execute the main script from the project root
python main.py
```

**Expected Output:**

The script will log its progress, read the target file, send the content to the agent, and write the updated content back to the file.

```
INFO - Running 'commentator' agent on: src/tools/shell_tools.py
INFO - Running 'commentator' with a simplified payload.
INFO - Tool execution completed for agent 'commentator'.
INFO - Successfully updated comments in: src/tools/shell_tools.py
```

### 2. Adapt for a Custom File

To use the `commentator` on any file in your project, modify the `main.py` script to target your desired file path:

```python
# In main.py, modify the main function:

async def main() -> None:
    """Main function to demonstrate running the commentator tool."""
    # Change this line to your target file
    target_file = "path/to/your/new_module.py" 
    logger.info(f"Demonstration: Running commentator on '{target_file}'")
    await commentator_tool(target_file)
```

Then, run `python main.py`. The agent will read the file, apply documentation and import cleanup, and overwrite the original file content.

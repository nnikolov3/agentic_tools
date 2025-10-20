# Agentic Tools

## Project Description

Agentic Tools is a powerful and flexible agentic toolchain designed for the comprehensive lifecycle of code development. It facilitates architecting, designing, validating, and approving code through a series of chained, intelligent agents and tools. This project aims to automate and enhance various aspects of the software development process, ensuring adherence to design principles and coding standards.

## Key Features and Capabilities

- **Modular Agent System**: A robust `Agent` class allows for the dynamic execution of various tasks, providing a flexible framework for extending capabilities.
- **Extensible Toolset**: The `Tool` class orchestrates a suite of specialized tools, including `ShellTools`, `ApiTools`, and `QdrantTools`, enabling complex operations.
- **Comprehensive File Processing**: `ShellTools` intelligently traverses specified project directories, concatenates relevant file contents, and applies filters based on file extensions, size, and exclusion rules.
- **Git Information Retrieval**: Seamlessly integrates with Git to extract repository information, such as the Git username and remote URL, enriching contextual data for agents.
- **LLM Provider Agnostic API Layer**: `ApiTools` provides a unified interface for interacting with various Large Language Model (LLM) providers, including Google Gemini, Groq, Cerebras, and SambaNova, allowing for flexible model selection.
- **Vector Database Integration**: `QdrantTools` enables the storage and retrieval of generated content (e.g., READMEs, design documents) into Qdrant, facilitating Retrieval Augmented Generation (RAG) and maintaining historical context.
- **Dynamic Configuration**: Utilizes a `Configurator` to load and manage project settings from TOML configuration files, ensuring parameterization and adaptability.
- **Automated Documentation Generation**: Includes a `readme_writer` agent capable of automatically generating and updating the `README.md` based on the project's source code, configuration, and design principles.
- **Strict Adherence to Standards**: Enforces foundational design principles and language-specific coding standards (as defined in `DESIGN_PRINCIPLES_GUIDE.md` and `CODING_FOR_LLMs.md`) to ensure high-quality, maintainable, and robust code.

## Prerequisites

Before installing and running Agentic Tools, ensure you have the following:

- **Python 3.11+**: The project uses `tomllib` which is built-in from Python 3.11 onwards.
  - Verify your Python version: `python3 --version`
- **Git**: Required for cloning the repository and for the `ShellTools` to gather repository information.
  - Verify Git installation: `git --version`
- **LLM API Keys**: Depending on the LLM provider you intend to use, you will need to set the corresponding API key as an environment variable.
  - **Google Gemini**: `GEMINI_API_KEY`
  - **Groq**: `GROQ_API_KEY`
  - **Cerebras**: `CEREBRAS_API_KEY`
  - **SambaNova**: `SAMBANOVA_API_KEY`
- **Qdrant Service (Optional)**: If you plan to utilize the Qdrant vector database integration, a Qdrant instance must be running and accessible, typically at `http://localhost:6333`.

## Installation

Follow these steps to get Agentic Tools up and running:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Create and Activate a Python Virtual Environment**:
   It's highly recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

1. **Install Dependencies**:
   Install the required Python packages.

   ```bash
   pip install mdformat groq qdrant-client google-generativeai
   # If you plan to use Cerebras or SambaNova, install their respective SDKs:
   # pip install cerebras-sdk
   # pip install sambanova-sdk
   ```

1. **Set Environment Variables**:
   Set the API keys for the LLM providers you intend to use. Replace `YOUR_API_KEY` with your actual keys.

   ```bash
   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   export GROQ_API_KEY="YOUR_GROQ_API_KEY"
   # export CEREBRAS_API_KEY="YOUR_CEREBRAS_API_KEY"
   # export SAMBANOVA_API_KEY="YOUR_SAMBANOVA_API_KEY"
   ```

## Usage Examples

This section demonstrates how to use Agentic Tools, focusing on the `readme_writer` agent as a practical example.

### 1. Configure the Agent (agentic_tools.toml)

Ensure your `agentic_tools.toml` file (located in the project root) is configured for the `readme_writer` agent. An example snippet is shown below:

```toml
# agentic_tools.toml
[agentic-tools]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md", "AGENTS.md"]
source = ["src"]
project_root = "."
tests_directory = ["tests"]
project_directories = ["conf", "docs", "src"]
include_extensions = [".py",  ".md", ".toml"]
exclude_files = ["__init__.py"]
exclude_directories = [".qwen", ".gemini",".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]
recent_minutes = 10
max_file_bytes = 262144
max_total_bytes = 10485760


#####################################################################
# README Writer Configuration
#####################################################################
[agentic-tools.readme_writer]
prompt = """
* You are an expert technical writer.
* Create excellent, concise,and practical README documentation based on the project's source code, configuration, and conventions.

__Generate a comprehensive yet simple README.md that includes:__

- Project title and description based on actual project
- Key features and capabilities
- Prerequisites with specific requirements (not generic placeholders like 'apt-get')
- Installation instructions specific to this project
- Usage examples based on actual code and functionality
- Configuration details from actual configuration files

* Use the github information provided.
* Focus how to onboard a new user.
* Focus on simplicity, clarity, and utility.
* Provide concrete, actionable examples based on the actual project structure and code, not generic placeholders.
* Make it helpful and accurate.

"""
model_name = "gemini-2.5-flash"
temperature = 0.1
description = "Generates high-quality README documentation"
model_provider = ["google"]
alternative_model = "gemini-2.5-flash"
alternative_model_provider = ["google"]
skills = [
    "technical writing",
    "documentation",
    "readme creation",
    "information synthesis",
    "content organization",
    "clarity and precision"
]
qdrant_embedding="all-MiniLM-L6-v2"
embedding_size = 384
```

### 2. Run an Agent

To execute an agent, you'll typically use a main entry point script. Assuming you have a `run.py` (or similar) at the project root that orchestrates agent execution, you would run it like this:

First, ensure your virtual environment is activated and API keys are set.

```bash
# Activate your virtual environment if not already active
source .venv/bin/activate

# Ensure API keys are set (as described in Prerequisites/Installation)
export GEMINI_API_KEY="..."

# Example: Running the readme_writer agent
# This assumes a simplified `run.py` might look like the Python snippet below
python run.py readme_writer
```

**Example `run.py` (for demonstration of internal workings):**

This hypothetical `run.py` demonstrates how you might programmatically invoke an agent:

```python
# run.py (Illustrative example, create this file if you want to run this)
import os
import sys
from src.configurator import Configurator
from src.agents.agent import Agent
import logging

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <agent_name>")
        sys.exit(1)

    agent_to_run = sys.argv[1]
    config_file_path = "agentic_tools.toml" # Path to your main configuration file

    try:
        # 1. Load configuration
        configurator = Configurator(config_file_path)
        project_config = configurator.get_config_dictionary()

        # 2. Instantiate and run the specified agent
        agent_instance = Agent(project_config)
        print(f"Executing agent: {agent_to_run}...")
        output = agent_instance.run_agent(agent_to_run)

        print(f"
--- Agent '{agent_to_run}' Output ---")
        # For readme_writer, the output is the generated README content
        # It also writes to README.md internally.
        print(output[:1000] + "...") # Print first 1000 chars for brevity

    except ValueError as e:
        logging.error(f"Configuration or Agent Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

When you run `python run.py readme_writer`, the `readme_writer` agent will:

1. Collect all relevant files from `project_directories` as specified in `agentic_tools.toml`.
1. Gather Git information (username, repository URL).
1. Send this context along with the `prompt` defined in the `readme_writer` configuration to the configured LLM (e.g., Google Gemini).
1. Receive the generated `README.md` content from the LLM.
1. Format the content using `mdformat`.
1. Write the formatted content to the `README.md` file in the project root.
1. Optionally, embed the content into Qdrant for future use.

## Configuration Details

All primary configuration for Agentic Tools is managed via the `agentic_tools.toml` file in the project root. This TOML file defines global project settings and specific configurations for each agent.

### Global `[agentic-tools]` Section

This section defines project-wide parameters:

- `project_name` (string): The name of your project.
- `project_description` (string): A brief description of the project.
- `design_docs` (array of strings): Paths to core design documentation files (e.g., `docs/DESIGN_PRINCIPLES_GUIDE.md`). These are included as context for agents.
- `source` (array of strings): Directories containing the main source code.
- `project_root` (string): The root directory of the project (typically `.`).
- `tests_directory` (array of strings): Directories containing project tests.
- `project_directories` (array of strings): All directories to be traversed and included as context for agents (e.g., `conf`, `docs`, `src`).
- `include_extensions` (array of strings): File extensions to include during file traversal (e.g., `".py"`, `".md"`, `".toml"`).
- `exclude_files` (array of strings): Specific filenames to exclude from traversal (e.g., `"__init__.py"`).
- `exclude_directories` (array of strings): Directories to exclude from traversal (e.g., `".git"`, `"venv"`, `"__pycache__"`).
- `recent_minutes` (integer): (Currently unused but present in config) Defines a time window for recent file changes.
- `max_file_bytes` (integer): Maximum size (in bytes) for an individual file to be included as context. Files exceeding this limit are skipped.
- `max_total_bytes` (integer): (Currently unused but present in config) Maximum total bytes for all concatenated files.

### Agent-Specific Configuration (e.g., `[agentic-tools.readme_writer]`)

Each agent (e.g., `readme_writer`, `approver`) has its own subsection under `[agentic-tools.<agent_name>]`. These settings define the behavior of the specific agent:

- `prompt` (string): The primary instruction prompt given to the LLM for this agent's task.
- `model_name` (string): The specific LLM model to use (e.g., `"gemini-2.5-flash"`).
- `temperature` (float): Controls the randomness of the LLM's output (0.0 for deterministic, higher for more creative).
- `description` (string): A short description of the agent's purpose.
- `model_provider` (array of strings): The LLM provider(s) to use (e.g., `["google"]`). The first provider in the list is used.
- `alternative_model` (string): An alternative LLM model name, for fallback or testing.
- `alternative_model_provider` (array of strings): An alternative LLM provider.
- `skills` (array of strings): A list of skills attributed to the agent, providing context for its capabilities.
- `qdrant_embedding` (string): The embedding model to use when storing content in Qdrant (e.g., `"all-MiniLM-L6-v2"`).
- `embedding_size` (integer): The dimension of the embeddings generated by `qdrant_embedding`.

______________________________________________________________________

This README provides a solid foundation for understanding, installing, and utilizing Agentic Tools. For more in-depth information on design philosophy and coding standards, please refer to the `docs/DESIGN_PRINCIPLES_GUIDE.md` and `docs/CODING_FOR_LLMs.md` files respectively.

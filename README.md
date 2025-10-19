
# Agentic Tools


Agentic Tools is a powerful framework designed to build sophisticated agentic toolchains. It enables architects, designers, and developers to automate the process of designing, validating, and approving code through a series of chained tools. By integrating Large Language Models (LLMs) with custom and external functionalities, Agentic Tools streamlines complex development workflows, ensuring adherence to design principles and coding standards.

## Key Features and Capabilities

- **Automated Code Lifecycle**: Facilitates automated architecting, design, validation, and approval of code.
- **Modular Agent System**: Easily define and chain agents to perform specific tasks. Examples include `readme_writer` for documentation and `approver` for quality gates.
- **LLM Integration**: Seamlessly connect with various LLM providers (Google Gemini, Groq, Cerebras, SambaNova) to leverage their capabilities within agent workflows.
- **Flexible Tooling**: Define custom tools (Python functions) that agents can invoke, providing extensibility and integration with any external system.
- **Comprehensive File System Utilities**: Agents can interact with the file system to read source code, configuration, design documents, and write outputs like generated documentation.
- **Git Integration**: Retrieve Git repository information (username, URL) for context-aware operations.
- **Configuration Management**: Robust configuration loading using TOML files, allowing for easy setup and customization of agents and tools.
- **Qdrant Integration**: Utilize Qdrant as a vector database for semantic memory, enabling agents to store and retrieve contextual information (e.g., approver decisions, patches, design documents) for enhanced decision-making and knowledge retention.
- **FastMCP Framework Compatibility**: Built to integrate with the FastMCP (Model Context Protocol) framework, ensuring standardized interaction with AI models and external systems.
- **Quality Enforcement**: Incorporates strict design principles and coding standards (via `DESIGN_PRINCIPLES_GUIDE.md` and `CODING_FOR_LLMs.md`) to ensure high-quality, maintainable, and secure generated code.

## Prerequisites

Before you can install and use Agentic Tools, ensure you have the following:

- **Python**: Version 3.9 or higher.
- **Git**: Installed and configured on your system.
- **Virtual Environment**: Strongly recommended to manage dependencies.
- **LLM API Keys**: Depending on the LLM providers you intend to use, you'll need API keys. Create a `.env` file in the project root with the following (or set them as environment variables):
  ```ini
  GEMINI_API_KEY="your_google_gemini_api_key"
  GROQ_API_KEY="your_groq_api_key"
  CEREBRAS_API_KEY="your_cerebras_api_key"
  SAMBANOVA_API_KEY="your_sambanova_api_key"

  # Optional Qdrant Configuration
  QDRANT_URL="http://localhost:6333" # Or your cloud Qdrant URL
  QDRANT_API_KEY="your_qdrant_api_key" # Only for cloud Qdrant
  QDRANT_LOCAL_PATH=".qdrant_db" # For local persistent Qdrant
  COLLECTION_NAME="agentic_decisions"
  EMBEDDING_MODEL="all-MiniLM-L6-v2"
  ```

## Installation

Follow these steps to set up Agentic Tools on your local machine.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   # If you are using uv, you can use the following command:
   # uv pip sync
   ```

1. **Set Up Environment Variables**: Create a `.env` file in the project root as described in the [Prerequisites](#prerequisites) section, or export them in your shell session.

## Usage

### Running Tests

To run the tests, use the following command:

```bash
python -m unittest discover
```

Agentic Tools are driven by agents configured in `agentic_tools.toml`. You interact with the system by running a specific agent. Here's how to run the `readme_writer` agent to generate/update your project's `README.md`.

1. **Ensure Configuration**: Make sure your `agentic_tools.toml` is correctly configured for the `readme_writer` (see [Configuration Details](#configuration-details)).

1. **Execute the Agent**: From the project root, you would typically have an entry point that initializes the `Agent` class and calls the desired agent method. A common pattern is to have a `main.py` or `run.py` that looks something like this:

   ```python
   # main.py
   from __future__ import annotations
   from src.configurator import Configurator
   from src.agents.agent import Agent
   from fastmcp import FastMCP
   import os
   from typing import Any
   from pathlib import Path

   cwd = os.getcwd()
   configuration_path = Path(f"{cwd}/conf/agentic_tools.toml")
   configurator = Configurator(configuration_path)
   configuration = configurator.get_config_dictionary()
   mcp_name = "Agentic Tools"
   mcp = FastMCP(mcp_name)


   def main():
       @mcp.tool(
           description="Walks the project directories, gets github information, and updates directly the README.md file"
       )
       async def readme_writer_tool() -> Any:
           print("readme_writer_tool")
           agent = Agent(configuration["agentic-tools"])  # Agent class
           return agent.run_agent("readme_writer")


   if __name__ == "__main__":
       mcp.run_async()
   ```

1. **Run the script**:

   ```bash
   python run.py
   ```

   This will execute the `readme_writer` agent, which will traverse your project files, gather information, send it to the configured LLM, and write the generated `README.md` to your project root.

### Qdrant Integration Example

Agentic Tools integrates with Qdrant for semantic memory. For example, the `approver` agent might store its decisions in Qdrant. While the direct `approver` agent execution is not shown, you can infer how a `QdrantIntegration` object is used:

```python
# Example of using QdrantIntegration (from qdrant_tools.py)
from src.tools.qdrant_tools import QdrantIntegration
import uuid # To generate unique IDs for decisions

# Assuming configuration is loaded as 'config'
# model_sizes = config.get("embedding_model_sizes", {})
# qdrant_config = {
#     "local_path": ".qdrant_db", # or url and api_key
#     "collection_name": "agentic_decisions",
#     "embedding_model": "all-MiniLM-L6-v2",
#     "model_sizes": model_sizes
# }

# qdrant_client = QdrantIntegration(**qdrant_config)

# Example decision data (hypothetical from an 'approver' agent)
# decision_data = {
#     "decision": "APPROVED",
#     "summary": "All code changes comply with design principles.",
#     "positive_points": ["Clean architecture", "High test coverage"],
#     "negative_points": [],
#     "required_actions": [],
#     "timestamp": "2023-10-27T10:00:00Z",
#     "user_chat": "User asked for a final review."
# }
# decision_id = str(uuid.uuid4())
# content_for_embedding = decision_data["summary"] + " ".join(decision_data["positive_points"])

# if qdrant_client.store_approver_decision(decision_data, decision_id, content_for_embedding):
#     print(f"Decision {decision_id} stored in Qdrant.")

# To search for similar decisions:
# query = "recent approved changes with high quality"
# similar_decisions = qdrant_client.search_similar_decisions(query, limit=2)
# print(f"Found similar decisions: {similar_decisions}")
```

## Configuration Details

The core configuration for Agentic Tools resides in `agentic_tools.toml`.

```toml
# agentic_tools.toml
[agentic-tools]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."

# Paths to design documentation (e.g., DESIGN_PRINCIPLES_GUIDE.md, CODING_FOR_LLMs.md)
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md", "AGENTS.md"]

# Source code directories to include (e.g., "src")
source = ["src"]

# Project root directory (usually ".")
project_root = "."

# Directories containing tests (e.g., "tests")
tests_directory = ["tests"]

# Other project directories to be processed (e.g., for README generation)
project_directories = ["conf", "docs", "src"]

# File extensions to include when processing files
include_extensions = [".py", ".md", ".toml"]

# Specific files to exclude from processing
exclude_files = ["__init__.py"]

# Directories to exclude from recursive processing
exclude_directories = [".qwen", ".gemini", ".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]

# How many minutes ago a file must have been modified to be considered 'recent' (currently not actively used by readme_writer but available)
recent_minutes = 10

# Maximum file size in bytes to include for processing (e.g., for LLM context)
max_file_bytes = 262144 # 256 KB

# Maximum total bytes across all concatenated files (currently not actively used but available)
max_total_bytes = 10485760 # 10 MB

# Qdrant embedding model to vector size mappings (used by QdrantIntegration)
[agentic-tools.embedding_model_sizes]
"all-MiniLM-L6-v2" = 384
"all-MiniLM-L12-v2" = 384

#####################################################################
# README Writer Configuration
#####################################################################
[agentic-tools.readme_writer]
# The specific prompt given to the LLM for README generation
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



#####################################################################
# Approver (final gate) Configuration
#####################################################################
[agentic-tools.approver]
# Prompt for the Approver LLM agent
prompt = """
You are the final gatekeeper in a software development pipeline. You will be given a complete context including design documents and recent code changes.

Your decision-making process:
1. Thoroughly analyze the changes for quality, design, and adherence to principles
2. Identify all positive and negative points with specific details
3. Only approve if the changes meet the highest standards and have no critical negative points
4. For any design flaws, code quality issues, or principle violations, return CHANGES_REQUESTED with specific required actions
5. Be rigorous in your evaluation - quality over speed

Return ONLY a single, valid JSON object with this exact structure:
{
  "decision": "APPROVED" | "CHANGES_REQUESTED",
  "summary": "string",
  "positive_points": ["string"],
  "negative_points": ["string"],
  "required_actions": ["string"]
}
"""
model_name = "gemini-2.5-pro"
temperature = 0.1
description = "Final approval decision"
model_provider = ["google"]
alternative_model = "gemini-2.5-flash"
alternative_model_provider = ["google"]
skills = [
    "code review",
    "quality assurance",
    "decision making",
    "technical analysis", 
    "standards compliance",
    "risk assessment",
    "context analysis"
]

# Agentic Tools

## Project Title

**Agentic Tools**

## Project Description

Agentic Tools is an advanced agentic toolchain designed for the end-to-end code lifecycle management. It provides a robust framework for architecting, designing, validating, and approving code, leveraging chained tools and intelligent AI models. This system aims to streamline software development processes by integrating AI capabilities into critical stages, ensuring high-quality, maintainable, and well-documented software.

## Key Features and Capabilities

- **Modular Agent Architecture**: Build and deploy intelligent agents designed for specific development tasks (e.g., `readme_writer`, `approver`).
- **Extensible Tooling**: Integrate seamlessly with LLM APIs (Google, Groq, Cerebras, SambaNova), file system operations, and external services through a flexible tool framework.
- **Configuration-Driven**: Define agent behaviors, LLM models, project parameters, and tool settings using a centralized `agentic_tools.toml` configuration file.
- **AI-Powered Code Lifecycle**: Facilitate design analysis, automated code generation, rigorous validation, and structured approval workflows.
- **Automated Documentation**: Generate comprehensive and accurate `README.md` files based on project context, code, and configuration, ensuring documentation stays up-to-date.
- **Semantic Memory Integration (Qdrant)**: Utilize Qdrant as a vector database for efficient semantic search, contextual recall, and persistent memory across agent operations. This supports advanced use cases like code search, documentation retrieval, and knowledge base management.
- **Model Context Protocol (FastMCP)**: Leverages `FastMCP` to provide a structured and discoverable interface for LLMs to interact with custom tools and data resources, enhancing AI-tool interoperability.
- **Strict Quality Enforcement**: Adheres to comprehensive coding standards for LLM-generated code, ensuring simplicity, readability, test-driven correctness, and automated quality checks.
- **File System & Git Operations**: Tools for traversing project directories, reading/writing files, and extracting Git repository information (username, URL).

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher**: The project uses `tomllib` which is built-in Python 3.11+.
- **pip**: Python's package installer.
- **git**: Version control system, required for cloning the repository and extracting project metadata.
- **LLM API Keys**: Depending on the models you intend to use, you will need API keys configured as environment variables:
  - `GEMINI_API_KEY` (for Google's Gemini models)
  - `GROQ_API_KEY` (for Groq models)
  - `CEREBRAS_API_KEY` (for Cerebras models)
  - `SAMBANOVA_API_KEY` (for SambaNova models)
- **(Optional) Docker**: If you plan to use Qdrant for semantic memory, Docker is recommended for easily running a local Qdrant instance.

## Installation Instructions

Follow these steps to set up and install Agentic Tools on your local machine.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```

1. **Create and Activate a Virtual Environment**:
   It's highly recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .\.venv\Scriptsctivate
   ```

1. **Install Project Dependencies**:
   Install the necessary Python packages.

   ```bash
   pip install mdformat qdrant-client[fastembed] fastmcp google-generativeai groq cerebras-cloud-sdk sambanova
   ```

1. **Configure Environment Variables**:
   Create a `.env` file in the project root or set the required API keys directly in your shell environment. Replace the placeholder values with your actual API keys.

   ```ini
   # .env example
   GEMINI_API_KEY="your_google_gemini_api_key"
   GROQ_API_KEY="your_groq_api_key"
   CEREBRAS_API_KEY="your_cerebras_api_key"
   SAMBANOVA_API_KEY="your_sambanova_api_key"

   # Optional Qdrant Configuration
   # QDRANT_URL="http://localhost:6333"
   # QDRANT_API_KEY="your_qdrant_api_key"
   # EMBEDDING_MODEL="all-MiniLM-L6-v2"
   ```

1. **(Optional) Set up Qdrant (for semantic memory)**:
   If you plan to use Qdrant, you can run it via Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334       -v $(pwd)/qdrant_storage:/qdrant/storage       qdrant/qdrant
   ```

   This will start a Qdrant instance accessible at `http://localhost:6333` (HTTP) and `localhost:6334` (gRPC).

## Usage Examples

Agentic Tools operates by running specific agents defined in its configuration. Here's how to interact with the system, focusing on the `readme_writer` agent.

### 1. Running an Agent

The primary way to use Agentic Tools is by executing an agent through a main script (e.g., `main.py`). This script would load the configuration and invoke the specified agent.

Assuming a `main.py` entry point in your project root:

```python
# main.py (conceptual example, not provided in sources but necessary for execution)
import sys
from src.configurator import Configurator
from src.agents.agent import Agent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <agent_name>")
        sys.exit(1)

    agent_name = sys.argv[1]
    
    try:
        # Load configuration
        config_path = "agentic_tools.toml"
        configurator = Configurator(config_path)
        # Ensure 'agentic-tools' is the top-level key in your config
        config = configurator.get_config_dictionary().get("agentic-tools", {}) 

        # Instantiate and run the agent
        agent_instance = Agent(config) 
        print(f"Executing agent: {agent_name}...")
        agent_instance.run_agent(agent_name)
        print(f"Agent '{agent_name}' finished execution.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

```

To generate the `README.md` for your project using the `readme_writer` agent:

```bash
python main.py readme_writer
```

This command will:

1. Read the project's configuration from `agentic_tools.toml`.
1. Traverse the directories and files specified in the configuration (`project_directories`, `include_extensions`, `exclude_directories`, `exclude_files`).
1. Gather Git repository information (username, URL).
1. Send this collected context, along with the `readme_writer`'s specific prompt, to the configured LLM.
1. Receive the generated README content.
1. Clean up any escaped characters and format the Markdown.
1. Write the final `README.md` file to the project root.

### 2. Example Agent (`readme_writer`) Functionality

The `readme_writer` agent's core logic, as seen in `src/tools/tool.py`, combines several internal tools:

```python
# From src/tools/tool.py
class Tool:
    # ...
    def readme_writer(self):
        """Execute readme writer logic"""
        # 1. Concatenate project files into a payload
        self.payload = self.shell_tools.concatenate_all_files() 
        
        # 2. Get Git repository information
        self.git_information = self.shell_tools.get_git_info()
        self.payload.update(self.git_information) # Add git info to payload
        
        # 3. Call the LLM API with the combined payload and agent's prompt
        self.response = self.api_tools.run_api(self.payload)
        
        # 4. Clean up and format the LLM's response
        self.response = self.shell_tools.cleanup_escapes(self.response)
        self.response = mdformat.text(self.response, options={"wrap": "preserve"})
        
        # 5. Write the generated content to README.md
        self.shell_tools.write_file("README.md", self.response)

        return self.response
```

### 3. FastMCP and Qdrant Integration

While end-users will primarily interact with `Agentic Tools` via its agents, the system internally leverages `FastMCP` for structured LLM interactions and `Qdrant` for semantic memory. For developers extending the framework:

- **Qdrant Tools**: The `QDRANT.md` document outlines two specific tools, `qdrant-store` and `qdrant-find`, which can be integrated into new agents to store and retrieve information semantically.
  - **`qdrant-store`**: Stores information and optional metadata into a specified Qdrant collection.
    ```python
    # Conceptual example within an Agent using a Qdrant-enabled tool
    # Assuming your agent uses a tool that wraps qdrant-store
    # await self.qdrant_tool.store_info(
    #    information="This is a key project design principle.", 
    #    metadata={"source": "DESIGN_PRINCIPLES_GUIDE.md"}, 
    #    collection_name="project_knowledge"
    # )
    ```
  - **`qdrant-find`**: Retrieves information relevant to a given query from a Qdrant collection.
    ```python
    # Conceptual example within an Agent using a Qdrant-enabled tool
    # relevant_docs = await self.qdrant_tool.find_info(
    #    query="What are the core design principles?", 
    #    collection_name="project_knowledge"
    # )
    # for doc in relevant_docs:
    #    print(doc)
    ```

## Configuration Details

The project's behavior is controlled by the `agentic_tools.toml` file located in the root directory. This TOML file organizes various settings for the project and its agents.

### `[agentic-tools]` Section

This section contains global project settings:

- **`project_name`**: (`str`) The main name of the project.
  - `project_name = "Agentic Tools"`
- **`project_description`**: (`str`) A brief overview of the project's purpose.
  - `project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."`
- **`design_docs`**: (`list` of `str`) List of paths to design documentation files. These are included in the context for agents.
  - `design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md", "AGENTS.md"]`
- **`source`**: (`list` of `str`) List of paths to the main source code directories.
  - `source = ["src"]`
- **`project_root`**: (`str`) The root directory of the project (usually `"."`).
  - `project_root = "."`
- **`tests_directory`**: (`list` of `str`) List of paths to test directories.
  - `tests_directory = ["tests"]`
- **`project_directories`**: (`list` of `str`) Specific directories to traverse and include content from for agents.
  - `project_directories = ["conf", "docs", "src"]`
- **`include_extensions`**: (`list` of `str`) File extensions to include when scanning directories.
  - `include_extensions = [".py", ".md", ".toml"]`
- **`exclude_files`**: (`list` of `str`) Specific filenames to exclude from content concatenation.
  - `exclude_files = ["__init__.py"]`
- **`exclude_directories`**: (`list` of `str`) Directories to exclude from scanning (e.g., `.git`, `venv`).
  - `exclude_directories = [".qwen", ".gemini",".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]`
- **`recent_minutes`**: (`int`) Placeholder for future functionality to filter files by recent modification.
  - `recent_minutes = 10`
- **`max_file_bytes`**: (`int`) Maximum size of a file (in bytes) to include in the context. Larger files will be skipped.
  - `max_file_bytes = 262144` (256 KB)
- **`max_total_bytes`**: (`int`) Maximum total size of all concatenated files (in bytes) to send as context.
  - `max_total_bytes = 10485760` (10 MB)

### `[agentic-tools.embedding_model_sizes]` Section

This section maps Qdrant embedding model names to their corresponding vector sizes. This is crucial for correctly configuring Qdrant collections.

- `"all-MiniLM-L6-v2" = 384`
- `"all-MiniLM-L12-v2" = 384`

### Agent-Specific Sections (e.g., `[agentic-tools.readme_writer]`, `[agentic-tools.approver]`)

Each agent defined in the system has its own configuration block. These blocks specify how the agent should behave, which LLM to use, and other agent-specific parameters.

Example for `readme_writer` agent:

```toml
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
```

Common fields in agent configurations:

- **`prompt`**: (`str`) The specific instructions or prompt given to the LLM for this agent's task.
- **`model_name`**: (`str`) The name of the primary LLM model to use (e.g., "gemini-2.5-flash").
- **`temperature`**: (`float`) Controls the randomness of the LLM's output. Lower values mean more deterministic results.
- **`description`**: (`str`) A brief description of the agent's purpose.
- **`model_provider`**: (`list` of `str`) The primary LLM provider(s) for this agent (e.g., `["google"]`, `["groq"]`).
- **`alternative_model`**: (`str`, optional) An alternative LLM model to use if the primary is unavailable.
- **`alternative_model_provider`**: (`list` of `str`, optional) An alternative LLM provider.
- **`skills`**: (`list` of `str`) A list of skills or capabilities associated with this agent, useful for meta-reasoning.

### Environment Variable Overrides

Certain configurations, especially sensitive ones like API keys or Qdrant connection details, can be provided via environment variables. For example, `QDRANT_URL`, `QDRANT_API_KEY`, `EMBEDDING_PROVIDER`, and `EMBEDDING_MODEL` for Qdrant setup are specified in `QDRANT.md` as environment variables. The `api_tools.py` also demonstrates the use of environment variables for LLM API keys.

By leveraging these configuration details, you can fine-tune the behavior of Agentic Tools to suit your specific project requirements and development workflows.

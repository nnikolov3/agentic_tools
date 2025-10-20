# Agentic Tools

Agentic Tools is a powerful and flexible agentic toolchain designed to streamline the software development lifecycle. It enables the architecture, design, validation, and approval of code through a series of chained tools, fostering high-quality, maintainable, and robust systems.

## Key Features and Capabilities

- **Modular Agent System**: Execute specific development tasks (e.g., `readme_writer`, `approver`) using a pluggable agent architecture.
- **Dynamic Tool Execution**: Agents can dynamically invoke configured tools (`shell_tools`, `api_tools`, `qdrant_tools`) based on their designated responsibilities.
- **Comprehensive Code & Documentation Context**: Gathers project source code, configuration files, and design documentation to provide rich context for agents.
- **Automated README Generation**: The `readme_writer` agent can automatically generate or update project `README.md` files based on the codebase and configuration.
- **Code Quality & Approval Workflow**: The `approver` agent facilitates automated code review and approval decisions by leveraging design documents, recent code changes (via git diff), and configurable LLM models.
- **LLM Integration**: Seamlessly integrates with various LLM providers (e.g., Google's Gemini) for intelligent code generation, analysis, and decision-making.
- **Persistent Knowledge Base**: Utilizes Qdrant for storing and retrieving vectorized project information, enabling advanced semantic search and context awareness for agents.
- **Robust Configuration Management**: A dedicated `Configurator` class handles loading and validating project configurations from `agentic_tools.toml`.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python 3.11+**: The project relies on `tomllib`, which is built-in to Python 3.11 and later.
- **pip**: Python package installer, usually included with Python.
- **Git**: Required for version control operations, especially for the `approver` agent's patch generation.
- **Qdrant Server**: A running instance of Qdrant (self-hosted or cloud) accessible at `http://localhost:6333` (default for the local client). Follow the [Qdrant documentation](https://qdrant.tech/documentation/quick-start/) for installation.
- **API Keys**:
  - **Google Gemini API Key**: Set as an environment variable `GOOGLE_API_KEY`. This is essential for agents interacting with Google's LLM models.

## Installation

Follow these steps to set up and install Agentic Tools:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/agentic-tools.git
   cd agentic-tools
   ```

1. **Create a Virtual Environment**:
   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate # On Windows, use `.\.venv\Scriptsctivate`
   ```

1. **Install Dependencies**:
   Create a `requirements.txt` file at the project root with the following contents:

   ```
   mdformat
   google-generativeai
   qdrant-client
   ```

   Then install them:

   ```bash
   pip install -r requirements.txt
   ```

1. **Set Environment Variables**:
   Set your Google Gemini API key. Replace `YOUR_GEMINI_API_KEY` with your actual key.

   ```bash
   export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
   # On Windows (Command Prompt): `set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"`
   # On Windows (PowerShell): `$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"`
   ```

   For persistent environment variables, consider adding this to your shell's profile file (e.g., `.bashrc`, `.zshrc`, `config.fish`, or system environment settings).

## Usage Examples

This section demonstrates how to initialize the system and execute agents for common tasks.

First, ensure your `agentic_tools.toml` configuration file is properly set up in the project root.

### Running the README Writer Agent

To generate or update your project's `README.md` file:

```python
# main_readme.py
import logging
from src.configurator import Configurator
from src.agents.agent import Agent

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_readme():
    """Initializes the configurator and runs the readme_writer agent."""
    try:
        config_path = "agentic_tools.toml"
        logging.info(f"Loading configuration from {config_path}")
        config_loader = Configurator(config_path)
        project_config = config_loader.get_config_dictionary()

        logging.info("Initializing Agent with readme_writer tool...")
        readme_agent = Agent(project_config)
        
        # The 'readme_writer' agent collects project files, git info,
        # sends it to an LLM, formats the response, writes README.md,
        # and stores it in Qdrant.
        readme_content = readme_agent.run_agent("readme_writer")
        
        logging.info("README.md generated and stored successfully!")
        print("
--- Generated README Content (excerpt) ---
")
        print(readme_content[:1000]) # Print an excerpt for verification
        print("
------------------------------------------")

    except Exception as e:
        logging.error(f"Failed to generate README: {e}", exc_info=True)

if __name__ == "__main__":
    generate_readme()
```

Run this script from your project root:

```bash
python main_readme.py
```

This will create (or update) a `README.md` file in your project root, and store its embedding in Qdrant.

### Running the Approver Agent

The `approver` agent audits code changes and provides a final decision. It typically operates on a `git diff`.

```python
# main_approver.py
import logging
from src.configurator import Configurator
from src.agents.agent import Agent

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_approver():
    """Initializes the configurator and runs the approver agent."""
    try:
        config_path = "agentic_tools.toml"
        logging.info(f"Loading configuration from {config_path}")
        config_loader = Configurator(config_path)
        project_config = config_loader.get_config_dictionary()

        logging.info("Initializing Agent with approver tool...")
        approver_agent = Agent(project_config)
        
        # The 'approver' agent creates a git patch, combines it with design docs,
        # sends to an LLM for review, and returns the approval decision.
        approval_result = approver_agent.run_agent("approver")
        
        logging.info("Approver agent execution complete.")
        print("
--- Approval Result ---
")
        print(approval_result)
        print("
-----------------------")

    except Exception as e:
        logging.error(f"Failed to run approver agent: {e}", exc_info=True)

if __name__ == "__main__":
    run_approver()
```

To run the `approver` agent, ensure you have pending git changes or a specific `git diff` command configured (e.g., `git diff staging`). Then execute:

```bash
python main_approver.py
```

The `approver` agent will output a JSON object indicating the approval decision, summary, positive points, negative points, and required actions.

## Configuration Details

The core configuration for Agentic Tools is managed in the `agentic_tools.toml` file. This file specifies project metadata, file inclusion/exclusion rules, and agent-specific parameters.

Here's a breakdown of the key sections and parameters in `agentic_tools.toml`:

```toml
# agentic_tools.toml

[agentic-tools]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/PROVIDERS_SDK.md","docs/CODING_FOR_LLMs.md", "AGENTS.md"]
source = ["src"] # Directories containing primary source code
project_root = "." # Root directory of the project
docs = "docs" # Directory containing general documentation
tests_directory = ["tests"] # Directory for test files
project_directories = ["conf", "docs", "src"] # All directories to be processed for context
include_extensions = [".py", ".md", ".toml"] # File extensions to include in context
exclude_files = ["__init__.py"] # Specific files to exclude
exclude_directories = [".qwen", ".gemini",".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"] # Directories to exclude from processing
recent_minutes = 10 # Not currently used by existing tools, but can be for future "recent changes" logic
max_file_bytes = 262144 # Maximum size of a file (in bytes) to be included in context (256KB)
max_total_bytes = 10485760 # Maximum total context size (in bytes) for all concatenated files (10MB)

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
model_name = "gemini-2.5-flash" # LLM model to use for this agent
temperature = 0.1 # LLM generation temperature
description = "Generates high-quality README documentation"
model_provider = ["google"] # LLM provider
alternative_model = "gemini-2.5-flash" # Alternative LLM model
alternative_model_provider = ["google"] # Alternative LLM provider
skills = [
    "technical writing",
    "documentation",
    "readme creation", 
    "information synthesis",
    "content organization",
    "clarity and precision"
]
qdrant_embedding="all-MiniLM-L6-v2" # Embedding model for Qdrant
embedding_size = 384 # Vector size for Qdrant embeddings

#####################################################################
# Approver (final gate)
[agentic-tools.approver]
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
qdrant_embedding="all-MiniLM-L6-v2"
embedding_size = 384
git_diff_command = ["git", "diff", "staging"] # Command to generate the git diff patch
skills = [
    "code review",
    "quality assurance",
    "decision making",
    "technical analysis", 
    "standards compliance",
    "risk assessment",
    "context analysis"
]
```

### Explanation of Configuration Parameters:

- **`[agentic-tools]` section**:

  - `project_name`: The display name of your project.
  - `project_description`: A brief overview of your project's purpose.
  - `design_docs`: A list of paths to important design documents that provide overarching context for agents.
  - `source`: Directories containing your primary application logic.
  - `project_root`: The root directory from which paths are resolved.
  - `docs`: The directory for general documentation.
  - `tests_directory`: The directory containing your test files.
  - `project_directories`: A list of all directories that should be scanned to gather context for the agents.
  - `include_extensions`: A list of file extensions to include when gathering project context (e.g., `.py`, `.md`, `.toml`).
  - `exclude_files`: Specific file names to ignore during context gathering (e.g., `__init__.py`).
  - `exclude_directories`: Directories to completely skip during context gathering (e.g., `venv`, `.git`).
  - `max_file_bytes`: Prevents large individual files from being processed, avoiding excessive LLM token usage.
  - `max_total_bytes`: Sets an overall limit on the combined size of all file contents fed to the LLM.

- **`[agentic-tools.readme_writer]` and `[agentic-tools.approver]` sections**:
  These are agent-specific configurations. Each agent can have its own settings:

  - `prompt`: The specific instruction given to the LLM for this agent's task. This defines the agent's role and expected output.
  - `model_name`: The name of the LLM model to use (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`).
  - `temperature`: Controls the randomness of the LLM's output. Lower values mean more deterministic results.
  - `description`: A short description of the agent's function.
  - `model_provider`: The LLM provider (e.g., `google`).
  - `alternative_model`, `alternative_model_provider`: Fallback options for LLM model and provider.
  - `skills`: A list of skills or capabilities associated with the agent.
  - `qdrant_embedding`: The name of the embedding model used for Qdrant.
  - `embedding_size`: The dimension of the vectors stored in Qdrant.
  - `git_diff_command` (specific to `approver`): The shell command used to generate the code patch for review. By default, `["git", "diff", "staging"]` compares current changes to the `staging` branch. Adjust as needed (e.g., `["git", "diff", "main"]` or `["git", "diff"]` for uncommitted changes).

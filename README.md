# Multi-Agent Code Quality Enforcer

A robust, multi-stage agentic workflow designed to automatically enforce high code quality standards, including linting, formatting, docstring generation, and import organization. This system uses a fault-tolerant validation and fixing pipeline to ensure all generated code adheres to strict project standards.

## Key Features

The core functionality is provided by the **Code Quality Enforcer Workflow**, a three-stage, multi-agent pipeline that guarantees code correctness and quality.

| Feature | Description | Agents Involved |
| :--- | :--- | :--- |
| **Pre-Fixing Loop** | Automatically fixes existing linting, formatting (`black`, `ruff`), and typing (`mypy`) errors before the main transformation. This ensures the Commentator agent receives clean, valid code. | `developer` |
| **Code Enhancement** | The primary stage where the `commentator` agent enhances docstrings, organizes imports, and applies structural improvements based on project standards. | `commentator` |
| **Post-Fixing Loop** | A critical, fault-tolerant loop that validates the Commentator's output and uses the `developer` agent to fix any validation errors (e.g., syntax or new linting issues) introduced by the LLM. | `developer` |
| **Dependency Consistency** | Pre-fixed files are immediately written to disk, resolving recursive fixing issues and ensuring subsequent files processed in a directory run against the latest clean dependencies. | N/A |
| **Validation Service** | Utilizes a dedicated `ValidationService` to run external tools (`black --check`, `ruff check`, `mypy`) on in-memory code efficiently. | N/A |

______________________________________________________________________

## Prerequisites

To run the agents and the quality enforcement workflow, you need the following:

1. **Python Environment:** Python 3.10 or higher.
1. **LLM Access:** A configured API key for the Google Gemini API.
   - The agent configuration expects the key to be set as an environment variable (e.g., `GEMINI_API_KEY`).
1. **Code Quality Tools:** The external tools used by the `ValidationService` must be installed and accessible in your environment:
   - `black` (Code Formatter)
   - `ruff` (Linter)
   - `mypy` (Static Type Checker)

## Installation

Assuming you have cloned the repository and set up a virtual environment:

```bash
# 1. Install project dependencies
pip install -r requirements.txt

# 2. Set your API Key (replace GEMINI_API_KEY with the actual name used in your config)
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

## Usage

The Code Quality Enforcer is exposed via the `commentator_tool`. It can be run on a single file or an entire directory.

### 1. Run on a Single File

To enforce quality standards on a specific Python file:

```bash
# Example: Running the workflow on a file named 'my_module.py'
python main.py commentator_tool --path src/my_module.py
```

### 2. Run on a Directory

To process all Python files recursively within a directory, ensuring dependency consistency between files:

```bash
# Example: Running the workflow on the entire 'src' directory
python main.py commentator_tool --path src/
```

## Configuration

The workflow is configured primarily in the project's TOML configuration file (e.g., `agentic_tools.toml`).

### Workflow Settings (`[code_quality_enforcer]`)

This section controls the behavior of the multi-stage pipeline:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_fix_attempts` | `int` | `3` | Maximum number of times the `developer` agent will attempt to fix validation errors in a loop before giving up on a file. |
| `run_pre_validation` | `bool` | `true` | If `true`, the Pre-Fixing Loop runs to clean the code before the `commentator` agent is invoked. Highly recommended. |

### Agent Configuration

The workflow relies on two specialized agents:

| Agent | Purpose | Key Configuration |
| :--- | :--- | :--- |
| `developer` | Used exclusively for the deterministic fixing loops (pre- and post-commenting). It is configured with a constrained prompt and low temperature (`temperature=0.0`) to ensure reliable, non-creative fixes based purely on validation errors. | `[developer]` section |
| `commentator` | The primary agent responsible for enhancing docstrings, organizing imports, and applying high-level code quality improvements. | `[commentator]` section |

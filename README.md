# Multi-Agent Model Context Protocol (MCP) System

**An agentic toolchain for architecting, designing, validating, and approving code via chained tools that leverage the Model Context Protocol.**

---

## Table of Contents
1. [Key Features](#key-features)
2. [Prerequisites & System Requirements](#prerequisites--system-requirements)
3. [Installation](#installation)
4. [Quick Start / Usage](#quick-start--usage)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## Key Features

| Feature | What it does | Why it matters |
|---------|--------------|----------------|
| **FastMCP framework** | Registers each tool (`approver`, `readme_writer`, etc.) as a callable endpoint | Simple, uniform interface for LLMs to access tools |
| **Multi-agent pipeline** | Separate agents for *Architect*, *Designer*, *Developer*, *Debugger*, *Validator*, *Triager*, *Writer*, *Version Control*, *Grapher*, *Auditor*, *Readme Writer*, and *Approver* | Clear single-responsibility components; easy to extend or replace |
| **Context-aware tooling** | Each agent receives project context including source code, configuration, and documentation | Guarantees accurate, project-specific responses |
| **Policy-driven context assembly** | `ContextPolicy` controls which files are collected, size limits, and discovery rules | Prevents resource exhaustion and keeps LLM calls focused |
| **Robust error handling** | All LLM calls are wrapped, errors are chained, and failures abort fast | Guarantees traceable failures and no silent corruption |
| **Test-driven development** | Comprehensive test suites ensure correctness and regression safety | Maintains high quality standards as the system evolves |

---

## Prerequisites & System Requirements

| Requirement | Minimum version / note |
|-------------|------------------------|
| **Python** | 3.10 or newer (type-hint rich, `from __future__ import annotations`) |
| **LLM provider credentials** | API keys for at least one of the configured providers (`google`, `groq`, `cerebras`, `sambanova`). Keys are read from environment variables (`GOOGLE_API_KEY`, `GROQ_API_KEY`, etc.) |
| **Git** | Required for the *Version Control* agent (standard `git` CLI) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nnikolov3/multi-agent-mcp.git
cd multi-agent-mcp

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .
# Or more specifically, install the requirements based on uv.lock
uv pip install -r requirements.txt
```

**Note:** The exact dependency set is managed by `uv.lock`. You can use `uv` for faster installation or regular `pip` tools.

---

## Quick Start / Usage

### Running the MCP server

```bash
# Start the FastMCP server (exposes the registered tools)
python main.py
```

This will start the server on the default port, exposing both the `approver_tool` and `readme_writer_tool` as MCP tools.

### Using the Readme Writer tool

The `readme_writer_tool` can be called (via MCP protocol) to generate project documentation:

```python
from main import readme_writer_tool

result = readme_writer_tool()
print(result["data"]["readme_content"])  # Generated README content
```

### Using the Approver tool

The `approver_tool` can be called to get approval decisions on code changes:

```python
from main import approver_tool

result = approver_tool("Add a new feature to the system")
print(result)  # JSON decision object
```

---

## Configuration

All runtime settings live in **`conf/mcp.toml`**. This file defines:

- Agent-specific configurations (prompts, models, providers)
- Context assembly policies (file patterns, exclusions, size limits)
- LLM provider settings and fallback strategies

Key sections for the readme_writer_tool:

```toml
[multi-agent-mcp.readme_writer]
prompt = """
You are an expert technical writer. Create excellent, concise, and practical README documentation...
"""
model_name = "gpt-oss-120b"
temperature = 0.3
description = "Generates high-quality README documentation"
model_providers = ["groq", "cerebras", "sambanova"]
alternative_model = "models/gemini-2.5-flash"
alternative_model_provider = ["google"]
```

### Context-assembly policy (shared by all agents)

```toml
# Context assembly policy (applies to every tool)
recent_minutes = 10
src_dir = "src"
include_extensions = [".py", ".rs", ".go", ".ts", ".tsx", ".js", ".json", ".md", ".toml", ".yml", ".yaml"]
exclude_dirs = [".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]
max_file_bytes = 262144          # 256 KB per file
max_total_bytes = 1048576       # 1 MB total payload
docs_paths = ["DESIGN_PRINCIPLES_GUIDE.md", "CODING_FOR_LLMs.md"]
```

* **`recent_minutes`** – Only files modified in the last 10 minutes are sent to the LLM (reduces token usage)  
* **`include_extensions`** – Limits the payload to relevant source and documentation files  
* **`exclude_dirs`** – Prevents leaking internal or generated artifacts (e.g., `.git`)  

---

## Project Structure

```
multi-agent-mcp/
├── main.py                     # FastMCP entry-point; registers tools
├── conf/
│   └── mcp.toml                # Central configuration (agents, policies)
├── src/
│   ├── __init__.py
│   ├── _api.py                 # Unified LLM API wrapper (providers, retry, etc.)
│   ├── approver.py             # Final gatekeeper implementation
│   ├── configurator.py         # Loads TOML, builds ContextPolicy objects
│   ├── readme_writer_tool.py   # Intelligent README generator
│   └── shell_tools.py          # Helpers for file discovery & source collection
├── docs/
│   ├── DESIGN_PRINCIPLES_GUIDE.md
│   ├── CODING_FOR_LLMs.md
│   └── PROMPT_ENGINEERING.md
├── tests/
│   ├── test_api.py
│   ├── test_approver.py
│   ├── test_approver_after_fixes.py
│   ├── test_readme_writer_tool.py
│   └── test_google_integration.py
├── .mega-linter.yml            # CI linting configuration
├── mypy.ini                    # Type checking configuration
├── uv.lock                     # Locked dependency set for `uv`
└── README.md                   # This file
```

* **`src/`** – All Python implementation files. Each agent lives in its own module following the *single-responsibility* principle.
* **`conf/`** – Declarative configuration; no code changes needed to switch providers or adjust policies.
* **`docs/`** – Design-principles, coding standards, and best-practice guides that the tools can reference.
* **`tests/`** – Full test suite; every public method is exercised and linted.

---

## Contributing

1. **Fork the repository** and create a feature branch
2. **Write tests first** (TDD). Place them under `tests/` and ensure they fail before implementation.
3. **Implement the feature** respecting the *Foundational Design Principles* (simplicity, explicitness, SRP, etc.)
4. **Run the full quality suite:**
   
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Run type checker
   python -m mypy src/
   
   # Run linters as configured
   # (The system uses Mega-Linter as defined in .mega-linter.yml)
   ```

5. **Commit with a clear message** that follows conventional commits if possible
6. **Open a Pull Request** – the system will validate compliance

### Code Style

* **Formatting** – Follow existing patterns in the codebase
* **Linting** – Adhere to the rules enforced by the Mega-Linter configuration
* **Typing** – Full type hints required; run `mypy --strict`
* **Immutability** – Prefer frozen dataclasses and immutable data structures where appropriate

---

## License

This project is released under the **MIT License**. See the `LICENSE` file at the repository root for the full text.

---

*Generated by the **Multi-Agent MCP System** using the readme_writer_tool.*
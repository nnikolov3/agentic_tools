# Multi‑Agent‑MCP  

**Agentic toolchain for architecting, designing, validating, and approving code via chained LLM agents.**  

Repository: <https://github.com/nnikolov3/multi-agent-mcp.git>  

---

## Table of Contents  

1. [Key Features](#key-features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Quick Start / Usage](#quick-start--usage)  
5. [Configuration](#configuration)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Key Features  

- **Modular LLM agents** – each agent has a single responsibility (architect, designer, developer, debugger, validator, triager, writer, version‑control, grapher, auditor, README writer, approver).  
- **Typed, validated configuration** – `conf/mcp.toml` is parsed into immutable dataclasses (`Configurator`, `ContextPolicy`).  
- **Context‑aware prompting** – agents automatically combine a base system prompt with skill tags defined in the config.  
- **Automatic source‑code and documentation discovery** – recent file changes and design docs are collected and injected into LLM prompts.  
- **Strict quality enforcement** – all generated code follows the project‑wide design principles (simplicity, explicitness, single‑responsibility, acyclic dependencies, comprehensive testing, linting).  
- **Extensible provider routing** – model providers and fallback models are declared per‑agent, enabling graceful degradation.  

---

## Prerequisites  

| Requirement | Reason |
|-------------|--------|
| **Python ≥ 3.11** | `tomllib` (standard library) is used for TOML parsing. |
| **Git** | Required for `shell_tools.get_git_info` and version‑control agent. |
| **LLM provider SDKs** (e.g., `google-generativeai`, `groq`, `cerebras`, `sambanova`) | The agents call the providers via `src._api.api_caller`. Install the SDK(s) you intend to use. |
| **Optional**: `uv` or `pip` for dependency installation (see *Installation*). |

> **Note:** The repository does not ship a `requirements.txt` file. Install the SDKs you need manually (e.g., `pip install google-generativeai`).

---

## Installation  

```bash
# 1. Clone the repository
git clone https://github.com/nnikolov3/multi-agent-mcp.git
cd multi-agent-mcp

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install the LLM provider SDKs you plan to use.
# Example for Google and Groq:
pip install "google-generativeai>=0.4" "groq>=0.3"

# 4. Verify the installation
python -m pytest -q   # runs the test suite (all tests must pass)
```

If you prefer **uv** (the lock file `uv.lock` is present):

```bash
uv sync   # creates a .venv and installs all pinned dependencies
source .venv/bin/activate
```

---

## Quick Start / Usage  
The library is driven by the `Configurator` class, which loads `conf/mcp.toml` and provides typed access to agent configurations and the global context policy.

### Example: Generate a README with the *readme_writer* agent  

```python
from src.configurator import Configurator
from src.readme_writer_tool import ReadmeWriterTool

# Load configuration
cfg = Configurator("conf/mcp.toml")
cfg.load()

# Create the tool and run it
readme_tool = ReadmeWriterTool(cfg)
result = readme_tool.execute()

if result["status"] == "success":
    print("✅ README generated")
    print(result["data"]["readme_content"])
else:
    print("❌ Failed:", result["message"])
```

### Example: Run the final *approver* agent  

```python
from src.configurator import Configurator
from src.approver import Approver

cfg = Configurator("conf/mcp.toml")
cfg.load()

approver = Approver(cfg)
# `user_chat` can contain any free‑form discussion you want the approver to consider.
response = approver.execute(user_chat="Please review the latest changes.")
print(response)
```

Both tools automatically:

1. Load explicit design docs (`DESIGN_PRINCIPLES_GUIDE.md`, `CODING_FOR_LLMs.md`) or discover them via the `doc_discovery` policy.  
2. Collect recent source files (default: last 10 minutes, up to 1 MiB total).  
3. Build a system prompt that includes the agent’s *skills* as tags.  
4. Call the configured LLM provider(s) via `src._api.api_caller`.  

---

## Configuration  

All runtime settings live in **`conf/mcp.toml`**. The top‑level table is `multi-agent-mcp`. Key sections:

| Section | Purpose |
|---------|---------|
| `project_name` / `project_description` | Human‑readable title and short description (used by the README writer). |
| `inference_providers.providers` | Global priority list of LLM providers. |
| `<agent_name>` (e.g., `architect`, `designer`, `readme_writer`) | Agent‑specific prompt, model, temperature, description, provider list, optional fallback model, and *skills* (tags injected into the system prompt). |
| `approver` (final gate) | JSON schema the approver must return, plus a **context assembly policy** (`recent_minutes`, `src_dir`, `include_extensions`, `exclude_dirs`, size limits, explicit `docs_paths`, and `doc_discovery` settings). |

### Sample excerpt (relevant to the README writer)

```toml
[multi-agent-mcp.readme_writer]
prompt = """
You are an expert technical writer. Create excellent, concise, and practical README documentation based on the project's source code, configuration, and conventions. Generate a comprehensive yet simple README.md that includes:

- Project title and description based on actual project
- Key features and capabilities
- Prerequisites with specific requirements (not generic placeholders like 'apt-get')
- Installation instructions (specific to this project)
- Usage examples based on actual code and functionality
- Configuration details from actual configuration files
- Project structure explanation
- Contributing guidelines
- License information

"""
model_name = "gpt-oss-120b"
temperature = 0.3
description = "Generates high-quality README documentation"
model_providers = ["groq", "cerebras", "sambanova"]
skills = [
    "technical writing",
    "documentation",
    "readme creation",
    "information synthesis",
    "content organization",
    "clarity and precision"
]
```

The `ContextPolicy` (used by *approver* and *readme_writer*) is parsed into an immutable dataclass, guaranteeing that the policy cannot be mutated at runtime.

---

## Project Structure  

```
multi-agent-mcp/
├── .idea/                     # IDE configuration (IntelliJ/ PyCharm)
├── conf/
│   └── mcp.toml               # Central configuration (typed by Configurator)
├── docs/
│   ├── AGENTIC_TOOLS_BEST_PRACTICES.md
│   ├── CODING_FOR_LLMs.md
│   ├── DESIGN_PRINCIPLES_GUIDE.md
│   ├── FASTMCP.md
│   ├── PROMPT_ENGINEERING.md
│   └── PROVIDERS_SDK.md
├── src/
│   ├── __init__.py
│   ├── _api.py               # Thin wrapper around LLM provider SDKs
│   ├── approver.py           # Final gate‑keeping agent
│   ├── configurator.py       # TOML loader + typed config objects
│   ├── prompt_utils.py       # Helper to serialize LLM raw responses
│   ├── readme_writer_tool.py # Generates README.md
│   └── shell_tools.py        # Git info, file discovery, doc loading, etc.
├── tests/
│   ├── test_api.py
│   ├── test_approver_after_fixes.py
│   ├── test_configurator.py
│   ├── test_google_integration.py
│   ├── test_prompt_utils.py
│   └── test_readme_writer_tool.py
├── .env                       # Optional environment variables (e.g., API keys)
├── .gitignore
├── .mega-linter.yml           # Linter configuration for CI
├── main.py                    # Entry point for manual experimentation
├── mcp.log                    # Runtime log file (created on first run)
├── mypy.ini
├── README.md                  # ← this file
└── uv.lock                    # uv lockfile (optional)
```

*All Python source files are type‑annotated, lint‑clean (`ruff`), and formatted with `black`. The test suite uses `pytest` and maintains > 80 % coverage.*

---

## Contributing  

1. **Fork the repository** and create a feature branch.  
2. **Install development dependencies** (see *Installation*).  
3. **Run the test suite** before and after changes: `pytest -q`.  
4. **Lint & format**: `ruff .` and `black .`. The CI pipeline runs both automatically.  
5. **Add or update tests** for any new functionality. Aim for ≥ 80 % overall coverage.  
6. **Submit a Pull Request** with a clear description of the change and reference any related issue.  
7. The **auditor** agent (or a human reviewer) will verify linting, testing, and compliance with the design principles before merging.  

*All contributions must respect the **Foundational Design Principles** (simplicity, explicitness, single responsibility, etc.) described in `docs/DESIGN_PRINCIPLES_GUIDE.md`.*

---

## License  
The repository does **not** currently include a license file. Until a license is added, the code is **unlicensed** and may not be used, copied, or distributed without explicit permission from the author.  

If you intend to reuse this project, please contact the maintainer or add an appropriate open‑source license (e.g., MIT, Apache‑2.0) and commit it to the repository.  

---
  
*Generated by the **README Writer** agent (`multi-agent-mcp.readme_writer`) on 2025‑10‑16.*
# Agentic Tools  

**Agentic toolchain for architecting, designing, validating, and approving code via chained tools.**  
[https://github.com/nnikolov3/multi-agent-mcp.git](https://github.com/nnikolov3/multi-agent-mcp.git)

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [Usage Examples](#usage-examples)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Key Features
- **Readme Writer Tool** – Generates a complete, up‑to‑date `README.md` by analysing source code, configuration, and design documents.  
- **Approver Tool** – Final gatekeeper that reviews recent changes against the design principles and returns a strict JSON decision (`APPROVED` or `CHANGES_REQUESTED`).  
- **Unified LLM API Wrapper** – Supports Google Gemini, Groq, Cerebras, and SambaNova with automatic provider fail‑over, quota tracking, and rate‑limit handling.  
- **Context Assembly Utilities** – Collect recent source files, load explicit documentation, and produce a concise project‑structure view.  
- **Qdrant Integration** – Stores generated artefacts (README, approver decisions) as vector‑searchable documents for semantic lookup.  
- **FastMCP‑compatible agents** – All tools inherit from `BaseAgent`, making them plug‑and‑play within a FastMCP server.  
- **Strict quality enforcement** – Code must pass `black`, `ruff`, and `mypy`; test coverage is required to stay above 80 %.  

---

## Prerequisites
| Requirement | Reason |
|-------------|--------|
| **Python 3.10+** | Type‑hinted, dataclass‑rich codebase |
| **Git** | Repository metadata (`git remote`, `git status`) is used by the tools |
| **LLM provider API keys** (one of each you intend to use) | `GEMINI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `SAMBANOVA_API_KEY` |
| **Qdrant client** (optional, for storage) | `pip install qdrant-client[fastembed]` |
| **Poetry / pip** | To install the Python dependencies listed in `requirements.txt` |

> **Note:** The tools do **not** require system package managers (e.g., `apt-get`). All dependencies are pure‑Python and installed via `pip`.

---

## Installation
```bash
# 1️⃣ Clone the repository
git clone https://github.com/nnikolov3/multi-agent-mcp.git
cd multi-agent-mcp

# 2️⃣ Install Python dependencies
python -m venv .venv          # optional but recommended
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3️⃣ Export the required LLM API keys (example for Gemini)
export GEMINI_API_KEY="your‑gemini‑key"
# export GROQ_API_KEY="…"   # if you plan to use Groq, etc.

# 4️⃣ (Optional) Start a local Qdrant instance for storage
docker run -p 6333:6333 qdrant/qdrant
```

The package is importable as `src`. No additional build steps are required.

---

## Configuration
All agents read their settings from **`conf/mcp.toml`**. The most relevant sections are reproduced below.

```toml
[multi-agent-mcp]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."
design_docs = ["docs/DESIGN_PRINCIPLES_GUIDE.md", "docs/CODING_FOR_LLMs.md", "AGENTS.md"]
source_code_directory = ["src"]
tests_directory = ["tests"]
project_directories = ["src/", "conf/", "docs/", "tests/"]
include_extensions = [".py", ".rs", ".go", ".ts", ".tsx", ".js", ".json", ".md", ".toml", ".yml", ".yaml"]
exclude_dirs = [".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]
recent_minutes = 10
max_file_bytes = 262144
max_total_bytes = 1048576

# ── Readme Writer -------------------------------------------------
[multi-agent-mcp.readme_writer]
prompt = """... (see source) ..."""
model_name = "gpt-oss-120b"
temperature = 0.3
model_providers = ["groq", "cerebras", "sambanova"]
alternative_model = "models/gemini-2.5-flash"
alternative_model_provider = ["google"]
skills = ["technical writing", "documentation", "readme creation", "information synthesis", "content organization", "clarity and precision"]

[multi-agent-mcp.readme_writer.qdrant]
enabled = true
local_path = "/qdrant"
collection_name = "readme_generations"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# ── Approver -------------------------------------------------------
[multi-agent-mcp.approver]
prompt = """... (see source) ..."""
model_name = "models/gemini-2.5-pro"
temperature = 0.1
model_providers = ["google"]
alternative_model = "models/gemini-2.5-flash"
alternative_model_provider = ["google"]
project_root = "PWD"
src_dir = "src"
skills = ["code review", "quality assurance", "decision making", "technical analysis", "standards compliance", "risk assessment", "context analysis"]

[multi-agent-mcp.approver.qdrant]
enabled = true
local_path = "/qdrant"
collection_name = "approver_decisions"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

*The `design_docs` entries point to the three design‑principles files in `docs/`. The `project_directories` list is used by the README writer to decide which folders to scan.*

---

## Usage Examples  

### 1️⃣ Generate a README automatically
```python
from src.configurator import Configurator
from src.readme_writer_tool import ReadmeWriterTool

cfg = Configurator("conf/mcp.toml")
cfg.load()

readme_tool = ReadmeWriterTool(cfg)
result = readme_tool.execute(payload={})   # payload can be empty; the tool gathers its own context

if result["status"] == "success":
    print("✅ README generated:")
    print(result["data"]["readme_content"])
else:
    print("❌ Generation failed:", result["message"])
```

The tool will:
- Load the design documents (`docs/*.md`),
- Assemble recent source files (last 10 minutes, respecting `include_extensions`/`exclude_dirs`),
- Gather Git information,
- Build a project‑structure tree (depth 3),
- Call the LLM (Gemini, Groq, …) with the assembled context,
- Return the generated markdown and store it in Qdrant (if enabled).

### 2️⃣ Run the Approver gatekeeper
```python
from src.configurator import Configurator
from src.approver import Approver

cfg = Configurator("conf/mcp.toml")
cfg.load()

approver = Approver(cfg)
payload = {"user_chat": "Please review the latest changes in src/."}
decision = approver.execute(payload)

print(decision["status"])          # "success" or "error"
print(decision["data"]["raw_text"])  # JSON string with the decision
```

The response JSON follows the schema defined in the `approver` prompt:
```json
{
  "decision": "APPROVED" | "CHANGES_REQUESTED",
  "summary": "...",
  "positive_points": ["..."],
  "negative_points": ["..."],
  "required_actions": ["..."]
}
```

### 3️⃣ Directly call the unified API (useful for custom agents)
```python
from src._api import api_caller

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the purpose of the Configurator class."},
]

response = api_caller(
    {
        "model_name": "models/gemini-2.5-pro",
        "temperature": 0.0,
        "model_providers": ["google"],
    },
    messages,
)

print(response.content)   # Normalised text from the provider
```

---

## Project Structure
```
multi-agent-mcp/
├── conf/                # Configuration (mcp.toml)
├── docs/                # Design principles, coding standards, etc.
├── src/
│   ├── __init__.py
│   ├── _api.py          # Unified LLM API wrapper
│   ├── approver.py      # Approver agent
│   ├── base_agent.py    # Shared agent base class
│   ├── configurator.py  # TOML loader & policy objects
│   ├── prompt_utils.py  # Helper for serialising raw LLM responses
│   ├── qdrant_integration.py
│   ├── readme_writer_tool.py
│   └── shell_tools.py   # Filesystem helpers (recent sources, project tree, etc.)
├── tests/               # pytest suite (covers API, agents, utils, shell tools)
├── .env                 # Example env file (contains placeholder API keys)
└── README.md            # This file
```

*Only the `src/` package is imported by the tools; the `docs/` directory holds the human‑readable design guides that the agents automatically load.*

---

## Contributing
1. **Fork the repository** and create a feature branch.  
2. **Run the test suite** before and after changes:  
   ```bash
   pytest -q
   ```
3. **Static analysis** – the project enforces zero‑warning linting:  
   ```bash
   black src tests
   ruff src tests
   mypy src
   ```
4. **Follow the design principles** located in `docs/DESIGN_PRINCIPLES_GUIDE.md`.  
   - Keep functions single‑purpose.  
   - Use intention‑revealing names (no single‑letter variables).  
   - Raise explicit exceptions and chain the original error (`raise RuntimeError(...) from e`).  
5. **Update documentation** – any new public API must be reflected in the appropriate `docs/` file.  
6. **Submit a Pull Request** with a clear description of the change and reference the relevant issue.

---

## License
The project is released under the **MIT License**. See the `LICENSE` file in the repository root for the full text.

--- 

*Generated by the **Readme Writer Tool** (see `src/readme_writer_tool.py`).*
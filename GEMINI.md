# Project: Multi-Agent-MCP

## Project Overview

The `Multi-Agent-MCP` project is a Python-based multi-agent system designed to automate and streamline various software engineering tasks. It leverages a custom `FastMCP` framework to orchestrate a suite of specialized AI agents, each with a distinct role in the software development lifecycle. These roles include Architect, Designer, Developer, Debugger, Validator, Triager, Writer, Version Control, Grapher, and Auditor.

A central `approver` agent acts as a final gatekeeper, making critical decisions based on comprehensive context, including design documents, recent code changes, and user interactions. The system is built for resilience, utilizing multiple Large Language Model (LLM) providers (Groq, SambaNova, Cerebras, Google) with a sophisticated fallback mechanism to ensure continuous operation even if primary providers fail.

Key functionalities include:
*   **Agent Orchestration:** Managing and coordinating various AI agents for different tasks.
*   **Flexible Configuration:** Agents are configured via `conf/mcp.toml`, allowing for easy customization of prompts, models, and LLM providers.
*   **Resilient LLM API Calls:** The `src/_api.py` module provides dynamic and robust API calling logic with provider health checks, response normalization, and error handling.
*   **Context-Aware Decision Making:** The `approver` agent gathers extensive context (documentation, code, chat history) to inform its decisions.

## Building and Running

This project appears to be a Python application.

**Dependencies:**
Dependencies are managed using `uv` (indicated by `uv.lock`). To install dependencies, navigate to the project root and run:
```bash
uv sync
```

**Running the MCP Server:**
The main entry point is `main.py`. To start the Multi-Agent Control Plane (MCP) server, execute:
```bash
python main.py
```
This will start the `FastMCP` server, making the registered agents (like the `approver_tool`) available.

**Testing:**
(TODO: Add specific testing commands here. Look for a `pytest.ini` or `tests/` directory and infer commands.)

## Development Conventions

*   **Language:** Python
*   **Configuration:** Agent configurations are managed in `conf/mcp.toml`.
*   **LLM Integration:** Utilizes a custom API abstraction (`src/_api.py`) for interacting with various LLM providers.
*   **Agent Structure:** Agents are implemented as Python classes (e.g., `src/approver.py`) that take a configuration dictionary and implement an `execute` method.
*   **Type Checking:** The presence of `mypy.ini` suggests that MyPy is used for static type checking.
*   **Linting:** The `.mega-linter.yml` file indicates that Mega-Linter is used for code quality and style enforcement.

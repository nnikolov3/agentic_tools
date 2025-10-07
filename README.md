# Multi-Agent Development Team

## Overview

This project implements a multi-agent orchestration system for software development. It leverages a team of 22 specialized AI agents, each with a specific role (e.g., `tech_lead`, `architect`, `developer`, `qa_engineer`), to automate the software development lifecycle. The system uses a `FastMCP` server to expose its functionality through a set of tools.

## Core Features

- **Project Indexing:** The script automatically scans the project directory, identifies source code files (Go, Python, Shell, Markdown, etc.), and indexes them in a Qdrant vector database. This allows the agents to have contextual awareness of the existing codebase.
- **Incremental Updates:** It tracks file modification times to only re-index files that have been changed, making the process efficient.
- **Multi-Agent Team:** It defines a team of 22 specialized agents with different roles and capabilities, leveraging models from multiple API providers (OpenAI, Groq, Google, Cerebras, Qwen).
- **Role-Based Access Control (RBAC):** The system implements a robust authentication and authorization mechanism. Users are assigned roles (Admin, Developer, Tester, etc.) with specific permissions, controlling their access to different agents and tools.
- **Intelligent Routing:** The script can intelligently route tasks to the most appropriate agent based on the task requirements.
- **Comprehensive Memory:** The system uses Qdrant to store and retrieve information about tasks, architectures, implementations, code reviews, test strategies, and more. This allows the system to learn from past successes and failures.
- **Semantic Caching:** It uses a semantic cache to store the results of LLM calls, which can significantly reduce costs and improve performance by reusing previous results for similar prompts.
- **Tool-Based Interface:** The system exposes its functionality through a set of tools.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nnikolov3/multi-agent-mcp.git
    cd multi-agent-mcp
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add the following environment variables:
    ```
    GEMINI_API_KEY=your_gemini_api_key
    GOOGLE_API_KEY=your_google_api_key
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    CEREBRAS_API_KEY_PERSONAL=your_cerebras_api_key_personal
    CEREBRAS_API_KEY_BOOK_EXPERT=your_cerebras_api_key_book_expert
    DASHSCOPE_API_KEY=your_dashscope_api_key
    QDRANT_URL=your_qdrant_url
    QDRANT_API_KEY=your_qdrant_api_key
    ```
4.  **Run the installation script:**
    ```bash
    ./install.sh
    ```

## Usage

Once the installation is complete, you can use the `gemini` CLI to interact with the multi-agent development team.

### Available Tools

- `develop_feature`: A complete development workflow that uses the multi-agent team to develop a new feature.
- `rescan_project`: Manually triggers a rescan and re-indexing of the project directory.
- `create_user`: Allows an admin to create new users with specific roles.
- `list_team_members`: Lists all the available agents in the team.
- `get_api_usage_stats`: Provides statistics on API usage.
- `search_project_files`: Allows users to perform a semantic search on the indexed project files.

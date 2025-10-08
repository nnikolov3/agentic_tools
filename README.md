# Multi-Agent Development Team MCP Server

Version: 3.1.0  
Author: Nikolay Nikolov  
Date: October 7, 2025

## Overview

A production-ready MCP server that provides **30+ specialized AI agents** with the latest models (October 2025) for comprehensive software development workflows. Works with **Gemini CLI** and **Claude Desktop**.

## Features

✅ **30+ AI Agents** with latest models:
- OpenAI: GPT-5, GPT-5-mini, GPT-4.1, O3/O4 deep research, GPT-OSS-120B
- Groq: Compound, Llama 4 Maverick/Scout, Kimi K2 (256k context)
- Google: Gemini 2.5 Pro/Flash/Flash-Lite
- Cerebras: Llama 4, Qwen 3 235B/480B-Coder (with Groq fallback)
- Qwen/Dashscope: Qwen Max (1T+ params), Coder-480B, QwQ reasoning (with Groq fallback)

✅ **Comprehensive Chat Storage** - Every LLM interaction saved to Qdrant with full metadata

✅ **12 MCP Tools**:
1. `develop_feature` - 12-stage development workflow across all providers
2. `create_user` - User management with role-based access
3. `list_team_members` - View all 30+ agents with model info
4. `get_api_usage_stats` - Provider usage analytics
5. `rescan_project` - Reindex project files
6. `search_project_files` - Semantic codebase search
7. `test_all_providers` - Verify all API connections
8. `search_chats` - Semantic chat search
9. `get_chat_history` - User conversation history
10. `rate_chat` - Rate conversations (1-5 stars)
11. `analyze_chat_patterns` - Chat analytics and insights
12. `export_chats` - Export conversations to JSON

✅ **Project File Indexing** - Automatic scanning and semantic search

✅ **Role-Based Access Control** - Admin, Developer, Tester, Writer, Viewer

✅ **Intelligent Fallbacks** - Cerebras→Groq, Qwen→Groq when not configured

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file or set these in your config:

```bash
OPENAI_API_KEY="your_openai_key"
GROQ_API_KEY="your_groq_key"
GOOGLE_API_KEY="your_google_key"
GEMINI_API_KEY="your_gemini_key"
QDRANT_URL="your_qdrant_cloud_url"
QDRANT_API_KEY="your_qdrant_api_key"
PROJECT_ROOT="/path/to/your/project"

# Optional - for full feature access
CEREBRAS_API_KEY_PERSONAL="your_cerebras_key"
CEREBRAS_API_KEY_BOOK_EXPERT="your_cerebras_key_2"
DASHSCOPE_API_KEY="your_qwen_dashscope_key"
ADMIN_API_KEY="your_admin_api_key"
```

### 3. Configure Gemini CLI

**Location:** `~/.gemini/settings.json` (macOS/Linux) or `%USERPROFILE%\.gemini\settings.json` (Windows)

Copy contents from `gemini_config.json` and update:
- Replace `/absolute/path/to/multi_agent_dev_team.py` with actual path
- Replace all API keys with your real keys

### 4. Test Standalone

```bash
python multi_agent_dev_team.py
```

You should see initialization messages for all providers.

### 5. Restart Gemini CLI

Restart your terminal or Gemini CLI session.

### 6. Verify Tools

In Gemini CLI:
```
/tools
```

You should see all 12 tools listed.

## Usage Examples

### Ask Gemini CLI to use your agents:

**Example 1: List all agents**
```
"List all my available AI agents with their models"
```
→ Calls `list_team_members`

**Example 2: Develop a feature**
```
"Develop a feature: Create a REST API with user authentication, CRUD operations for posts, and rate limiting"
```
→ Calls `develop_feature` with 12-stage workflow using GPT-5, Gemini 2.5, Llama 4, Qwen Max, etc.

**Example 3: Search your codebase**
```
"Search my project files for authentication logic"
```
→ Calls `search_project_files` with semantic search

**Example 4: Test all providers**
```
"Test all my API providers and show which ones are working"
```
→ Calls `test_all_providers`

**Example 5: Get analytics**
```
"Show me my chat analytics and token usage"
```
→ Calls `analyze_chat_patterns` and `get_api_usage_stats`

## Architecture

### Chat Storage
Every agent call is stored in Qdrant with:
- Full prompt and response
- Agent name, model, provider
- Tokens used, temperature, timestamp
- Duration in milliseconds
- Task ID and stage name
- Context usage flag
- Cache hit/miss flag
- User rating and feedback

### 12-Stage Development Workflow

When you call `develop_feature`, it runs:

1. **O3 Deep Research** (OpenAI) - Comprehensive research
2. **GPT-5 Architecture** (OpenAI) - System design
3. **Qwen Coder 480B** (Cerebras/Groq) - Core implementation
4. **Qwen Max 1T+** (Qwen/Groq) - Algorithm optimization
5. **QwQ Reasoning** (Qwen/Groq) - Advanced reasoning
6. **Groq Compound** (Groq) - Rapid prototyping
7. **Llama 4 Maverick** (Groq) - Production code
8. **GPT-4.1 Review** (OpenAI) - Code review
9. **Gemini 2.5 Flash QA** (Google) - Testing strategy
10. **Gemini 2.5 Flash Security** (Google) - Security audit
11. **Gemini 2.5 Pro Docs** (Google) - Documentation
12. **GPT-5 Mini Polish** (OpenAI) - Final polish

## File Structure

```
.
├── multi_agent_dev_team.py     # Main MCP server
├── requirements.txt             # Python dependencies
├── gemini_config.json          # Gemini CLI configuration template
├── claude_desktop_config.json  # Claude Desktop configuration template
├── README.md                   # This file
└── multi_agent_dev_team.log   # Server logs (auto-generated)
```

## Configuration Files

### For Gemini CLI
Location: `~/.gemini/settings.json`

See `gemini_config.json` for template.

### For Claude Desktop
Location (macOS): `~/Library/Application Support/Claude/claude_desktop_config.json`  
Location (Windows): `%APPDATA%\Claude\claude_desktop_config.json`

See `claude_desktop_config.json` for template.

## Troubleshooting

### MCP server not showing up
- Check logs: `multi_agent_dev_team.log`
- Verify Python path is absolute in config
- Ensure all required dependencies are installed

### Tools not available
- Restart Gemini CLI / Claude Desktop
- Check config file syntax (valid JSON)
- Verify file permissions

### API key errors
- Ensure all required keys are set correctly
- Required: OpenAI, Groq, Google, Gemini, Qdrant
- Optional: Cerebras, Dashscope (will fallback to Groq)

### Qdrant connection failed
- Verify QDRANT_URL and QDRANT_API_KEY
- Check network connectivity to Qdrant Cloud
- Ensure Qdrant cluster is running

## Advanced Features

### Role-Based Access Control

5 roles with different permissions:
- **Admin** - Full access to all tools
- **Developer** - Development, QA, docs tools
- **Tester** - QA tools only
- **Writer** - Documentation tools only
- **Viewer** - Read-only access

Create users with `create_user` tool.

### Chat Analytics

Track and analyze:
- Total chats and tokens used
- Agent usage distribution
- Provider usage distribution
- Average ratings
- Cache hit rate

Use `analyze_chat_patterns` tool.

### Project File Indexing

Automatic indexing of:
- Source code (.go, .py, .rs, .c, .h)
- Documentation (.md, .txt)
- Configuration (.yaml, .json, .toml)

Semantic search with `search_project_files`.

## License

Private use for Nikolay Nikolov.

## Support

Check logs at `multi_agent_dev_team.log` for debugging information.

# Agentic Tools - Agents Documentation

This document provides comprehensive information about all the agents available in the Agentic Tools system, including their purpose, configuration, and usage guidelines.

## Table of Contents
1. [Overview](#overview)
2. [Available Agents](#available-agents)
3. [Agent Configuration](#agent-configuration)
4. [Usage Guidelines](#usage-guidelines)
5. [Integration with FastMCP](#integration-with-fastmcp)
6. [Development Guidelines](#development-guidelines)

---

## Overview

The Agentic Tools system is built around a collection of specialized agents that work together to perform various software development tasks. Each agent has a specific responsibility and follows the design principles outlined in the project documentation.

The system is built on the FastMCP (Model Context Protocol) framework, which allows AI models to call external tools to perform actions, access data sources, and receive parameterized prompts.

### Key Principles:
- **Single Responsibility**: Each agent has one clear purpose
- **Context Awareness**: Agents gather project context to provide accurate responses
- **Skill-Enhanced**: Agents can leverage defined skills to improve response quality
- **Policy Driven**: All agents follow configurable context assembly policies
- **Quality Focused**: All agents adhere to the project's coding and design standards

For a complete list of design principles, see DESIGN_PRINCIPLES_GUIDE.md.

---

## Available Agents

### 1. Approver Tool (`approver_tool`)
**Purpose**: Final gatekeeper in the software development pipeline that analyzes design documents, recent code changes, and user chat history to make approval decisions.

**Function Signatures**: `approver_tool(user_chat: str) -> Dict[str, Any]`

**Model**: `models/gemini-2.5-pro`
**Temperature**: 0.1 (conservative, precise responses)
**Providers**: `google` (primary), with fallbacks
**Description**: Reviews code changes against design principles and returns approval decisions
**Skills**: `["code review", "quality assurance", "decision making", "technical analysis", "standards compliance", "risk assessment", "context analysis"]`

**Configuration**:
- Project root: `PWD` (defaults to current directory)
- Context policy defined in conf/mcp.toml
- Includes recent source files, documentation, and configuration context
- Skills are injected as tags in system prompt to enhance decision-making

### 2. Readme Writer Tool (`readme_writer_tool`)
**Purpose**: Intelligent README generator that creates excellent documentation based on best practices, technical writing standards, source code analysis, and local information.

**Function Signature**: `readme_writer_tool() -> Dict[str, Any]`

**Model**: `gpt-oss-120b`
**Temperature**: 0.3 (balanced creativity and precision)
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Generates comprehensive, accurate, and useful README files
**Skills**: `["technical writing", "documentation", "readme creation", "information synthesis", "content organization", "clarity and precision"]`

**Configuration**:
- Gathers context from source files, project structure, configuration, and documentation
- Uses git information including project URL and branch status
- Follows technical writing best practices for documentation
- Skills are injected as tags in system prompt to enhance documentation quality

### 3. Architect Agent (`architect`)
**Purpose**: Researches and plans system architecture based on requirements and constraints.

**Model**: `qwen-3-235b Instruct`
**Temperature**: 0.5
**Providers**: `cerebras` (primary), with alternatives
**Description**: Produces system architecture and implementation plans
**Skills**: `["system architecture", "requirements analysis", "design patterns", "scalability planning", "technical research", "solution design"]`

### 4. Designer Agent (`designer`)
**Purpose**: Produces concrete, testable designs from architectural guidelines.

**Model**: `qwen-3-235b Thinking`
**Temperature**: 0.4
**Providers**: `cerebras` (primary), with alternatives
**Description**: Creates detailed technical designs with single responsibilities
**Skills**: `["software design", "component design", "interface design", "testability planning", "abstraction", "decoupling"]`

### 5. Developer Agent (`developer`)
**Purpose**: Implements designed components with clean, well-documented, type-safe code.

**Model**: `qwen-3-coder-480b`
**Temperature**: 0.3
**Providers**: `cerebras` (primary), with alternatives
**Description**: Creates implementation code that passes linters and tests
**Skills**: `["software development", "clean code", "type safety", "documentation", "testing", "linting compliance"]`

### 6. Debugger Agent (`debugger`)
**Purpose**: Identifies minimal, precise fixes to make code correct and standards-compliant.

**Model**: `gpt-oss-120b`
**Temperature**: 0.2
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Debugs implementations with minimal, precise fixes
**Skills**: `["debugging", "error identification", "root cause analysis", "fix implementation", "code quality", "standards compliance"]`

### 7. Validator Agent (`validator`)
**Purpose**: Verifies that fixes are correct and that requirements and standards are fully met.

**Model**: `gpt-oss-120b`
**Temperature**: 0.1
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Validates fixes and confirms standards compliance
**Skills**: `["validation", "verification", "quality assurance", "requirement checking", "standards compliance", "testing"]`

### 8. Triager Agent (`triager`)
**Purpose**: Identifies repeated issues, categorizes, prioritizes, and proposes resolution strategies.

**Model**: `qwen-3-32b`
**Temperature**: 0.2
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Triages repeated issues with categorization and prioritization
**Skills**: `["issue triage", "categorization", "prioritization", "pattern recognition", "resolution planning", "problem analysis"]`

### 9. Writer Agent (`writer`)
**Purpose**: Updates and improves documentation to reflect the current implementation and standards.

**Model**: `gpt-oss-120b`
**Temperature**: 0.3
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Updates documentation to match current implementation
**Skills**: `["technical writing", "documentation", "technical communication", "style guide adherence", "information synthesis", "clarity and precision"]`

### 10. Version Control Agent (`version_control`)
**Purpose**: Performs atomic, auditable Git operations with clear messages and branches.

**Model**: `qwen-3-32b`
**Temperature**: 0.1
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Handles Git operations with atomic, auditable changes
**Skills**: `["version control", "git operations", "branch management", "commit messaging", "merge strategies", "repository management"]`

### 11. Grapher Agent (`grapher`)
**Purpose**: Generates accurate Mermaid diagrams and visuals for systems and processes.

**Model**: `qwen-3-32b`
**Temperature**: 0.3
**Providers**: `groq, cerebras, sambanova` (primary), with Google fallback
**Description**: Creates system diagrams and visual representations
**Skills**: `["diagram creation", "visualization", "Mermaid syntax", "system modeling", "process mapping", "technical illustration"]`

### 12. Auditor Agent (`auditor`)
**Purpose**: Confirms linting, testing, and quality standards are fully met before commit.

**Model**: `DeepSeek-V3.1`
**Temperature**: 0.1
**Providers**: `sambanova` (primary), with alternatives
**Description**: Audits code pre-commit to ensure quality standards
**Skills**: `["code auditing", "quality assurance", "linting", "testing verification", "standards compliance", "pre-commit validation"]`

---

## Agent Configuration

### Configuration File: `conf/mcp.toml`

All agent configurations are stored in `conf/mcp.toml` using the following structure:

```toml
[multi-agent-mcp]
project_name = "Agentic Tools"
project_description = "Agentic toolchain for architecting, designing, validating, and approving code via chained tools."

[multi-agent-mcp.<agent_name>]
prompt = "Agent-specific system prompt that defines behavior and guidelines"
model_name = "model identifier"
temperature = 0.x (for appropriate creativity/precision balance)
description = "Brief description of the agent's purpose"
model_providers = ["primary_provider", "fallback_provider"]
alternative_model = "alternative_model_name"
alternative_model_provider = ["alternative_provider"]
skills = ["skill1", "skill2", "skill3"]  # Skills are injected as tags in system prompt

# Context assembly policy (shared by all agents)
recent_minutes = 10
src_dir = "src"
include_extensions = [".py", ".rs", ".go", ".ts", ".tsx", ".js", ".json", ".md", ".toml", ".yml", ".yaml"]
exclude_dirs = [".git", ".github", ".gitlab", "node_modules", "venv", ".venv", "dist", "build", "target", "__pycache__"]
max_file_bytes = 262144          # 256 KB per file
max_total_bytes = 1048576       # 1 MB total payload
docs_paths = ["DESIGN_PRINCIPLES_GUIDE.md", "CODING_FOR_LLMs.md"]
```

### Context Assembly Policy

All agents follow the same context assembly policy to ensure consistency:

- **Recent File Limits**: Only files modified within `recent_minutes` are included
- **Extension Filtering**: Only files with specified extensions are included
- **Directory Exclusion**: Certain directories are excluded to prevent information leakage
- **Size Limits**: Both per-file and total payload size limits to prevent resource exhaustion
- **Documentation Discovery**: Automatically discovers relevant documentation based on patterns and keywords
- **Skills Injection**: Agent skills are automatically included as tags in the system prompt to enhance LLM responses

---

## Usage Guidelines

### Best Practices for Using Agents

1. **Provide Comprehensive Context**: When using agents that require user input, provide as much relevant context as possible
2. **Verify Outputs**: Always review agent outputs for accuracy and appropriateness
3. **Follow Design Principles**: Ensure all agent interactions align with the project's design principles
4. **Respect Rate Limits**: Some agents may have rate limiting through their LLM providers
5. **Handle Errors Gracefully**: Agent calls may fail due to provider issues or other errors
6. **Leverage Skills**: Skills configured for each agent enhance their capabilities and focus

### Common Use Cases

- **Code Approval Pipeline**: Use approver_tool as a final gate before code changes
- **Documentation Generation**: Use readme_writer_tool to generate project documentation
- **Development Assistance**: Use architect, designer, and developer agents in sequence for new features
- **Quality Assurance**: Use debugger, validator, and auditor agents to ensure code quality

---

## Integration with FastMCP

### FastMCP Framework

The agents are integrated into the FastMCP (Model Context Protocol) framework:

- **Tools**: Each agent is exposed as a callable tool via `@mcp.tool` decorator
- **Resources**: Agents can access project resources as needed
- **Prompts**: Parameterized message templates for guiding LLMs, enhanced with skills tags
- **Context Server**: Access to advanced MCP features like progress reporting, LLM sampling, and user elicitation

### Main Entry Point (`main.py`)

```python
from fastmcp import FastMCP
from src.approver import Approver
from src.readme_writer_tool import ReadmeWriterTool
from src.configurator import Configurator

mcp = FastMCP("Agentic Tools")

# Load configuration
configurator = Configurator("conf/mcp.toml")
configurator.load()

# Register tools
@mcp.tool
def approver_tool(user_chat: str) -> Dict[str, Any]:
    agent = Approver(configurator)
    return agent.execute(user_chat=user_chat)

@mcp.tool
def readme_writer_tool() -> Dict[str, Any]:
    agent = ReadmeWriterTool(configurator)
    return agent.execute()

if __name__ == "__main__":
    mcp.run()
```

---

## Development Guidelines

### Creating New Agents

When creating new agents, follow these guidelines:

1. **Single Responsibility**: Each agent should have one clear, well-defined purpose
2. **Follow Existing Patterns**: Use the same architectural patterns as existing agents
3. **Respect Configuration**: Use the Configurator to load agent-specific configuration, including skills
4. **Implement Proper Error Handling**: Handle both expected and unexpected errors gracefully
5. **Follow Design Principles**: Adhere to all project design principles and coding standards
6. **Write Comprehensive Tests**: Provide thorough test coverage for the new agent
7. **Document Configuration**: Add appropriate configuration sections to `conf/mcp.toml`
8. **Define Skills**: Include relevant skills in the agent configuration to enhance performance

### Agent Architecture Pattern

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src._api import UnifiedResponse, api_caller
from src.configurator import Configurator, ContextPolicy

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AgentInputs:
    prompt: str
    model_name: str
    model_providers: List[str]
    temperature: float
    # Add other required fields

class NewAgent:
    """
    Agent description explaining its purpose and functionality.
    """
    
    def __init__(self, configurator: Configurator) -> None:
        self._configurator = configurator
    
    def _load_inputs(self) -> AgentInputs:
        # Load configuration from configurator
        pass
    
    def execute(self, *args) -> Dict[str, Any]:
        # Main execution logic
        # - Gather context
        # - Prepare messages for LLM (with skills injected into system prompt)
        # - Call api_caller
        # - Process response
        # - Return structured result
        pass
```

### Testing Guidelines

- Create comprehensive tests for each agent using pytest
- Test both success and error scenarios
- Mock external dependencies appropriately
- Verify that skills are properly incorporated into system prompts
- Verify that the agent follows design principles
- Test the agent's interaction with the configuration system

---

## Quality and Standards

### Design Principles Compliance

All agents must comply with the project's design principles:
- Simplicity is non-negotiable
- Methodical problem decomposition
- Explicit over implicit
- Self-documenting code
- Single responsibility and low complexity
- Acyclic dependencies
- Composition over inheritance
- Error handling excellence
- Test-driven correctness
- Verifiable truth and no deception
- Automated quality enforcement
- Immutability by default
- Efficient memory management
- Consistency reduces cognitive load
- No premature optimization
- Remove what isn't used

### Coding Standards

- Use type hints for all function signatures
- Write self-documenting code with intention-revealing names
- Follow the project's formatting and linting standards
- Chain exceptions to preserve context
- Use immutable data structures where appropriate

---

## Troubleshooting

### Common Issues

1. **No Recent Files Found**: The agent may return a "no_recent_files" status if no files match the context policy criteria
2. **Provider Issues**: Agent calls may fail if LLM providers are unavailable or rate-limited
3. **Configuration Problems**: Ensure the conf/mcp.toml file is properly formatted and contains all required sections including skills
4. **Context Limits**: Large projects may hit file size or total payload limits
5. **Skills Not Applied**: Verify that skills are properly configured and the combine_prompt_with_skills method is being used

### Performance Considerations

- Context assembly can be time-consuming for large projects
- Multiple agents calling the same information gathering functions (like get_git_info) may impact performance
- Subprocess calls for git information should be minimized

---

## Conclusion

The Agentic Tools system provides a comprehensive suite of specialized agents that work together to handle various aspects of software development. Each agent follows consistent patterns and design principles, making the system maintainable, extensible, and reliable.

By following the documented patterns and guidelines, new agents can be added to the system while maintaining consistency and quality. The skills-based approach enhances agent performance by providing explicit context about each agent's specialized capabilities.
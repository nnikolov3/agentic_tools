# Multi-Agent Development Team Context

## Project Overview
This is an enterprise-grade multi-agent orchestration system with:
- 22 specialized AI agents across 5 API providers
- Role-based access control (RBAC)
- Comprehensive Qdrant vector memory with semantic caching
- Intelligent agent routing based on task requirements

## Team Structure

### Leadership Agents (OpenAI)
- **tech_lead** (GPT-5): Strategic decisions, project kickoff
- **architect** (o1): Deep reasoning for system architecture

### Development Agents
- **rapid_developer** (Groq Compound): Ultra-fast development with 450 tps
- **senior_developer** (Cerebras): Production-grade implementation
- **code_specialist** (Qwen): Algorithm optimization

### Quality Assurance
- **fast_qa** (Groq Llama 3.1 8B): High-throughput testing
- **qa_engineer** (Gemini 2.5 Flash): Comprehensive QA strategy
- **security_auditor** (Gemini 2.5 Flash): Security analysis

### Documentation
- **technical_writer** (Gemini 2.5 Pro): READMEs and docs
- **diagram_specialist** (Gemini 2.5 Flash Preview): Mermaid diagrams

## Authentication

All tools require an API key. The admin API key is generated on first run.

### Roles:
- **admin**: Full access to all agents and admin functions
- **developer**: Can call development, QA, and docs agents
- **tester**: QA agents only
- **writer**: Documentation agents only
- **viewer**: Read-only access

## Usage Patterns

### For New Features:
1. Start with `tech_lead` for strategic analysis
2. Use `architect` for system design
3. Employ `rapid_developer` or `senior_developer` for implementation
4. Run `fast_qa` for testing
5. Call `security_auditor` for security review
6. Use `technical_writer` for documentation

### For Quick Prototypes:
- Use `rapid_prototyper` (Groq) for 10x faster iteration

### For Complex Algorithms:
- Use `code_specialist` (Qwen) for optimization

## Qdrant Memory

The system stores and learns from:
- All conversations
- Code implementations
- Success patterns
- Failure patterns
- Optimization techniques

Context is automatically retrieved before each task.

## Environment

Requires these environment variables (loaded from .env):
- GEMINI_API_KEY, GOOGLE_API_KEY
- OPENAI_API_KEY
- GROQ_API_KEY
- CEREBRAS_API_KEY_PERSONAL, CEREBRAS_API_KEY_BOOK_EXPERT
- DASHSCOPE_API_KEY
- QDRANT_URL, QDRANT_API_KEY

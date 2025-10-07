# Agent Selection Guide

## When to Call Which Agent

### Strategic Planning
- **tech_lead**: Project scope, risk assessment, team assignments
- **architect**: System architecture, component design

### Fast Development (Groq)
- **rapid_developer**: When speed matters, prototyping
- **agentic_developer**: Multi-step reasoning, complex workflows
- **fast_qa**: High-throughput testing

### Production Development
- **senior_developer**: Core features, production code
- **code_specialist**: Performance-critical algorithms

### Quality
- **fast_qa**: Unit tests, quick validation
- **qa_engineer**: Comprehensive test strategy
- **security_auditor**: Security reviews

### Documentation
- **technical_writer**: READMEs, API docs
- **diagram_specialist**: Architecture diagrams

## Agent Capabilities

Each agent has access to:
- Past successful implementations (via Qdrant)
- Failure patterns to avoid
- Optimization techniques
- Project-specific context

## Best Practices

1. **Always start with leadership** for major features
2. **Use Groq for speed** when iterating quickly
3. **Run security audits** before production
4. **Document as you go** with technical_writer
5. **Learn from failures** - all logged in Qdrant

# Agentic Tools: Best Practices and Usage Guide

This document provides comprehensive guidance on the available agentic tools in the multi-agent-mcp system, including best practices for their usage and implementation details.

## Available Agentic Tools

### 1. Approver Tool (`approver_tool`)

The `approver_tool` serves as the final gatekeeper in the software development pipeline, making critical decisions about code changes and project direction.

**Signature:** `approver_tool(user_chat: str) -> Dict[str, Any]`

**Purpose:**
- Reviews code changes against design principles and coding standards
- Makes final approval decisions (APPROVED/CHANGES_REQUESTED)
- Provides detailed feedback and actionable recommendations
- Ensures quality and consistency across the codebase

**Best Practices for Using Approver Tool:**
- Provide comprehensive context in the `user_chat` parameter
- Include design documents, recent code changes, and user requirements
- Use the tool as part of a review pipeline before merging changes
- Review the structured output (decision, summary, positive/negative points, required actions)
- Address all `required_actions` before requesting re-approval

**Implementation Details:**
- Uses Google's Gemini 2.5 Pro model for high-quality analysis
- Follows strict JSON response format for automation
- Gathers context from source files, documentation, and configuration
- Implements rate limiting and error handling for reliability

**Quality Gates:**
- Code must follow all design principles (simplicity, explicit over implicit, etc.)
- All required documentation must be present
- Test coverage must exceed 80%
- Type safety checks must pass

### 2. Readme Writer Tool (`readme_writer_tool`)

The `readme_writer_tool` is an intelligent README generator that creates excellent documentation based on best practices and project analysis.

**Signature:** `readme_writer_tool() -> Dict[str, Any]`

**Purpose:**
- Generates comprehensive, accurate, and useful README files
- Analyzes source code, configuration, and project structure
- Creates documentation that follows technical writing best practices
- Provides specific, actionable information (not generic placeholders)

**Best Practices for Using Readme Writer Tool:**
- Run the tool when starting a new project or significantly changing an existing one
- Review and customize the generated content as needed
- Ensure the generated README accurately reflects the current project state
- Use the tool to maintain consistent documentation quality across projects
- Verify that installation and usage instructions are accurate and complete

**Implementation Details:**
- Uses GPT-OSS 120B model with balanced creativity and precision
- Gathers context from source files, project structure, configuration, and documentation
- Follows the same architectural patterns as other tools in the system
- Includes comprehensive test coverage for reliability

**Documentation Standards:**
- Focus on simplicity and clarity over complexity
- Include specific examples based on the actual project (not generic placeholders)
- Provide clear prerequisites and system requirements
- Document the project structure and configuration options
- Include usage examples and contributing guidelines

## Common Best Practices for All Agentic Tools

### 1. Input Quality
- Provide comprehensive, well-formatted inputs to tools
- Ensure input data is clean and relevant
- Follow the expected data structures and formats

### 2. Error Handling
- Always check the `status` field in tool responses
- Handle `error` and `no_recent_files` statuses appropriately
- Implement retry logic for transient failures
- Log tool interactions for debugging and auditing

### 3. Configuration Management
- Maintain configuration in `conf/mcp.toml` for all tools
- Use environment-specific configuration when needed
- Validate configuration parameters during initialization
- Implement fallback mechanisms for high availability

### 4. Testing and Validation
- Create comprehensive test suites for all tool implementations
- Test both success and error scenarios
- Validate adherence to design principles and coding standards
- Ensure type safety with mypy validation

### 5. Performance Considerations
- Respect rate limits imposed by LLM providers
- Implement caching for repeated requests where appropriate
- Monitor tool execution time and resource usage
- Use appropriate models for different tasks (balance cost vs. quality)

### 6. Security Practices
- Validate all external inputs before processing
- Ensure sensitive information is not exposed in logs
- Use secure authentication for external services
- Implement input sanitization to prevent injection attacks

## Integration Guidelines

### FastMCP Framework Integration
- All tools should be decorated with `@mcp.tool`
- Follow the same architectural patterns as existing tools
- Use the Configurator for loading configuration
- Leverage the unified API system for LLM interactions
- Implement proper error handling and structured responses

### Context Assembly
- Use shell_tools for gathering project information
- Respect file size and count limits to prevent resource exhaustion
- Implement proper exclusion patterns to filter unwanted directories
- Assemble context from multiple sources (source files, docs, config)

## Troubleshooting

### Common Issues
- **Rate Limiting:** LLM providers may impose rate limits; monitor quota usage
- **Configuration Errors:** Ensure `conf/mcp.toml` is properly formatted
- **File Access Issues:** Verify file permissions and paths are accessible
- **API Key Issues:** Ensure required environment variables are set

### Performance Optimization
- Monitor tool execution times and optimize bottlenecks
- Use appropriate models for different tasks (balance cost vs. speed)
- Implement caching for frequently requested information
- Consider parallel execution where appropriate

## Conclusion

The agentic tools in this system provide powerful capabilities for software development automation while maintaining high standards for code quality and documentation. By following the best practices outlined in this document, you can effectively leverage these tools to improve your development workflow and maintain consistent quality across your projects.
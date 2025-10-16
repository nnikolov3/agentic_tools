# FastMCP Framework: A Comprehensive Guide

## Overview
FastMCP (Model Context Protocol) is a Python framework designed to create MCP (Model Context Protocol) applications that serve as intermediaries between AI models and external systems. The core component is the `FastMCP` server class that manages tools, resources, and prompts.

## Server Creation and Configuration

### Basic Server Setup
```python
from fastmcp import FastMCP

# Create a basic server instance
mcp = FastMCP(name="MyAssistantServer")

# Server with instructions
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)
```

### Constructor Parameters
The `FastMCP` constructor supports numerous parameters for configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"FastMCP"` | Human-readable server name |
| `instructions` | `str \| None` | `None` | Description of how to interact with the server |
| `auth` | `OAuthProvider \| TokenVerifier \| None` | `None` | Authentication provider for HTTP transports |
| `lifespan` | `AsyncContextManager \| None` | `None` | Async context manager for startup/shutdown |
| `tools` | `list[Tool \| Callable] \| None` | `None` | List of tools to add to server |
| `include_tags` | `set[str] \| None` | `None` | Only expose components with matching tags |
| `exclude_tags` | `set[str] \| None` | `None` | Hide components with matching tags |

## Core Components

### 1. Tools
Tools are functions that LLMs can call to perform actions:

```python
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b
```

#### Advanced Tool Configuration
```python
from typing import Annotated
from pydantic import Field

@mcp.tool(
    name="find_products",           # Custom tool name for the LLM
    description="Search the product catalog with optional category filtering.",
    tags={"catalog", "search"},      # Optional tags for organization/filtering
    meta={"version": "1.2", "author": "product-team"},  # Custom metadata
    enabled=True,                   # Enable/disable the tool
)
def search_products(
    query: str,
    max_results: Annotated[int, Field(description="Maximum number of results", ge=1, le=100)] = 10,
    category: str | None = None
) -> list[dict]:
    """Search the product catalog."""
    # Implementation...
```

#### Tool Context Access
Access advanced MCP features through the Context object:

```python
from fastmcp import FastMCP, Context

@mcp.tool
async def process_data(data_uri: str, ctx: Context) -> dict:
    """Process data from a resource with progress reporting."""
    await ctx.info(f"Processing data from {data_uri}")
    
    # Read a resource
    resource = await ctx.read_resource(data_uri)
    data = resource[0].content if resource else ""
    
    # Report progress
    await ctx.report_progress(progress=50, total=100)
    
    # Example request to the client's LLM for help
    summary = await ctx.sample(f"Summarize this in 10 words: {data[:200]}")
    
    await ctx.report_progress(progress=100, total=100)
    return {
        "length": len(data),
        "summary": summary.text
    }
```

### 2. Resources
Resources are read-only data sources that LLMs can access:

```python
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}
```

### 3. Resource Templates
Parameterized resources using RFC 6570 URI templates:

```python
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}
```

### 4. Prompts
Prompts are parameterized message templates to guide LLMs:

```python
@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"
```

## Context Server Capabilities

### Accessing Context
Context can be accessed via dependency injection or runtime functions:

```python
from fastmcp import FastMCP, Context

# Via dependency injection
@mcp.tool
async def process_file(file_uri: str, ctx: Context) -> str:
    """Processes a file, using context for logging and resource access."""
    await ctx.info(f"Processing file: {file_uri}")
    return "Processed file"

# Via runtime dependency function
from fastmcp.server.dependencies import get_context

async def process_data_in_nested_function(data: list[float]) -> dict:
    # This function retrieves the active context for the current tool execution via context variables
    ctx = get_context()    
    await ctx.info(f"Processing {len(data)} data points")
```

### Context Capabilities
The Context object provides access to:

#### Logging
```python
await ctx.debug("Starting analysis")
await ctx.info(f"Processing {len(data)} items") 
await ctx.warning("Deprecated parameter used")
await ctx.error("Processing failed")
```

#### Progress Reporting
```python
await ctx.report_progress(progress=50, total=100)  # 50% complete
```

#### LLM Sampling
```python
response = await ctx.sample("Analyze this data", temperature=0.7)
```

#### Client Elicitation
```python
from dataclasses import dataclass

@dataclass
class UserInfo:
    name: str
    age: int

result = await ctx.elicit("Enter your info:", response_type=UserInfo)
if result.action == "accept":
    user = result.data
```

#### Resource Access
```python
content_list = await ctx.read_resource("resource://config")
content = content_list[0].content
```

## Server Composition Features

### Importing Servers
One-time copy of components with static linking:

```python
main_server = FastMCP("MainServer")
utility_server = FastMCP("Utils")

# Import components (static copy)
main_server.import_server(utility_server, prefix="util")

# Changes to utility_server won't affect main_server
```

### Mounting Servers
Dynamic linking with live updates:

```python
main_server = FastMCP("MainServer")
tool_server = FastMCP("Tools")

# Mount server (live link)
main_server.mount(tool_server, prefix="tool")

# Changes to tool_server are reflected in main_server
```

### Proxy Servers
Bridge remote servers to local interfaces:

```python
from fastmcp import FastMCP, Client

# Create proxy for remote server
backend = Client("http://example.com/mcp")
proxy_server = FastMCP.as_proxy(backend, name="ProxyServer")
```

## Advanced Features

### Tag-Based Filtering
Control which components are exposed:

```python
# Tag your tools
@mcp.tool(tags={"public", "utility"})
def public_tool() -> str:
    return "This tool is public"

# Configure server to filter
mcp = FastMCP(include_tags={"public"})  # Only show public tools
```

### Authentication
Built-in enterprise authentication support:

```python
from fastmcp.auth import GitHubAuth

mcp = FastMCP(
    name="SecureServer",
    auth=GitHubAuth(  # Supports Google, Microsoft, Auth0, etc.
        client_id="your-client-id",
        client_secret="your-client-secret"
    )
)
```

### Transport Options
```python
# STDIO (default, for local tools)
mcp.run()  # Uses STDIO by default

# HTTP (recommended for web services)
mcp.run(transport="http", host="127.0.0.1", port=9000)
```

### Custom Routes
Add web endpoints alongside MCP functionality:

```python
from starlette.requests import Request
from starlette.responses import PlainTextResponse

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")
```

## Configuration

### Global Settings
Configure via environment variables (prefixed with `FASTMCP_`):
- `FASTMCP_LOG_LEVEL`
- `FASTMCP_MASK_ERROR_DETAILS`
- `FASTMCP_RESOURCE_PREFIX_FORMAT`
- `FASTMCP_STRICT_INPUT_VALIDATION`
- `FASTMCP_INCLUDE_FASTMCP_META`
- `FASTMCP_ENV_FILE`

### Custom Tool Serialization
Customize how tool return values are serialized:

```python
import yaml
from fastmcp import FastMCP

def yaml_serializer(data):
    return yaml.dump(data, sort_keys=False)

mcp = FastMCP(name="MyServer", tool_serializer=yaml_serializer)

@mcp.tool
def get_config():
    """Returns configuration in YAML format."""
    return {"api_key": "abc123", "debug": True, "rate_limit": 100}
```

## Best Practices

### 1. Type Safety
Use Python type hints for automatic schema generation:

```python
@mcp.tool
def calculate_stats(numbers: list[float], precision: int = 2) -> dict:
    """Calculate statistics with proper typing."""
    avg = round(sum(numbers) / len(numbers), precision)
    return {"average": avg, "count": len(numbers)}
```

### 2. Async Operations
Use async functions for I/O operations:

```python
import aiofiles

@mcp.resource("file://{path}")
async def read_file(path: str) -> str:
    """Read file asynchronously to avoid blocking."""
    
    async with aiofiles.open(path, "r") as f:
        return await f.read()
```

### 3. Error Handling
Handle errors gracefully within tools:

```python
from fastmcp.exceptions import ToolError

@mcp.tool
def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers with error handling."""
    if b == 0:
        raise ToolError("Division by zero is not allowed.")
    return a / b
```

### 4. Context Usage
Leverage context for advanced MCP features:

```python
@mcp.tool
async def complex_operation(ctx: Context, data: str) -> str:
    """Perform complex operation with progress reporting."""
    await ctx.info(f"Starting operation with {len(data)} chars")
    
    # Simulate work with progress
    for i in range(0, 101, 25):
        await ctx.report_progress(progress=i, total=100)
    
    result = process_data(data)
    await ctx.info("Operation completed successfully")
    return result
```

## Key Takeaways

1. **FastMCP** is a comprehensive framework for building MCP applications that connect AI models with external systems
2. **Core components** include tools (functions for LLMs to call), resources (data sources), and prompts (message templates)
3. **Context server** provides access to advanced MCP features like progress reporting, logging, LLM sampling, and user elicitation
4. **Type safety** is built-in through Python type hints and automatic schema generation
5. **Server composition** allows flexible architectures through mounting, importing, and proxying
6. **Transport flexibility** supports STDIO, HTTP, and custom routing
7. **Authentication** is built-in for enterprise use cases
8. **Tag-based filtering** allows component organization and access control

This framework enables rapid development of production-ready MCP applications while maintaining clean architecture and standardized interfaces for AI model interactions.
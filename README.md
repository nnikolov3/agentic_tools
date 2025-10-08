# Multi-Agent MCP MVP

## Project Summary
Multi-Agent MCP MVP provides the Go-based foundation for orchestrating specialized agent teams over NATS with Qdrant-backed memory.

## Detailed Description
The project reboots the previously Python-driven multi-agent control-plane as a Go 1.25 workspace. It focuses on two initial MCP services—leadership and development—that can delegate work packets, log deliberations in Qdrant, and enforce whitelisted command execution. The codebase currently provides validated configuration loading while the remaining orchestration components follow the design captured in `ARCHITECTURE_ADDENDUM.md` and the blueprint.

Key capabilities planned for the MVP:
- Leadership MCP to expose an MCP-compatible RPC server, accept tasks, decompose them, and assign work via NATS subjects.
- Development MCP to execute delegated work, record votes, and persist artefacts and chat transcripts inside Qdrant collections.
- Structured command execution with per-team whitelists and auditable logs.
- Iterative expansion to QA, Documentation, and DevOps MCPs once the core workflow is stable.

## Technology Stack
- Go 1.25
- NATS (messaging bus)
- Qdrant (vector store + structured metadata)
- golangci-lint
- testify (unit test assertions)

## Getting Started

### Prerequisites
- Install Go 1.25  
  ```bash
  brew install go@1.25 && brew link go@1.25 --force
  ```
  *If you use another platform or package manager, install Go 1.25 via the official binaries.*
- Install golangci-lint  
  ```bash
  curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b "$(go env GOPATH)/bin" latest
  ```

### Installation
1. Clone the repository  
   ```bash
   git clone https://github.com/nnikolov3/multi-agent-mcp.git
   cd multi-agent-mcp
   ```
2. Install Go module dependencies  
   ```bash
   cd go-mcp
   make install
   ```
3. Run the baseline TDD loop  
   ```bash
   make test
   make lint
   ```

## Usage
- Build binaries (installs them into `~/bin`):  
  ```bash
  cd go-mcp
  make build
  ```
- Run the leadership MCP entrypoint (currently a placeholder main awaiting orchestration wiring):  
  ```bash
  make run-leadership
  ```
- Run the development MCP entrypoint:  
  ```bash
  make run-development
  ```
The entrypoints presently exit immediately because the leadership and development services are being implemented incrementally under TDD.

## Testing
- Execute the unit test suite:  
  ```bash
  cd go-mcp
  make test
  ```
  A successful run reports all configuration tests passing.
- Run linting to enforce coding standards:  
  ```bash
  cd go-mcp
  make lint
  ```

## License
Distributed under the MIT License. See the LICENSE file for details.

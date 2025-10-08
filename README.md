# Multi-Agent MCP MVP

## Project Summary
Multi-Agent MCP MVP provides a Go-based foundation for orchestrating specialized agent teams over NATS with Qdrant-backed memory. The current implementation delivers configuration loading, reusable domain models, majority voting helpers, and adapters for messaging plus persistence so that the leadership and development MCPs can be wired next.

## Detailed Description
The project replaces the previous Python control plane with a Go 1.25 workspace. The initial focus covers two MCP services—leadership and development—that must:

- Delegate work packets via NATS subjects.
- Log deliberations, vote outcomes, command execution records, and artefacts inside Qdrant collections.
- Enforce per-team command whitelists with auditable execution logs.

Subsequent MCP teams (QA, Documentation, DevOps, etc.) reuse the same contracts once the leadership ↔ development workflow is stable. For the broader long-term vision, refer to `COMPLETE MULTI-AGENT MCP SYSTEM BLUEPRINT.md`.

## Current Implementation Status
- ✅ Go workspace, Makefile, and tooling target Go 1.25.
- ✅ `internal/config` loads environment-driven configuration with exhaustive validation tests.
- ✅ `pkg/model` and `pkg/voting` model work packets, chat messages, command execution records, and majority decisions.
- ✅ `internal/nats` wraps `nats.go` with publish/subscribe helpers plus embedded-server contract tests.
- ✅ `internal/qdrant` provides an HTTP adapter with integration-style tests for collection provisioning and chat upserts.
- ✅ `internal/command` executes whitelisted shell commands while capturing full audit metadata.
- ⏳ Leadership and development MCP entrypoints exist as stubs awaiting orchestration wiring and command whitelist enforcement.

## Architecture Overview
### MVP Services
- **Leadership MCP** – External entrypoint, task decomposition, cross-team orchestration, publishes assignments, aggregates completions.
- **Development MCP** – Executes work packets, coordinates sub-agents, performs majority voting, and returns artefacts.
- **QA / Documentation / DevOps MCPs** – Introduced iteratively after the core workflow stabilises.

### Command Execution Contract
- Each MCP exposes a constrained bash execution capability restricted by per-team command whitelists.
- Every invocation captures `command`, `arguments`, `stdout`, `stderr`, `exit_code`, `invoked_by`, `timestamp`, and optional metadata for audit purposes.
- Failures halt the active iteration and are escalated to the coordinator.

### Messaging & Persistence
- **NATS Subjects**
  - `leadership.task.request` – inbound work.
  - `leadership.task.assignment` – work packets dispatched to teams.
  - `team.<name>.work.ready` – completed deliverables.
  - `team.<name>.blocked` – blockers requiring leadership intervention.
  - `team.<name>.vote` – vote summaries and telemetry.
- **Qdrant Collections**
  - `work_packets` – shared task metadata.
  - `team_chat_<name>` – deliberations, command logs, vote records.
  - `artefacts_<name>` – persisted deliverables per team.
  - Collections are provisioned at boot and verified before use.

## Project Layout
```
multi-agent-mcp/
└── go-mcp/
    ├── Makefile
    ├── go.mod
    ├── cmd/
    │   ├── leadership-mcp/
    │   │   └── main.go
    │   └── development-mcp/
    │       └── main.go
    ├── internal/
    │   ├── config/
    │   ├── nats/
    │   └── qdrant/
    ├── pkg/
    │   ├── model/
    │   └── voting/
    └── tests/
        └── integration/
```

## Backlog Snapshot
- ✅ Bootstrap Go workspace and tooling.
- ✅ Ship configuration loader with tests.
- ✅ Define domain models and majority voting helpers.
- ✅ Implement NATS and Qdrant adapters with contract tests.
- ⏳ Wire the leadership → development workflow (assignment, execution, voting, response) with command whitelist enforcement.
- ⏳ Prototype the structured task-board CLI (see tooling wishlist).
- ⏳ Add end-to-end tests covering NATS message flow and Qdrant persistence.
- ⏳ Introduce MCP façade, command executor, and extend to QA/Documentation/DevOps teams.

## Tooling Wishlist
These supporting tools increase accuracy, speed, and auditability:

1. **Structured Task Board CLI** – Syncs open work packets with owning teams, highlights blockers, and surfaces TODO roll-ups via NATS + Qdrant.
2. **Command Replay Harness** – Deterministically replays whitelisted commands against captured contexts to reproduce failures before reassigning work.
3. **Agent Conversation Explorer** – Qdrant-backed browser that renders chat timelines, vote outcomes, and artefact diffs for rapid reviews.
4. **Coverage Heatmap Reporter** – Wraps `go test -coverprofile` to surface per-package coverage deltas and enforce the 80% target.
5. **Token Usage Auditor** – Optional MCP-side proxy that logs external API usage per tool call to avoid unexpected spend.

## Getting Started
### Prerequisites
- Install Go 1.25  
  ```bash
  brew install go@1.25 && brew link go@1.25 --force
  ```
  *(On other platforms, install Go 1.25 using the official binaries.)*
- Install golangci-lint  
  ```bash
  curl -sSfL https://raw.githubusercontent.com/golangci-lint/master/install.sh | sh -s -- -b "$(go env GOPATH)/bin" latest
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
- Build binaries (installs into `~/bin`):  
  ```bash
  cd go-mcp
  make build
  ```
- Run the leadership MCP entrypoint (stub):  
  ```bash
  make run-leadership
  ```
- Run the development MCP entrypoint (stub):  
  ```bash
  make run-development
  ```
The entrypoints currently exit immediately while orchestration is implemented under TDD.

## Testing
- Execute the unit test suite (config, domain models, voting, NATS, Qdrant):  
  ```bash
  cd go-mcp
  make test
  ```
- Enforce coding standards:  
  ```bash
  cd go-mcp
  make lint
  ```

## License
Distributed under the MIT License. See the LICENSE file for details.

# Architecture Addendum – Multi-Agent MCP MVP (Go)

## Purpose
This addendum clarifies the minimum viable product requirements for rebuilding the multi-agent MCP stack in Go with explicit support for NATS coordination, Qdrant persistence, and command-line tooling for each agent team. It supplements the existing blueprint documents without altering their long-term vision.

## MVP Services
- **Leadership MCP** – External entrypoint, task decomposition, cross-team orchestration, publishes assignments, aggregates completions.
- **Development MCP** – Implements work packets, coordinates sub-agents, performs majority vote, returns artefacts.
- **QA, Documentation, DevOps MCPs** – Share the same coordination/voting pattern and are introduced iteratively after the Development team stabilises.

Each MCP service exposes:
1. An MCP-compatible RPC interface for tools (Gemini CLI, Codex, etc.).
2. A NATS subscriber/publisher for inter-team communication.
3. A Qdrant chat room per team storing deliberations, votes, and artefacts.

## Command Execution Contract
- Every team MCP provides a constrained **bash execution capability** for automation tasks (e.g., `go test`, `make lint`).
- Allowed commands are whitelisted per team and executed inside an audited sandbox.
- All command invocations are captured as chat-room messages with fields: `command`, `arguments`, `stdout`, `stderr`, `exit_code`, `invoked_by`, `timestamp`.
- Failures immediately halt the current iteration and are surfaced to the coordinator for remediation.

## Messaging and Persistence
- **NATS Subjects**
  - `leadership.task.request` – external requests entering the system.
  - `leadership.task.assignment` – work packets dispatched to specific teams.
  - `team.<name>.work.ready` – team deliverables.
  - `team.<name>.blocked` – blockers, requiring leadership intervention.
  - `team.<name>.vote` – optional telemetry summarising vote outcomes.
- **Qdrant Collections**
  - `team_chat_<name>` – structured conversation history (prompt, response, command logs, vote records).
  - `work_packets` – task metadata shared across teams.
  - `artefacts_<name>` – persisted deliverables per team.
  - Collections are provisioned during service boot; existence is verified at runtime.

## Go Project Layout
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
    │   ├── qdrant/
    │   ├── command/
    │   └── leadership/
    ├── pkg/
    │   ├── model/
    │   └── voting/
    └── tests/
        └── integration/
```

- **cmd/** holds service entrypoints.
- **internal/** contains infrastructure adapters and business orchestration logic (kept private to the module).
- **pkg/** hosts reusable domain packages (models, vote evaluation).
- **tests/** captures integration suites executed via `go test ./...`.

## MCP Integration
- Leadership MCP exposes a JSON-RPC server that complies with the OpenAI MCP specification.
- Tool invocations are rate-limited, cached, and can run in “dry run” mode to reduce external token usage.
- The MCP server translates tool calls into internal leadership work packets and publishes them on NATS.

## Testing and Quality Gates
- **TDD Loop**: write failing unit tests → implement code → run `golangci-lint` → fix → run tests → repeat.
- Minimum 80% coverage target enforced via coverage reports in CI.
- Integration tests spin up ephemeral NATS servers and mock Qdrant responses to validate the hand-off between leadership and development.
- Command execution routines are tested with fake shells to avoid side effects during unit tests.

## Immediate Build Order
1. Bootstrap Go workspace (Makefile, go.mod, directory skeleton).
2. Implement `internal/config` with validation tests.
3. Implement `pkg/model` and `pkg/voting` via TDD.
4. Build Qdrant chat repository and NATS bus adapters with mocks.
5. Deliver Leadership → Development MVP workflow (assignment, execution, voting, response).
6. Add MCP server facade and CLI command whitelist enforcement.
7. Iterate to QA, Documentation, DevOps teams.

This addendum should be kept in sync with subsequent design updates to ensure the Go implementation remains aligned with the broader blueprint.


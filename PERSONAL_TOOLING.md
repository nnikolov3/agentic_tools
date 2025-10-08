# Personal Tooling Blueprint

## Purpose
This note captures the support tooling that would significantly improve the accuracy, speed, and auditability of the multi-agent MCP project. Each item is actionable and geared toward giving every agent team deterministic feedback loops.

## Tooling Wishlist
- **Structured Task Board** – Lightweight CLI that syncs open work packets with their owning teams and highlights blockers. Desired features: NATS subject subscription, Qdrant linkage, and automatic TODO roll-ups.
- **Command Replay Harness** – Deterministic runner that replays whitelisted commands against captured task contexts to reproduce failures locally before reassigning work.
- **Agent Conversation Explorer** – Qdrant-backed browser that renders chat-room timelines, vote records, and artefact diffs to make cross-team reviews faster.
- **Coverage Heatmap Reporter** – Integrates with `go test -coverprofile` and surfaces coverage deltas per package so that leadership can enforce the 80% target with evidence.
- **Token Usage Auditor** – Optional MCP-side proxy that logs token usage per tool call to prevent accidental external API burn rates.

## Immediate Steps
1. Prototype the structured task board as a read-only CLI built atop the existing config module.
2. Evaluate whether the command replay harness can share adapters with the upcoming command executor.
3. Design the conversation explorer’s schema so it reuses Qdrant collections without duplication.


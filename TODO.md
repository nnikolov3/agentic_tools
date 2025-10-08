# MCP MVP TODO List

- [ ] Update Go module configuration and tooling to target Go 1.25.
- [ ] Expand `internal/config` tests to cover invalid integers, booleans, and empty command lists.
- [ ] Redraft `README.md` to describe the Go-based MCP MVP, setup steps, and usage.
- [ ] Define domain models in `pkg/model` and majority-vote logic in `pkg/voting` using TDD.
- [ ] Implement Qdrant and NATS adapters with unit tests and contract fixtures.
- [ ] Wire the leadership → development MVP workflow, including command whitelist enforcement.
- [ ] Prototype the structured task-board CLI described in `PERSONAL_TOOLING.md`.
- [ ] Add integration tests covering NATS message flow and Qdrant persistence.

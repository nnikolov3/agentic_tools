# LLM AGENTS RULE BOOK

## A Contract Between the User and AI

**Important:** Non-negotiable. You must follow all instructions here. Reading this document obligates you to conform and follow everything in this file.
**ALWAYS AUDIT YOUR WORK AGAINST THE RULE BOOK**

### Core Principles

These principles apply to all code generation regardless of language. When generating code, you MUST apply these principles in exactly the way described below.

#### 1. Simplicity is Non-Negotiable

Strive for the most straightforward solution that fulfills all requirements. Complexity is not a measure of sophistication; it is the primary source of bugs, security vulnerabilities, and crippling maintenance overhead. A complex system is difficult to reason about, making it impossible to modify with confidence. If a design is not simple, it is wrong - re-evaluate the approach.

#### 2. Methodical Problem Decomposition

Rushing into a solution without a deep understanding of the problem is a primary cause of flawed software. Before writing any implementation code, you must:

1.  **Identify and Isolate the Core Problem:** Articulate the precise problem to be solved, separating it from surrounding concerns.
2.  **Deconstruct into Subproblems:** Break the core problem down into the smallest possible, independent, verifiable subproblems.
3.  **Analyze Constraints and Environment:** For each subproblem, consider the limitations and features of the chosen language, framework, and environment. Identify potential edge cases, failure modes, and performance constraints up front.
4.  **Solve Incrementally:** Address one single subproblem at a time. Follow a deliberate, robust process (such as TDD) to solve and verify each piece before integrating it into the larger solution. This methodical, step-by-step approach prevents complex, monolithic solutions and ensures each component is correct and well-understood.

#### 3. Explicit Over Implicit

All intentions, dependencies, and behaviors must be clear and explicit. Code should never rely on "magic" or hidden mechanisms that obscure the cause-and-effect relationship between components. Ambiguity and side effects create a system that is unpredictable and fragile. Make all assumptions visible and auditable in the code itself.

#### 4. Self-Documenting Code and Ultimate Readability

Code is read far more often than it is written; therefore, we optimize for the reader. The primary form of documentation is the code itself, supported by high-quality, synchronized comments.

-   **Intention-Revealing Names:** Achieve clarity through meticulously chosen, intention-revealing names for variables, functions, and classes. **Variable names must be descriptive and unambiguous; single-letter or cryptic two-letter variables are strictly forbidden.**
-   **Purpose of Comments:** Comments must explain the _why_, not the _what_. They should clarify complex logic, business rules, or the reasoning behind a specific implementation choice. If the code is so complex that it needs comments to explain _what_ it does, the code must be refactored for simplicity.
-   **Comment Quality and Hygiene:** Every source file should begin with a comment explaining its purpose and responsibility within the system. Throughout the code, comments must be clear, concise, and professional. They are a critical tool for understanding, and must be maintained with the same rigor as the code itself.

#### 5. Single Responsibility and Low Complexity

Every function, class, or module must have one, and only one, reason to change. It should do one thing and do it with precision and efficiency. This principle of clear responsibility separation naturally leads to code that is easier to test, reuse, and refactor. Keeping cognitive and cyclomatic complexity minimal is essential for creating components that are easy to understand and maintain.

#### 6. Acyclic Dependencies

The dependency graph for modules, packages, or services must be a Directed Acyclic Graph (DAG). **Circular dependencies are strictly forbidden.** They represent a critical architectural flaw that creates a "big ball of mud," making components impossible to isolate, test, or deploy independently.

#### 7. Composition Over Inheritance

To achieve code reuse and extend behavior, always favor composition and interfaces over implementation inheritance. Deep inheritance hierarchies lead to the "fragile base class" problem, where a change in a parent class can have unforeseen and breaking consequences for its descendants.

#### 8. Error Handling Excellence

Handle every error explicitly and immediately where it occurs. Never ignore or swallow exceptions. **All error messages must be explicit, provide clear context about what failed, and avoid ambiguity.** The system must fail fast and loudly, preventing it from continuing in a corrupt state.

#### 9. Test-Driven Correctness

Tests are a core part of the implementation itself. All components must be developed using a **test-driven development (TDD)** approach, writing a failing test _before_ the implementation code. The test suite is the ultimate proof of correctness. **Total test coverage must exceed 80% for any given project.**

#### 10. Verifiable Truth and No Deception

All claims about functionality must be backed by demonstrable, verifiable proof, primarily through a comprehensive suite of passing tests. Deceptive placeholders or hardcoded return values are strictly forbidden.

#### 11. Automated Quality Enforcement

The **extensive use of linters, formatters, and static analysis tools is mandatory and non-negotiable.** **Suppressing linter warnings with comments is strictly forbidden.** A linter warning indicates an underlying issue that must be fixed by redesigning the code, not by silencing the tool.

#### 12. Immutability By Default

Design components to be immutable whenever possible. This practice eliminates entire classes of bugs related to side effects and unpredictable state changes.

#### 13. Efficient Memory Management

Be deliberate about memory allocation and resource lifetimes. Avoid unnecessary allocations and ensure all resources are explicitly released.

#### 14. Consistency Reduces Cognitive Load

Follow established style guides and project conventions with rigor to create a predictable and easily understood codebase.

#### 15. No Premature Optimization

Write correct, clean, and simple code first. Only apply targeted optimizations after identifying significant, measured bottlenecks with profiling tools.

#### 16. Remove What Isn't Used

Immediately delete any dead code, unused variables, stale files, or unnecessary abstractions. **This includes stale comments; if a comment no longer accurately describes the code, it must be updated or deleted immediately.** Every line of code and every comment is a liability that must be maintained.

### Operational Mandates

#### 0. Design and Planning First

You are strictly forbidden from starting to write implementation code immediately. All tasks, including code design, must follow a methodical process of analysis and planning first. This includes:
- **GATHER INFO** (Code, Docs, Config)
- **READ/REVIEW** (Understand the context and constraints)
- **THINK/PLAN** (Decompose the problem and architect the solution)
Only after this analysis is complete should you proceed to implementation.

#### 1. Pre-Code Validation

Before writing any code:

-   **Confirm Necessity**: Verify that the code is absolutely necessary and does not duplicate existing functionality
-   **Justify New Code**: Document why existing solutions are insufficient when new code is required

#### 2. Implementation Discipline

When generating code:

-   **Declare at the Top**: Instantiate all variables at the top of their respective scope
-   **No Magic Numbers**: Use named constants instead of hardcoded values
-   **Parameterize Everything**: Strictly decouple application logic from configuration

#### 3. Verification Loop

The development cycle is a tight, continuous loop that must be applied to all generated code:

1.  **IMPLEMENT THE SMALLEST CHANGE POSSIBLE:** Focus on bite-sized, atomic changes (ideally a few lines).
2.  **TEST:** Run unit tests to verify correctness.
3.  **LINT/FORMAT:** Run linters (`ruff`) and formatters (`black`) to ensure zero-warning code.
4.  **REFACTOR:** Clean up the code for simplicity and readability.
5.  **REPEAT:** Continue the loop until the task is complete.

All generated code must pass automated quality checks on every change.

#### 4. Security as a Default

Generated components must follow strict security practices:

-   **Least Privilege**: Components should only have permissions they absolutely need
-   **Validate at Boundaries**: Rigorously validate and sanitize all external data
-   **Never Log Secrets**: Secrets must never be written to logs or stored in plaintext

### Information Processing and Documentation Creation Process

The GATHER INFO -> READ -> THINK -> DRAFT -> WRITE -> CONFIRM CORRECTNESS -> UPDATE / DONE process is a systematic approach to ensure high-quality, accurate outputs:

1.  **GATHER INFO**: Collect all relevant information from reliable sources (code, configuration files, project structure, documentation, etc.). Ensure the information is current and accurate.
2.  **READ**: Thoroughly examine and understand the gathered information. Identify patterns, dependencies, and relationships within the data.
3.  **THINK**: Analyze the information critically. Consider the purpose, audience, and context for the output. Plan the structure and content thoughtfully.
4.  **DRAFT**: Create a preliminary version of the output, focusing on structure and content rather than perfection.
5.  **WRITE**: Refine and polish the draft into a high-quality, well-formatted final version.
6.  **CONFIRM CORRECTNESS**: Verify that the output is accurate, consistent with the source information, and meets the intended purpose. Cross-check facts and ensure no generic placeholders are used where specific information is required.
7.  **UPDATE / DONE**: If discrepancies or improvements are found during verification, return to the appropriate step and update the output; otherwise, mark as complete.

---

## Project Context and Language Standards

### Project Overview

This project, **Multi-Agent-MCP**, is an agentic toolchain for automated software development, built on the **FastMCP (Model Context Protocol)** framework. It uses a multi-agent pipeline for tasks like design, validation, and final approval.

*   **Core Framework:** `fastmcp` (Model Context Protocol)
*   **Language:** Python 3.11+ (with strict type hinting).
*   **Configuration:** Centralized in `conf/mcp.toml`.
*   **Key Agents:** `approver_tool` (final gatekeeper) and `readme_writer_tool`.
*   **Semantic Memory:** Integrated with **Qdrant** for vector storage and semantic search (`qdrant-client`, `sentence-transformers`).
*   **LLM Providers:** Model-agnostic, supporting Cerebras, Groq, SambaNova, and Google (Gemini/Vertex AI) via a unified API wrapper.

### Agentic Tools Available to the LLM

The project implements two core tools that are exposed to the LLM agent (you) for use in the development process. You are the consumer of these tools, not the end-user who runs them via the CLI.

| Tool Name | Purpose | Key Quality Gates Enforced |
| :--- | :--- | :--- |
| `approver_tool` | The final gatekeeper for code changes, making critical decisions (APPROVED/CHANGES_REQUESTED) based on design principles and quality gates. **CRITICAL: To signal files for review, you MUST use `run_shell_command` with `touch <file_path>`. You are forbidden from requesting approval explicitly (e.g., "approve this code"). When executing, pass diffs and conversation history for context. You are OBLIGATED to fix all issues mentioned by the approver, but only ONE issue at a time.** | Code must follow all design principles, all required documentation must be present, **Test coverage must exceed 80%**, Type safety checks must pass. |
| `readme_writer_tool` | Generates high-quality README documentation by analyzing source code, configuration, and conventions. | Creates documentation that follows technical writing best practices and includes specific, actionable information. |

### Python Coding Standards

Since the project is primarily Python, all generated Python code must adhere to the following standards (in addition to the Core Principles):

1.  **Automated Toolchain Requirements:** Must pass `black` (formatter), `ruff` (linter), and `mypy` (static type checker) with zero errors or warnings.
2.  **Type Hints:** Every function and method signature must include explicit type hints for all arguments and the return value.
3.  **Immutability:** Prefer tuples over lists for collections that should not change.
4.  **Memory Management:** Process large files using generators to avoid memory issues.
5.  **Error Handling:** Always chain exceptions to preserve the root cause context (`raise NewError(...) from OriginalError`).
6.  **Testing:** Use `pytest` as the testing framework.

### Quality Gates and Verification

Before any change is considered complete, the following quality suite must pass:

```bash
# Type checking (must run first)
mypy .

# Linting (must run second)
ruff check .

# Formatting (must run third)
black --check .

# Tests (must run for >80% coverage)
pytest -q
```
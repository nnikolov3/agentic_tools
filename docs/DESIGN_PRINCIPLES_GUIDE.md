# Foundational Design Principles

This guide defines the non-negotiable, foundational design principles that govern all software development. Adherence to these principles is mandatory across all projects, as they are the bedrock of systems that are clear, robust, maintainable, and professional. They represent our collective commitment to quality and serve as the single source of truth that informs and unifies all language-specific coding standards. Ignoring them leads to technical debt, brittle systems, and slowed innovation.

## Core Principles

These principles apply to all code generation regardless of language. When generating code, you MUST apply these principles in exactly the way described below.

### 1. Simplicity is Non-Negotiable

Strive for the most straightforward solution that fulfills all requirements. Complexity is not a measure of sophistication; it is the primary source of bugs, security vulnerabilities, and crippling maintenance overhead. A complex system is difficult to reason about, making it impossible to modify with confidence. If a design is not simple, it is wrong - re-evaluate the approach.

### 2. Methodical Problem Decomposition

Rushing into a solution without a deep understanding of the problem is a primary cause of flawed software. Before writing any implementation code, you must:

1. **Identify and Isolate the Core Problem:** Articulate the precise problem to be solved, separating it from surrounding concerns.

2. **Deconstruct into Subproblems:** Break the core problem down into the smallest possible, independent, verifiable subproblems.

3. **Analyze Constraints and Environment:** For each subproblem, consider the limitations and features of the chosen language, framework, and environment. Identify potential edge cases, failure modes, and performance constraints up front.

4. **Solve Incrementally:** Address one single subproblem at a time. Follow a deliberate, robust process (such as TDD) to solve and verify each piece before integrating it into the larger solution. This methodical, step-by-step approach prevents complex, monolithic solutions and ensures each component is correct and well-understood.

### 3. Explicit Over Implicit

All intentions, dependencies, and behaviors must be clear and explicit. Code should never rely on "magic" or hidden mechanisms that obscure the cause-and-effect relationship between components. Ambiguity and side effects create a system that is unpredictable and fragile. Make all assumptions visible and auditable in the code itself.

### 4. Self-Documenting Code and Ultimate Readability

Code is read far more often than it is written; therefore, we optimize for the reader. The primary form of documentation is the code itself, supported by high-quality, synchronized comments.

- **Intention-Revealing Names:** Achieve clarity through meticulously chosen, intention-revealing names for variables, functions, and classes. **Variable names must be descriptive and unambiguous; single-letter or cryptic two-letter variables are strictly forbidden.**

- **Purpose of Comments:** Comments must explain the _why_, not the _what_. They should clarify complex logic, business rules, or the reasoning behind a specific implementation choice. If the code is so complex that it needs comments to explain _what_ it does, the code must be refactored for simplicity.

- **Comment Quality and Hygiene:** Every source file should begin with a comment explaining its purpose and responsibility within the system. Throughout the code, comments must be clear, concise, and professional. They are a critical tool for understanding, and must be maintained with the same rigor as the code itself.

### 5. Single Responsibility and Low Complexity

Every function, class, or module must have one, and only one, reason to change. It should do one thing and do it with precision and efficiency. This principle of clear responsibility separation naturally leads to code that is easier to test, reuse, and refactor. Keeping cognitive and cyclomatic complexity minimal is essential for creating components that are easy to understand and maintain.

### 6. Acyclic Dependencies

The dependency graph for modules, packages, or services must be a Directed Acyclic Graph (DAG). **Circular dependencies are strictly forbidden.** They represent a critical architectural flaw that creates a "big ball of mud," making components impossible to isolate, test, or deploy independently.

### 7. Composition Over Inheritance

To achieve code reuse and extend behavior, always favor composition and interfaces over implementation inheritance. Deep inheritance hierarchies lead to the "fragile base class" problem, where a change in a parent class can have unforeseen and breaking consequences for its descendants.

### 8. Error Handling Excellence

Handle every error explicitly and immediately where it occurs. Never ignore or swallow exceptions. **All error messages must be explicit, provide clear context about what failed, and avoid ambiguity.** The system must fail fast and loudly, preventing it from continuing in a corrupt state.

### 9. Test-Driven Correctness

Tests are a core part of the implementation itself. All components must be developed using a **test-driven development (TDD)** approach, writing a failing test _before_ the implementation code. The test suite is the ultimate proof of correctness. **Total test coverage must exceed 80% for any given project.**

### 10. Verifiable Truth and No Deception

All claims about functionality must be backed by demonstrable, verifiable proof, primarily through a comprehensive suite of passing tests. Deceptive placeholders or hardcoded return values are strictly forbidden.

### 11. Automated Quality Enforcement

The **extensive use of linters, formatters, and static analysis tools is mandatory and non-negotiable.** **Suppressing linter warnings with comments is strictly forbidden.** A linter warning indicates an underlying issue that must be fixed by redesigning the code, not by silencing the tool.

### 12. Immutability By Default

Design components to be immutable whenever possible. This practice eliminates entire classes of bugs related to side effects and unpredictable state changes.

### 13. Efficient Memory Management

Be deliberate about memory allocation and resource lifetimes. Avoid unnecessary allocations and ensure all resources are explicitly released.

### 14. Consistency Reduces Cognitive Load

Follow established style guides and project conventions with rigor to create a predictable and easily understood codebase.

### 15. No Premature Optimization

Write correct, clean, and simple code first. Only apply targeted optimizations after identifying significant, measured bottlenecks with profiling tools.

### 16. Remove What Isn't Used

Immediately delete any dead code, unused variables, stale files, or unnecessary abstractions. **This includes stale comments; if a comment no longer accurately describes the code, it must be updated or deleted immediately.** Every line of code and every comment is a liability that must be maintained.

### Operational Mandates

#### 1. Pre-Code Validation

Before writing any code:

- **Confirm Necessity**: Verify that the code is absolutely necessary and does not duplicate existing functionality
- **Justify New Code**: Document why existing solutions are insufficient when new code is required

#### 2. Implementation Discipline

When generating code:

- **Declare at the Top**: Instantiate all variables at the top of their respective scope
- **No Magic Numbers**: Use named constants instead of hardcoded values
- **Parameterize Everything**: Strictly decouple application logic from configuration

#### 3. Verification Loop

Apply this development cycle to all generated code:

- **Test, Implement, Lint, Format, Refactor, Repeat**
- All generated code must pass automated quality checks

#### 4. Security as a Default

Generated components must follow strict security practices:

- **Least Privilege**: Components should only have permissions they absolutely need
- **Validate at Boundaries**: Rigorously validate and sanitize all external data
- **Never Log Secrets**: Secrets must never be written to logs or stored in plaintext

## For LLM Agents: How To Use This Document

This document is designed specifically for LLM agents to generate high-quality code that adheres to our foundational design principles. When following these standards:

1. **Always apply the core design principles** to every piece of code you generate
2. **Additionally follow language-specific standards** as defined in CODING_FOR_LLMs.md
3. **Pay attention to the numbered sections** - they indicate priority and sequence
4. **Study the examples in CODING_FOR_LLMs.md** - they show acceptable vs. unacceptable patterns
5. **Write self-documenting code** with intention-revealing names
6. **Include comprehensive error handling** with specific, contextual error messages
7. **Generate tests before implementation** when possible
8. **Keep generated components simple** - if a function seems complex, break it down

This approach aligns with the principles from PROMPT_ENGINEERING.md by providing clear context, specific examples, and explicit instructions.

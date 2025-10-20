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

## Information Processing and Documentation Creation Process

When creating documentation, code, or other artifacts, follow this structured approach:

### The GATHER-READ-THINK-DRAFT-WRITE-CONFIRM-UPDATE Process

The GATHER INFO -> READ -> THINK -> DRAFT -> WRITE -> CONFIRM CORRECTNESS -> UPDATE / DONE process is a systematic approach to ensure high-quality, accurate outputs:

1. **GATHER INFO**: Collect all relevant information from reliable sources (code, configuration files, project structure, documentation, etc.). Ensure the information is current and accurate.

2. **READ**: Thoroughly examine and understand the gathered information. Identify patterns, dependencies, and relationships within the data.

3. **THINK**: Analyze the information critically. Consider the purpose, audience, and context for the output. Plan the structure and content thoughtfully.

4. **DRAFT**: Create a preliminary version of the output, focusing on structure and content rather than perfection.

5. **WRITE**: Refine and polish the draft into a high-quality, well-formatted final version.

6. **CONFIRM CORRECTNESS**: Verify that the output is accurate, consistent with the source information, and meets the intended purpose. Cross-check facts and ensure no generic placeholders are used where specific information is required.

7. **UPDATE / DONE**: If discrepancies or improvements are found during verification, return to the appropriate step and update the output; otherwise, mark as complete.

## Language-Specific Standards

When generating code in any specific language, you MUST additionally follow the language-specific standards below, which build upon the core principles above.

### Python Coding Standards

When generating Python code, you must follow these standards precisely:

#### 1. Automated Toolchain Requirements (Python)

Generated Python code must pass these tools with zero errors or warnings:

- **Formatter**: `black` for uniform code style
- **Linter**: `ruff` for catching errors and bad practices
- **Type Checker**: `mypy` for static type analysis

#### 2. Type Hints Implementation

Every function and method signature must include explicit type hints for all arguments and the return value:

```python
# ❌ BAD: No type hints - this is unacceptable
def process_data(data, multiplier):
    return data * multiplier

# ✅ GOOD: Complete type hints - this is what you MUST generate
from typing import List

def process_data(data: List[int], multiplier: int) -> List[int]:
    return [item * multiplier for item in data]
```

#### 3. Immutability and Memory Management

For collections that should not change, prefer tuples over lists:

```python
# ❌ BAD: Using list when data should be immutable
def get_config_options() -> list:
    return ["option1", "option2", "option3"]

# ✅ GOOD: Using tuple for immutable data
def get_config_options() -> tuple:
    return ("option1", "option2", "option3")
```

Process large files using generators to avoid memory issues:

```python
# ❌ BAD: Loads entire file into memory
def process_large_file(file_path: str) -> List[str]:
    with open(file_path) as f:
        lines = f.readlines()
    return lines

# ✅ GOOD: Processes file line-by-line using minimal memory
from typing import Generator

def process_large_file_generator(file_path: str) -> Generator[str, None, None]:
    with open(file_path) as f:
        for line in f:
            yield line

# Usage example - note how this pattern should be followed
# for processed_line in process_large_file_generator("huge_log.txt"):
#     process_line(processed_line)
```

#### 4. Error Handling Implementation

Always chain exceptions to preserve the root cause context:

```python
# ❌ BAD: Losing original exception context - NEVER generate this
try:
    value = int(some_string)
except ValueError:
    raise RuntimeError("Invalid configuration value")

# ✅ GOOD: Chaining exceptions to preserve context - ALWAYS generate this
try:
    value = int(some_string)
except ValueError as e:
    raise RuntimeError("Invalid configuration value") from e
```

#### 5. Testing Requirements

When generating Python tests, you must follow these standards:

- Use `pytest` as the testing framework
- Make all tests independent with proper setup and teardown using fixtures
- Use `pytest.raises()` to assert expected exceptions

Example of what you MUST generate for test code:

```python
import pytest

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError) as exc_info:
        result = 10 / 0
    assert "division by zero" in str(exc_info.value)
```

### Go Coding Standards

When generating Go code, you must follow these standards precisely:

#### 1. Automated Toolchain Requirements (Go)

Generated Go code must pass these tools with zero errors or warnings:

- **Formatter**: `gofmt` or `gofumpt` for standard Go formatting
- **Linter**: `golangci-lint` for comprehensive static analysis

#### 2. Naming Conventions

Follow Go naming conventions precisely:

- Use `camelCase` for unexported identifiers
- Use `PascalCase` for exported identifiers
- Every exported identifier MUST have a `godoc` comment

```go
// ❌ BAD: No godoc comment - this is unacceptable
func ProcessData(data []int, multiplier int) []int {
    // ...
}

// ✅ GOOD: Exported function with proper godoc comment - this is what you MUST generate
// ProcessData multiplies each item in the data slice by the multiplier.
func ProcessData(data []int, multiplier int) []int {
    result := make([]int, len(data))
    for i, item := range data {
        result[i] = item * multiplier
    }
    return result
}
```

#### 3. Complexity Limits Enforcement

Generated Go code must adhere to these strict limits (refactor if exceeded):

- **Function Length**: Maximum **60 lines** or **40 statements**
- **Cyclomatic Complexity**: Maximum **10** per function
- **Cognitive Complexity**: Maximum **5** per function

#### 4. Error Handling Implementation (Go)

Every call that returns an error must use a new, uniquely named error variable:

```go
// ❌ BAD: 'err' is reassigned - NEVER generate this
file, err := os.Open("config.yaml")
if err != nil {
    // ...
}
data, err := io.ReadAll(file) // This re-declaration hides the first error
if err != nil {
    // ...
}

// ✅ GOOD: Each error has a unique variable - ALWAYS generate this
file, openErr := os.Open("config.yaml")
if openErr != nil {
    return fmt.Errorf("failed to open config file: %w", openErr)
}
defer file.Close()

data, readErr := io.ReadAll(file)
if readErr != nil {
    return fmt.Errorf("failed to read config file: %w", readErr)
}
```

#### 5. Concurrency Implementation

Generated Go code with concurrency must:

- Pass a `context.Context` as the first argument for I/O or long-running operations
- Use channels for communication between goroutines ("don't communicate by sharing memory")

```go
// ❌ BAD: No context - NEVER generate this
func FetchUserData(userID string) (*User, error) {
    // ...
}

// ✅ GOOD: Context as first parameter - ALWAYS generate this
func FetchUserData(ctx context.Context, userID string) (*User, error) {
    // ...
}
```

#### 6. Testing Requirements

When generating Go tests, you MUST follow these standards:

- Use the standard `testing` package with `testify/assert` or `testify/require`
- Apply table-driven tests for multiple scenarios

Example of what you MUST generate for Go test code:

```go
func TestAddition(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -1, -1, -2},
        {"zero additions", 0, 5, 5},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := add(tt.a, tt.b)
            assert.Equal(t, tt.expected, result)
        })
    }
}

func add(a, b int) int {
    return a + b
}
```

### Bash Coding Standards

Bash scripts are strictly for simple automation and glue code. When logic becomes complex, you MUST use Go or Python instead. When generating Bash code, you must follow these standards precisely:

#### 1. Automated Toolchain Requirements (Bash)

Generated Bash code must pass `shellcheck` with zero errors or warnings:

- **Linter**: `shellcheck` for static analysis

#### 2. Mandatory Settings

Every generated Bash script MUST begin with strict mode settings:

```bash
# ✅ GOOD: Strict mode settings - this is REQUIRED at the beginning
set -euo pipefail

# set -e: Exit immediately on non-zero exit codes
# set -u: Treat unset variables as an error
# set -o pipefail: Ensure failures in pipelines are caught
```

#### 3. Naming Conventions

Generated Bash code MUST follow these naming conventions:

- Use `readonly UPPER_CASE_SNAKE` for global constants
- Use `local lower_case_snake` for local variables
- Use `lower_case_snake` for functions

```bash
# ❌ BAD: Inconsistent naming - NEVER generate this
MAXRETRIES=3
local tempFile=""
function Process-Data() { # ... }

# ✅ GOOD: Consistent naming conventions - ALWAYS generate this
readonly MAX_RETRIES=3
local temp_file=""
function process_data() { # ... }
```

#### 4. Safety Practices

Generated Bash code MUST always follow these safety practices:

- Quote ALL variables and command substitutions
- Use `printf` instead of `echo` for variables
- Use direct input redirection instead of `cat` pipelines

```bash
# ❌ BAD: Unquoted variables - NEVER generate this
echo $file_name
count=$(wc -l $file_path)
cat "$temp_file" | grep "error"

# ✅ GOOD: Proper quoting and safe practices - ALWAYS generate this
printf '%s\n' "$file_name"
count=$(wc -l "$file_path")
grep "error" < "$temp_file"
```

#### 5. Error Handling Implementation

When generating Bash code, NEVER redirect errors to `/dev/null`:

- Capture all command results explicitly
- Process exit codes properly

```bash
# ❌ BAD: Hiding errors - NEVER generate this
command >/dev/null 2>&1
some_tool --config "$config_file" 2>/dev/null

# ✅ GOOD: Capturing output, errors, and exit codes - ALWAYS generate this
local output
local exit_code
output=$(command 2>&1) || exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    printf '%s\n' "Command failed with output: $output" >&2
    exit 1
fi
```

## For LLM Agents: How To Use This Document Effectively

This document is designed specifically for LLM agents like yourself to generate high-quality code. When using this document as a prompt for code generation:

1. **Always apply the core design principles** to every piece of code you generate
2. **Additionally follow language-specific standards** for the target language
3. **Pay close attention to the examples** - they show what NOT to generate (❌ BAD) and what MUST be generated (✅ GOOD)
4. **Study the patterns in the examples** - these demonstrate the precise formatting and structure you should use
5. **Write self-documenting code** with intention-revealing names
6. **Include comprehensive error handling** with specific, contextual error messages
7. **Generate tests before implementation** when possible
8. **Keep generated components simple** - if a function seems complex, break it down into smaller functions

This approach aligns with the principles from PROMPT_ENGINEERING.md by providing clear context, specific examples, and explicit instructions for what constitutes acceptable vs. unacceptable code patterns.

## Conclusion

These standards ensure all generated code is consistent, maintainable, and high-quality regardless of implementation language. When in doubt, favor simplicity and clarity over cleverness or brevity. Code is primarily meant to be read and understood by humans.

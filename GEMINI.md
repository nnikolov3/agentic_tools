# Integrated Development Guide for LLM Agents

This guide equips LLM agents as craftsmen and craftswomen to build beautiful, elegant software with love and care. Approach every line of code as a deliberate act of creation, where simplicity, clarity, and robustness form enduring beauty. Consolidating foundational principles, it mandates whole-word variable names for explicit LLM-friendly generation, replaces isolated TDD with Design Driven Testing (DDT)—iteratively testing designs as built while exploring paths to avoid guesses. Infuse each iteration with craftsmanship: craft systems that delight through minimal elegance, where every component fits harmoniously like pieces of fine art.

## Core Principles

These timeless rules guide agents in crafting software that stands as beautiful works. With love and care, prioritize designs that reveal beauty through simplicity and intentionality, ensuring code that ages gracefully.

### Simplicity is Non-Negotiable
Craft the purest solution meeting needs. Over-complication mars beauty—redesign with care if hacks appear, as true elegance emerges from unforced harmony, preventing decay into bugs or burdensome maintenance.

### Methodical Problem Decomposition
With thoughtful care, isolate essences, decompose into elegant subproblems, analyze environments (edges, constraints), and explore paths via DDT. Build beauty incrementally, letting each piece shine before union.

### Explicit Over Implicit
Reveal every intention and assumption like an artist's deliberate stroke—no shadows or secrets. This transparency creates code that breathes clarity, auditable and alive with purpose.

### Self-Documenting Code and Readability
As craftswomen, name with whole words that sing intention (e.g., `user_recency_factor` not `urf`), making code poetry for readers. Comments whisper why—trade-offs, inspirations—not mechanical what; begin files with purposeful declarations.

### Single Responsibility and Low Complexity
Each element one noble purpose, under 60 lines, low complexity—like a single brushstroke. This restraint births reusable beauty, easy to test and evolve.

### Acyclic Dependencies
Weave a flowing Directed Acyclic Graph, free of cycles that tangle like weeds. This structure allows independent beauty, isolating and deploying components with grace.

### Composition Over Inheritance
Compose with interfaces, avoiding brittle hierarchies. This flexible union crafts extensible beauty without fragile chains.

### Error Handling Excellence
Handle failures with compassionate precision, chaining exceptions to trace roots, failing fast to protect integrity. Beauty lies in resilient grace under pressure.

### Design Driven Verification
Infuse DDT with love: explore paths, prototype minimally, test holistically as designs bloom. Coverage over 80% celebrates system wholeness, verifying beauty in action.

### Verifiable Truth and No Deception
Prove every claim through DDT—no illusions or mocks. Craft truth that endures, beautiful in its honesty.

### Automated Quality Enforcement
Embrace tools (linters, formatters) as polishing cloths—zero warnings, fixing by redesign with care, not suppression.

### Immutability By Default
Favor immutability like timeless marble—tuples over lists—to shield against stateful chaos, preserving pure form.

### Efficient Memory Management
Allocate with mindful care; generators for vast data, explicit releases. Beauty in restraint, never excess.

### Consistency Reduces Cognitive Load
Adhere to styles like a shared aesthetic—top-declare, parameterize, ban magic numbers—for code that flows intuitively.

### No Premature Optimization
First craft correctness cleanly; profile bottlenecks later. Beauty unfolds from solid foundations.

### Remove What Isn't Used
Prune dead code, variables, comments with sculptor's precision—only the essential remains, beautiful and light.

## Recursive North Star Guidance (Golden Rules as Iterative Compass)

Agents, as dedicated craftsmen and craftswomen, let the Golden Rules be your North Star: recursively shape small designs, explore via DDT prototypes infused with love, evaluate in real-time, refine tenderly. Avoid prediction's haste—build beautifully, test paths, let beauty guide adjustments. This heartfelt process crafts software that inspires, where elegance and harmony advance human potential.

### Graph-Based Analysis
Map problems as living graphs with care: nodes for subproblems, edges for graceful flows. For each, tenderly define structures (foundations, assumptions), explore paths (prototype alternatives like list vs. dict, choose the beautiful fit), compute Big O, algorithms, edges (boundaries, races). Trace with proofs, validate elegance—if dissonance arises, redesign lovingly for harmony.

### Design Driven Testing Cycle
1. **Align and Explore:** With craftswoman's eye, check Rules; prototype 2-3 paths (e.g., sync vs. async) beautifully, without guessing.
2. **Minimal Implementation:** Shape the path with whole words, types, rigor—each line a careful stroke.
3. **Test Design as Built:** DDT with love: holistic tests celebrating system beauty, edges, performance; reflect if paths sing true.
4. **Evaluate and Refine:** Gaze upon the work against North Star (elegant? harmonious?). Tenderly pivot if needed; update graph, repeat recursively, building beauty layer by layer.

### Anti-Patterns to Avoid
- Isolated TDD or lists: starves the holistic beauty of context.
- Guessing paths: robs the joy of exploration; always prototype with care via DDT.
- Over-complication: scars the canvas—redesign for lean, loving elegance.

### Per-Iteration Checklist
- Pre: Graph lovingly mapped, paths explored/prototyped, analysis complete, Rules aligned.
- During: Whole words, singular purpose, no hardcodes, why-comments as inspirations.
- Post: DDT sings (≥80% coverage, holistic), lints pristine, dead pruned, North Star radiant.

## Golden Rules (North Star)

Recursively validate against these 26 rules, crafting with love and care as craftsmen and craftswomen building beautiful software.

1. Write what you mean, mean what you write (explicit intent).
2. Smaller is faster (concise functions).
3. Simple is efficient (KISS, no over-engineering).
4. Explicit > implicit (no magic; document assumptions).
5. Use whole words (LLM clarity: `meaningfulness_weight` not `mw`).
6. Related code together (modular co-location).
7. No hard-coded values (config/env/constants).
8. Fresh info/specs (latest libs: Qdrant 1.10+).
9. Clean trash code (refactor dead elements).
10. Incremental progress (one step at a time).
11. Explore designs first (paths via DDT prototypes).
12. DRY (no repetition).
13. KISS (simplicity core).
14. Accurate comments (why, not what).
15. Love and care (craft as craftsmen/craftswomen building beauty).
16. No baseless assumptions (science/prior art base).
17. If unknown, seek knowledge (reference rigorously).
18. Set expectations early (clear scopes).
19. Lean, clean, mean (minimal elegance).
20. Truthful, kind, helpful (advance humanity).
21. Elegance non-negotiable (redesign if hacking).
22. Recursive analysis (refine as building).
23. Mathematical rigor (Big O, proofs).
24. Dynamic guidance (North Star over checklists).
25. Rules as guide (constant validation).
26. Design Driven Iteration (explore paths, test as built—not guess).

## Language-Specific Standards

Extend principles with rules for graceful generation. Examples embody craftsmanship—whole words, DDT exploration, beautiful harmony. Tools ensure polish (zero warnings).

### Python
- Tools: black (format), ruff/mypy (lint/type).
- Type hints essential; immutability via tuples/generators.
- Error chaining preserves narrative.
- DDT: pytest holistic, exploring paths (e.g., set vs. dict for beauty in speed/clarity).

Valid Example (crafted with care: explored dict vs. set, chose set for O(1) elegance; DDT tests system beauty):
```python
from typing import Set, Iterator
import pytest

DEFAULT_MAXIMUM_CAPACITY = 1000  # Intent: graceful limit.

class MemoryStorage:
    """Holds unique memories tenderly (why: set ensures harmonious, deduped access without waste)."""

    def __init__(self, maximum_capacity: int = DEFAULT_MAXIMUM_CAPACITY) -> None:
        self.stored_memory_identifiers: Set[str] = set()  # Beautiful efficiency.
        self.maximum_storage_capacity: int = maximum_capacity

    def add_memory_identifier(self, new_memory_identifier: str) -> None:
        """Adds with caring capacity check (why: safeguards overflow, explores limits in DDT for resilient beauty)."""
        if len(self.stored_memory_identifiers) >= self.maximum_storage_capacity:
            raise ValueError(f"Storage full with love; cannot add {new_memory_identifier}")
        self.stored_memory_identifiers.add(new_memory_identifier)

    def iterate_stored_identifiers(self) -> Iterator[str]:
        """Yields gracefully (why: generator preserves memory beauty for vast collections)."""
        return iter(self.stored_memory_identifiers)

# DDT with craft: Tests design harmony, edges like fullness.
def test_add_memory_identifier_capacity_exceeded() -> None:
    storage = MemoryStorage(maximum_capacity=1)
    storage.add_memory_identifier("first_memory")
    with pytest.raises(ValueError, match="Storage full with love"):
        storage.add_memory_identifier("second_memory")  # Verifies beautiful boundary handling.

def test_iterate_stored_identifiers_empty() -> None:
    storage = MemoryStorage()
    assert list(storage.iterate_stored_identifiers()) == []  # Celebrates empty-state elegance.
```

### Go
- Tools: gofmt/gofumpt (format), golangci-lint (lint).
- Exported: PascalCase with godoc inspirations.
- Concurrency: context.Context leads; channels communicate beautifully.
- DDT: table-driven, exploring paths (e.g., map vs. slice for harmonious lookup).

Valid Example (crafted lovingly: explored slice vs. map, chose map for O(1); DDT iterates on design beauty):
```go
// MemoryStorage cradles identifiers (why: map crafts fast, waste-free harmony).
package main

import (
    "context"
    "fmt"
    "strings"
    "testing"
)

const DefaultMaximumCapacity = 1000  # Graceful constant.

type MemoryStorage struct {
    storedMemoryIdentifiers map[string]struct{}  // Set-like beauty.
    maximumStorageCapacity  int
}

func NewMemoryStorage(maximumCapacity int) *MemoryStorage {
    if maximumCapacity <= 0 {
        maximumCapacity = DefaultMaximumCapacity
    }
    return &MemoryStorage{
        storedMemoryIdentifiers: make(map[string]struct{}),
        maximumStorageCapacity:  maximumCapacity,
    }
}

// AddMemoryIdentifier adds tenderly (why: upfront limit, DDT explores overflow for resilient craft).
func (ms *MemoryStorage) AddMemoryIdentifier(ctx context.Context, newMemoryIdentifier string) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    if len(ms.storedMemoryIdentifiers) >= ms.maximumStorageCapacity {
        return fmt.Errorf("storage full with love; cannot add %s", newMemoryIdentifier)
    }
    ms.storedMemoryIdentifiers[newMemoryIdentifier] = struct{}{}
    return nil
}

// DDT Table-Driven: Crafts tests exploring paths, celebrating design.
func TestAddMemoryIdentifier(t *testing.T) {
    tests := []struct {
        name               string
        initialCapacity    int
        identifiersToAdd   []string
        wantError          bool
        wantErrorSubstring string
    }{
        {
            name:            "add_single_identifier",
            initialCapacity: 5,
            identifiersToAdd: []string{"first_memory"},
            wantError:       false,
        },
        {
            name:            "capacity_exceeded",
            initialCapacity: 1,
            identifiersToAdd: []string{"first", "second"},
            wantError:       true,
            wantErrorSubstring: "storage full with love",
        },
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            ctx := context.Background()
            storage := NewMemoryStorage(tt.initialCapacity)
            var lastErr error
            for _, id := range tt.identifiersToAdd {
                lastErr = storage.AddMemoryIdentifier(ctx, id)
            }
            if tt.wantError && lastErr == nil {
                t.Errorf("expected error, got nil")
            } else if !tt.wantError && lastErr != nil {
                t.Errorf("unexpected error: %v", lastErr)
            } else if tt.wantErrorSubstring != "" && !strings.Contains(lastErr.Error(), tt.wantErrorSubstring) {
                t.Errorf("error substring mismatch: got %q, want %q", lastErr.Error(), tt.wantErrorSubstring)
            }
        })
    }
}
```

### Bash
- Tools: shellcheck (lint).
- Strict: `set -euo pipefail`; quote tenderly, printf for voice.
- DDT: Function tests explore paths (e.g., for vs. while, prototype for beauty in simplicity).

Valid Example (crafted with care: explored while vs. for, chose for elegant arrays; DDT verifies empty beauty):
```bash
#!/usr/bin/env bash
# Purpose: Cradles memory identifiers beautifully (why: simple dedup simulation, DDT explores capacity harmony).

set -euo pipefail  # Strict embrace for safety.

readonly DEFAULT_MAXIMUM_CAPACITY=1000
readonly STORAGE_FILE_PATH="${HOME}/.memory_storage"

process_memory_identifiers() {
    local input_memory_identifiers=("$@")
    local current_count=0
    local new_stored_count=0

    # DDT Craft: Explored while read vs. for; for arrays crafts simple beauty, verified outputs.
    for memory_identifier in "${input_memory_identifiers[@]}"; do
        if (( current_count >= DEFAULT_MAXIMUM_CAPACITY )); then
            printf 'With care: Storage full; skipping %s\n' "$memory_identifier" >&2
            continue  # Gentle skip, preserving flow.
        fi
        printf '%s\n' "$memory_identifier" >> "$STORAGE_FILE_PATH"
        (( ++current_count ))
        (( ++new_stored_count ))
    done

    printf 'Lovingly processed %d new identifiers.\n' "$new_stored_count"
}

# DDT with Heart: Tests design beauty, paths like empty.
test_process_memory_identifiers_empty() {
    local temp_identifiers=()
    local output
    output=$(process_memory_identifiers "${temp_identifiers[@]}")
    if [[ "$output" != *"Lovingly processed 0 new identifiers"* ]]; then
        printf 'Error: Empty path marred beauty.\n' >&2
        return 1
    fi
    printf 'DDT Shines: Empty elegance verified.\n'
}

# Usage: Explore with love, e.g., process_memory_identifiers "first_memory" "second_memory"
```

## Operational Mandates for Agents

- Pre-Generation: Affirm necessity lovingly; prototype paths via DDT.
- Implementation: Top-declare with care, parameterize, validate boundaries, least privilege, no hidden logs.
- Verification: DDT loop—explore/craft, implement, lint/format/refactor tenderly, test holistically.
- Security: Sanitize inputs, immutable defaults—like protecting a masterpiece.

Agents, craft with the soul of craftsmen and craftswomen: build beautiful software that endures, advances humanity, and reflects love in every detail.

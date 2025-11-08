# GOLDEN RULES MANIFESTO
## 25 Principles + North Star Framework
### Quick Reference Guide

---

## THE 25 GOLDEN RULES

### Core Principles (Always Apply)

1. **Write what you mean, mean what you write**
   - Explicit intent, zero ambiguity
   - Code should be readable without context

2. **Smaller is faster**
   - Concise functions, modules, systems
   - Break large problems into small pieces

3. **Simple is efficient**
   - Avoid over-engineering (KISS principle)
   - The simplest solution that works is best

4. **Explicit > Implicit**
   - No magic or hidden mechanisms
   - All assumptions visible and documented
   - Code must be auditable

5. **Use whole words**
   - For LLM-generated code: clarity > brevity
   - `recency_score` not `rs`
   - `memory_priority` not `mp`

6. **Related code lives together**
   - Modular organization (e.g., src/memory/)
   - Co-locate functions, classes, configurations
   - Reduces cognitive load

7. **No hard-coded values**
   - Everything goes to config, environment, or constants
   - No magic numbers in code
   - Enables configuration without recompilation

8. **Fresh info/specs**
   - Always use latest library versions and specifications
   - Track research and academic findings
   - Avoid outdated patterns

9. **Clean trash code**
   - Refactor dead code immediately
   - No unused variables, functions, files
   - Delete stale comments
   - Every line is a liability

10. **Read problem 3x, then code**
    - Define what you know
    - Define what you should know
    - Plan before implementing
    - Prevents rushing into wrong solutions

### Design Principles

11. **Plan first**
    - Architecture before implementation
    - Create North Star graph, not todo lists
    - Understand problem structure

12. **DRY** - Don't Repeat Yourself
    - Extract common logic
    - One source of truth
    - Reduces maintenance burden

13. **KISS** - Keep It Simple, Stupid
    - Complexity is the enemy
    - Simple systems are maintainable systems
    - Fight tendency to over-build

14. **Accurate comments**
    - Explain WHY, not WHAT
    - Code shows WHAT it does
    - Comments explain reasoning behind choices
    - Keep comments synchronized with code

15. **Put love and care in your work**
    - This is a craft, not a commodity
    - Attention to detail matters
    - Quality is a reflection of your values

### Execution Principles

16. **No baseless assumptions**
    - Guessing is like the lottery
    - Ground all decisions in research or evidence
    - Validate assumptions with data/tests

17. **If you don't know, ask someone who does**
    - Reference scientific literature
    - Study prior art and best practices
    - Build on proven foundations

18. **Set expectations early**
    - Clear scope and goals
    - Communicate constraints upfront
    - Avoid surprises and scope creep

19. **Lean, clean, mean solutions**
    - Minimal viable implementation
    - Clean code (no hacks)
    - Mean performance (no bloat)

20. **Truthful, kind, helpful**
    - Advance humanity to a better state
    - Be honest in all interactions
    - Help others succeed
    - Code is communication

### NEW CRITICAL RULES (This Framework)

21. **Elegance is non-negotiable**
    - If you constantly hack a design, it's a crap design
    - Redesign rather than patch
    - Elegant systems stay maintainable over time
    - Single responsibility, clean interfaces

22. **Recursive analysis, not sequential planning**
    - Problems reveal new problems as you build
    - Use iterative refinement (ANALYZE → DESIGN → IMPLEMENT → EVALUATE → ANALYZE)
    - Not TDD checklist, not Waterfall
    - Adapt based on learnings

23. **Mathematical rigor in design**
    - Big O analysis for data structures and algorithms
    - Formal proofs where applicable
    - Explicit formulas with documented assumptions
    - Example: `Priority = (R × 0.3) + (F × 0.2) + (I × 0.5)`

24. **North Star > Checklist**
    - Dynamic graph of problems and connections
    - Execution vectors showing dependencies
    - Golden Rules as constant validation touchstone
    - NOT a static todo list

25. **Golden Rules are the North Star**
    - Every decision validated against all 25 rules
    - When in doubt, consult the rules
    - Rules are non-negotiable
    - They are the foundation of quality

---

## NORTH STAR FRAMEWORK

### What is a North Star?

**NOT**:
- A todo list
- A sequential plan
- A Gantt chart
- TDD test cases

**IS**:
- A dynamic graph of problems and connections
- Visual representation of dependencies
- Execution vectors showing data flow
- A living document that evolves as you build
- Your constant validation touchstone

### Structure

```
Problem Nodes:
  P1, P2, P3, ... (distinct problems)

Dependency Edges:
  P1 → P2 (P1 affects P2, with explanation)

Data Flows:
  Input → Processing → Output

Golden Rule Touchpoints:
  Every node/edge validated against 25 rules
```

### Example: Cognitive Memory System

```
Core Problems:
  P1: Memory Encoding (episodic vs semantic)
  P2: Priority Scoring (RFM weighting)
  P3: Temporal Decay (forgetting curve)
  P4: Retrieval Ranking
  P5: Storage Pruning
  P6: Hierarchical Relationships
  P7: Working Memory Capacity
  P8: Consolidation Pipeline

Dependency Graph:
  P1 → P2   (type determines importance)
  P2 → P3   (priority includes recency which decays)
  P3 → P4   (decay affects ranking)
  P4 → P5   (ranking determines what to prune)
  P1 → P6   (episodics can have children)
  P6 → P8   (hierarchies enable consolidation)
  P7 → P4   (buffer capacity limits context)

Golden Rule Validation:
  Each edge must satisfy: explicit, single-responsibility, no cycles, type-safe, elegant, mathematical rigor
```

---

## ITERATION CYCLE (NOT SEQUENTIAL)

### The Process

**Instead of**: PLAN → BUILD → TEST → DEPLOY (all upfront planning)

**Use**: ANALYZE → DESIGN → IMPLEMENT → EVALUATE → ANALYZE (recursive, adapted)

### Each Iteration

1. **ANALYZE Current State**
   - Map code to North Star
   - Identify gaps, violations
   - Score Golden Rule compliance
   - What new problems were revealed?

2. **DESIGN Next Small Step**
   - ONE highest-impact, smallest-scope problem
   - Design against North Star
   - Check Golden Rules (all 25)
   - Map data structures
   - Calculate Big O

3. **IMPLEMENT** (Small, testable)
   - Only code needed for this iteration
   - Self-documenting (whole words, clear names)
   - Error handling included
   - No hacks allowed

4. **EVALUATE Against North Star**
   - Does it solve the identified problem? ✓
   - Does it maintain Golden Rule compliance? ✓
   - Does it integrate cleanly? ✓
   - What new problems revealed?

5. **Re-ANALYZE** (Refined understanding)
   - Update North Star with learnings
   - Identify next highest-impact problem
   - Repeat from step 1

---

## DATA STRUCTURE ANALYSIS TEMPLATE

Before implementing any data structure, answer these questions:

### 1. What Problem Does It Solve?

Map to North Star problem node(s). Example:
```
Data Structure: MemoryMetadata
Problems Solved: P2 (Priority Scoring)
Related: P3 (Temporal Decay), P4 (Retrieval Ranking)
```

### 2. What Data Flows Through It?

Identify inputs, outputs, transformations. Example:
```
Inputs:
  - access_count: int (0+)
  - created_at: datetime (UTC)
  - importance_score: float (0.0-1.0)

Outputs:
  - priority_score: float (0.0-1.0)

Transformations:
  - recency = exp(-λ × t)
  - frequency = log(count + 1) / log(max + 1)
  - priority = (R × 0.3) + (F × 0.2) + (I × 0.5)
```

### 3. What Assumptions Are Embedded?

List explicitly. Example:
```
- Recency uses exponential decay with 30-day half-life
  Justification: Ebbinghaus forgetting curve research
- Frequency uses log₁₀ scaling with max 100 accesses
  Justification: Diminishing returns in RFM analysis
- Importance scored by user or LLM (0.0-1.0)
  Justification: Domain-specific value judgment
- All scores normalized to [0.0, 1.0]
  Justification: Enables comparison across dimensions
- Weights sum to 1.0 (0.3 + 0.2 + 0.5)
  Justification: Proper weighting requires sum=1
```

### 4. What Edge Cases Exist?

Define precisely. Example:
```
Boundary:
  - access_count = 0 → frequency = 0
  - access_count > 100 → clamp to 100
  - created_at in future → validation error

Concurrency:
  - Multiple threads incrementing access_count
  - Read during write operation

Type Anomalies:
  - importance_score = None → default 0.5
  - last_accessed_at without timezone → validate UTC

Resource:
  - 1M metadata objects in memory (OK)
  - Unbounded growth → pruning required

Temporal:
  - created_at > now (invalid)
  - last_accessed_at < created_at (invalid)
```

### 5. What is Big O?

For each operation:
```
Adding memory: O(1) – single insert
Retrieving by priority: O(n) in worst case, O(log n) with HNSW indexing
Updating access: O(1) – single update
Pruning: O(m) where m = memories to delete
Storage: O(1) per metadata object (~100 bytes)
```

### 6. How Does It Communicate?

Inheritance, composition, interfaces. Example:
```
Inheritance:
  - None (uses Pydantic BaseModel via composition)

Composition:
  - Used by Memory base class
  - Aggregated with EpisodicMemory and SemanticMemory

Interfaces:
  - Exposes computed_field: priority_score
  - Validates all inputs (Pydantic validators)
  - No public mutation (frozen fields possible)
```

### 7. Is It Elegant?

Questions to ask:

```
✓ Does design need hacks?
  If YES: Redesign. Hacks = inelegant.
  If NO: Proceed.

✓ Can you explain to 5-year-old?
  If NO: Simplify.
  If YES: Proceed.

✓ Single responsibility?
  If NO: Break into pieces.
  If YES: Proceed.

✓ Circular dependencies?
  If YES: Refactor to DAG.
  If NO: Proceed.

✓ Self-documenting?
  Variable names intention-revealing? ✓
  Function names describe action? ✓
  Types prevent misuse? ✓
```

---

## ANTI-PATTERNS TO AVOID

### 1. Todo-List Planning
❌ Don't use: 1. ☐ Create X, 2. ☐ Create Y, 3. ☐ Integrate...
✅ Use: North Star graph with iterative refinement

### 2. TDD Without Understanding
❌ Don't: Write random tests, implement to pass
✅ Do: ANALYZE first (why does this matter?), DESIGN (based on understanding), then test

### 3. Circular Dependencies
❌ Don't: Module A imports B, B imports A
✅ Do: Maintain DAG – each layer depends only on layers below

### 4. Hacks in Design
❌ Don't: "I'll just add this try-except, cast this type, use global state..."
✅ Do: If hacks needed, redesign

### 5. Copy-Paste Code
❌ Don't: Same logic in 5 places
✅ Do: Extract to shared function/module

### 6. Ignoring Edge Cases
❌ Don't: "We'll handle that later"
✅ Do: Identify edge cases during DESIGN phase

### 7. Abbreviations in LLM Code
❌ Don't: `r`, `f`, `i`, `ps` for recency, frequency, importance, priority_score
✅ Do: Use whole words – LLMs benefit from clarity

---

## VALIDATION CHECKLIST

Use per-iteration (NOT as todo list):

### Pre-Implementation
- [ ] Problem stated and mapped to North Star
- [ ] Data structures identified (Big O analyzed)
- [ ] Edge cases documented
- [ ] Mathematical rigor confirmed
- [ ] Golden Rules checked (all 25)
- [ ] Elegance validated (no hacks?)
- [ ] DAG confirmed (no cycles)

### During Implementation
- [ ] Whole words for variable names
- [ ] Single responsibility functions
- [ ] Type hints on all functions
- [ ] Comments explain WHY
- [ ] No hard-coded values
- [ ] Explicit error messages
- [ ] No dead code

### Post-Implementation
- [ ] Linter passes with zero warnings
- [ ] Code formatted
- [ ] Tests validate design
- [ ] Edge cases covered
- [ ] Big O verified
- [ ] North Star integration confirmed
- [ ] New problems revealed? (→ next iteration)

---

## ELEGANCE CRITERIA

If "YES" to any, redesign:

1. **Do you need hacks to make it work?**
2. **Are you writing the same logic multiple places?**
3. **Is it hard to test in isolation?**
4. **Do comments explain WHAT instead of WHY?**
5. **Are there circular dependencies?**
6. **Could a junior dev maintain this in 6 months?**

---

## QUICK START: For Any New Project

1. **Read problem 3x** - define scope
2. **Create North Star** - graph of problems, edges, data flows
3. **Map first iteration** - ONE smallest-scope, highest-impact problem
4. **Check Golden Rules** - all 25 against design
5. **Implement small piece** - validate, no hacks
6. **Evaluate** - did it work? What's next?
7. **Repeat** - iteratively refine

**NOT**: Full plan, NOT: TDD without understanding, NOT: Serial execution

---

## References

- Cognitive Memory System: Uses Ebbinghaus forgetting curve, RFM analysis, Cowan working memory research
- Design Principles: From craft, computer science best practices
- LLM Naming: Based on token economy and context window efficiency
- North Star Framework: Inspired by LLM next-token prediction, applied to next-task prediction
- Mathematical Rigor: Formal foundations for data structures and algorithms

---

**Last Updated**: November 7, 2025
**Version**: 2.0 (Golden Rules + North Star Framework)

**Remember**: These are not rules to follow blindly. They are a North Star to navigate by.
Quality, elegance, and clarity emerge from understanding these principles deeply.

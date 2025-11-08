**UNIFIED DEVELOPMENT MANIFESTO**

**Design, Planning, and Execution Framework** **Version 2.0: North Star Navigation with Recursive Problem Analysis** **SECTION 1: THE NORTH STAR PARADIGM**

**What is a North Star? **

A North Star is not a checklist, todo list, or serialized plan. It is a **living, dynamic graph** that represents: 1. **Problem Nodes**: Each distinct problem or concern in your system 2. **Execution Paths**: Vectors between nodes showing how problems connect and influence each other 3. **Data Flow**: The movement of information through the system 4. **Golden Rule Compliance**: Every decision constantly validated against principles 5. **Recursive Depth**: The understanding that solving one problem reveals new problems **Why Not Sequential Planning? **

Sequential planning \(traditional Waterfall or even TDD checklists\) assumes: All problems are known upfront

Solutions are independent

Changes can be made in isolation

**Reality**: Solutions create new problems. A change to memory data structures cascades to retrieval. A retrieval optimization exposes pruning inefficiencies. 

**The LLM Strategy Analogy**

Language models predict the next token based on: 1. Current context \(what you know now\)

2. Probability distribution \(what options exist\) 3. Recursive evaluation \(each token generates new context for the next\) **Next-task prediction must work the same way**: 1. Analyze current state against North Star 2. Identify the highest-impact, smallest-scope next action 3. Implement it \(not plan it\)

4. Re-evaluate against North Star with new context 5. Repeat

**SECTION 2: THE NORTH STAR CONSTRUCTION FOR COGNITIVE MEMORY SYSTEM**

**Layer 1: Problem Graph \(Abstract\)**

┌─────────────────────────────────────────────────────────────────┐

│ COGNITIVE MEMORY SYSTEM │

│ North Star Graph │

└─────────────────────────────────────────────────────────────────┘

Core Problems:

P1: Memory Encoding \(episodic vs semantic\) P2: Priority Scoring \(RFM weighting\)

P3: Temporal Decay \(forgetting curve\)

P4: Retrieval Ranking \(which memories matter\) P5: Storage Management \(pruning old memories\) P6: Hierarchical Relationships \(parent-child linking\) P7: Working Memory Capacity \(4-chunk buffer\) P8: Consolidation Pipeline \(episodic→semantic\) Dependency Graph \(Execution Vectors\):

P1 → P2 \(type determines importance scoring\) P2 → P3 \(priority includes recency which decays\) P3 → P4 \(decay affects ranking\)

P4 → P5 \(ranking determines what to prune\) P1 → P6 \(episodic memories can have children\) P6 → P8 \(hierarchies enable consolidation\) P2 → P8 \(priority guides which episodics to consolidate\) P7 → P4 \(buffer capacity limits what's in context\) Data Flow:

EpisodicMemory ─→ RFM\_Calculator ─→ Priority\_Score Priority\_Score ─→ Retrieval\_Ranker ─→ Context\_Buffer Context\_Buffer ─→ Working\_Memory \(max 4 chunks\) Old\_Episodics ─→ Consolidation ─→ SemanticMemory SemanticMemory ─→ Knowledge\_Bank ─→ Retrieval Golden Rule Touch Points \(every node must satisfy\):

✓ Explicit intent \(no magic\)

✓ Single responsibility \(each node does one thing\)

✓ No circular dependencies \(DAG structure\)

✓ Type safety \(Pydantic models\)

✓ Elegant design \(no hacks\)

✓ Mathematical rigor \(Ebbinghaus, RFM formulas\) **Layer 2: Data Structure Decisions**

**Before Implementation**: Ask these questions for EACH data structure: 1. **What problem does it solve? ** \(map to problem node\) 2. **What data flows through it? ** \(validate against data flow\) 3. **What assumptions are embedded? ** \(list explicitly\) 4. **What edge cases exist? ** \(define precisely\) 5. **What is its Big O? ** \(storage, retrieval, update\) 6. **How does it communicate? ** \(inheritance, composition, interfaces\)

**Example: MemoryMetadata**

**Problem solved**: P2 \(RFM Priority Scoring\) **Data flowing through**:

recency\_score: float \(0.0-1.0\)

frequency\_score: float \(0.0-1.0\)

importance\_score: float \(0.0-1.0\)

access\_count: int \(≥0\)

last\_accessed\_at: Optional\[datetime\]

**Assumptions** \(must be documented\):

Recency uses exponential decay with 30-day half-life Frequency uses log₁₀ scaling with max 100 accesses Importance is user-defined or LLM-classified All scores normalized to \[0.0, 1.0\]

Priority = \(R × 0.3\) \+ \(F × 0.2\) \+ \(I × 0.5\) **Edge cases**:

What if access\_count > 100? \(clamp at 100\) What if created\_at is in future? \(abs the delta\) What if metadata fields are None? \(default values\) What if weights don't sum to 1.0? \(validation error\) **Big O Analysis**:

Computing recency: O\(1\) – single exponential calculation Computing frequency: O\(1\) – logarithm calculation Computing priority: O\(1\) – three multiplications and two additions Storage: O\(1\) – fixed 5 float fields \+ 1 int \+ 1 datetime **Communication** \(How does it fit into inheritance tree?\): Composition: Used by Memory base class

No inheritance \(Pydantic BaseModel is composition\) Interfaces: Exposes computed\_field priority\_score **Elegance check**:

✓ Does what it says \(transparent\)

✓ No hacks needed \(formulas are clean\)

✓ Extensible \(can add fields without breaking\)

✓ Type-safe \(Pydantic validates\)

**SECTION 3: UNIFIED GOLDEN RULES \(UPDATED\)** **The 25 Golden Rules**

1. **Write what you mean, mean what you write** - Explicit intent, no ambiguity 2. **Smaller is faster** - Concise functions, modules, systems 3. **Simple is efficient** - Avoid over-engineering \(KISS\) 4. **Explicit > Implicit** - No magic; document assumptions 5. **Use whole words** - LLM-written code benefits from clarity: recency\_score not rs 6. **Related code lives together** - Modular, co-located \(e.g., src/memory/\) 7. **No hard-coded values** - Everything in config, environment, or named constants 8. **Fresh info/specs** - Latest libraries, specifications, research 9. **Clean trash code** - Refactor on sight; no dead variables or functions 10. **Read problem 3x** - Define what you know, what you should know before coding 11. **Plan first** - Architecture before implementation \(via North Star\) 12. **DRY** - Don't Repeat Yourself

13. **KISS** - Keep It Simple, Stupid

14. **Accurate comments** - Explain WHY, not WHAT

15. **Put love and care in your work** - This is a craft 16. **No baseless assumptions** - Guessing is like the lottery 17. **If you don't know, ask someone who does** - Reference science, prior art 18. **Set expectations early** - Clear scope and goals 19. **Lean, clean, mean solutions** - Minimal viable elegance 20. **Truthful, kind, helpful** - Advance humanity to better state **NEW RULES \(Added for this approach\)**

21. **Elegance is non-negotiable** - If you constantly hack a design, it's crap; redesign it 22. **Recursive analysis, not sequential planning** - Iteratively refine as you build 23. **Mathematical rigor in data structures** - Big O analysis, proofs of correctness 24. **North Star > Checklist** - Dynamic graph, not todo list 25. **Golden Rules are the North Star** - Every decision validated against these 25 rules **SECTION 4: PROBLEM ANALYSIS FRAMEWORK \(NOT TDD, NOT SEQUENTIAL\)** **Step 1: Graph-Based Problem Decomposition** **Input**: High-level goal or requirement **Process**: Create problem graph with: **Nodes**: Distinct problems/concerns

**Edges**: Dependencies and data flows

**Weights**: Impact on overall system \(high/medium/low\) **Validation**: Each node cross-checked against Golden Rules

**Output**: Visual representation of problem space **Example for Memory System**:

High-Level Goal: "Store memories with RFM prioritization and forgetting" 

Graph Nodes:

┌─────────────────┐

│ P1: Encoding │──\(determines\)──┐

└─────────────────┘ │

┌──────────────────┐

│ P2: Priority │

│ Scoring │

└──────────────────┘

│

├──\(affects\)──→ P4: Retrieval

│

└──\(determines\)──→ P5: Pruning

Edge Validation:

- P1→P2: "Encoding type \(episodic vs semantic\) determines importance scoring method" ✓

- P2→P4: "RFM priority directly ranks retrieval order" ✓

- P2→P5: "Low priority scores are pruning candidates" ✓

Golden Rule Check:

- Explicit intent: Each edge explains WHY

- Single responsibility: Each node solves one thing

- No cycles: DAG structure confirmed

**Step 2: Data Structure Mapping**

For each problem node, answer:

1. **What data structures are needed? **

P1 \(Encoding\): EpisodicMemory, SemanticMemory classes P2 \(Priority\): MemoryMetadata with recency/frequency/importance P4 \(Retrieval\): MemoryQuery with filters; SearchResult with scores 2. **What mathematical operations? **

P2 \(Priority\): 

recency = exp\(-λt\) where λ = ln\(2\) / τ

frequency = log\(count \+ 1\) / log\(max \+ 1\)

priority = \(R × 0.3\) \+ \(F × 0.2\) \+ \(I × 0.5\) 3. **What are Big O complexities? **

Add memory: O\(1\) – single insert into Qdrant Retrieve: O\(n\) – vector search, but with HNSW → O\(log n\) Update priority: O\(1\) – compute new scores Prune: O\(m\) – m = memories to delete

4. **What assumptions embedded? **

- Recency uses 30-day half-life \(why? based on Ebbinghaus studies\)

- Frequency maxes at 100 accesses \(why? diminishing returns\)

- Working memory = 4 chunks \(why? Cowan 2001 psychology\)

**Step 3: Edge Case Identification** **Structured approach**:

1. **Boundary conditions**: Min/max values, empty states, overflow 2. **Concurrency issues**: Async operations, race conditions 3. **Type mismatches**: None values, wrong types 4. **Resource exhaustion**: Memory limits, vector space overflow 5. **Temporal anomalies**: Backwards time, missing timestamps **Example for MemoryMetadata**:

Boundary Conditions:

- access\_count = 0 → frequency = 0 \(edge case: newly created\)

- access\_count &gt; 100 → clamp to 100 \(saturation\)

- recency\_score = 1.0 → memory just created \(fresh\)

- recency\_score → 0.0 → memory very old \(forgotten\)

- priority\_score must be in \[0.0, 1.0\] \(validated by Pydantic\) Concurrency Issues:

- Multiple threads incrementing access\_count \(use atomic operations\)

- Reading stale frequency while write in progress \(lock on update\) Type Mismatches:

- What if importance\_score = None? \(default to 0.5\)

- What if last\_accessed\_at is not UTC? \(validate timezone\) Resource Exhaustion:

- Storing 1 million MemoryMetadata objects \(each ~100 bytes, OK for RAM\)

- What if metadata grows unbounded? \(pruning mechanism\) Temporal Anomalies:

- created\_at in future? \(validate against current time, raise error\)

- last\_accessed\_at before created\_at? \(impossible, validation fails\) **Step 4: Elegance Assessment**

**Before implementing**, ask:

1. **Does the design need hacks? **

If YES: Redesign. Hacks = inelegant design. 

If NO: Proceed. 

2. **Can I explain it to a 5-year-old? **

If NO: Simplify or decompose further. 

If YES: Proceed. 

3. **Does every component have single responsibility? **

If NO: Break into smaller pieces. 

If YES: Proceed. 

4. **Are there circular dependencies? **

If YES: Refactor to eliminate cycles \(apply Dependency Inversion\) If NO: Proceed. 

5. **Is the code self-documenting? **

Variable names intention-revealing? ✓

Function names describe action? ✓

Types prevent misuse? ✓

**SECTION 5: THE ITERATIVE IMPROVEMENT CYCLE \(NOT SERIALIZED\)** **The Process**

Instead of:

PLAN → BUILD → TEST → DEPLOY

\(Sequential, all planning upfront\)

Use:

ANALYZE → DESIGN → IMPLEMENT \(small\) → EVALUATE → ANALYZE \(refined\)

↑\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_↓

\(Recursive: each cycle reveals new problems\) **Concrete Steps for Each Iteration**

**Iteration N**:

1. **ANALYZE Current State**

Map current code to North Star graph

Identify gaps or violations

Score each Golden Rule compliance

Example:

Current: MemoryMetadata calculates priority in add\_memory\(\) Issue: "Elasticity" – should be calculated whenever needed Golden Rule 23 \(Mathematical Rigor\): Priority = f\(R,F,I\) always Issue: Calculation scattered; not DRY \(Rule 12\) 2. **DESIGN Next Small Step**

Identify ONE highest-impact, smallest-scope problem Design solution against North Star

Check Golden Rules compliance

Map data structures

Calculate Big O

Example:

Problem: "Extract priority calculation to separate module" 

Scope: Create RFMCalculator class

North Star: Solves P2, doesn't break P3-P8

Golden Rules:

✓ 1: Explicit intent \(module name = functionality\)

✓ 5: Whole words \(recency\_score, not rs\)

✓ 21: Elegant \(no hacks, clean math\)

✓ 23: Mathematical rigor \(formulas documented\) Big O: O\(1\) for each calculation

3. **IMPLEMENT** \(Small, testable piece\)

Write only the code needed for this iteration Make it self-documenting

Include error handling

No hacks allowed

4. **EVALUATE Against North Star**

Does it solve the identified problem? ✓

Does it maintain Golden Rule compliance? ✓

Does it integrate cleanly? ✓

What new problems does it reveal? 

Example:

New problem revealed: "Access tracking needs automatic frequency update" 

→ Creates next iteration target

5. **Re-ANALYZE** \(Refined understanding\) Update North Star graph with learnings

Identify next highest-impact problem

Repeat

**Why This Works**

1. **Captures Unknowns**: Each iteration reveals what sequential planning couldn't 2. **Maintains Quality**: Every step validated against Golden Rules \+ North Star 3. **Avoids Over-Engineering**: Only solve problems you have, not predict them 4. **LLM-Compatible**: Next-task prediction works on current context \+ graph 5. **Pragmatic**: Small changes, fast feedback, quick course correction **SECTION 6: ANTI-PATTERNS TO AVOID**

**Pattern 1: Todo-List Planning**

❌ **BAD**:

1. Create Memory model ☐

2. Create RFM calculator ☐

3. Create Qdrant manager ☐

4. Create add\_memory function ☐

5. Create retrieve function ☐

6. ...50 more items... 

Issues:

- Assumes 100% knowledge upfront

- Discovers problems too late

- Can't adapt to learnings

- Serialized – no parallelization of thought

✅ **GOOD**:

NORTH STAR GRAPH:

P1 \(Encoding\) → P2 \(Priority\) → P4 \(Retrieval\)

↓

P5 \(Pruning\)

CURRENT ITERATION: "Implement P2 \+ validate P2→P4 connection" 

- Create MemoryMetadata with RFM scores

- Create RFMCalculator with formulas

- Test: priority calculation works

- NEW PROBLEM FOUND: "Access tracking needed for frequency" 

- → Next iteration

NOT: "Follow todo list slavishly" 

**Pattern 2: TDD Without Context**

❌ **BAD**:

\# Write test first \(without understanding\) def test\_priority\_calculation\(\):

assert calculate\_priority\(0.5, 0.5, 0.5\) == 0.5

\# Implement to pass \(correct result, wrong understanding\) def calculate\_priority\(r, f, i\):

return \(r \+ f \+ i\) / 3 \# WRONG FORMULA\! 

✅ **GOOD**:

\# ANALYZE first: Why does priority matter? Where used? 

\# North Star: P2 determines ranking in P4

\# Mathematical basis: RFM weights based on marketing science

\# Edge cases: What if R=0? F=0? I=0? 

\# DESIGN: RFMCalculator with explicit formulas \+ assumptions class RFMCalculator:

"""Calculates priority via RFM weighting \(Recency×0.3 \+ Frequency×0.2 \+ Importance×0.5\) Assumptions documented:

- Recency uses exponential decay \(Ebbinghaus 1885\)

- Frequency uses logarithmic scaling \(diminishing returns\)

- Weights sum to 1.0 \(validated\)

""" 

\# THEN write tests that validate design

def test\_recency\_decay\(\):

"""At 30-day half-life, R\(30d\) should = 0.5""" 

assert 0.49 &lt; calc.recency\(30\_days\_ago\) &lt; 0.51

def test\_priority\_formula\(\):

"""Priority = \(R×0.3\) \+ \(F×0.2\) \+ \(I×0.5\)""" 

\# NOT "guess" – based on mathematical proof assert calc.priority\(0.9, 0.8, 1.0\) == 0.93

**Pattern 3: Circular Dependencies**

❌ **BAD**:

\# qdrant\_memory.py imports from retrieval.py from retrieval import retrieve\_memories

\# retrieval.py imports from qdrant\_memory.py from qdrant\_memory import QdrantMemory

\# PROBLEM: Circular import, can't test in isolation

✅ **GOOD**:

\# DAG structure:

\# models.py \(no imports from other memory modules\)

\# ↓

\# rfm\_calculator.py \(imports only models\)

\# ↓

\# qdrant\_memory.py \(imports models \+ rfm\_calculator\)

\# ↓

\# memory\_system.py \(imports qdrant\_memory, coordinates\)

\# No circles. Each layer depends only on layers below. 

**Pattern 4: Ignoring Elegance**

❌ **BAD**:

\# Hack 1: Global state

\_memory\_cache = \{\}

\# Hack 2: Magic numbers

if priority &gt; 0.6666: \# Why 0.666? No explanation

\# Hack 3: Copy-paste code

def add\_episodic\(...\):

\# 50 lines of logic

def add\_semantic\(...\):

\# 47 lines almost identical logic

\# Issue: Design is inelegant, will require constant hacks

✅ **GOOD**:

\# Elegance 1: Dependency injection \(no globals\) class MemorySystem:

def \_\_init\_\_\(self, storage: StorageBackend\): self.storage = storage

\# Elegance 2: Named constants

PRIORITY\_THRESHOLD\_HIGH = 2/3 \# Two-thirds, based on RFM weighting

\# Elegance 3: DRY – extract common logic

def add\_memory\(self, memory: Memory\) -&gt; str:

"""Add any memory type \(episodic, semantic, working\) Calculates RFM scores based on type. 

Validates against North Star constraints. 

""" 

metadata = self.\_calculate\_metadata\(memory\) return self.\_store\_with\_vectors\(memory, metadata\)

\# Result: Clean, extensible, no hacks needed

**SECTION 7: VARIABLE NAMING FOR LLM-WRITTEN CODE**

**Whole Words, Every Time**

**Reasoning**: Most code is now LLM-generated. LLMs excel with explicit, unambiguous names. Cryptic abbreviations defeat that advantage. 

❌ **BAD** \(traditional, developer-written code\): r = 0.5 \# recency

f = 0.3 \# frequency

i = 0.2 \# importance

ps = r \* 0.3 \+ f \* 0.2 \+ i \* 0.5 \# priority score cur\_mem = memories\[i\] \# current memory

✅ **GOOD** \(LLM-compatible, explicit\):

recency\_score = 0.5

frequency\_score = 0.3

importance\_score = 0.2

priority\_score = \(recency\_score \* 0.3\) \+ \(frequency\_score \* 0.2\) \+ \(importance\_score \* 0.5\) current\_memory = memories\[index\]

**Benefits**:

LLM context window better utilized \(name ≈ documentation\) Self-documenting \(no comments needed for obvious names\) Reduces errors \(typos in short names hide bugs\) Maintainability \(future readers don't need to decode abbreviations\) **Naming Conventions**

Element

Pattern

Example

Variables

lower\_case\_snake

memory\_priority, recency\_score

Constants

UPPER\_CASE\_SNAKE

MAX\_WORKING\_MEMORY\_CAPACITY, DEFAULT\_HALF\_LIFE\_DAYS

Functions

verb\_noun\_lowercase

calculate\_priority, retrieve\_memories

Classes

PascalCase

MemoryMetadata, RFMCalculator

Private

\_leading\_underscore

\_calculate\_recency\_internal

Boolean

is\_/has\_/can\_

is\_valid\_memory, has\_children, can\_consolidate **SECTION 8: IMPLEMENTATION CHECKLIST \(NOT A TODO LIST\)** Use this to validate each iteration, NOT as a linear plan:

**Pre-Implementation**

\[ \] Problem clearly stated and mapped to North Star

\[ \] Data structures identified \(with Big O analysis\)

\[ \] Edge cases documented

\[ \] Mathematical rigor confirmed \(formulas, proofs if applicable\)

\[ \] Golden Rules compliance checked \(all 25\)

\[ \] Design elegance validated \(no hacks needed?\)

\[ \] Dependencies mapped \(DAG confirmed, no circles\) **During Implementation**

\[ \] Use whole words for variable names \(LLM-compatible\)

\[ \] Every function has single responsibility

\[ \] Type hints on all functions \(Python\) or complete types \(Go\)

\[ \] Comments explain WHY, not WHAT

\[ \] No hard-coded values \(use constants or config\)

\[ \] Error messages explicit and contextual

\[ \] No dead code, unused variables, stale comments **After Implementation**

\[ \] Code passes linter without warnings \(ruff for Python, golangci-lint for Go\)

\[ \] Code is formatted \(black for Python, gofmt for Go\)

\[ \] Tests validate design \(not just "happy path"\)

\[ \] Edge cases covered in tests

\[ \] Big O verified \(profiling if applicable\)

\[ \] Integration with North Star confirmed

\[ \] New problems revealed? \(→ next iteration\) **SECTION 9: ELEGANCE CRITERIA**

If you answer "YES" to any of these, the design needs rework: 1. **Do you need hacks to make it work? **

Examples: Global state, type casting to bypass checks, try-except to hide errors Hacks = design flaw

2. **Are you writing the same logic in multiple places? **

Violates DRY principle

Sign of missing abstraction

3. **Is the code hard to test in isolation? **

Indicates tight coupling

Violates single responsibility

4. **Does it require extensive comments to explain WHAT it does? **

Code should be self-documenting

Comments explain WHY, not WHAT

5. **Are there circular dependencies? **

Critical flaw: breaks isolation and testing Must refactor to DAG

6. **Could a junior dev maintain this 6 months from now? **

If NO: Not elegant

Elegance = understandable without expert knowledge **CONCLUSION: THE MANIFESTO**

**This is not a process to follow blindly. It is a North Star to navigate by. **

**Start with understanding** \(read problem 3x, create graph\) **Build incrementally** \(small, validated steps\) **Check constantly** \(against Golden Rules, North Star\) **Adapt dynamically** \(new problems → new iterations\) **Value elegance** \(if you hack, redesign\) **Use whole words** \(LLM-written code deserves clarity\) **Trust mathematics** \(not guesses or cargo-cult practices\) **The goal**: Systems that are clear, robust, maintainable, and beautiful. 

Not because perfectionism matters, but because **elegance is how good systems stay good over time**. 

\[1\] \[2\]

⁂

1. CODING\_STANDARDS.md

2. DESIGN\_PRINCIPLES\_GUIDE.md




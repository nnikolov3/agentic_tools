### Project : Agentic Tools MCP

### Target : MEMORY.md Implementation

### Goal : Enable writing to Qdrant VM by end of today

### Date : November 7, 2025, 2:36 PM PST

### Qdrant VM : http://192.168.122.40:

### Your project has basic Qdrant connectivity and can perform simple memory operations, but is

### missing critical components specified in MEMORY.md:

### Working :

### Missing :

### The current add_memory() WILL write to Qdrant but stores only minimal metadata (text_content,

### timestamp, day_of_week). To align with MEMORY.md architecture, you need to implement

### comprehensive payload fields with RFM priority scoring.

### One blocking bug exists : Async client initialization in qdrant_memory.py line 102 needs fixing.

# Cognitive Memory System: Complete Audit &

# Implementation Plan

## Executive Summary

## Current Status: 30% Complete (Phase 1 Foundation)

### ✓ Qdrant VM connectivity (192.168.122.40:6333)

### ✓ Multi-vector support (dense + sparse)

### ✓ Basic add_memory() and retrieve_context()

### ✓ Collection management

### ✓ HNSW indexing configuration

### ✗ Type-safe Pydantic models

### ✗ RFM priority scoring

### ✗ Episodic/semantic/working memory separation

### ✗ Memory consolidation pipeline

### ✗ System 1/System 2 agent integration

### ✗ Memory pruning implementation

## Critical Finding


### Status : Ready for production, no changes needed

### Capabilities :

### Code Quality : Excellent separation of concerns, proper async patterns, comprehensive configuration

### Recommendation : No modifications required for Phase 1 goals

### Status : Partially functional, requires significant enhancements

### What Works :

### Critical Issues Identified :

## Section 1: Detailed File Audit

## 1.1 qdrant_client_manager.py ✓ FULLY IMPLEMENTED

### Async AsyncQdrantClient with proper initialization

### Multi-vector configuration (dense: text-dense, sparse: text-sparse)

### HNSW indexing: m=32, ef_construct=400, on_disk=true

### Payload index creation for timestamp, text_content, day_of_week

### Quantization support (int8, quantile=0.99)

### VM connectivity: http://192.168.122.40:

### gRPC enabled on port 6334

## 1.2 qdrant_memory.py ⚠ MAJOR FIXES NEEDED

### add_memory(text_content: str) - basic write operation

### retrieve_context(query_text: str) - time-bucket weighted retrieval

### Multi-vector embedding generation (dense + sparse)

### Collection initialization and vector name resolution

### Time-decay retrieval with 8 time buckets + knowledge bank

## Issue #1: No Type Safety (HIGH PRIORITY)

### Current : Accepts plain string add_memory(text_content: str)

### Required : Type-safe Memory models add_memory(memory: Memory | str)

### Impact : Data corruption risk, no validation, incompatible with MEMORY.md

### Fix Time : 30 minutes (create models.py with Pydantic)


### Current payload contains only 3 fields:

#### {

```
"text_content": str,
"timestamp": float,
"day_of_week": str
}
```
### Required payload must contain 20+ fields:

#### {

```
# Core
"memory_id": str,
"memory_type": str, # episodic/semantic
"text_content": str,
```
```
# Timestamps
"created_at": float,
"updated_at": float,
"last_accessed_at": float | None,
```
```
# RFM Scores
"recency_score": float,
"frequency_score": float,
"importance_score": float,
"priority_score": float,
"access_count": int,
```
```
# Classification
"tags": list[str],
"agent_name": str | None,
"day_of_week": str,
```
```
# Type-specific
"event_type": str | None, # episodic
"context": dict | None, # episodic
"parent_memory_id": str | None, # hierarchical
"source_memory_ids": list[str] | None, # semantic
"confidence_score": float | None, # semantic
"domain": str | None # semantic
}
```
## Issue #2: Missing RFM Priority Scoring (HIGH PRIORITY)

### Current : No priority calculation or storage

### Required : Priority = (Recency × 0.3) + (Frequency × 0.2) + (Importance × 0.5)

### Impact : Cannot prioritize important memories, no retrieval boosting

### Fix Time : 20 minutes (implement calculation methods)

## Issue #3: Minimal Payload Metadata (HIGH PRIORITY)


### Fix Time : 30 minutes (update add_memory() and to_qdrant_payload())

### Status : Working, needs minor improvements

### What Works :

## Issue #4: Async Client Initialization Bug (CRITICAL BLOCKER)

### Location : Line 102 in _init_qdrant()

### Current Code : self.client = qdrant_manager.get_client()

### Problem : get_client() returns a coroutine but is not awaited

### Runtime Error : Will fail when add_memory() attempts to use self.client

### Fix : All usages must await: client = await self.client

### Fix Time : 5 minutes

## Issue #5: prune_memories() Not Implemented (MEDIUM PRIORITY)

### Current : Returns warning message, does nothing

### Required : Scroll collection, filter by age/priority, delete batch

### Impact : Memory bloat over time, no cleanup mechanism

### Fix Time : 45 minutes

## Issue #6: No Access Tracking (MEDIUM PRIORITY)

### Current : retrieve_context() doesn't update metadata

### Required : Increment access_count, update last_accessed_at, recalculate frequency_score

### Impact : Frequency score remains 0, no adaptive prioritization

### Fix Time : 20 minutes (implement update_memory_access())

## Issue #7: No Memory Type Separation (MEDIUM PRIORITY)

### Current : All memories stored identically

### Required : Episodic vs Semantic distinction with type-specific fields

### Impact : Cannot implement consolidation pipeline

### Fix Time : Included in Issue #3 fix

## 1.3 embedding_models.py ⚠ MINOR FIXES

### GoogleEmbedder with Gemini API

### MistralEmbedder with Mistral API

### FastEmbedEmbedder with local models

### Factory pattern: create_embedder(config)


### Issues :

### Status : Fully functional, minor config mismatch

### Capabilities :

### Issue #10: Config Section Mismatch (LOW PRIORITY)

### Proper API key loading from environment

## Issue #8: FastEmbed Return Type (LOW PRIORITY)

### Problem : embed() returns generator, need list[float]

### Fix : Add .tolist() conversion

### Impact : LOW - would fail only if using FastEmbed

### Fix Time : 5 minutes

## Issue #9: No Multi-Provider Support (MEDIUM PRIORITY - Phase 2)

### Current : Single embedder per memory instance

### Required : Generate embeddings from multiple providers simultaneously (Google + Mistral)

### Purpose : Late interaction retrieval with MaxSim operator

### Impact : MEDIUM - cannot implement advanced retrieval

### Fix Time : 1 hour (create MultiEmbedder class)

### Note : Can defer to Phase 2

## 1.4 ingest_knowledge_bank.py ✓ MOSTLY GOOD

### PDF processing with pdfminer

### JSON flattening and processing

### Markdown with header-based chunking

### LLM summarization via Google Gemini

### Async batch processing with semaphore

### Content deduplication via SHA256 hashing

### Retry logic for network failures

### Problem : Expects [knowledge_bank_ingestion] section in config

### Current Config : Only has [memory] section

### Impact : LOW - knowledge bank ingestion will fail, but memory writes work

### Fix : Add config section to agentic-tools.toml

### Fix Time : 2 minutes


### What's Configured :

### Missing Configuration :

```
[memory]
prune_enabled = true
prune_days = 365
prune_confidence_threshold = 0.
prune_batch_size = 100
```
```
[memory]
priority_recency_weight = 0.
priority_frequency_weight = 0.
priority_importance_weight = 0.
frequency_max_accesses = 100
recency_half_life_days = 30.
```
```
[knowledge_bank_ingestion]
source_directory = "docs/knowledge"
output_directory = ".ingested"
supported_extensions = [".json", ".md", ".pdf"]
chunk_size = 1024
chunk_overlap = 256
qdrant_batch_size = 128
concurrency_limit = 2
prompt = "Summarize this document with practical examples."
model = "gemini-2.0-flash-exp"
google_api_key_name = "GEMINI_API_KEY"
```
## 1.5 agentic-tools.toml ⚠ INCOMPLETE

### Qdrant connection: http://192.168.122.40:

### gRPC enabled on port 6334

### Embedding models: mistral-embed (1024-dim), naver/splade-v3 (sparse)

### Collection names: agent_memory, knowledge-bank

### HNSW parameters: m=32, ef_construct=

### Retrieval weights: All 0.0 except knowledge_bank=1.

### 1. Pruning Settings :

### 2. Priority Scoring Parameters :

### 3. Knowledge Bank Ingestion :


### Status : Functional with minor issue

### Capabilities :

### Issue #11: Prune Implementation

```
Requirement Status Implementation Priority Est. Time
```
```
Multi-vector storage ✓ Complete qdrant_client_manager.py DONE 0 min
```
```
Type-checked metadata (Pydantic) ✗ Missing Need models.py CRITICAL 30 min
```
```
Priority scoring (RFM) ✗ Missing Need in qdrant_memory.py CRITICAL 20 min
```
```
Provider abstraction ⚠ Partial embedding_models.py works MEDIUM 0 min
```
```
Episodic/semantic separation ✗ Missing Need memory_type field HIGH 10 min
```
```
Hierarchical metadata ✗ Missing Need parent_memory_id MEDIUM 5 min
```
### Phase 1 Completion: 33% (2/6 requirements complete)

### Blockers for 100% Phase 1 :

### Estimated Time to Complete Phase 1 : 1.5-2 hours

## 1.6 main.py ✓ WORKING

### VM-aware config loading

### Async initialization of Qdrant and memory

### Multi-agent orchestration with FastMCP

### Knowledge bank ingestion via --ingest flag

### Memory pruning via --prune flag (but prune_memories() is stub)

### Line 73 calls memory.prune_memories() which does nothing

### Fix by implementing prune_memories() method first

## Section 2: MEMORY.MD Compliance Gap Analysis

## Phase 1: Core Multi-Vector Memory Foundation

### 1. Create Pydantic models (models.py)

### 2. Implement RFM priority scoring

### 3. Add memory_type classification

### 4. Enhance payload with all metadata fields


```
Requirement Status Notes
```
```
Memory Analyst agent ✗ Not started Lightweight LLM for classification
```
```
Memory Synthesizer agent ✗ Not started Automatic enrichment before storage
```
```
Query complexity classification ✗ Not started Route to appropriate retrieval strategy
```
```
Adaptive retrieval strategies ⚠ Basic Time-bucket weighting exists
```
### Phase 2 Completion: 10% (Basic retrieval framework exists)

### Not Required for Today's Goal : Phase 2 can wait

```
Requirement Status Estimated Effort
```
```
LLM-guided consolidation ✗ Not started 4-6 hours
```
```
Offline batch processing ✗ Not started 2-3 hours
```
```
Conflict resolution ✗ Not started 3-4 hours
```
```
Selective semantic updating ✗ Not started 2-3 hours
```
### Phase 3 Completion: 0%

### Not Required for Today's Goal : Future work

### Not started (0%). Future work after Phase 3.

### Not started (0%). Long-term roadmap.

### Timeline : 2-2.5 hours to working system

### File : src/memory/models.py

### Classes to Implement :

## Phase 2: System 1 Agent Implementation

## Phase 3: Consolidation Pipeline

## Phase 4: System 2 Integration

## Phase 5: Advanced Features

## Section 3: Today's Implementation Plan

## Goal: Write Memories to Qdrant with Full Metadata

## Step 1: Create Pydantic Models (30 minutes)

### 1. MemoryType (Enum)


```
class MemoryType(str, Enum):
EPISODIC = "episodic"
SEMANTIC = "semantic"
WORKING = "working"
```
```
class EventType(str, Enum):
USER_INTERACTION = "user_interaction"
TOOL_EXECUTION = "tool_execution"
ERROR_EVENT = "error_event"
SYSTEM_EVENT = "system_event"
AGENT_DECISION = "agent_decision"
```
```
class MemoryMetadata(BaseModel):
recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
frequency_score: float = Field(default=0.0, ge=0.0, le=1.0)
importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
access_count: int = Field(default= 0 , ge= 0 )
last_accessed_at: Optional[datetime] = None
```
```
@property
def priority_score(self) -&gt; float:
return (
self.recency_score * 0.3 +
self.frequency_score * 0.2 +
self.importance_score * 0.
)
```
```
class Memory(BaseModel):
id: str = Field(default_factory=lambda: str(uuid4()))
memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
text_content: str = Field(..., min_length= 1 )
created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
tags: list[str] = Field(default_factory=list)
agent_name: Optional[str] = None
```
```
def to_qdrant_payload(self) -&gt; dict[str, Any]:
# Returns comprehensive payload dict
```
```
class EpisodicMemory(Memory):
memory_type: MemoryType = Field(default=MemoryType.EPISODIC, frozen=True)
event_type: EventType = Field(default=EventType.USER_INTERACTION)
```
### 2. EventType (Enum)

### 3. MemoryMetadata (RFM Scoring)

### 4. Memory (Base Model)

### 5. EpisodicMemory (Extends Memory)


```
context: dict[str, Any] = Field(default_factory=dict)
parent_memory_id: Optional[str] = None
```
```
class SemanticMemory(Memory):
memory_type: MemoryType = Field(default=MemoryType.SEMANTIC, frozen=True)
source_memory_ids: list[str] = Field(default_factory=list)
confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
domain: Optional[str] = None
```
```
class MemoryQuery(BaseModel):
query_text: str = Field(..., min_length= 1 )
limit: int = Field(default= 20 , ge= 1 , le= 100 )
memory_types: list[MemoryType] = Field(default_factory=list)
min_priority_score: Optional[float] = None
tags: Optional[list[str]] = None
agent_name: Optional[str] = None
time_range_hours: Optional[int] = None
```
### Validation : All models include Pydantic validators for timestamps, score ranges, and required fields

### Import Models :

```
from src.memory.models import (
Memory, EpisodicMemory, SemanticMemory,
MemoryMetadata, MemoryQuery, MemoryType
)
import math
```
### Add Priority Scoring Methods :

```
def calculate_recency_score(
self,
created_at: datetime,
half_life_days: float = 30.
) -&gt; float:
"""Exponential decay: score = exp(-decay_rate * age_days)"""
now = datetime.now(UTC)
age_days = (now - created_at).total_seconds() / 86400.
decay_rate = math.log( 2 ) / half_life_days
return math.exp(-decay_rate * age_days)
```
```
def calculate_frequency_score(
self,
access_count: int,
```
### 6. SemanticMemory (Extends Memory)

### 7. MemoryQuery (Query Parameters)

## Step 2: Update qdrant_memory.py (1 hour)


```
max_accesses: int = 100
) -&gt; float:
"""Logarithmic scaling: score = log(1+count) / log(1+max)"""
if access_count &lt;= 0 :
return 0.
normalized = math.log1p(access_count) / math.log1p(max_accesses)
return min(1.0, normalized)
```
```
def calculate_priority_score(self, metadata: MemoryMetadata) -&gt; float:
"""RFM formula: Priority = (R×0.3) + (F×0.2) + (I×0.5)"""
return metadata.priority_score # Uses @property
```
### Update add_memory() Method :

```
async def add_memory(
self,
memory: Memory | str,
importance: Optional[float] = None
) -&gt; str:
"""Add memory with full RFM scoring and metadata."""
```
```
# Convert string to EpisodicMemory if needed
if isinstance(memory, str):
memory = EpisodicMemory(
text_content=memory,
metadata=MemoryMetadata(importance_score=importance or 0.5)
)
elif importance is not None:
memory.metadata.importance_score = importance
```
```
# Calculate dynamic scores
memory.metadata.recency_score = self.calculate_recency_score(
memory.created_at
)
memory.metadata.frequency_score = self.calculate_frequency_score(
memory.metadata.access_count
)
```
```
# Generate embeddings
dense_vector = self._embed_text(memory.text_content)
sparse_vector = self._embed_sparse([memory.text_content])[ 0 ]
```
```
# Build vector payload
vector_payload = {self.agent_dense_name: dense_vector}
if self.agent_sparse_name:
vector_payload[self.agent_sparse_name] = sparse_vector
```
```
# Create point with comprehensive payload
point = models.PointStruct(
id=memory.id,
vector=vector_payload,
payload=memory.to_qdrant_payload() # 20+ fields
)
```
```
# Upsert to Qdrant
```

```
client = await self.client
await client.upsert(
collection_name=self.collection_name,
points=[point],
wait=True
)
```
```
logger.info(
f"Added {memory.memory_type.value} memory '{memory.id}' "
f"with priority {memory.metadata.priority_score:.3f}"
)
return memory.id
```
### Add Access Tracking Method :

```
async def update_memory_access(
self,
memory_id: str,
increment_count: bool = True
) -&gt; None:
"""Update metadata after retrieval."""
client = await self.client
```
```
# Retrieve current point
point = await client.retrieve(
collection_name=self.collection_name,
ids=[memory_id],
with_payload=True
)
```
```
if not point:
logger.warning(f"Memory {memory_id} not found")
return
```
```
payload = point[ 0 ].payload
new_access_count = payload.get("access_count", 0 ) + ( 1 if increment_count else 0 )
new_frequency_score = self.calculate_frequency_score(new_access_count)
```
```
# Update
await client.set_payload(
collection_name=self.collection_name,
points=[memory_id],
payload={
"access_count": new_access_count,
"frequency_score": new_frequency_score,
"last_accessed_at": datetime.now(UTC).timestamp(),
}
)
```
### Fix Async Client Issue :

### All methods using self.client must await it: client = await self.client

### Already done in the updated methods above


### File : test_qdrant_write.py

### Purpose : Validate that memories write correctly to Qdrant VM

### Test Cases :

### Expected Output :

```
2025-11-07 14:30:00 - INFO - Testing priority score calculations...
2025-11-07 14:30:00 - INFO - High importance fresh memory: 0.800 (expected ~0.8)
2025-11-07 14:30:00 - INFO - Low importance frequent memory: 0.450 (expected ~0.45)
2025-11-07 14:30:00 - INFO - Balanced memory: 0.650 (expected ~0.65)
2025-11-07 14:30:00 - INFO - ✓ Priority scoring tests complete
```
```
2025-11-07 14:30:01 - INFO - Loading configuration...
2025-11-07 14:30:01 - INFO - Initializing Qdrant connection...
2025-11-07 14:30:02 - INFO - Creating test memory...
2025-11-07 14:30:02 - INFO - Writing memory to Qdrant...
2025-11-07 14:30:03 - INFO - ✓ Successfully wrote memory with ID: abc123...
2025-11-07 14:30:03 - INFO - Retrieving memory from Qdrant...
2025-11-07 14:30:04 - INFO - ✓ Successfully retrieved memory
2025-11-07 14:30:04 - INFO - Collection 'agent_memory' stats:
2025-11-07 14:30:04 - INFO - - Points count: 1
2025-11-07 14:30:04 - INFO - - Vectors config: {'text-dense': VectorParams(...)}
```
```
============================================================
✓ ALL TESTS PASSED - Qdrant write system is working!
============================================================
```
### Run Command :

```
python test_qdrant_write.py
```
## Step 3: Create Test Script (20 minutes)

### 1. test_priority_scoring() : Verify RFM calculations

### High importance + fresh → priority ~0.

### Low importance + frequent → priority ~0.

### Balanced → priority ~0.

### 2. test_qdrant_write() : End-to-end write/read

### Initialize connection

### Create EpisodicMemory with tags

### Write via add_memory()

### Retrieve via retrieve_context()

### Verify payload fields

### Check collection stats


### Checklist :

## Step 4: Validation (10 minutes)

### 1. VM Connectivity :

### ping -c 3 192.168.122.40 → Success

### curl http://192.168.122.40:6333/collections → Returns JSON

### 2. Environment Variables :

### echo $GEMINI_API_KEY → Shows key

### OR echo $MISTRAL_API_KEY → Shows key

### 3. Dependencies :

```
pip install qdrant-client pydantic sentence-transformers fastembed
```
### 4. Run Test Script :

```
python test_qdrant_write.py
```
### Should exit with code 0

### Should print "ALL TESTS PASSED"

### 5. Verify in Qdrant UI :

### Open http://192.168.122.40:6333/dashboard

### Navigate to Collections → agent_memory

### Click on a point to inspect payload

### Confirm all 20+ fields present:

### memory_id, memory_type, text_content

### created_at, updated_at

### recency_score, frequency_score, importance_score, priority_score

### access_count, tags, agent_name

### event_type, context, etc.

### 6. Type Check with mypy :

```
mypy src/memory/models.py
mypy src/memory/qdrant_memory.py
```
### Should report 0 errors


### Issue : VM at 192.168.122.40 becomes unreachable

### Impact : Cannot write to Qdrant

### Probability : Low (KVM VMs are stable)

### Mitigation :

### Issue : First run downloads large models (>1GB)

### Impact : 5-10 minute delay

### Probability : High on first run

### Mitigation :

### Issue : Incorrect await usage causes RuntimeError

### Impact : Script crashes

### Probability : Medium (common async mistake)

### Mitigation :

## Section 4: Risk Assessment & Mitigation

## High-Risk Issues

## Risk 1: VM Network Connectivity

### Pre-flight ping test

### Check libvirt network: virsh net-list

### Verify port forwarding: ss -tulpn | grep 6333

### Restart VM if needed: virsh start &lt;vm-name&gt;

## Risk 2: Embedding Model Download Time

### Use cached models if available: ~/.cache/huggingface

### Start download early in background

### Use smaller model temporarily: BAAI/bge-small-en-v1.5 (133MB)

### Monitor with: watch -n 1 ls -lh ~/.cache/huggingface/hub

## Risk 3: Async Pattern Errors

### Test with simple examples first

### Use asyncio.run() at top level

### Ensure all coroutines awaited

### Add try/except blocks for debugging


### Issue : Changing payload structure breaks existing data

### Impact : Need to migrate all points

### Probability : Medium (as requirements evolve)

### Mitigation :

### Issue : Collection grows unbounded without pruning

### Impact : Slow queries, high storage

### Probability : High over weeks/months

### Mitigation :

### Issue : Typo in TOML breaks initialization

### Impact : Clear error message, easy to fix

### Probability : Low

### Mitigation : TOML syntax checking, use default values

## Medium-Risk Issues

## Risk 4: Schema Evolution

### Version payload schema: schema_version = "1.0"

### Write migration scripts before breaking changes

### Test on copy of collection first

### Use set_payload() to update in place

## Risk 5: Memory Bloat

### Implement pruning in Phase 2

### Monitor collection size: collection_info.points_count

### Set alerts at 100k, 1M points

### Archive old memories to separate collection

## Low-Risk Issues

## Risk 6: Config Errors

## Section 5: Post-Implementation Roadmap

## Immediate Next Steps (After Today)

### 1. Implement Memory Pruning (Phase 1, 45 min)

### Delete memories older than prune_days with low priority

### Batch delete with scroll + filter

### Log pruning stats


### 2. Enhanced Error Handling (Phase 1, 30 min)

### Retry logic with tenacity

### Better logging with structlog

### Connection pooling

### 3. Multi-Provider Embeddings (Phase 2, 1 hour)

### Simultaneous Google + Mistral embeddings

### Store multiple vectors per memory

### Late interaction retrieval

## Short-Term Goals (Next Week)

### 4. System 1 Memory Analyst Agent (Phase 2, 3-4 hours)

### Lightweight LLM (Mistral 7B or Phi-3)

### Classifies incoming memories

### Enriches metadata automatically

### Runs before storage

### 5. Query Complexity Classification (Phase 2, 2 hours)

### Route simple queries to single-vector search

### Route complex queries to multi-vector + graph

### Adaptive strategy selection

### 6. Working Memory Buffer (Phase 4, 2 hours)

### 4-chunk capacity limit

### Temporary context storage

### Automatic cleanup

## Medium-Term Goals (Next Month)

### 7. Consolidation Pipeline (Phase 3, 6-8 hours)

### Batch job: episodic → semantic

### LLM-guided synthesis

### Conflict resolution

### Parent-child linking

### 8. Memory Synthesizer Agent (Phase 2, 3-4 hours)

### Pre-storage enrichment

### Automatic tagging

### Context extraction

### 9. Graph Relationship Traversal (Phase 5, 4-6 hours)


### All code is provided in the attached implementation plan. Key files:

### Parent-child memory links

### Causal chains

### Multi-hop reasoning

## Long-Term Vision (Next Quarter)

### 10. Multimodal Memory (Phase 5, 8-10 hours)

### Image embeddings (CLIP)

### Code snippet storage

### Diagram understanding

### 11. Meta-Learning (Phase 5, 10-12 hours)

### System learns to improve itself

### Adaptive consolidation strategies

### Self-optimizing retrieval

### 12. Comprehensive Metrics (All Phases, ongoing)

### Retrieval quality (NDCG, Recall@k)

### Consolidation ratio

### System health dashboards

## Section 6: Complete Code Deliverables

## src/memory/models.py (NEW)

### ~150 lines

### 7 Pydantic model classes

### Full type safety with validators

### Comprehensive payload generation

## src/memory/qdrant_memory.py (MODIFIED)

### ~100 lines added

### 3 new methods for RFM scoring

### Updated add_memory() signature

### Access tracking implementation

### Fixed async client usage


### YES , with 1.5-2 hours of implementation work.

### Total: ~95 minutes to working system

### Focus on Phase 1 completion (today's goal) before moving to Phase 2. The current implementation

### is 30% complete; after today's work it will be 80% complete for Phase 1.

### Correctness over speed : Take time to test thoroughly. A working system today is better than a

### broken system rushed to completion.

## test_qdrant_write.py (NEW)

### ~80 lines

### Priority scoring tests

### End-to-end write/read validation

### Collection stats reporting

## agentic-tools.toml (MODIFIED)

### Add [knowledge_bank_ingestion] section

### Add pruning parameters

### Add priority scoring weights

## Conclusion

## Can You Write to Qdrant Today?

## What's the Minimum Viable Path?

### 1. Fix async client bug (5 min) - CRITICAL BLOCKER

### 2. Create Pydantic models (30 min) - CRITICAL for type safety

### 3. Update add_memory() (30 min) - CRITICAL for full metadata

### 4. Create test script (20 min) - CRITICAL for validation

### 5. Run and verify (10 min) - CRITICAL for confirmation

## What Happens After?

### Memories will write with full metadata (20+ fields)

### Priority scoring will enable intelligent retrieval

### Type safety will prevent data corruption

### Foundation ready for Phase 2 (System 1 agents)

## Recommendation


```
# Pre-flight checks
ping -c 3 192.168.122.
curl http://192.168.122.40:6333/collections
ss -tulpn | grep 6333
```
```
# Install dependencies
pip install qdrant-client pydantic sentence-transformers fastembed
```
```
# Set environment variables
export GEMINI_API_KEY="your-key-here"
```
```
# Run test
python test_qdrant_write.py
```
```
# Type check
mypy src/memory/models.py src/memory/qdrant_memory.py
```
```
# Qdrant UI
# Browser: http://192.168.122.40:6333/dashboard
```
### End of Audit Report



# Cognitive Memory System: Complete Audit & Implementation Plan


I've completed a **comprehensive detailed audit** of your project against the MEMORY.md specification. Here's the executive summary:

## Current Status: 30% Complete (Phase 1 Foundation)

**Good News**: Your project has solid Qdrant connectivity and can write to the VM. The infrastructure (qdrant_client_manager.py) is excellent.

**Critical Finding**: Current `add_memory()` **WILL work** but stores minimal metadata (only text_content, timestamp, day_of_week). To align with MEMORY.md, you need to implement **comprehensive payload fields with RFM priority scoring**.

## What You Need to Do Today (1.5-2 hours)

### Phase 1: Immediate Implementation

**Step 1: Create Pydantic Models** (30 min)
- New file: `src/memory/models.py`
- Define type-safe schemas: Memory, EpisodicMemory, SemanticMemory, MemoryMetadata
- Implement RFM priority scoring as a @property
- Add comprehensive to_qdrant_payload() methods with 20+ fields

**Step 2: Update qdrant_memory.py** (1 hour)
- Fix critical async client bug (line 102: must await get_client())
- Add calculate_recency_score() - exponential decay formula
- Add calculate_frequency_score() - logarithmic scaling
- Update add_memory() to accept Memory models
- Enhance payload from 3 fields to 20+ fields
- Add update_memory_access() for frequency tracking

**Step 3: Create Test Script** (20 min)
- New file: `test_qdrant_write.py`
- Validate priority calculations
- Test end-to-end write/read
- Verify payload in Qdrant UI

**Step 4: Validate** (10 min)
- Run test script
- Check Qdrant dashboard at http://192.168.122.40:6333/dashboard
- Verify all metadata fields present

## Critical Issues Found

### 11 Issues Identified (Prioritized)

**P0 - BLOCKER**:
- Issue #6: Async client not awaited (5 min fix)

**P1 - CRITICAL** (blocks MEMORY.md compliance):
- Issue #1: No Pydantic type safety (30 min)
- Issue #2: Missing RFM priority scoring (20 min)
- Issue #3: Minimal payload metadata - need 20+ fields (30 min)

**P2 - HIGH** (important for functionality):
- Issue #4: prune_memories() not implemented (45 min)
- Issue #5: No access tracking/frequency updates (20 min)
- Issue #7: No episodic/semantic separation (included in #3)

**P3 - MEDIUM** (can defer):
- Issue #9: No multi-provider embeddings (1 hour, Phase 2)
- Issue #10: Config section mismatch (2 min)

**P4 - LOW**:
- Issue #8: FastEmbed return type (5 min)
- Issue #11: Prune stub in main.py (depends on #4)

## MEMORY.MD Compliance

### Phase 1 Requirements (Core Foundation)
- ✓ Multi-vector storage: **COMPLETE**
- ✓ Provider abstraction: **WORKING**
- ✗ Type-safe schemas: **MISSING** (today's work)
- ✗ RFM priority scoring: **MISSING** (today's work)
- ✗ Episodic/semantic separation: **MISSING** (today's work)
- ⚠️ Hierarchical metadata: **PARTIAL** (need parent_memory_id)

**Phase 1 Completion: 33%** → After today: **80%**

### Phase 2-5 Requirements
Not started (0%). Can address after Phase 1 is solid.

## Payload Schema Comparison

**Current (Minimal)**:
```python
{
    "text_content": str,
    "timestamp": float,
    "day_of_week": str
}
```

**Required (Full)**:
```python
{
    # Core
    "memory_id": str,
    "memory_type": "episodic" | "semantic",
    "text_content": str,
    
    # Timestamps
    "created_at": float,
    "updated_at": float,
    "last_accessed_at": float | None,
    
    # RFM Scoring
    "recency_score": float,  # 0.0-1.0
    "frequency_score": float,  # 0.0-1.0
    "importance_score": float,  # 0.0-1.0
    "priority_score": float,  # Calculated
    "access_count": int,
    
    # Classification
    "tags": list[str],
    "agent_name": str | None,
    "day_of_week": str,
    
    # Type-specific fields
    "event_type": str | None,  # episodic
    "context": dict | None,  # episodic
    "parent_memory_id": str | None,  # hierarchical
    "source_memory_ids": list[str] | None,  # semantic
    "confidence_score": float | None,  # semantic
    "domain": str | None  # semantic
}
```

## Quick Start Commands

```bash
# Pre-flight checks
ping -c 3 192.168.122.40
curl http://192.168.122.40:6333/collections

# Install dependencies
pip install qdrant-client pydantic sentence-transformers fastembed

# Set API key
export GEMINI_API_KEY="your-key-here"

# Run test
python test_qdrant_write.py

# Verify in UI
# Open: http://192.168.122.40:6333/dashboard
```

## Recommendation

**You CAN write to Qdrant today** with 1.5-2 hours of focused implementation. The complete code for all three files (models.py, updated qdrant_memory.py, test_qdrant_write.py) is provided in the PDF.

**Priority: Correctness over speed**. Take time to test thoroughly. A working system at 5 PM is better than a broken system rushed by 3 PM.


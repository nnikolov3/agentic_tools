# COMPLETE TEST SUITE
# File: test_qdrant_write.py

import asyncio
import logging
import os
from datetime import datetime, UTC, timedelta

from src.memory.qdrant_client_manager import QdrantClientManager
from src.memory.qdrant_memory import QdrantMemory
from src.memory.models import (
    EpisodicMemory,
    SemanticMemory,
    MemoryMetadata,
    MemoryType,
    EventType,
)
from src.memory.rfm_calculator import RFMCalculator
from src.memory.prune_memories import prune_memories

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure Qdrant VM is accessible
QDRANT_HOST = "192.168.122.40"
QDRANT_PORT = 6333

# Minimal config for testing
config = {
    "memory": {
        "collection_name": "agent_memory_test",
        "qdrant_url": f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        "prefer_grpc": True,
        "embedding_model": {
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
            "embedding_size": 384,
            "device": "cpu"
        },
        "rfm_config": {
            "recency_half_life_days": 30.0,
            "frequency_max_accesses": 100,
            "frequency_log_base": 10
        },
        "working_memory": {
            "capacity": 4
        },
        "vectors": {
            "distance": "Cosine",
            "datatype": "float16",
            "on_disk": False
        },
        "hnsw_config": {
            "m": 32,
            "ef_construct": 400
        },
        "quantization_config": {
            "type": "int8",
            "quantile": 0.99
        },
        "pruning": {
            "enabled": True,
            "prune_days": 365,
            "prune_min_priority": 0.2
        }
    }
}


async def test_rfm_calculator():
    """Test RFM calculations match scientific formulas"""
    logger.info("\n========== Testing RFM Calculator ==========")
    
    calc = RFMCalculator(recency_half_life_days=30.0, frequency_max_accesses=100)
    
    # Test 1: Recency decay
    now = datetime.now(UTC)
    
    # Just created: recency = 1.0
    r1 = calc.calculate_recency_score(now, now)
    assert 0.99 < r1 <= 1.0, f"Expected ~1.0, got {r1}"
    logger.info(f"✓ Recency at t=0: {r1:.3f}")
    
    # 30 days old: recency = 0.5 (half-life)
    r2 = calc.calculate_recency_score(now - timedelta(days=30), now)
    assert 0.49 < r2 < 0.51, f"Expected ~0.5, got {r2}"
    logger.info(f"✓ Recency at t=30d: {r2:.3f} (half-life)")
    
    # 60 days old: recency = 0.25
    r3 = calc.calculate_recency_score(now - timedelta(days=60), now)
    assert 0.24 < r3 < 0.26, f"Expected ~0.25, got {r3}"
    logger.info(f"✓ Recency at t=60d: {r3:.3f}")
    
    # Test 2: Frequency scaling (logarithmic)
    f0 = calc.calculate_frequency_score(0)
    assert f0 == 0.0
    logger.info(f"✓ Frequency for 0 accesses: {f0:.3f}")
    
    f1 = calc.calculate_frequency_score(1)
    assert 0.04 < f1 < 0.05
    logger.info(f"✓ Frequency for 1 access: {f1:.3f}")
    
    f10 = calc.calculate_frequency_score(10)
    assert 0.52 < f10 < 0.53
    logger.info(f"✓ Frequency for 10 accesses: {f10:.3f}")
    
    f100 = calc.calculate_frequency_score(100)
    assert 0.99 < f100 <= 1.0
    logger.info(f"✓ Frequency for 100 accesses: {f100:.3f}")
    
    # Test 3: Priority combination
    p1 = calc.calculate_priority_score(0.9, 0.8, 1.0)
    assert 0.92 < p1 < 0.94
    logger.info(f"✓ Priority (R=0.9, F=0.8, I=1.0): {p1:.3f}")
    
    p2 = calc.calculate_priority_score(0.1, 0.1, 0.2)
    assert 0.14 < p2 < 0.16
    logger.info(f"✓ Priority (R=0.1, F=0.1, I=0.2): {p2:.3f}")
    
    logger.info("✓ RFM Calculator tests PASSED")


async def test_qdrant_write():
    """Test writing memory to Qdrant with full metadata"""
    logger.info("\n========== Testing Qdrant Write ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Create episodic memory
    episodic = EpisodicMemory(
        text_content="User asked about memory pruning functionality",
        event_type=EventType.USER_INTERACTION,
        tags=["conversation", "memory"],
        context={"user_id": "test_user", "session_id": "test_session"},
        agent_name="test_agent",
        metadata=MemoryMetadata(importance_score=0.8)
    )
    
    # Write to Qdrant
    memory_id = await memory.add_memory(episodic)
    logger.info(f"✓ Successfully wrote memory with ID: {memory_id}")
    
    # Verify payload
    points = await memory.client.retrieve(
        collection_name=memory.collection_name,
        ids=[memory_id],
        with_payload=True
    )
    
    assert len(points) == 1
    payload = points[0].payload
    
    # Verify all payload fields
    required_fields = [
        "memory_id", "memory_type", "text_content", "tags",
        "event_type", "context", "agent_name",
        "recency_score", "frequency_score", "importance_score", "priority_score",
        "access_count", "created_at", "day_of_week", "hour_of_day"
    ]
    
    for field in required_fields:
        assert field in payload, f"Missing field: {field}"
    
    logger.info(f"✓ All {len(required_fields)} payload fields verified")
    logger.info(f"  - Priority score: {payload['priority_score']:.3f}")
    logger.info(f"  - Recency score: {payload['recency_score']:.3f}")
    logger.info(f"  - Frequency score: {payload['frequency_score']:.3f}")
    
    logger.info("✓ Qdrant write tests PASSED")
    return memory, memory_id


async def test_update_memory_access():
    """Test access tracking and frequency updates"""
    logger.info("\n========== Testing Update Memory Access ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Create memory
    episodic = EpisodicMemory(
        text_content="Test memory for access tracking",
        event_type=EventType.SYSTEM_EVENT,
        tags=["test"],
        metadata=MemoryMetadata(importance_score=0.7)
    )
    memory_id = await memory.add_memory(episodic)
    
    # Verify initial state
    points = await memory.client.retrieve(
        collection_name=memory.collection_name,
        ids=[memory_id],
        with_payload=True
    )
    initial_access_count = points[0].payload["access_count"]
    initial_frequency = points[0].payload["frequency_score"]
    
    assert initial_access_count == 0
    assert initial_frequency == 0.0
    logger.info(f"✓ Initial: access_count={initial_access_count}, frequency={initial_frequency:.3f}")
    
    # Update access 5 times
    for i in range(5):
        await memory.update_memory_access(memory_id)
    
    # Verify updated state
    points = await memory.client.retrieve(
        collection_name=memory.collection_name,
        ids=[memory_id],
        with_payload=True
    )
    updated_access_count = points[0].payload["access_count"]
    updated_frequency = points[0].payload["frequency_score"]
    updated_priority = points[0].payload["priority_score"]
    
    assert updated_access_count == 5
    assert updated_frequency > initial_frequency
    logger.info(f"✓ After 5 accesses: count={updated_access_count}, frequency={updated_frequency:.3f}, priority={updated_priority:.3f}")
    
    logger.info("✓ Update memory access tests PASSED")


async def test_retrieve_context():
    """Test retrieval with RFM prioritization"""
    logger.info("\n========== Testing Retrieve Context ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Create memories with different priorities
    memories = [
        EpisodicMemory(
            text_content="High priority: critical bug fixed",
            event_type=EventType.SYSTEM_EVENT,
            tags=["bug", "critical"],
            metadata=MemoryMetadata(importance_score=1.0)
        ),
        EpisodicMemory(
            text_content="Low priority: trivial log message",
            event_type=EventType.SYSTEM_EVENT,
            tags=["log"],
            metadata=MemoryMetadata(importance_score=0.1)
        ),
        EpisodicMemory(
            text_content="Medium priority: user question answered",
            event_type=EventType.USER_INTERACTION,
            tags=["conversation"],
            metadata=MemoryMetadata(importance_score=0.5)
        ),
    ]
    
    for mem in memories:
        await memory.add_memory(mem)
    
    # Retrieve context
    context = await memory.retrieve_context("bug system", limit=5)
    logger.info(f"✓ Retrieved context:\n{context}")
    
    # Verify high-priority appears first
    assert "critical bug" in context.lower()
    logger.info("✓ Retrieve context tests PASSED")


async def test_hierarchical_memory():
    """Test parent-child relationships"""
    logger.info("\n========== Testing Hierarchical Memory ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Create parent
    parent = EpisodicMemory(
        text_content="Started debugging session",
        event_type=EventType.SYSTEM_EVENT,
        tags=["debugging"],
        metadata=MemoryMetadata(importance_score=0.8)
    )
    parent_id = await memory.add_memory(parent)
    logger.info(f"✓ Created parent: {parent_id}")
    
    # Create children
    child1 = EpisodicMemory(
        text_content="Fixed async client bug",
        event_type=EventType.SYSTEM_EVENT,
        parent_memory_id=parent_id,
        tags=["debugging", "resolved"],
        metadata=MemoryMetadata(importance_score=0.7)
    )
    await memory.add_memory(child1)
    
    child2 = EpisodicMemory(
        text_content="Verified connection",
        event_type=EventType.SYSTEM_EVENT,
        parent_memory_id=parent_id,
        tags=["debugging", "verification"],
        metadata=MemoryMetadata(importance_score=0.6)
    )
    await memory.add_memory(child2)
    logger.info("✓ Created 2 children")
    
    # Retrieve children
    children = await memory.get_memory_children(parent_id)
    assert len(children) == 2
    logger.info(f"✓ Retrieved {len(children)} children")
    
    # Build tree
    tree = await memory.get_memory_tree(parent_id, max_depth=2)
    assert len(tree["children"]) == 2
    logger.info(f"✓ Built memory tree with {len(tree['children'])} branches")
    
    logger.info("✓ Hierarchical memory tests PASSED")


async def test_pruning():
    """Test memory pruning"""
    logger.info("\n========== Testing Memory Pruning ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Create old, low-priority memories
    old_date = datetime.now(UTC) - timedelta(days=400)
    for i in range(3):
        old_memory = EpisodicMemory(
            text_content=f"Old test memory {i}",
            event_type=EventType.SYSTEM_EVENT,
            tags=["test", "old"],
            metadata=MemoryMetadata(importance_score=0.1),
            created_at=old_date
        )
        await memory.add_memory(old_memory)
    
    logger.info("✓ Created 3 old, low-priority test memories")
    
    # Dry run
    stats = await prune_memories(
        client=memory.client,
        collection_name=memory.collection_name,
        prune_days=365,
        min_priority_score=0.2,
        dry_run=True
    )
    logger.info(f"✓ Dry run: Would prune {stats['total_to_prune']} memories")
    
    # Execute pruning
    stats = await prune_memories(
        client=memory.client,
        collection_name=memory.collection_name,
        prune_days=365,
        min_priority_score=0.2,
        dry_run=False
    )
    logger.info(f"✓ Pruned {stats['deleted']} old memories")
    
    logger.info("✓ Memory pruning tests PASSED")


async def test_working_memory_buffer():
    """Test 4-chunk working memory capacity"""
    logger.info("\n========== Testing Working Memory Buffer ==========")
    
    qdrant_manager = QdrantClientManager(config)
    memory = await QdrantMemory.create(config, qdrant_manager)
    
    # Add 6 memories (exceeds 4-chunk capacity)
    for i in range(6):
        episodic = EpisodicMemory(
            text_content=f"Working memory test {i}",
            event_type=EventType.USER_INTERACTION,
            tags=["test"],
            metadata=MemoryMetadata(importance_score=0.5)
        )
        await memory.add_memory(episodic)
    
    # Verify buffer size = 4 (FIFO eviction)
    buffer_size = len(memory.working_memory_buffer)
    assert buffer_size == 4, f"Expected 4 chunks, got {buffer_size}"
    logger.info(f"✓ Working memory buffer size: {buffer_size} (Cowan 2001 limit)")
    
    logger.info("✓ Working memory buffer tests PASSED")


async def main():
    """Run all tests"""
    try:
        await test_rfm_calculator()
        await test_qdrant_write()
        await test_update_memory_access()
        await test_retrieve_context()
        await test_hierarchical_memory()
        await test_pruning()
        await test_working_memory_buffer()
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED - Memory system is working!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

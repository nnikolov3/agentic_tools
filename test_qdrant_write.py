import asyncio
import logging
import os
from datetime import datetime, UTC, timedelta
import math

from qdrant_client import models

from src.memory.qdrant_client_manager import QdrantClientManager
from src.memory.qdrant_memory import QdrantMemory
from src.memory.models import (
    Memory,
    EpisodicMemory,
    MemoryMetadata,
    MemoryType,
    EventType,
)
from src.configurator import find_config, get_config_dictionary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure Qdrant VM is running and accessible
QDRANT_HOST = "192.168.122.40"
QDRANT_PORT = 6333
QDRANT_GRPC_PORT = 6334

if not os.getenv("HF_TOKEN"):
    logger.warning(
        "HF_TOKEN environment variable not set. Sparse embedding model download might fail."
    )


async def test_priority_scoring():
    logger.info("Testing priority score calculations...")

    # Test case 1: High importance, fresh memory
    metadata1 = MemoryMetadata(
        recency_score=1.0,
        frequency_score=0.5,
        importance_score=0.9,  # Very recent
    )
    # Expected: (1.0 * 0.3) + (0.5 * 0.2) + (0.9 * 0.5) = 0.3 + 0.1 + 0.45 = 0.85
    logger.info(
        f"High importance fresh memory: {metadata1.priority_score:.3f} (expected ~0.85)"
    )
    assert math.isclose(metadata1.priority_score, 0.85, abs_tol=0.01)

    # Test case 2: Low importance, frequent memory
    metadata2 = MemoryMetadata(
        recency_score=0.3,
        frequency_score=0.9,
        importance_score=0.1,  # Very frequent
    )
    # Expected: (0.3 * 0.3) + (0.9 * 0.2) + (0.1 * 0.5) = 0.09 + 0.18 + 0.05 = 0.32
    logger.info(
        f"Low importance frequent memory: {metadata2.priority_score:.3f} (expected ~0.32)"
    )
    assert math.isclose(metadata2.priority_score, 0.32, abs_tol=0.01)

    # Test case 3: Balanced memory
    metadata3 = MemoryMetadata(
        recency_score=0.7, frequency_score=0.6, importance_score=0.5
    )
    # Expected: (0.7 * 0.3) + (0.6 * 0.2) + (0.5 * 0.5) = 0.21 + 0.12 + 0.25 = 0.58
    logger.info(f"Balanced memory: {metadata3.priority_score:.3f} (expected ~0.58)")
    assert math.isclose(metadata3.priority_score, 0.58, abs_tol=0.01)

    logger.info("✓ Priority scoring tests complete")


async def test_qdrant_write():
    logger.info("Loading configuration...")
    config_path = find_config()
    config = get_config_dictionary(config_path)

    # Override Qdrant host for testing
    config["memory"]["qdrant_host"] = QDRANT_HOST
    config["memory"]["qdrant_port"] = QDRANT_PORT
    config["memory"]["qdrant_grpc_port"] = QDRANT_GRPC_PORT

    logger.info("Initializing Qdrant connection...")
    qdrant_manager = QdrantClientManager(config)
    qdrant_memory = await QdrantMemory.create(config, qdrant_manager)

    test_collection_name = qdrant_memory.collection_name

    # Clear collection for a clean test run
    logger.info(f"Clearing collection '{test_collection_name}' for a clean test run...")
    client = qdrant_memory.client
    await client.delete_collection(collection_name=test_collection_name)
    # Re-create collection after deletion
    await qdrant_manager.ensure_collection_exists(
        collection_name=test_collection_name,
        embedding_size=qdrant_memory.embedding_size,
    )

    logger.info("Creating test memory...")
    test_memory = EpisodicMemory(
        text_content="This is a test memory about a user interaction.",
        tags=["test", "user_interaction"],
        agent_name="TestAgent",
        event_type=EventType.USER_INTERACTION,
        context={"user_id": "123", "session_id": "abc"},
        metadata=MemoryMetadata(importance_score=0.7),
    )

    # Simulate an older memory for recency score testing
    old_created_at = datetime.now(UTC) - timedelta(days=10)
    test_memory.created_at = old_created_at
    test_memory.updated_at = old_created_at

    logger.info("Writing memory to Qdrant...")
    memory_id = await qdrant_memory.add_memory(test_memory)
    logger.info(f"✓ Successfully wrote memory with ID: {memory_id}")

    logger.info("Retrieving memory from Qdrant...")
    # Retrieve directly to check payload
    retrieved_points = await client.retrieve(
        collection_name=test_collection_name,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )

    assert len(retrieved_points) == 1
    retrieved_payload = retrieved_points[0].payload
    logger.info("✓ Successfully retrieved memory")

    logger.info("Verifying payload fields...")
    assert retrieved_payload["memory_id"] == str(test_memory.id)
    assert retrieved_payload["memory_type"] == test_memory.memory_type.value
    assert retrieved_payload["text_content"] == test_memory.text_content
    assert math.isclose(
        retrieved_payload["created_at"], test_memory.created_at.timestamp()
    )
    assert math.isclose(
        retrieved_payload["updated_at"], test_memory.updated_at.timestamp()
    )
    assert retrieved_payload["last_accessed_at"] is None  # Should be None initially
    assert math.isclose(
        retrieved_payload["recency_score"],
        test_memory.metadata.recency_score,
        abs_tol=0.01,
    )
    assert math.isclose(
        retrieved_payload["frequency_score"],
        test_memory.metadata.frequency_score,
        abs_tol=0.01,
    )
    assert math.isclose(
        retrieved_payload["importance_score"],
        test_memory.metadata.importance_score,
        abs_tol=0.01,
    )
    assert math.isclose(
        retrieved_payload["priority_score"],
        test_memory.metadata.priority_score,
        abs_tol=0.01,
    )
    assert retrieved_payload["access_count"] == test_memory.metadata.access_count
    assert retrieved_payload["tags"] == test_memory.tags
    assert retrieved_payload["agent_name"] == test_memory.agent_name
    assert retrieved_payload["event_type"] == test_memory.event_type.value
    assert retrieved_payload["context"] == test_memory.context
    assert retrieved_payload["parent_memory_id"] is None
    logger.info("✓ All payload fields verified.")

    logger.info("Testing update_memory_access...")
    await qdrant_memory.update_memory_access(memory_id)
    updated_points = await client.retrieve(
        collection_name=test_collection_name,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )
    updated_payload = updated_points[0].payload
    assert updated_payload["access_count"] == 1
    assert updated_payload["last_accessed_at"] is not None
    assert updated_payload["frequency_score"] > 0
    logger.info("✓ update_memory_access verified.")

    logger.info("Testing retrieve_context...")
    context = await qdrant_memory.retrieve_context("test memory")
    assert test_memory.text_content in context
    logger.info("✓ retrieve_context verified.")

    # Check collection stats
    collection_info = await client.get_collection(collection_name=test_collection_name)
    logger.info(f"Collection '{test_collection_name}' stats:")
    logger.info(f"- Points count: {collection_info.points_count}")
    logger.info(f"- Vectors config: {collection_info.config.params.vectors}")
    logger.info("✓ Collection stats checked.")

    logger.info(f"Cleaning up collection '{test_collection_name}'...")
    await client.delete_collection(collection_name=test_collection_name)
    logger.info(f"✓ Collection '{test_collection_name}' deleted.")


async def main():
    await test_priority_scoring()
    await test_qdrant_write()
    logger.info("\n============================================================")
    logger.info("✓ ALL TESTS PASSED - Qdrant write system is working!")
    logger.info("============================================================")


if __name__ == "__main__":
    asyncio.run(main())

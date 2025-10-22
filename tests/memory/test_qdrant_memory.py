# tests/memory/test_qdrant_memory.py

"""
Purpose:
This module contains unit tests for the QdrantMemory class.

Description:
The tests cover the initialization, memory addition, and context retrieval functionalities
of the QdrantMemory class. It uses pytest for the testing framework and unittest.mock
for mocking asynchronous interactions with the Qdrant client, ensuring that the memory
component behaves as expected without requiring a live Qdrant instance.
"""

import math
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

# Explicitly import the class to be tested.
# Acyclic Dependencies: This test module depends on the source, not the other way around.
from src.memory.qdrant_memory import QdrantMemory


# Simplicity is Non-Negotiable: A fixture provides a clean, reusable config.
@pytest.fixture
def memory_config():
    """Provides a default configuration dictionary for QdrantMemory."""
    return {
        "qdrant_url": "http://localhost:6333",
        "collection_name": "test_agent_memory",
        "embedding_model": "test-embedding-model",
        "embedding_size": 128,
        "short_term_weight": 0.7,
        "long_term_weight": 0.3,
        "total_memories_to_retrieve": 10,
    }


# Test case 1: Initialization
@pytest.mark.asyncio
@patch("src.memory.qdrant_memory.AsyncQdrantClient")
async def test_create_initialization(mock_qdrant_client, memory_config):
    """
    Tests that the QdrantMemory.create factory method correctly initializes
    the class and calls _ensure_collection_exists.
    """
    # Arrange: Mock the async method that ensures the collection exists.
    # This isolates the test to the factory method's logic.
    with patch.object(
        QdrantMemory, "_ensure_collection_exists", new_callable=AsyncMock
    ) as mock_ensure_collection:
        # Act: Call the asynchronous factory method.
        memory_instance = await QdrantMemory.create(memory_config)

        # Assert:
        # 1. The instance is of the correct type.
        assert isinstance(memory_instance, QdrantMemory)

        # 2. The internal async setup method was called exactly once.
        mock_ensure_collection.assert_awaited_once()

        # 3. Configuration values were correctly assigned to instance attributes.
        # Explicit Over Implicit: Verify each configuration value is set as expected.
        assert memory_instance.collection_name == "test_agent_memory"
        assert memory_instance.total_memories_to_retrieve == 10
        assert memory_instance.short_term_weight == 0.7


# Test case 2: `add_memory`
@pytest.mark.asyncio
@patch("src.memory.qdrant_memory.AsyncQdrantClient")
async def test_add_memory_calls_qdrant_add(mock_qdrant_client, memory_config):
    """
    Tests that add_memory calls the qdrant_client.add method with the correct parameters.
    """
    # Arrange
    # We can instantiate directly and mock the client, bypassing the `create` factory.
    memory_instance = QdrantMemory(memory_config)
    memory_instance.client = AsyncMock()
    test_content = "This is a new memory."

    # Act
    await memory_instance.add_memory(test_content)

    # Assert
    # Single Responsibility: This test focuses solely on the `add` method call.
    memory_instance.client.add.assert_awaited_once_with(
        collection_name=memory_config["collection_name"],
        documents=[test_content],
        # The payload includes the content and a dynamically generated timestamp.
        # `ANY` is used to match the timestamp without needing its exact value.
        payload=[{"text_content": test_content, "timestamp": ANY}],
        model=memory_config["embedding_model"],
        wait=True,
    )


# Test case 3: `retrieve_context` (No Memories)
@pytest.mark.asyncio
@patch("src.memory.qdrant_memory.AsyncQdrantClient")
async def test_retrieve_context_with_no_memories(mock_qdrant_client, memory_config):
    """
    Tests that retrieve_context returns an empty string when Qdrant finds no memories.
    """
    # Arrange
    memory_instance = QdrantMemory(memory_config)
    mock_response = MagicMock()
    mock_response.points = []
    # Configure the mock client to return a response with an empty list of points.
    memory_instance.client.query_points = AsyncMock(return_value=mock_response)

    # Act
    context = await memory_instance.retrieve_context("some query")

    # Assert
    # Error Handling Excellence: The system should gracefully handle no results.
    assert context == ""


# Test case 4: `retrieve_context` (70/30 Split)
@pytest.mark.asyncio
@patch("src.memory.qdrant_memory.AsyncQdrantClient")
async def test_retrieve_context_70_30_split(mock_qdrant_client, memory_config):
    """
    Tests the core logic of retrieve_context, ensuring it queries for short-term
    and long-term memories with the correct limits and filters, and formats the
    output correctly.
    """
    # --- Arrange ---
    # Self-Documenting Code: Clear variable names for mock data.
    short_term_memory_text = "This happened today."
    long_term_memory_text = "This happened yesterday."

    # Create mock Qdrant points to be returned by the client.
    short_term_point = models.ScoredPoint(
        id="st_id_1",
        version=1,
        score=0.9,
        payload={"text_content": short_term_memory_text},
    )
    long_term_point = models.ScoredPoint(
        id="lt_id_1",
        version=1,
        score=0.8,
        payload={"text_content": long_term_memory_text},
    )

    # Create mock responses for the two separate queries (short-term and long-term).
    short_term_response = MagicMock()
    short_term_response.points = [short_term_point]
    long_term_response = MagicMock()
    long_term_response.points = [long_term_point]

    memory_instance = QdrantMemory(memory_config)
    # Use `side_effect` to return different results for consecutive calls.
    memory_instance.client.query_points = AsyncMock(
        side_effect=[short_term_response, long_term_response]
    )

    # --- Act ---
    query = "What happened?"
    context = await memory_instance.retrieve_context(query)

    # --- Assert ---
    # 1. Verify that `query_points` was called twice.
    assert memory_instance.client.query_points.await_count == 2
    calls = memory_instance.client.query_points.await_args_list

    # 2. Analyze the first call (short-term memories).
    short_term_call_args, short_term_call_kwargs = calls[0]
    num_short_term = math.ceil(
        memory_config["total_memories_to_retrieve"] * memory_config["short_term_weight"]
    )
    assert short_term_call_kwargs["collection_name"] == memory_config["collection_name"]
    assert short_term_call_kwargs["query"] == query
    assert short_term_call_kwargs["limit"] == num_short_term  # 7
    # Check that the filter is for recent timestamps (gte).
    short_term_filter = short_term_call_kwargs["query_filter"]
    assert isinstance(short_term_filter, models.Filter)
    assert short_term_filter.must[0].key == "timestamp"
    assert short_term_filter.must[0].range.gte is not None
    assert short_term_filter.must[0].range.lt is None

    # 3. Analyze the second call (long-term memories).
    long_term_call_args, long_term_call_kwargs = calls[1]
    num_long_term = math.floor(
        memory_config["total_memories_to_retrieve"] * memory_config["long_term_weight"]
    )
    assert long_term_call_kwargs["collection_name"] == memory_config["collection_name"]
    assert long_term_call_kwargs["query"] == query
    assert long_term_call_kwargs["limit"] == num_long_term  # 3
    # Check that the filter is for older timestamps (lt).
    long_term_filter = long_term_call_kwargs["query_filter"]
    assert isinstance(long_term_filter, models.Filter)
    assert long_term_filter.must[0].key == "timestamp"
    assert long_term_filter.must[0].range.lt is not None
    assert long_term_filter.must[0].range.gte is None

    # 4. Verify the final formatted context string.
    # Consistency Reduces Cognitive Load: The format should be predictable.
    expected_context = (
        "--- Relevant Memories Retrieved ---"
        f"{short_term_memory_text}"
        f"{long_term_memory_text}"
    )
    assert context == expected_context

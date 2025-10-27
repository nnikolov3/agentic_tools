# src/memory/test_qdrant_memory.py

"""Unit tests for QdrantClientManager and QdrantMemory.

This module contains comprehensive unit tests for the Qdrant memory management
components. It verifies the functionality of both the QdrantClientManager,
responsible for client and collection setup, and QdrantMemory, which handles
the logic for storing and retrieving context for agents.

The tests use the pytest framework and pytest-asyncio for handling asynchronous
operations, with extensive mocking to isolate components and ensure predictable
behavior.
"""

# Standard Library Imports
import asyncio
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

# Third-Party Library Imports
import pytest
from qdrant_client import models

# Local Application/Module Imports
from src.memory.qdrant_client_manager import QdrantClientManager
from src.memory.qdrant_memory import QdrantMemory


@pytest.fixture
def qdrant_config() -> dict:
    """Provides a standard configuration for QdrantClientManager."""
    return {
        "qdrant_url": "http://localhost:6333",
        "timeout": 30.0,
        "hnsw_config": {"m": 16, "ef_construct": 100},
        "optimizers_config": {},
        "wal_config": {"wal_capacity_mb": 32},
        "quantization_config": {
            "scalar_type": "int8",
            "quantile": 0.99,
            "always_ram": True,
        },
    }


@pytest.fixture
def memory_config() -> dict:
    """Provides a standard configuration for QdrantMemory."""
    return {
        "collection_name": "test_agent_memory",
        "knowledge_bank": "test_knowledge_bank",
        "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
        "sparse_embedding_model": "naver/splade-v3",
        "device": "cpu",
        "reranker": {"enabled": True},
    }


@pytest.fixture
def mock_qdrant_manager() -> tuple[MagicMock, AsyncMock]:
    """Provides a mocked QdrantClientManager and its associated client."""
    qdrant_manager_mock = MagicMock(spec=QdrantClientManager)
    qdrant_client_mock = AsyncMock()
    qdrant_manager_mock.get_client.return_value = qdrant_client_mock
    return qdrant_manager_mock, qdrant_client_mock


class TestQdrantClientManager:
    """Tests for the QdrantClientManager class."""

    def test_qdrant_manager_initialization(self, qdrant_config: dict) -> None:
        """
        Tests that the QdrantClientManager initializes correctly with a given config.
        """
        # Act
        manager = QdrantClientManager(qdrant_config)

        # Assert
        assert manager.qdrant_url == qdrant_config["qdrant_url"]
        assert manager.timeout == qdrant_config["timeout"]
        assert manager.client is not None

    @patch("os.cpu_count", return_value=8)
    def test_qdrant_manager_init_dynamic_threads(
        self, mock_cpu_count: MagicMock, qdrant_config: dict
    ) -> None:
        """
        Tests that thread counts are dynamically set to os.cpu_count() when
        configured with -1.
        """
        # Arrange
        config = qdrant_config.copy()
        config["hnsw_config"] = {"max_indexing_threads": -1}
        config["optimizers_config"] = {"max_optimization_threads": -1}

        # Act
        manager = QdrantClientManager(config)

        # Assert
        assert manager.hnsw_config["max_indexing_threads"] == 8
        assert manager.optimizers_config["max_optimization_threads"] == 8
        assert mock_cpu_count.call_count == 2

    @patch("os.cpu_count", return_value=None)
    def test_qdrant_manager_init_dynamic_threads_no_cpu_count(
        self, mock_cpu_count: MagicMock, qdrant_config: dict
    ) -> None:
        """
        Tests that thread counts default to 0 if os.cpu_count() returns None.
        """
        # Arrange
        config = qdrant_config.copy()
        config["hnsw_config"] = {"max_indexing_threads": -1}

        # Act
        manager = QdrantClientManager(config)

        # Assert
        assert manager.hnsw_config["max_indexing_threads"] == 0
        mock_cpu_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_when_it_does(
        self, qdrant_config: dict
    ) -> None:
        """
        Tests that create_collection is not called if the collection already exists.
        """
        # Arrange
        manager = QdrantClientManager(qdrant_config)
        client_mock = AsyncMock()
        manager.client = client_mock
        client_mock.collection_exists.return_value = True
        collection_name = "existing_collection"

        # Act
        await manager.ensure_collection_exists(collection_name, 128)

        # Assert
        client_mock.collection_exists.assert_awaited_once_with(
            collection_name=collection_name
        )
        client_mock.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_creates_new_collection(
        self, qdrant_config: dict
    ) -> None:
        """
        Tests that a new collection and its payload indexes are created when
        the collection does not exist.
        """
        # Arrange
        manager = QdrantClientManager(qdrant_config)
        client_mock = AsyncMock()
        manager.client = client_mock
        client_mock.collection_exists.return_value = False
        collection_name = "new_collection"
        embedding_size = 128
        payload_indexes: List[Tuple[str, Union[str, Dict[str, Any]]]] = [
            ("metadata.source", "keyword")
        ]

        # Act
        await manager.ensure_collection_exists(
            collection_name, embedding_size, payload_indexes
        )

        # Assert
        client_mock.collection_exists.assert_awaited_once_with(
            collection_name=collection_name
        )
        client_mock.create_collection.assert_awaited_once()
        client_mock.create_payload_index.assert_awaited_once_with(
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema="keyword",
        )

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_raises_runtime_error_on_failure(
        self, qdrant_config: dict
    ) -> None:
        """
        Tests that a RuntimeError is raised if collection creation fails.
        """
        # Arrange
        manager = QdrantClientManager(qdrant_config)
        client_mock = AsyncMock()
        manager.client = client_mock
        client_mock.collection_exists.return_value = False
        client_mock.create_collection.side_effect = Exception("Qdrant connection error")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Collection creation failed"):
            await manager.ensure_collection_exists("failing_collection", 128)


@patch("src.memory.qdrant_memory.SparseEncoder")
@patch("src.memory.qdrant_memory.TextEmbedding")
class TestQdrantMemory:
    """Tests for the QdrantMemory class."""

    @pytest.mark.asyncio
    async def test_resolve_vector_names_with_named_vectors(
        self,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Tests that vector names are correctly resolved from a collection
        that uses named dense and sparse vectors.
        """
        # Arrange
        mock_text_embedding.return_value.embedding_size = 1024
        qdrant_manager, qdrant_client = mock_qdrant_manager
        memory = QdrantMemory(memory_config, qdrant_manager)

        collection_info = MagicMock()
        collection_info.config.params.vectors = {
            "dense-vector": models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            )
        }
        collection_info.config.params.sparse_vectors = {
            "sparse-vector": models.SparseVectorParams()
        }
        qdrant_client.get_collection.return_value = collection_info

        # Act
        dense_name, sparse_name = await memory._resolve_vector_names("test_collection")

        # Assert
        assert dense_name == "dense-vector"
        assert sparse_name == "sparse-vector"

    @pytest.mark.asyncio
    async def test_resolve_vector_names_with_unnamed_dense_vector(
        self,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Tests resolving vector names from a collection with a single, unnamed
        dense vector and no sparse vectors.
        """
        # Arrange
        mock_text_embedding.return_value.embedding_size = 1024
        qdrant_manager, qdrant_client = mock_qdrant_manager
        memory = QdrantMemory(memory_config, qdrant_manager)

        collection_info = MagicMock()
        collection_info.config.params.vectors = models.VectorParams(
            size=1024, distance=models.Distance.COSINE
        )
        collection_info.config.params.sparse_vectors = None
        qdrant_client.get_collection.return_value = collection_info

        # Act
        dense_name, sparse_name = await memory._resolve_vector_names("test_collection")

        # Assert
        assert dense_name == ""
        assert sparse_name is None

    @pytest.mark.asyncio
    async def test_resolve_vector_names_with_preferred_names(
        self,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Tests that vector names from the config are preferred if they exist
        in the collection.
        """
        # Arrange
        config = memory_config.copy()
        config["dense_vector_name"] = "preferred_dense"
        config["sparse_vector_name"] = "preferred_sparse"
        mock_text_embedding.return_value.embedding_size = 1024
        qdrant_manager, qdrant_client = mock_qdrant_manager
        memory = QdrantMemory(config, qdrant_manager)

        collection_info = MagicMock()
        collection_info.config.params.vectors = {
            "other_dense": models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
            "preferred_dense": models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        }
        collection_info.config.params.sparse_vectors = {
            "other_sparse": models.SparseVectorParams(),
            "preferred_sparse": models.SparseVectorParams(),
        }
        qdrant_client.get_collection.return_value = collection_info

        # Act
        dense_name, sparse_name = await memory._resolve_vector_names("test_collection")

        # Assert
        assert dense_name == "preferred_dense"
        assert sparse_name == "preferred_sparse"

    def test_make_prefetch_with_sparse_name(
        self,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Tests that a Prefetch object with RRF fusion is created when a
        sparse vector name is provided.
        """
        # Arrange
        mock_text_embedding.return_value.embedding_size = 1024
        qdrant_manager, _ = mock_qdrant_manager
        memory = QdrantMemory(memory_config, qdrant_manager)
        dense_query_vector = [0.1] * 1024
        sparse_query_vector = models.SparseVector(indices=[1, 2], values=[0.3, 0.4])

        # Act
        prefetch = memory._make_prefetch(
            dense_query_vector, sparse_query_vector, 10, "dense_name", "sparse_name"
        )

        # Assert
        assert prefetch is not None
        assert isinstance(prefetch, models.Prefetch)
        assert isinstance(prefetch.query, models.FusionQuery)
        assert prefetch.query.fusion == models.Fusion.RRF

    def test_make_prefetch_without_sparse_name(
        self,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Tests that no Prefetch object is created when the sparse vector name is None.
        """
        # Arrange
        mock_text_embedding.return_value.embedding_size = 1024
        qdrant_manager, _ = mock_qdrant_manager
        memory = QdrantMemory(memory_config, qdrant_manager)
        dense_query_vector = [0.1] * 1024
        sparse_query_vector = models.SparseVector(indices=[1, 2], values=[0.3, 0.4])

        # Act
        prefetch = memory._make_prefetch(
            dense_query_vector, sparse_query_vector, 10, "dense_name", None
        )

        # Assert
        assert prefetch is None

    @pytest.mark.asyncio
    @patch("src.memory.qdrant_memory.TextCrossEncoder")
    async def test_retrieve_context_end_to_end(
        self,
        mock_reranker: MagicMock,
        mock_text_embedding: MagicMock,
        mock_sparse_encoder: MagicMock,
        memory_config: dict,
        mock_qdrant_manager: tuple[MagicMock, AsyncMock],
    ) -> None:
        """
        Performs an end-to-end test of the retrieve_context method, including
        querying, result processing, and reranking.
        """
        # Arrange
        mock_text_embedding.return_value.embedding_size = 1024
        mock_embedding_result = MagicMock()
        mock_embedding_result.tolist.return_value = [0.1] * 1024
        mock_text_embedding.return_value.embed.return_value = iter(
            [mock_embedding_result]
        )
        mock_sparse_encoder.return_value.encode_query.return_value = MagicMock()
        mock_reranker.return_value.rerank.return_value = [
            ("memory 1", 0.9),
            ("memory 2", 0.8),
            ("memory 3", 0.7),
        ]

        qdrant_manager, qdrant_client = mock_qdrant_manager
        memory = QdrantMemory(memory_config, qdrant_manager)
        memory.agent_dense_name = "dense_vector"
        memory.agent_sparse_name = "sparse_vector"
        memory.kb_dense_name = ""
        memory.kb_sparse_name = None
        memory.reranker = mock_reranker.return_value

        mock_points = [
            models.ScoredPoint(
                id="1", version=1, score=0.9, payload={"text_content": "memory 1"}
            ),
            models.ScoredPoint(
                id="2", version=1, score=0.8, payload={"text_content": "memory 2"}
            ),
            models.ScoredPoint(
                id="3", version=1, score=0.7, payload={"text_content": "memory 3"}
            ),
        ]

        # Create 9 mocked responses for the 9 query tasks
        mock_response_1 = MagicMock()
        mock_response_1.points = mock_points
        mock_response_empty = MagicMock()
        mock_response_empty.points = []

        # The side_effect should be a list of the results, not Futures
        results = [mock_response_1] + [mock_response_empty] * 8

        qdrant_client.query_points.side_effect = results

        # Act
        context = await memory.retrieve_context("test query")

        # Assert
        assert QdrantMemory.CONTEXT_HEADER in context
        assert "memory 1" in context
        assert "memory 2" in context
        assert "memory 3" in context
        mock_reranker.return_value.rerank.assert_called_once()

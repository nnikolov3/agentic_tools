"""
Purpose:
This module provides a self-contained memory management system using Qdrant as the vector database.
It handles storing memories, embedding them via FastEmbed, and retrieving contextually relevant short-term and long-term memories based on a query.
The design follows foundational principles: simplicity through focused responsibilities, explicit configuration handling, and robust error management to ensure reliable agent context retrieval without cascading failures.
Requires 'qdrant-client[fastembed]' for automatic text-to-vector embedding with specified models.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models
import uuid

# Self-Documenting Code: A dedicated logger for this module improves traceability and debugging in distributed systems.
logger: logging.Logger = logging.getLogger(__name__)


class QdrantMemory:
    """
    Manages agent memories using a Qdrant vector database with FastEmbed integration.

    This class encapsulates all interactions with Qdrant, including collection management,
    memory storage (embedding via Document and upload), and context retrieval. It implements a
    retrieval strategy that combines recent (short-term) and older (long-term) memories
    to provide a comprehensive context for the agent, prioritizing recency for short-term
    relevance while incorporating historical context for depth.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the QdrantMemory instance with explicit configuration values.

        This constructor is intended to be called via the `create` classmethod,
        which handles the asynchronous setup of the Qdrant collection. Configuration
        parameters are stored as instance attributes for explicit access, avoiding
        repeated dictionary lookups and ensuring immutability where possible.

        Args:
            config: A dictionary containing the memory configuration, typically loaded from a TOML file.
        """
        # Explicit Over Implicit: Store config values directly for clarity and performance.
        self.qdrant_url: str = config.get("qdrant_url", "http://localhost:6333")
        self.collection_name: str = config.get("collection_name", "agent_memory")
        self.embedding_model: str = config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_size: Optional[int] = config.get(
            "embedding_size"
        )  # Will auto-detect if None.
        self.short_term_weight: float = config.get("short_term_weight", 0.7)
        self.long_term_weight: float = config.get("long_term_weight", 0.3)
        self.total_memories_to_retrieve: int = config.get(
            "total_memories_to_retrieve", 10
        )

        # Efficient Memory Management: Initialize the client once to minimize resource overhead.
        self.client: AsyncQdrantClient = AsyncQdrantClient(url=self.qdrant_url)

    @classmethod
    async def create(cls, config: Dict[str, Any]) -> "QdrantMemory":
        """
        Asynchronously creates and initializes a QdrantMemory instance.

        This factory method ensures the Qdrant collection is created or verified before
        returning the instance, preventing runtime failures due to missing infrastructure.
        It follows the factory pattern to encapsulate async initialization logic.

        Args:
            config: The memory configuration dictionary.

        Returns:
            A fully initialized QdrantMemory instance ready for use.

        Raises:
            RuntimeError: If collection setup fails, chaining the original exception for context.
        """
        instance: QdrantMemory = cls(config)
        await instance._ensure_collection_exists()
        return instance

    async def _ensure_collection_exists(self) -> None:
        """
        Ensures the configured Qdrant collection exists, creating it if necessary.

        This method uses COSINE distance for vector similarity, as it is optimal for
        normalized embeddings in semantic search tasks like memory retrieval.
        Embedding size is auto-detected via the model if not explicitly configured.

        Raises:
            RuntimeError: If collection check or creation fails, providing detailed context.
        """
        try:
            collection_exists: bool = await self.client.collection_exists(
                collection_name=self.collection_name
            )
            if not collection_exists:
                # Auto-detect embedding size if not provided.
                if self.embedding_size is None:
                    self.embedding_size = self.client.get_embedding_size(
                        self.embedding_model
                    )

                logger.info(
                    f"Collection '{self.collection_name}' not found. Creating it now with size {self.embedding_size}."
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_size, distance=models.Distance.COSINE
                    ),
                )
                logger.info(
                    f"Successfully created collection '{self.collection_name}'."
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as setup_error:
            # Error Handling Excellence: Fail fast with explicit context to prevent partial setups.
            error_message: str = (
                f"Failed to ensure Qdrant collection '{self.collection_name}' exists: {setup_error}"
            )
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from setup_error

    async def add_memory(self, text_content: str) -> None:
        """
        Embeds and stores a new memory in the Qdrant collection using FastEmbed.

        The text is wrapped in a Document for automatic embedding, and a Unix timestamp payload
        enables time-based filtering for short-term (recent) vs. long-term (historical)
        retrieval. This supports the agent's need for temporally aware context.

        Args:
            text_content: The string content representing the memory to store.

        Raises:
            RuntimeError: If embedding or storage fails, chaining the original exception.
        """
        try:
            # Prepare payload with zone-aware timestamp.
            memory_payload: Dict[str, str | float] = {
                "text_content": text_content,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }

            # Use Document for FastEmbed integration to specify model explicitly.
            document: models.Document = models.Document(
                text=text_content, model=self.embedding_model
            )

            # Upload single point; generate UUID for ID.
            point_id: str = str(uuid.uuid4())
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=document,  # FastEmbed embeds on upload.
                        payload=memory_payload,
                    )
                ],
            )
            logger.info(
                f"Successfully added memory to collection '{self.collection_name}' with ID '{point_id}'."
            )
        except Exception as storage_error:
            # Error Handling Excellence: Raise explicitly to signal failure, allowing upstream recovery.
            error_message: str = (
                f"Failed to add memory to Qdrant collection '{self.collection_name}': {storage_error}"
            )
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from storage_error

    async def retrieve_context(self, query_text: str) -> str:
        """
        Retrieves and combines relevant short-term and long-term memories for the given query.

        Short-term memories (last 24 hours) are weighted higher for recency bias, while
        long-term memories provide historical depth. Uses FastEmbed for query embedding.
        Results are deduplicated by ID and formatted for direct use in agent prompts.

        Args:
            query_text: The query string to embed and match against stored memories.

        Returns:
            A formatted string of unique, relevant memories, or an empty string if none found or retrieval fails.
        """
        # Calculate retrieval limits based on weights to balance recency and history.
        num_short_term_memories: int = math.ceil(
            self.total_memories_to_retrieve * self.short_term_weight
        )
        num_long_term_memories: int = math.floor(
            self.total_memories_to_retrieve * self.long_term_weight
        )

        if num_short_term_memories == 0 and num_long_term_memories == 0:
            return ""

        current_time: datetime = datetime.now(timezone.utc)
        short_term_threshold: datetime = current_time - timedelta(hours=24)
        short_term_timestamp: float = short_term_threshold.timestamp()

        # Explicit filters using timestamp ranges for precise temporal segmentation.
        short_term_filter: models.Filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=short_term_timestamp),
                )
            ]
        )

        long_term_filter: models.Filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(lt=short_term_timestamp),
                )
            ]
        )

        # Prepare query Document for FastEmbed embedding.
        query_document: models.Document = models.Document(
            text=query_text, model=self.embedding_model
        )

        # Prepare concurrent search coroutines only for requested memory types.
        # Explicit Over Implicit: Type as list of coroutines for clarity in async handling.
        search_coroutines: List[Coroutine[Any, Any, Any]] = []
        if num_short_term_memories > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_document,
                    query_filter=short_term_filter,
                    limit=num_short_term_memories,
                    with_payload=True,
                )
            )
        if num_long_term_memories > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_document,
                    query_filter=long_term_filter,
                    limit=num_long_term_memories,
                    with_payload=True,
                )
            )

        try:
            # Run searches concurrently for efficiency in high-throughput scenarios.
            search_results: List[Any] = await asyncio.gather(
                *search_coroutines, return_exceptions=True
            )
        except Exception as retrieval_error:
            # Graceful degradation: Log and return empty context to avoid blocking agent operation.
            error_message: str = (
                f"Failed to retrieve memories from Qdrant: {retrieval_error}"
            )
            logger.error(error_message, exc_info=True)
            return ""

        # Aggregate points from all successful searches, filtering out exceptions.
        all_points: List[Any] = []
        for result in search_results:
            if isinstance(result, Exception):
                logger.warning(f"Individual search failed: {result}")
                continue
            if hasattr(result, "points") and result.points:
                all_points.extend(result.points)

        if not all_points:
            return ""

        # Deduplicate by point ID to ensure uniqueness across short- and long-term results.
        seen_point_ids: set = set()
        unique_memory_texts: List[str] = []
        for point in all_points:
            if point.id not in seen_point_ids:
                seen_point_ids.add(point.id)
                payload: Optional[Dict[str, Any]] = getattr(point, "payload", None)
                if payload and "text_content" in payload:
                    unique_memory_texts.append(payload["text_content"])

        if not unique_memory_texts:
            return ""

        # Format as a simple, prompt-ready string with a clear header for context separation.
        context_header: str = "--- Relevant Memories Retrieved ---\n"
        context_body: str = "\n".join(unique_memory_texts)
        return f"{context_header}{context_body}"

# src/memory/qdrant_memory.py
"""
Purpose:
This module provides a self-contained memory management system using Qdrant as the vector database.
It handles storing memories, embedding them via FastEmbed, and retrieving contextually relevant
memories based on a weighted, four-part strategy (today, monthly, long-term, and knowledge bank).
The design follows foundational principles: simplicity through focused responsibilities, explicit
configuration handling, and robust error management to ensure reliable agent context retrieval.
"""

import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models

# Self-Documenting Code: A dedicated logger for this module improves traceability.
logger: logging.Logger = logging.getLogger(__name__)


class QdrantMemory:
    """
    Manages agent memories using Qdrant, supporting a weighted, multi-source retrieval strategy.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the QdrantMemory instance with explicit configuration values.
        Reads from [agentic-tools.memory] section of the configuration.
        """
        # Qdrant connection settings
        self.qdrant_url: str = config.get("qdrant_url", "http://localhost:6333")
        self.collection_name: str = config.get("collection_name", "agent_memory")
        self.knowledge_bank_collection_name: str = config.get(
            "knowledge_bank", "knowledge_bank"
        )

        # Embedding settings
        self.embedding_model: str = config.get(
            "embedding_model", "mixedbread-ai/mxbai-embed-large-v1"
        )
        self.embedding_size: Optional[int] = config.get("embedding_size")

        # Retrieval settings
        self.total_memories_to_retrieve: int = config.get(
            "total_memories_to_retrieve", 10
        )

        # Weighted retrieval parameters
        self.today_retrieval_weight: float = config.get("today_retrieval_weight", 0.4)
        self.monthly_retrieval_weight: float = config.get(
            "monthly_retrieval_weight", 0.2
        )
        self.long_term_retrieval_weight: float = config.get(
            "long_term_retrieval_weight", 0.2
        )
        self.knowledge_bank_retrieval_weight: float = config.get(
            "knowledge_bank_retrieval_weight", 0.2
        )

        # Validate that weights sum to approximately 1.0
        total_weight = (
                self.today_retrieval_weight
                + self.monthly_retrieval_weight
                + self.long_term_retrieval_weight
                + self.knowledge_bank_retrieval_weight
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            logger.warning(
                f"Memory retrieval weights sum to {total_weight}, not 1.0. This may lead to unexpected retrieval counts."
            )

        # Efficient Memory Management: Initialize the client once.
        self.client: AsyncQdrantClient = AsyncQdrantClient(url=self.qdrant_url)

    @classmethod
    async def create(cls, config: Dict[str, Any]) -> "QdrantMemory":
        """
        Asynchronously creates and initializes a QdrantMemory instance.
        Ensures both agent_memory and knowledge_bank collections exist.
        """
        instance: QdrantMemory = cls(config)
        # Ensure both required collections exist before proceeding.
        await instance._ensure_collection_exists(instance.collection_name)
        await instance._ensure_collection_exists(instance.knowledge_bank_collection_name)
        return instance

    async def _ensure_collection_exists(self, collection_name: str) -> None:
        """
        Ensures the specified Qdrant collection exists, creating it if necessary.
        """
        try:
            collection_exists: bool = await self.client.collection_exists(
                collection_name=collection_name
            )
            if not collection_exists:
                # Determine embedding size dynamically if not specified
                if self.embedding_size is None:
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer(self.embedding_model)
                    embedding_size = embedder.get_sentence_embedding_dimension()
                else:
                    embedding_size = self.embedding_size

                logger.info(
                    f"Collection '{collection_name}' not found. Creating it with size {embedding_size}."
                )
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size, distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Successfully created collection '{collection_name}'.")
        except Exception as setup_error:
            error_message = f"Failed to ensure Qdrant collection '{collection_name}' exists: {setup_error}"
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from setup_error

    async def add_memory(self, text_content: str) -> None:
        """
        Embeds and stores a new memory in the `agent_memory` collection.
        """
        try:
            memory_payload: Dict[str, Any] = {
                "text_content": text_content,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }
            document = models.Document(text=text_content, model=self.embedding_model)
            point_id = str(uuid.uuid4())
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id, vector=document, payload=memory_payload
                    )
                ],
            )
            logger.info(
                f"Successfully added memory to collection '{self.collection_name}' with ID '{point_id}'."
            )
        except Exception as storage_error:
            error_message = f"Failed to add memory to Qdrant: {storage_error}"
            logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from storage_error

    async def retrieve_context(self, query_text: str) -> str:
        """
        Retrieves context using a four-part weighted strategy:
        1. Today's memories (time-filtered from agent_memory)
        2. Monthly memories (time-filtered from agent_memory)
        3. Long-term memories (time-filtered from agent_memory)
        4. Knowledge bank entries (from knowledge_bank collection)

        Returns a formatted string with all relevant retrieved content.
        """
        # Step 1: Calculate Retrieval Limits based on weights
        num_today = math.ceil(
            self.total_memories_to_retrieve * self.today_retrieval_weight
        )
        num_monthly = math.ceil(
            self.total_memories_to_retrieve * self.monthly_retrieval_weight
        )
        num_long_term = math.ceil(
            self.total_memories_to_retrieve * self.long_term_retrieval_weight
        )
        num_kb = math.ceil(
            self.total_memories_to_retrieve * self.knowledge_bank_retrieval_weight
        )

        # Step 2: Define Time-Based Filters for agent_memory
        current_time = datetime.now(timezone.utc)
        today_ts = (current_time - timedelta(days=1)).timestamp()
        month_ts = (current_time - timedelta(days=30)).timestamp()

        today_filter = models.Filter(
            must=[models.FieldCondition(key="timestamp", range=models.Range(gte=today_ts))]
        )
        monthly_filter = models.Filter(
            must=[
                models.FieldCondition(key="timestamp", range=models.Range(lt=today_ts)),
                models.FieldCondition(key="timestamp", range=models.Range(gte=month_ts)),
            ]
        )
        long_term_filter = models.Filter(
            must=[models.FieldCondition(key="timestamp", range=models.Range(lt=month_ts))]
        )

        # Step 3: Prepare Concurrent Queries
        query_document = models.Document(text=query_text, model=self.embedding_model)
        search_coroutines: List[Coroutine[Any, Any, Any]] = []

        # Query agent_memory with time filters
        if num_today > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_document,
                    query_filter=today_filter,
                    limit=num_today,
                    with_payload=True,
                )
            )
        if num_monthly > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_document,
                    query_filter=monthly_filter,
                    limit=num_monthly,
                    with_payload=True,
                )
            )
        if num_long_term > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_document,
                    query_filter=long_term_filter,
                    limit=num_long_term,
                    with_payload=True,
                )
            )

        # Query knowledge_bank collection (no time filter)
        if num_kb > 0:
            search_coroutines.append(
                self.client.query_points(
                    collection_name=self.knowledge_bank_collection_name,
                    query=query_document,
                    limit=num_kb,
                    with_payload=True,
                )
            )

        if not search_coroutines:
            return ""

        # Step 4: Execute and Process Results
        try:
            search_results = await asyncio.gather(
                *search_coroutines, return_exceptions=True
            )
        except Exception as retrieval_error:
            logger.error(f"Failed to retrieve memories from Qdrant: {retrieval_error}")
            return ""

        all_points: List[Any] = []
        retrieved_counts = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.warning(f"A memory search query failed: {result}")
                retrieved_counts.append(0)
                continue
            points = getattr(result, "points", [])
            all_points.extend(points)
            retrieved_counts.append(len(points))

        # Log retrieval statistics
        logger.info(
            f"Retrieved memories - Today: {retrieved_counts[0] if num_today > 0 else 0}, "
            f"Monthly: {retrieved_counts[1] if num_monthly > 0 and len(retrieved_counts) > 1 else 0}, "
            f"Long-Term: {retrieved_counts[2] if num_long_term > 0 and len(retrieved_counts) > 2 else 0}, "
            f"Knowledge Bank: {retrieved_counts[-1] if num_kb > 0 and len(retrieved_counts) > 0 else 0}"
        )

        if not all_points:
            return ""

        # Step 5: Deduplicate and Extract Text
        seen_point_ids: set = set()
        unique_memory_texts: List[str] = []
        for point in all_points:
            if point.id not in seen_point_ids:
                seen_point_ids.add(point.id)
                payload = getattr(point, "payload", {})
                # Check for different payload field names (agent_memory vs knowledge_bank)
                text_content = payload.get("text_content") or payload.get("summary_text")
                if text_content:
                    unique_memory_texts.append(text_content)

        if not unique_memory_texts:
            return ""

        # Step 6: Format and Return Context
        context_header = "--- Relevant Memories Retrieved ---\n"
        context_body = "\n".join(unique_memory_texts)
        return f"{context_header}{context_body}"

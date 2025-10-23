# src/memory/qdrant_memory.py

"""
Purpose:
Complete Qdrant memory management with FastEmbed reranker integration and optimizations.
"""

import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional, cast

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models
from src.memory.qdrant_client_manager import QdrantClientManager
from fastembed import TextEmbedding

logger: logging.Logger = logging.getLogger(__name__)


class QdrantMemory:
    """
    Memory management with FastEmbed reranker for relevance boost.
    """

    def __init__(
        self, config: Dict[str, Any], qdrant_manager: QdrantClientManager
    ) -> None:
        """
        Initialize with configuration from TOML.
        """
        self.collection_name: str = config.get("collection_name", "agent_memory")
        self.knowledge_bank_collection_name: str = config.get(
            "knowledge_bank", "knowledge-bank"
        )
        self.embedding_model: str = config.get(
            "embedding_model", "mixedbread-ai/mxbai-embed-large-v1"
        )
        self.device: str = config.get("device", "cpu")
        self.embedder = TextEmbedding(
            model_name=self.embedding_model, device=self.device
        )
        self.embedding_size: int = self.embedder.embedding_size  # 1024
        self.total_memories_to_retrieve: int = config.get(
            "total_memories_to_retrieve", 20
        )
        self.search_hnsw_ef: int = config.get("search_hnsw_ef", 128)

        # Weights from TOML
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

        self.qdrant_manager = qdrant_manager
        self.client = self.qdrant_manager.get_client()

        # FastEmbed reranker
        self.reranker: Optional[TextCrossEncoder] = None
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True):
            model_name = reranker_config.get(
                "model_name", "jinaai/jina-reranker-v2-base-multilingual"
            )
            self.reranker = TextCrossEncoder(model_name=model_name, device=self.device)

    @classmethod
    async def create(cls, config: Dict[str, Any]) -> "QdrantMemory":
        """Create memory instance with collections."""
        qdrant_manager = QdrantClientManager(config)
        instance = cls(config, qdrant_manager)
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.collection_name,
            embedding_size=instance.embedding_size,
        )
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.knowledge_bank_collection_name,
            embedding_size=instance.embedding_size,
        )
        return instance

    async def add_memory(self, text_content: str) -> None:
        """Add memory to collection."""
        try:
            payload = {
                "text_content": text_content,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }
            vector = list(self.embedder.embed(documents=[text_content]))[0].tolist()
            point_id = str(uuid.uuid4())
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                ],
                wait=True,
            )
            logger.info(f"Added memory ID '{point_id}' to '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise RuntimeError(f"Memory addition failed: {e}") from e

    async def retrieve_context(self, query_text: str) -> str:
        """Retrieve with vector search and reranking."""
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

        current_time = datetime.now(timezone.utc)
        today_ts = (current_time - timedelta(days=1)).timestamp()
        month_ts = (current_time - timedelta(days=30)).timestamp()

        query_vector = list(self.embedder.embed(documents=[query_text]))[0].tolist()
        search_params = models.SearchParams(hnsw_ef=self.search_hnsw_ef, exact=False)

        # Time-based filters
        today_filter = models.Filter(
            must=[
                models.FieldCondition(key="timestamp", range=models.Range(gte=today_ts))
            ]
        )
        monthly_filter = models.Filter(
            must=[
                models.FieldCondition(key="timestamp", range=models.Range(lt=today_ts)),
                models.FieldCondition(
                    key="timestamp", range=models.Range(gte=month_ts)
                ),
            ]
        )
        long_term_filter = models.Filter(
            must=[
                models.FieldCondition(key="timestamp", range=models.Range(lt=month_ts))
            ]
        )

        # Concurrent searches
        search_coroutines: List[Coroutine[Any, Any, List[models.ScoredPoint]]] = []
        if num_today > 0:
            search_coroutines.append(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=today_filter,
                    limit=num_today,
                    search_params=search_params,
                )
            )
        if num_monthly > 0:
            search_coroutines.append(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=monthly_filter,
                    limit=num_monthly,
                    search_params=search_params,
                )
            )
        if num_long_term > 0:
            search_coroutines.append(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=long_term_filter,
                    limit=num_long_term,
                    search_params=search_params,
                )
            )
        if num_kb > 0:
            search_coroutines.append(
                self.client.search(
                    collection_name=self.knowledge_bank_collection_name,
                    query_vector=query_vector,
                    limit=num_kb,
                    search_params=search_params,
                )
            )

        if not search_coroutines:
            return ""

        try:
            search_results = await asyncio.gather(
                *search_coroutines, return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ""

        all_points: List[models.ScoredPoint] = []
        for result in search_results:
            if isinstance(result, Exception):
                logger.warning(f"Search failed: {result}")
                continue
            all_points.extend(cast(List[models.ScoredPoint], result))

        # Deduplicate
        seen_ids = set()
        unique_memory_texts = []
        for point in all_points:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                text_content = ""
                if point.payload is not None:
                    text_content = point.payload.get("text_content", "")
                if text_content:
                    unique_memory_texts.append(text_content)

        if not unique_memory_texts:
            return ""

        # FastEmbed reranker
        if self.reranker:
            rerank_scores = list(self.reranker.rerank(query_text, unique_memory_texts))
            ranked = sorted(
                zip(unique_memory_texts, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            unique_memory_texts = [text for text, score in ranked]

        header = "--- FastEmbed Reranked Memories Retrieved ---\n"
        context_body = "\n".join(unique_memory_texts)
        return f"{header}{context_body}"

"""
Retrieval strategy components for the memory system.

This module encapsulates the logic for constructing complex, time-weighted,
hybrid search queries. By isolating this logic, we keep the main QdrantMemory
class cleaner and more focused on the orchestration of memory operations,
adhering to the Single Responsibility Principle.
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client import models

logger = logging.getLogger(__name__)


class TimeWeightedHybridRetriever:
    """
    Builds a set of time-weighted, hybrid (dense + sparse) search queries.
    
    This class reads the retrieval weights from the configuration and calculates
    the number of memories to retrieve from different time buckets (e.g., last
    hour, last day, last week) and from the permanent knowledge bank.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initializes the retriever with retrieval weights from the config.

        Args:
            config: The [memory] section of the configuration dictionary.
        """
        self.total_memories_to_retrieve = config.get("total_memories_to_retrieve", 20)
        self.weights = {
            "hourly": config.get("hourly_retrieval_weight", 0.1),
            "daily": config.get("daily_retrieval_weight", 0.2),
            "weekly": config.get("weekly_retrieval_weight", 0.3),
            "monthly": config.get("monthly_retrieval_weight", 0.1),
            "knowledge_bank": config.get("knowledge_bank_retrieval_weight", 0.3),
        }
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Ensures that the retrieval weights sum to 1.0 for proper distribution."""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            logger.warning("All retrieval weights are zero. No memories will be retrieved.")

    def _calculate_retrieval_limits(self) -> dict[str, int]:
        """
        Calculates the number of memories to retrieve for each time bucket.
        """
        limits = {
            key: math.ceil(self.total_memories_to_retrieve * weight)
            for key, weight in self.weights.items()
        }
        logger.debug("Calculated retrieval limits: %s", limits)
        return limits

    def build_query_requests(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        agent_collection: str,
        kb_collection: str,
        search_params: models.SearchParams,
    ) -> list[models.QueryRequest]:
        """
        Constructs a list of Qdrant QueryRequest objects for batch searching.

        Args:
            dense_vector: The dense query vector.
            sparse_vector: The sparse query vector.
            agent_collection: The name of the agent's memory collection.
            kb_collection: The name of the knowledge bank collection.
            search_params: The search parameters (e.g., hnsw_ef).

        Returns:
            A list of QueryRequest objects ready for a batch query.
        """
        limits = self._calculate_retrieval_limits()
        now = datetime.now(UTC)

        time_filters = {
            "hourly": models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.DatetimeRange(gte=(now - timedelta(hours=1))),
                    )
                ]
            ),
            "daily": models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.DatetimeRange(
                            lt=(now - timedelta(hours=1)),
                            gte=(now - timedelta(days=1)),
                        ),
                    )
                ]
            ),
            "weekly": models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.DatetimeRange(
                            lt=(now - timedelta(days=1)),
                            gte=(now - timedelta(weeks=1)),
                        ),
                    )
                ]
            ),
            "monthly": models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.DatetimeRange(
                            lt=(now - timedelta(weeks=1)),
                            gte=(now - timedelta(days=30)),
                        ),
                    )
                ]
            ),
        }

        requests = []
        # Agent memory queries
        for bucket, time_filter in time_filters.items():
            limit = limits.get(bucket, 0)
            if limit > 0:
                requests.append(
                    models.QueryRequest(
                        collection_name=agent_collection,
                        query=models.HybridQuery(
                            dense=dense_vector, sparse=sparse_vector
                        ),
                        filter=time_filter,
                        limit=limit,
                        params=search_params,
                        with_payload=True,
                    )
                )

        # Knowledge bank query
        kb_limit = limits.get("knowledge_bank", 0)
        if kb_limit > 0:
            requests.append(
                models.QueryRequest(
                    collection_name=kb_collection,
                    query=models.HybridQuery(dense=dense_vector, sparse=sparse_vector),
                    limit=kb_limit,
                    params=search_params,
                    with_payload=True,
                )
            )

        return requests

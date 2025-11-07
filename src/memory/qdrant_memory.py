from __future__ import annotations

import asyncio
import logging
import math
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from qdrant_client import models
from sentence_transformers import SparseEncoder

from src.memory.embedding_models import create_embedder
from src.memory.qdrant_client_manager import QdrantClientManager

logger = logging.getLogger(__name__)





class QdrantMemory:

    CONTEXT_HEADER = "--- Memories Retrieved ---\n"



    def __init__(

        self,

        config: dict[str, Any],

        qdrant_manager: QdrantClientManager,

        agent_name: str | None = None,

    ) -> None:

        self._init_config(config, agent_name)

        self._init_embedding_models(config)

        self._init_qdrant(qdrant_manager)



    def _init_config(self, config: dict[str, Any], agent_name: str | None) -> None:

        mem_config = config.get("memory", config)

        self.collection_name = mem_config.get("collection_name", "agent_memory")

        self.knowledge_bank_collection_name = mem_config.get(

            "knowledge_bank", "knowledge-bank"

        )

        self.total_memories_to_retrieve = mem_config.get(

            "total_memories_to_retrieve", 20

        )

        self.query_points_hnsw_ef = mem_config.get("query_points_hnsw_ef", 128)

        self._set_retrieval_weights(config, agent_name)



    def _set_retrieval_weights(

        self, config: dict[str, Any], agent_name: str | None

    ) -> None:

        def get_weight(key: str, default: float) -> float:

            agent_cfg = (

                config.get(agent_name, {}).get("memory", {}) if agent_name else {}

            )

            return float(

                agent_cfg.get(key, config.get("memory", config).get(key, default))

            )



        weight_keys = [

            "hourly",

            "daily",

            "weekly",

            "two_weeks",

            "monthly",

            "ninety_days",

            "one_eighty_days",

            "three_sixty_days",

            "knowledge_bank",

        ]

        self.retrieval_weights = {

            k: get_weight(f"{k}_retrieval_weight", 0.1 if k == "hourly" else 0.05)

            for k in weight_keys

        }



    def _init_embedding_models(self, config: dict[str, Any]) -> None:

        mem_config = config.get("memory", config)

        self.embedding_model = create_embedder(mem_config.get("embedding_model", {}))

        self.embedding_size = mem_config.get("embedding_size", 1024)

        self.sparse_embedder = SparseEncoder(

            mem_config.get("sparse_embedding_model", "naver/splade-v3")

        )



    def _init_qdrant(self, qdrant_manager: QdrantClientManager) -> None:

        self.qdrant_manager = qdrant_manager

        self.client = qdrant_manager.get_client()

        self.agent_dense_name = None

        self.agent_sparse_name = None

        self.kb_dense_name = None

        self.kb_sparse_name = None



    @lru_cache(maxsize=1000)

    def _embed_text(self, text: str) -> list[float]:

        if not self.embedding_model:

            logger.warning("Embedding model not initialized. Returning zero vector.")

            return [0.0] * self.embedding_size

        try:

            return self.embedding_model.embed(text)

        except Exception as e:

            logger.error(f"Embedding failed: {e}")

            raise RuntimeError("Embedding generation failed") from e



    def _embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:

        embeddings = self.sparse_embedder.encode_document(texts)

        return [

            models.SparseVector(

                indices=emb.indices().squeeze().tolist(),

                values=emb.values().squeeze().tolist(),

            )

            for emb in embeddings

        ]

    async def create(
        self, config: dict[str, Any], agent_name: str | None = None
    ) -> QdrantMemory:
        instance = self.__class__(config, self.qdrant_manager, agent_name)
        await instance._setup_collections()
        return instance

    async def _setup_collections(self) -> None:
        await self.qdrant_manager.ensure_collection_exists(
            self.collection_name, self.embedding_size
        )
        await self.qdrant_manager.ensure_collection_exists(
            self.knowledge_bank_collection_name, self.embedding_size
        )
        self.agent_dense_name, self.agent_sparse_name = (
            await self._resolve_vector_names(self.collection_name)
        )
        self.kb_dense_name, self.kb_sparse_name = await self._resolve_vector_names(
            self.knowledge_bank_collection_name
        )

    async def _resolve_vector_names(self, collection: str) -> tuple[str, str | None]:
        info = await self.client.get_collection(collection_name=collection)
        params = info.config.params

        dense_name = (
            next(iter(params.vectors.keys()))
            if isinstance(params.vectors, dict)
            else ""
        )
        sparse_name = (
            next(iter(params.sparse_vectors.keys()))
            if hasattr(params, "sparse_vectors") and params.sparse_vectors
            else None
        )

        return dense_name, sparse_name

    async def add_memory(self, text_content: str) -> None:
        try:
            dense_vector = self._embed_text(text_content)
            sparse_vector = self._embed_sparse([text_content])[0]

            point_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            payload = {
                "text_content": text_content,
                "timestamp": now.timestamp(),
                "day_of_week": now.strftime("%A"),
            }

            vector_payload = {self.agent_dense_name: dense_vector}
            if self.agent_sparse_name:
                vector_payload[self.agent_sparse_name] = sparse_vector

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id, vector=vector_payload, payload=payload
                    )
                ],
                wait=True,
            )
            logger.info(f"Added memory ID '{point_id}' to '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise RuntimeError("Memory addition failed") from e

    async def retrieve_context(self, query_text: str) -> str:
        limits = self._calculate_retrieval_limits()
        if not sum(limits):
            return ""

        timestamps = self._get_time_boundaries()
        dense_query_vector = self._embed_text(query_text)
        sparse_query_vector = self._embed_sparse([query_text])[0]

        query_tasks = self._build_query_tasks(
            dense_query_vector, sparse_query_vector, limits, timestamps
        )
        if not query_tasks:
            return ""

        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        unique_texts = self._process_results(query_results)
        if not unique_texts:
            return ""

        return f"{self.CONTEXT_HEADER}{' '.join(unique_texts)}"

    def _calculate_retrieval_limits(self) -> tuple[int, ...]:
        return tuple(
            math.ceil(self.total_memories_to_retrieve * w)
            for w in self.retrieval_weights.values()
        )

    def _get_time_boundaries(self) -> tuple[float, ...]:
        now = datetime.now(UTC)
        return tuple(
            (now - timedelta(**{unit: value})).timestamp()
            for unit, value in [
                ("hours", 1),
                ("days", 1),
                ("weeks", 1),
                ("weeks", 2),
                ("days", 30),
                ("days", 90),
                ("days", 180),
                ("days", 365),
            ]
        )

    def _build_query_tasks(
        self, dense_vec, sparse_vec, limits, timestamps
    ) -> list[Any]:
        tasks = []
        time_filters = self._create_time_filters(timestamps)

        for i, (limit, time_filter) in enumerate(zip(limits, time_filters)):
            if limit > 0:
                collection = (
                    self.knowledge_bank_collection_name
                    if i == len(limits) - 1
                    else self.collection_name
                )
                dense_name = (
                    self.kb_dense_name
                    if i == len(limits) - 1
                    else self.agent_dense_name
                )
                sparse_name = (
                    self.kb_sparse_name
                    if i == len(limits) - 1
                    else self.agent_sparse_name
                )

                tasks.append(
                    self.client.query_points(
                        collection_name=collection,
                        query=dense_vec,
                        using=dense_name,
                        query_filter=time_filter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                        search_params=models.SearchParams(
                            hnsw_ef=self.query_points_hnsw_ef, exact=False
                        ),
                        prefetch=self._create_prefetch(
                            dense_vec, sparse_vec, limit, dense_name, sparse_name
                        ),
                    )
                )

        return tasks

    def _create_time_filters(self, timestamps) -> list[models.Filter | None]:
        filters = []
        for i in range(len(timestamps) - 1):
            filters.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp", range=models.Range(lt=timestamps[i])
                        )
                    ]
                )
            )

        filters.append(None)  # No filter for knowledge bank
        return filters

    def _create_prefetch(
        self,
        dense_vec,
        sparse_vec,
        limit,
        dense_name,
        sparse_name,
    ) -> models.Prefetch | None:
        if not sparse_name:
            return None
        return models.Prefetch(
            prefetch=[
                models.Prefetch(query=dense_vec, using=dense_name, limit=limit),
                models.Prefetch(query=sparse_vec, using=sparse_name, limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

    def _process_results(self, query_results) -> list[str]:
        points = []
        for result in query_results:
            if isinstance(result, BaseException):
                logger.warning(f"Query failed: {result}")
                continue
            points.extend(result.points)

        seen_ids = set()
        unique_texts = []
        for point in points:
            if point.id not in seen_ids and (text := point.payload.get("text_content")):
                seen_ids.add(point.id)
                unique_texts.append(str(text))
        return unique_texts



    async def prune_memories(self) -> int:
        logger.warning("Memory pruning is not yet implemented.")
        return 0


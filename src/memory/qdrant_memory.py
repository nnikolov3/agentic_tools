# src/memory/qdrant_memory.py
import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union, cast

from faker import Faker
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from qdrant_client import models
from sentence_transformers import SparseEncoder  # used
from src.memory.qdrant_client_manager import QdrantClientManager

logger: logging.Logger = logging.getLogger(__name__)


class QdrantMemory:
    CONTEXT_HEADER: str = "--- FastEmbed Reranked Memories Retrieved ---\n"

    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_manager: QdrantClientManager,
        agent_name: Optional[str] = None,
    ) -> None:
        # Determine the source of the global memory configuration (full project config vs. memory-only config)
        global_memory_config = config.get("memory", config)

        self.collection_name: str = global_memory_config.get(
            "collection_name", "agent_memory"
        )
        self.knowledge_bank_collection_name: str = global_memory_config.get(
            "knowledge_bank",
            "knowledge-bank",
        )

        # Embeddings
        self.embedding_model: str = global_memory_config.get(
            "embedding_model",
            "mixedbread-ai/mxbai-embed-large-v1",
        )
        self.device: str = global_memory_config.get("device", "cpu")
        self.embedder = TextEmbedding(model_name=self.embedding_model)

        self.sparse_embedding_model: str = global_memory_config.get(
            "sparse_embedding_model",
            "naver/splade-v3",
        )
        self.sparse_embedder = SparseEncoder(self.sparse_embedding_model)

        self.embedding_size: int = self.embedder.embedding_size
        self.total_memories_to_retrieve: int = global_memory_config.get(
            "total_memories_to_retrieve", 20
        )
        self.query_points_hnsw_ef: int = global_memory_config.get(
            "query_points_hnsw_ef", 128
        )

        # Optional preferred names from config (used as hints if they exist in the collection)
        self.preferred_dense_vector_name: Optional[str] = global_memory_config.get(
            "dense_vector_name"
        )
        self.preferred_sparse_vector_name: Optional[str] = global_memory_config.get(
            "sparse_vector_name"
        )

        agent_specific_config = config.get(agent_name, {}) if agent_name else {}
        agent_memory_config = agent_specific_config.get("memory", {})

        # Helper function to get weight with fallback: Agent -> Global -> Hardcoded
        def get_weight(key: str, default: float) -> float:
            return float(
                agent_memory_config.get(
                    key,
                    global_memory_config.get(key, default),
                )
            )

        self.hourly_retrieval_weight: float = get_weight("hourly_retrieval_weight", 0.1)
        self.daily_retrieval_weight: float = get_weight("daily_retrieval_weight", 0.2)
        self.weekly_retrieval_weight: float = get_weight("weekly_retrieval_weight", 0.3)
        self.two_weeks_retrieval_weight: float = get_weight(
            "two_weeks_retrieval_weight", 0.1
        )
        self.monthly_retrieval_weight: float = get_weight(
            "monthly_retrieval_weight", 0.1
        )
        self.ninety_days_retrieval_weight: float = get_weight(
            "ninety_days_retrieval_weight", 0.05
        )
        self.one_eighty_days_retrieval_weight: float = get_weight(
            "one_eighty_days_retrieval_weight", 0.05
        )
        self.three_sixty_days_retrieval_weight: float = get_weight(
            "three_sixty_days_retrieval_weight", 0.05
        )
        self.knowledge_bank_retrieval_weight: float = get_weight(
            "knowledge_bank_retrieval_weight", 0.05
        )

        self.qdrant_manager = qdrant_manager
        self.client = self.qdrant_manager.get_client()

        # Reranker
        self.reranker: Optional[TextCrossEncoder] = None
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True):
            model_name = reranker_config.get(
                "model_name",
                "jinaai/jina-reranker-v2-base-multilingual",
            )
            self.reranker = TextCrossEncoder(model_name=model_name, device=self.device)

        self.faker = Faker()

        # Resolved names per collection (filled in create)
        self.agent_dense_name: Optional[str] = None
        self.agent_sparse_name: Optional[str] = None
        self.kb_dense_name: Optional[str] = None
        self.kb_sparse_name: Optional[str] = None

    @classmethod
    async def create(
        cls,
        config: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> "QdrantMemory":
        qdrant_manager = QdrantClientManager(config)
        instance = cls(config, qdrant_manager, agent_name)

        # Respect existing collections; just ensure they exist (no schema change)
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.collection_name,
            embedding_size=instance.embedding_size,
        )
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.knowledge_bank_collection_name,
            embedding_size=instance.embedding_size,
        )

        # Resolve actual vector names from each collection
        instance.agent_dense_name, instance.agent_sparse_name = (
            await instance._resolve_vector_names(instance.collection_name)
        )
        instance.kb_dense_name, instance.kb_sparse_name = (
            await instance._resolve_vector_names(
                instance.knowledge_bank_collection_name
            )
        )

        logger.info(
            f"Resolved vector names: agent_memory(dense='{instance.agent_dense_name}', sparse='{instance.agent_sparse_name}'), "
            f"knowledge_bank(dense='{instance.kb_dense_name}', sparse='{instance.kb_sparse_name}')"
        )

        return instance

    async def _resolve_vector_names(self, collection: str) -> Tuple[str, Optional[str]]:
        """
        Inspect an existing collection and return:
        - dense_name: name of the dense vector to use. Empty string "" if the collection has a single unnamed dense vector.
        - sparse_name: name of the sparse vector if configured, else None.
        Preference order for dense:
          1) preferred_dense_vector_name if present in collection
          2) first named dense vector
          3) "" (unnamed dense)
        Preference order for sparse:
          1) preferred_sparse_vector_name if present
          2) first configured sparse name
          3) None (no sparse configured)
        """
        info = await self.client.get_collection(collection_name=collection)
        params = info.config.params

        # Dense
        dense_name: str
        if isinstance(params.vectors, dict):
            if (
                self.preferred_dense_vector_name
                and self.preferred_dense_vector_name in params.vectors
            ):
                dense_name = self.preferred_dense_vector_name
            else:
                dense_name = next(iter(params.vectors.keys()))
        else:
            dense_name = ""  # unnamed dense vector

        # Sparse
        sparse_name: Optional[str] = None
        if (
            hasattr(params, "sparse_vectors")
            and isinstance(params.sparse_vectors, dict)
            and params.sparse_vectors
        ):
            if (
                self.preferred_sparse_vector_name
                and self.preferred_sparse_vector_name in params.sparse_vectors
            ):
                sparse_name = self.preferred_sparse_vector_name
            else:
                sparse_name = next(iter(params.sparse_vectors.keys()))

        return dense_name, sparse_name

    def _embed_text(self, text: str) -> List[float]:
        try:
            embedding_generator = self.embedder.embed(documents=[text])
            return next(embedding_generator).tolist()  # type: ignore
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError("Embedding generation failed") from e

    def _embed_sparse_documents(
        self, documents: List[str]
    ) -> List[models.SparseVector]:
        sparse_embeddings = self.sparse_embedder.encode_document(documents)
        return [
            models.SparseVector(
                indices=cast(Any, embedding).coalesce().indices().squeeze().tolist(),
                values=cast(Any, embedding).coalesce().values().squeeze().tolist(),
            )
            for embedding in sparse_embeddings
        ]

    def _embed_sparse_query(self, query: str) -> models.SparseVector:
        sparse_embedding = self.sparse_embedder.encode_query(query)
        return models.SparseVector(
            indices=cast(Any, sparse_embedding).coalesce().indices().squeeze().tolist(),
            values=cast(Any, sparse_embedding).coalesce().values().squeeze().tolist(),
        )

    async def add_memory(self, text_content: str) -> None:
        """
        Upserts using whatever names the agent_memory collection supports.
        If the collection has no sparse vector configured, only upsert the dense vector.
        If the collection uses an unnamed dense vector, use key "" for the dense vector.
        """
        try:
            dense_vector = self._embed_text(text_content)
            sparse_vector = self._embed_sparse_documents([text_content])[0]

            point_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            payload = {
                "text_content": text_content,
                "timestamp": now.timestamp(),
                "day_of_week": now.strftime("%A"),
                "random_fact": self.faker.text(),
            }

            vector_payload: Dict[str, Any] = {}
            # Dense vector name can be "" (unnamed)
            if self.agent_dense_name is None:
                raise RuntimeError(
                    "Dense vector name for agent collection is not resolved"
                )
            vector_payload[self.agent_dense_name] = dense_vector

            # Only include sparse if the collection supports it
            if self.agent_sparse_name:
                vector_payload[self.agent_sparse_name] = sparse_vector

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector_payload,
                        payload=payload,
                    ),
                ],
                wait=True,
            )
            logger.info(f"Added memory ID '{point_id}' to '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise RuntimeError("Memory addition failed") from e

    def _calculate_retrieval_limits(
        self,
    ) -> Tuple[int, int, int, int, int, int, int, int, int]:
        n = self.total_memories_to_retrieve
        return (
            math.ceil(n * self.hourly_retrieval_weight),
            math.ceil(n * self.daily_retrieval_weight),
            math.ceil(n * self.weekly_retrieval_weight),
            math.ceil(n * self.two_weeks_retrieval_weight),
            math.ceil(n * self.monthly_retrieval_weight),
            math.ceil(n * self.ninety_days_retrieval_weight),
            math.ceil(n * self.one_eighty_days_retrieval_weight),
            math.ceil(n * self.three_sixty_days_retrieval_weight),
            math.ceil(n * self.knowledge_bank_retrieval_weight),
        )

    def _make_prefetch(
        self,
        dense_query_vector: List[float],
        sparse_query_vector: models.SparseVector,
        limit: int,
        dense_name: str,
        sparse_name: Optional[str],
    ) -> Optional[models.Prefetch]:
        """
        Build Fusion+Prefetch only if sparse_name is available for the target collection.
        """
        if not sparse_name:
            return None
        return models.Prefetch(
            prefetch=[
                models.Prefetch(
                    query=dense_query_vector,
                    using=dense_name,  # may be "" for unnamed
                    limit=limit,
                ),
                models.Prefetch(
                    query=sparse_query_vector,
                    using=sparse_name,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

    def _query_args_for_collection(
        self,
        collection: str,
        dense_vec: List[float],
        sparse_vec: models.SparseVector,
        limit: int,
        qfilter: Optional[models.Filter],
        dense_name: str,
        sparse_name: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            collection_name=collection,
            query=dense_vec,
            using=dense_name,  # "" if unnamed dense
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(
                hnsw_ef=self.query_points_hnsw_ef, exact=False
            ),
        )
        prefetch = self._make_prefetch(
            dense_vec, sparse_vec, limit, dense_name, sparse_name
        )
        if prefetch:
            kwargs["prefetch"] = [prefetch]
        return kwargs

    def _build_query_tasks(
        self,
        dense_query_vector: List[float],
        sparse_query_vector: models.SparseVector,
        limits: Tuple[int, int, int, int, int, int, int, int, int],
        timestamps: Tuple[float, float, float, float, float, float, float, float],
    ) -> List[Coroutine[Any, Any, models.QueryResponse]]:
        (
            num_hourly,
            num_daily,
            num_weekly,
            num_two_weeks,
            num_monthly,
            num_ninety_days,
            num_one_eighty_days,
            num_three_sixty_days,
            num_kb,
        ) = limits
        (
            hour_ts,
            day_ts,
            week_ts,
            two_weeks_ts,
            month_ts,
            ninety_days_ts,
            one_eighty_days_ts,
            three_sixty_days_ts,
        ) = timestamps

        if self.agent_dense_name is None or self.kb_dense_name is None:
            logger.error(
                "Dense vector names were not resolved for one or more collections."
            )
            return []

        tasks: List[Coroutine[Any, Any, models.QueryResponse]] = []

        # Helper to make filters
        def rng(
            gte: Optional[float] = None, lt: Optional[float] = None
        ) -> models.Filter:
            must = []
            if lt is not None:
                must.append(
                    models.FieldCondition(key="timestamp", range=models.Range(lt=lt))
                )
            if gte is not None:
                must.append(
                    models.FieldCondition(key="timestamp", range=models.Range(gte=gte))
                )
            return models.Filter(must=cast(List[Any], must))

        # agent_memory tasks (time buckets)
        if num_hourly > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_hourly,
                        rng(gte=hour_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_daily > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_daily,
                        rng(gte=day_ts, lt=hour_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_weekly > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_weekly,
                        rng(gte=week_ts, lt=day_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_two_weeks > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_two_weeks,
                        rng(gte=two_weeks_ts, lt=week_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_monthly > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_monthly,
                        rng(gte=month_ts, lt=two_weeks_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_ninety_days > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_ninety_days,
                        rng(gte=ninety_days_ts, lt=month_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_one_eighty_days > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_one_eighty_days,
                        rng(gte=one_eighty_days_ts, lt=ninety_days_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )
        if num_three_sixty_days > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_three_sixty_days,
                        rng(gte=three_sixty_days_ts, lt=one_eighty_days_ts),
                        self.agent_dense_name,
                        self.agent_sparse_name,
                    )
                )
            )

        # knowledge_bank query (no time filter, independent schema)
        if num_kb > 0:
            tasks.append(
                self.client.query_points(
                    **self._query_args_for_collection(
                        self.knowledge_bank_collection_name,
                        dense_query_vector,
                        sparse_query_vector,
                        num_kb,
                        None,
                        self.kb_dense_name,
                        self.kb_sparse_name,
                    )
                )
            )

        return tasks

    @staticmethod
    def _process_and_deduplicate_results(
        query_results: List[Union[models.QueryResponse, BaseException]],
    ) -> List[str]:
        all_points: List[models.ScoredPoint] = []
        for result in query_results:
            if isinstance(result, BaseException):
                logger.warning(
                    f"A Qdrant query failed during memory retrieval: {result}",
                )
                continue
            all_points.extend(result.points)

        seen_ids: set[Union[str, int]] = set()
        unique_memory_texts: List[str] = []
        for point in all_points:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                if point.payload and (
                    text_content := point.payload.get("text_content")
                ):
                    unique_memory_texts.append(str(text_content))
        return unique_memory_texts

    def _rerank_results(self, query_text: str, memory_texts: List[str]) -> List[str]:
        if not self.reranker:
            return memory_texts
        try:
            rerank_scores = list(self.reranker.rerank(query_text, memory_texts))
            ranked_pairs = sorted(
                zip(memory_texts, rerank_scores),
                key=lambda item: item[1],
                reverse=True,
            )
            return [text for text, _ in ranked_pairs]
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order. Error: {e}")
            return memory_texts

    async def retrieve_context(self, query_text: str) -> str:
        limits = self._calculate_retrieval_limits()
        if sum(limits) == 0:
            return ""

        current_time = datetime.now(timezone.utc)
        hour_ts = (current_time - timedelta(hours=1)).timestamp()
        day_ts = (current_time - timedelta(days=1)).timestamp()
        week_ts = (current_time - timedelta(weeks=1)).timestamp()
        two_weeks_ts = (current_time - timedelta(weeks=2)).timestamp()
        month_ts = (current_time - timedelta(days=30)).timestamp()
        ninety_days_ts = (current_time - timedelta(days=90)).timestamp()
        one_eighty_days_ts = (current_time - timedelta(days=180)).timestamp()
        three_sixty_days_ts = (current_time - timedelta(days=365)).timestamp()

        try:
            dense_query_vector = self._embed_text(query_text)
            sparse_query_vector = self._embed_sparse_query(query_text)
        except RuntimeError:
            return ""

        query_tasks = self._build_query_tasks(
            dense_query_vector,
            sparse_query_vector,
            limits,
            (
                hour_ts,
                day_ts,
                week_ts,
                two_weeks_ts,
                month_ts,
                ninety_days_ts,
                one_eighty_days_ts,
                three_sixty_days_ts,
            ),
        )
        if not query_tasks:
            return ""

        try:
            query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(
                f"Unexpected error during asyncio.gather for Qdrant queries: {e}",
            )
            return ""

        unique_memory_texts = self._process_and_deduplicate_results(query_results)
        if not unique_memory_texts:
            return ""

        ranked_texts = self._rerank_results(query_text, unique_memory_texts)
        context_body = "\n".join(ranked_texts)
        return f"{self.CONTEXT_HEADER}{context_body}"

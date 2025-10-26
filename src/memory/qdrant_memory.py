# src/memory/qdrant_memory.py
import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

from faker import Faker
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

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
        self.collection_name: str = config.get("collection_name", "agent_memory")
        self.knowledge_bank_collection_name: str = config.get(
            "knowledge_bank",
            "knowledge-bank",
        )
        self.embedding_model: str = config.get(
            "embedding_model",
            "mixedbread-ai/mxbai-embed-large-v1",
        )
        self.device: str = config.get("device", "cpu")
        self.embedder = TextEmbedding(
            model_name=self.embedding_model,
            device=self.device,
        )
        self.embedding_size: int = self.embedder.embedding_size
        self.total_memories_to_retrieve: int = config.get(
            "total_memories_to_retrieve",
            20,
        )
        self.query_points_hnsw_ef: int = config.get("query_points_hnsw_ef", 128)

        agent_memory_config = (
            config.get(agent_name, {}).get(
                "memory",
                {},
            )
            if agent_name
            else {}
        )

        self.hourly_retrieval_weight: float = agent_memory_config.get(
            "hourly_retrieval_weight",
            config.get("hourly_retrieval_weight", 0.1),
        )
        self.daily_retrieval_weight: float = agent_memory_config.get(
            "daily_retrieval_weight",
            config.get("daily_retrieval_weight", 0.2),
        )
        self.weekly_retrieval_weight: float = agent_memory_config.get(
            "weekly_retrieval_weight",
            config.get("weekly_retrieval_weight", 0.3),
        )
        self.two_weeks_retrieval_weight: float = agent_memory_config.get(
            "two_weeks_retrieval_weight",
            config.get("two_weeks_retrieval_weight", 0.1),
        )
        self.monthly_retrieval_weight: float = agent_memory_config.get(
            "monthly_retrieval_weight",
            config.get("monthly_retrieval_weight", 0.1),
        )
        self.ninety_days_retrieval_weight: float = agent_memory_config.get(
            "ninety_days_retrieval_weight",
            config.get("ninety_days_retrieval_weight", 0.05),
        )
        self.one_eighty_days_retrieval_weight: float = agent_memory_config.get(
            "one_eighty_days_retrieval_weight",
            config.get("one_eighty_days_retrieval_weight", 0.05),
        )
        self.three_sixty_days_retrieval_weight: float = agent_memory_config.get(
            "three_sixty_days_retrieval_weight",
            config.get("three_sixty_days_retrieval_weight", 0.05),
        )
        self.knowledge_bank_retrieval_weight: float = agent_memory_config.get(
            "knowledge_bank_retrieval_weight",
            config.get("knowledge_bank_retrieval_weight", 0.05),
        )

        self.qdrant_manager = qdrant_manager
        self.client = self.qdrant_manager.get_client()

        self.reranker: Optional[TextCrossEncoder] = None
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True):
            model_name = reranker_config.get(
                "model_name",
                "jinaai/jina-reranker-v2-base-multilingual",
            )
            self.reranker = TextCrossEncoder(
                model_name=model_name,
                device=self.device,
            )
        self.faker = Faker()

    @classmethod
    async def create(
        cls,
        config: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> "QdrantMemory":
        qdrant_manager = QdrantClientManager(config)
        instance = cls(config, qdrant_manager, agent_name)
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.collection_name,
            embedding_size=instance.embedding_size,
        )
        await qdrant_manager.ensure_collection_exists(
            collection_name=instance.knowledge_bank_collection_name,
            embedding_size=instance.embedding_size,
        )
        return instance

    def _embed_text(self, text: str) -> List[float]:
        try:
            embedding_generator = self.embedder.embed(documents=[text])
            return next(embedding_generator).tolist()  # type: ignore
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError("Embedding generation failed") from e

    async def add_memory(self, text_content: str) -> None:
        try:
            vector = self._embed_text(text_content)
            point_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            payload = {
                "text_content": text_content,
                "timestamp": now.timestamp(),
                "day_of_week": now.strftime("%A"),
                "random_fact": self.faker.text(),
            }

            vec_payload: Any
            vector_name = getattr(self.qdrant_manager, "vector_name", None)
            vec_payload = {vector_name: vector} if vector_name else vector

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vec_payload,
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

    def _build_query_tasks(
        self,
        query_vector: List[float],
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

        search_params = models.SearchParams(
            hnsw_ef=self.query_points_hnsw_ef,
            exact=False,
        )
        using_vec = getattr(self.qdrant_manager, "vector_name", None)
        tasks: List[Coroutine[Any, Any, models.QueryResponse]] = []

        if num_hourly > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=hour_ts),
                            ),
                        ],
                    ),
                    limit=num_hourly,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_daily > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(lt=hour_ts),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=day_ts),
                            ),
                        ],
                    ),
                    limit=num_daily,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_weekly > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(lt=day_ts),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=week_ts),
                            ),
                        ],
                    ),
                    limit=num_weekly,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_two_weeks > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(lt=week_ts),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=two_weeks_ts,
                                ),
                            ),
                        ],
                    ),
                    limit=num_two_weeks,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_monthly > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    lt=two_weeks_ts,
                                ),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=month_ts),
                            ),
                        ],
                    ),
                    limit=num_monthly,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_ninety_days > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(lt=month_ts),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=ninety_days_ts,
                                ),
                            ),
                        ],
                    ),
                    limit=num_ninety_days,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_one_eighty_days > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    lt=ninety_days_ts,
                                ),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=one_eighty_days_ts,
                                ),
                            ),
                        ],
                    ),
                    limit=num_one_eighty_days,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_three_sixty_days > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=using_vec,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    lt=one_eighty_days_ts,
                                ),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=three_sixty_days_ts,
                                ),
                            ),
                        ],
                    ),
                    limit=num_three_sixty_days,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        if num_kb > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.knowledge_bank_collection_name,
                    query=query_vector,
                    using=using_vec,
                    limit=num_kb,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                ),
            )
        return tasks

    @staticmethod
    def _process_and_deduplicate_results(
        query_results: List[Union[models.QueryResponse, BaseException]],
    ) -> List[str]:
        all_points: List[models.ScoredPoint] = []
        for result in query_results:
            # Necessary because asyncio.gather with return_exceptions=True can return
            # instances of BaseException (e.g., CancelledError).
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
            query_vector = self._embed_text(query_text)
        except RuntimeError:
            return ""

        query_tasks = self._build_query_tasks(
            query_vector,
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

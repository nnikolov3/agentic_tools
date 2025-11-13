"""
This module defines the three-layer memory system for agents, driven by
configuration to ensure explicitness and adaptability.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models


logger = logging.getLogger(__name__)


class Memory:
    """
    Manages a three-layer memory system: episodic, working, and semantic.

    This class provides a unified interface for adding memories to specific
    layers and retrieving a weighted, time-filtered context based on an
    agent's configuration. This design makes memory operations explicit and
    dynamically adaptable without code changes.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_client: AsyncQdrantClient,
        providers: Dict[str, Any],
    ):
        """
        Initializes the Memory system with explicit dependencies.

        Why: Injecting clients (Qdrant, Google) follows the principle of
        Composition over Inheritance and makes dependencies explicit. The full
        configuration is passed to ensure all necessary parameters are available.
        """
        self.config = config
        self.client = qdrant_client
        self.providers = providers

        mem_cfg = self.config.get("memory", {})
        self.episodic_collection: str = mem_cfg.get("episodic_collection", "episodic")
        self.working_collection: str = mem_cfg.get("working_collection", "working")
        self.semantic_collection: str = mem_cfg.get("semantic_collection", "semantic")

        self.embedding_provider_name: str = mem_cfg.get("embedding_provider")
        self.embedding_model_name: str = mem_cfg.get("embedding_model")
        self.embedding_size: int = mem_cfg.get("embedding_size")

        if self.embedding_provider_name not in self.providers:
            raise ValueError(
                f"Embedding provider '{self.embedding_provider_name}' not found in available providers."
            )
        self.embedding_provider = self.providers[self.embedding_provider_name]

    @classmethod
    async def create(
        cls, config: Dict[str, Any], providers: Dict[str, Any]
    ) -> "Memory":
        """
        Factory method to create and initialize a Memory instance.

        Why: A factory centralizes the complex initialization logic, including
        client setup and collection verification. This simplifies object
        creation for the caller and ensures the Memory instance is always
        in a valid state.
        """
        mem_cfg = config.get("memory", {})
        qdrant = AsyncQdrantClient(
            host=mem_cfg.get("host", "192.168.122.40"),
            grpc_port=int(mem_cfg.get("grpc_port", 6334)),
            prefer_grpc=mem_cfg.get("prefer_grpc", True),
            timeout=int(mem_cfg.get("timeout", 30)),
        )

        instance = cls(config, qdrant, providers)

        for collection_name in [
            instance.episodic_collection,
            instance.working_collection,
            instance.semantic_collection,
        ]:
            await instance._ensure_collection(collection_name)

        return instance

    async def _ensure_collection(self, name: str) -> None:
        """
        Ensures a Qdrant collection exists, creating it if necessary.

        Why: This idempotent operation prevents errors from attempting to use
        a non-existent collection and centralizes the collection schema,
        making it consistent and easy to manage.
        """
        if await self.client.collection_exists(name):
            return

        vectors_config = {
            "embedding": models.VectorParams(
                size=self.embedding_size,
                distance=models.Distance.COSINE,
            )
        }

        await self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )
        logger.info("Created Qdrant collection: %s", name)

    async def add(
        self,
        text_content: str,
        collection: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Adds a memory to a specified collection with a timestamp.

        Why: This unified 'add' method is explicit, requiring the caller to
        specify the target collection. This removes ambiguity and prevents
        accidental writes to the wrong memory layer, adhering to the
        "Explicit over Implicit" principle.
        """
        vector = self.embedding_provider.embed(
            text=text_content,
            model_name=self.embedding_model_name,
            output_dimensionality=self.embedding_size,
        )
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).timestamp()

        payload = {"text": text_content, "timestamp": now}
        if metadata:
            payload.update(metadata)

        await self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"embedding": vector},
                    payload=payload,
                )
            ],
            wait=True,
        )
        logger.info("Added memory %s to collection '%s'.", point_id, collection)
        return point_id

    async def retrieve_context(self, query: str, agent_config: Dict[str, Any]) -> str:
        """
        Retrieves a weighted, time-filtered context for an agent.

        Why: This method is configuration-driven, respecting the weights and
        limits defined in the agent's configuration. This makes the retrieval
        logic adaptable and transparent. Temporal filtering ensures that each
        memory layer serves its intended purpose.
        """
        if not agent_config.get("memory_enabled", False):
            return ""

        vector = self.embedding_provider.embed(
            text=query,
            model_name=self.embedding_model_name,
            task="RETRIEVAL_QUERY",
            output_dimensionality=self.embedding_size,
        )
        total_limit = agent_config.get("total_memories_to_retrieve", 20)

        episodic_weight = agent_config.get("agent_episodic_weight", 0.0)
        working_weight = agent_config.get("agent_working_weight", 0.0)
        semantic_weight = agent_config.get("agent_semantic_weight", 0.0)

        now = datetime.now(timezone.utc)
        episodic_ts = (now - timedelta(hours=24)).timestamp()
        working_ts = (now - timedelta(days=30)).timestamp()

        tasks = []
        if episodic_weight > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.episodic_collection,
                    query=vector,
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=episodic_ts),
                            )
                        ]
                    ),
                    limit=int(total_limit * episodic_weight),
                )
            )

        if working_weight > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.working_collection,
                    query=vector,
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=working_ts, lt=episodic_ts),
                            )
                        ]
                    ),
                    limit=int(total_limit * working_weight),
                )
            )

        if semantic_weight > 0:
            tasks.append(
                self.client.query_points(
                    collection_name=self.semantic_collection,
                    query=vector,
                    limit=int(total_limit * semantic_weight),
                )
            )

        if not tasks:
            return ""

        responses = await asyncio.gather(*tasks)

        seen_ids = set()
        unique_texts: List[str] = []
        for response in responses:
            if response:
                for point in response.points:
                    if point.id not in seen_ids:
                        seen_ids.add(point.id)
                        if point.payload and "text" in point.payload:
                            unique_texts.append(point.payload["text"])

        return "\n---\n".join(unique_texts) if unique_texts else ""

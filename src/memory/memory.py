"""Three-layer memory (config-driven, no hardcoding)."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from qdrant_client import AsyncQdrantClient, models

from src.apis.google_client import GoogleClient

logger = logging.getLogger(__name__)


class Memory:
    """Three-layer memory: episodic, working, semantic."""

    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_client: AsyncQdrantClient,
        google_client: GoogleClient,
    ):
        mem_cfg = config.get("memory", {})
        self.episodic = mem_cfg.get("episodic_collection", "episodic")
        self.working = mem_cfg.get("working_collection", "working")
        self.semantic = mem_cfg.get("semantic_collection", "semantic")
        self.client = qdrant_client
        self.google = google_client

    @classmethod
    async def create(cls, config: Dict[str, Any]) -> "Memory":
        """Factory: initialize Memory from config."""
        google = GoogleClient(config)

        mem_cfg = config.get("memory", {})
        qdrant = AsyncQdrantClient(
            url=mem_cfg.get("url", "http://192.168.122.40:6334"),
            timeout=int(mem_cfg.get("timeout", 30)),
        )

        instance = cls(config, qdrant, google)

        for coll in [instance.episodic, instance.working, instance.semantic]:
            await instance._ensure_collection(coll)

        return instance

    async def _ensure_collection(self, name: str) -> None:
        if await self.client.collection_exists(name):
            return

        vectors_config = {
            "embedding": models.VectorParams(
                size=self.google.size,
                distance=models.Distance.COSINE,
            )
        }

        await self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )

    async def retrieve_context(self, query: str) -> str:
        """Retrieve memories for agent."""
        vector = self.google.embed(query)
        results = []

        for limit, collection in [
            (10, self.episodic),
            (5, self.working),
            (5, self.semantic),
        ]:
            response = await self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            results.extend(response.points)

        seen = set()
        texts = []
        for point in results:
            if point.id not in seen:
                seen.add(point.id)
                if point.payload and "text" in point.payload:
                    texts.append(point.payload["text"])

        return "\n---\n".join(texts) if texts else ""

    async def add_memory(self, text_content: str) -> str:
        """Store agent response."""
        vector = self.google.embed(text_content)
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).timestamp()

        await self.client.upsert(
            collection_name=self.episodic,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"embedding": vector},
                    payload={"text": text_content, "timestamp": now},
                )
            ],
            wait=True,
        )
        return point_id

    async def add_to_semantic(
        self, text_content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Store to semantic layer."""
        vector = self.google.embed(text_content)
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).timestamp()

        payload = {"text": text_content, "timestamp": now}
        if metadata:
            payload.update(metadata)

        await self.client.upsert(
            collection_name=self.semantic,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"embedding": vector},
                    payload=payload,
                )
            ],
            wait=True,
        )
        return point_id

"""Three-layer memory system."""

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
        agent_name: str,
    ):
        mem_cfg = config.get("memory", {})
        self.agent_name = agent_name
        self.episodic = mem_cfg.get("episodic_collection", "episodic")
        self.working = mem_cfg.get("working_collection", "working")
        self.semantic = mem_cfg.get("semantic_collection", "semantic")

        self.client = qdrant_client
        self.google = google_client

    @classmethod
    async def create(cls, config: Dict[str, Any], agent_name: str) -> "Memory":
        """Initialize memory."""
        mem_cfg = config.get("memory", {})

        google = GoogleClient(
            api_key_env=mem_cfg.get("google_api_key_env", "GEMINI_API_KEY"),
            model=mem_cfg.get("embedding_model", "gemini-embedding-001"),
            size=int(mem_cfg.get("embedding_size", 3072)),
        )

        qdrant = AsyncQdrantClient(
            url=mem_cfg.get("url", "http://localhost:6333"),
            timeout=int(mem_cfg.get("timeout", 30)),
        )

        instance = cls(config, qdrant, google, agent_name)

        for coll in [instance.episodic, instance.working, instance.semantic]:
            await instance._ensure_collection(coll)

        return instance

    async def _ensure_collection(self, name: str) -> None:
        """Create collection if missing."""
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

        for field, field_type in [("timestamp", "float"), ("agent_name", "keyword")]:
            try:
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=field_type,
                )
            except:
                pass

    async def retrieve_context(self, query: str) -> str:
        """Retrieve memories for agent context."""
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
        """Store memory."""
        vector = self.google.embed(text_content)
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).timestamp()

        await self.client.upsert(
            collection_name=self.episodic,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"embedding": vector},
                    payload={
                        "text": text_content,
                        "timestamp": now,
                        "agent_name": self.agent_name,
                    },
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

        payload = {
            "text": text_content,
            "timestamp": now,
            "agent_name": self.agent_name,
        }
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

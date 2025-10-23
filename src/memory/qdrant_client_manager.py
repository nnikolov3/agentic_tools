# src/memory/qdrant_client_manager.py

"""
Purpose:
Centralized Qdrant client manager with timeout handling, gRPC support, and all optimizations from agentic_tools.toml.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, models

logger: logging.Logger = logging.getLogger(__name__)


class QdrantClientManager:
    """
    Manages Qdrant client with robust timeout and all performance optimizations.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with timeout and optimization config from TOML.
        """
        self.qdrant_url: str = config.get("qdrant_url", "http://localhost:6333")
        self.timeout: float = config.get("timeout", 60.0)
        hnsw_config = config.get("hnsw_config", {})
        optimizers_config = config.get("optimizers_config", {})
        wal_config = config.get("wal_config", {})
        quantization_config = config.get("quantization_config", {})

        # Dynamic threading for RV
        if hnsw_config.get("max_indexing_threads") == -1:
            hnsw_config["max_indexing_threads"] = os.cpu_count() or 0
        if optimizers_config.get("max_optimization_threads") == -1:
            optimizers_config["max_optimization_threads"] = os.cpu_count() or 0

        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config

        # Client with timeout for stable connections
        self.client = AsyncQdrantClient(url=self.qdrant_url, timeout=self.timeout)

    def get_client(self) -> AsyncQdrantClient:
        """Return the client instance."""
        return self.client

    async def ensure_collection_exists(
        self,
        collection_name: str,
        embedding_size: int,
        payload_indexes: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Ensure collection exists with optimized parameters.
        """
        try:
            if await self.client.collection_exists(collection_name=collection_name):
                logger.debug(f"Collection '{collection_name}' already exists.")
                return

            logger.info(
                f"Creating optimized collection '{collection_name}' with embedding size {embedding_size}."
            )
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                    on_disk=True,  # Memory optimization
                ),
                hnsw_config=models.HnswConfigDiff(**self.hnsw_config),
                optimizers_config=models.OptimizersConfigDiff(**self.optimizers_config),
                wal_config=models.WalConfigDiff(**self.wal_config),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=self.quantization_config.get("scalar_type", "int8"),
                        quantile=self.quantization_config.get("quantile", 0.99),
                        always_ram=self.quantization_config.get("always_ram", True),
                    )
                ),
            )
            logger.info(f"Created optimized collection '{collection_name}'.")

            if payload_indexes:
                for field_name, field_schema in payload_indexes:
                    await self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                    logger.info(
                        f"Created payload index '{field_name}' for '{collection_name}'."
                    )

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise RuntimeError(f"Collection creation failed: {e}") from e

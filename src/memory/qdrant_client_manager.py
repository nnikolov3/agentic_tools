# src/memory/qdrant_client_manager.py
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import AsyncQdrantClient, models

logger: logging.Logger = logging.getLogger(__name__)


class QdrantClientManager:
    QUANTIZATION_TYPE_MAP: Dict[str, Optional[models.ScalarType]] = {
        "int8": models.ScalarType.INT8,
        "none": None,
    }

    PayloadSchema = Union[str, Dict[str, Any]]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.qdrant_url: str = config.get("qdrant_url", "http://localhost:6333")
        self.timeout: int = int(config.get("timeout", 60))
        self.prefer_grpc: bool = bool(config.get("prefer_grpc", False))
        self.api_key: Optional[str] = config.get("api_key")
        self.grpc_options: Optional[Dict[str, Any]] = config.get("grpc_options")

        self.replication_factor: Optional[int] = config.get("replication_factor")
        self.write_consistency_factor: Optional[int] = config.get(
            "write_consistency_factor",
        )
        self.on_disk_payload: Optional[bool] = config.get("on_disk_payload")

        self.vector_name: Optional[str] = config.get("vector_name")
        self.vectors_config_raw: Optional[Union[Dict[str, Any], Any]] = config.get(
            "vectors_config"
        ) or config.get("vectors")
        self.sparse_vectors_config: Optional[Dict[str, Any]] = config.get(
            "sparse_vectors_config",
        )

        hnsw_config: Dict[str, Any] = config.get("hnsw_config", {})
        optimizers_config: Dict[str, Any] = config.get("optimizers_config", {})
        wal_config: Dict[str, Any] = config.get("wal_config", {})
        self.quantization_config: Dict[str, Any] = config.get("quantization_config", {})

        if hnsw_config.get("max_indexing_threads") == -1:
            hnsw_config["max_indexing_threads"] = os.cpu_count() or 1
        if optimizers_config.get("max_optimization_threads") == -1:
            optimizers_config["max_optimization_threads"] = os.cpu_count() or 1

        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config

        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
            api_key=self.api_key,
            grpc_options=self.grpc_options,
        )

    def get_client(self) -> AsyncQdrantClient:
        return self.client

    async def ensure_collection_exists(
        self,
        collection_name: str,
        embedding_size: int,
        payload_indexes: Optional[List[Tuple[str, PayloadSchema]]] = None,
    ) -> None:
        try:
            if await self.client.collection_exists(collection_name=collection_name):
                logger.debug(f"Collection '{collection_name}' already exists.")
                return

            logger.info(
                f"Creating optimized collection '{collection_name}' with embedding size {embedding_size}.",
            )
            await self._create_collection(collection_name, embedding_size)
            logger.info(f"Created optimized collection '{collection_name}'.")

            if payload_indexes:
                for field_name, field_schema in payload_indexes:
                    await self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                    logger.info(
                        f"Created payload index '{field_name}' for '{collection_name}'.",
                    )
        except Exception as error:
            logger.error(
                f"Failed to create or configure collection '{collection_name}': {error}",
            )
            raise RuntimeError(
                f"Collection creation failed for '{collection_name}'",
            ) from error

    @staticmethod
    def _to_vector_params(spec: Dict[str, Any]) -> models.VectorParams:
        dist = str(spec.get("distance", "COSINE")).upper()
        return models.VectorParams(
            size=int(spec["size"]),
            distance=getattr(models.Distance, dist),
            on_disk=bool(spec.get("on_disk", True)),
        )

    def _build_vectors_config(
        self,
        default_size: int,
    ) -> Union[models.VectorParams, Dict[str, models.VectorParams]]:
        if (
            isinstance(
                self.vectors_config_raw,
                dict,
            )
            and "size" not in self.vectors_config_raw
        ):
            return {
                name: self._to_vector_params(v)
                for name, v in self.vectors_config_raw.items()
            }
        if (
            isinstance(
                self.vectors_config_raw,
                dict,
            )
            and "size" in self.vectors_config_raw
        ):
            return self._to_vector_params(self.vectors_config_raw)
        return models.VectorParams(
            size=int(default_size),
            distance=models.Distance.COSINE,
            on_disk=True,
        )

    async def _create_collection(
        self,
        collection_name: str,
        embedding_size: int,
    ) -> None:
        q_type_str: str = str(self.quantization_config.get("type", "int8")).lower()
        q_enum = self.QUANTIZATION_TYPE_MAP.get(q_type_str)

        quantization_cfg = None
        if q_enum is not None:
            quantization_cfg = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=q_enum,
                    quantile=float(self.quantization_config.get("quantile", 0.99)),
                    always_ram=bool(self.quantization_config.get("always_ram", True)),
                ),
            )

        vectors_config = self._build_vectors_config(embedding_size)

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=self.sparse_vectors_config,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=(
                models.HnswConfigDiff(
                    **self.hnsw_config,
                )
                if self.hnsw_config
                else None
            ),
            optimizers_config=(
                models.OptimizersConfigDiff(
                    **self.optimizers_config,
                )
                if self.optimizers_config
                else None
            ),
            wal_config=(
                models.WalConfigDiff(
                    **self.wal_config,
                )
                if self.wal_config
                else None
            ),
            quantization_config=quantization_cfg,
        )

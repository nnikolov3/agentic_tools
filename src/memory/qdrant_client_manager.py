import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)


class QdrantClientManager:
    QUANTIZATION_TYPE_MAP = {
        "int8": models.ScalarType.INT8,
        "none": None,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self._init_connection_config(config)
        self._init_collection_config(config)
        self._init_indexing_config(config)
        self.client = None  # Will be initialized async

    async def initialize(self) -> None:
        """Async initialization of the Qdrant client"""
        self.client = await self._create_client_async()

    async def _create_client_async(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=self.qdrant_url,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
            api_key=self.api_key,
            grpc_options=self.grpc_options,
        )

    def _init_connection_config(self, config: Dict[str, Any]):
        mem_config = config.get("memory", config)
        self.qdrant_url = os.getenv(
            "QDRANT_URL", mem_config.get("qdrant_url", "http://192.168.122.40:6333")
        )
        self.timeout = int(mem_config.get("timeout", 60))
        self.prefer_grpc = bool(mem_config.get("prefer_grpc", True))
        self.api_key = None  # No API key as per requirements
        self.grpc_options = mem_config.get("grpc_options")

    def _init_collection_config(self, config: Dict[str, Any]):
        mem_config = config.get("memory", config)
        self.dense_vector_name = mem_config.get("dense_vector_name", "text-dense")
        self.sparse_vector_name = mem_config.get("sparse_vector_name", "text-sparse")
        self.vector_name = mem_config.get("vector_name", self.dense_vector_name)
        self.vectors_config = mem_config.get("vectors_config") or mem_config.get(
            "vectors"
        )
        self.sparse_vectors_config = mem_config.get("sparse_vectors_config")
        self.on_disk = True
        self.replication_factor = mem_config.get("replication_factor", 1)
        self.write_consistency_factor = mem_config.get("write_consistency_factor", 1)

    def _init_indexing_config(self, config: Dict[str, Any]):
        mem_config = config.get("memory", config)
        default_hnsw = {
            "m": 32,
            "ef_construct": 400,
            "max_indexing_threads": 16,
            "on_disk": True,
        }
        self.hnsw_config = self._normalize_config(
            mem_config.get("hnsw_config", default_hnsw)
        )
        self.optimizers_config = self._normalize_config(
            mem_config.get("optimizers_config", {})
        )
        self.wal_config = self._normalize_config(mem_config.get("wal_config", {}))
        self.quantization_config = mem_config.get("quantization_config", {})

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if config.get("max_indexing_threads") == -1:
            config["max_indexing_threads"] = os.cpu_count() or 0
        if config.get("max_optimization_threads") == -1:
            config["max_optimization_threads"] = os.cpu_count() or 0
        return config

    async def get_client(self) -> AsyncQdrantClient:
        """Async client getter that ensures initialization"""
        if self.client is None:
            await self.initialize()
        return self.client

    async def ensure_collection_exists(
        self,
        collection_name: str,
        embedding_size: int,
        payload_indexes: Optional[List[Tuple[str, Union[str, Dict[str, Any]]]]] = None,
    ) -> None:
        client = await self.get_client()

        try:
            if await client.collection_exists(collection_name=collection_name):
                logger.debug("Collection '%s' already exists", collection_name)
                return

            logger.info(
                "Creating collection '%s' with embedding size %d",
                collection_name,
                embedding_size,
            )
            await self._create_collection_async(client, collection_name, embedding_size)

            if payload_indexes:
                await self._create_payload_indexes_async(
                    client, collection_name, payload_indexes
                )

            logger.info("Created collection '%s'", collection_name)
        except Exception as error:
            logger.error("Failed to create collection '%s': %s", collection_name, error)
            raise RuntimeError(
                f"Collection creation failed for '{collection_name}'"
            ) from error

    async def _create_collection_async(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embedding_size: int,
    ) -> None:
        quantization_cfg = self._create_quantization_config()
        vectors_config = self._build_vectors_config(embedding_size)
        sparse_vectors_config = self._build_sparse_vectors_config()

        await client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=True,
            hnsw_config=(
                models.HnswConfigDiff(**self.hnsw_config) if self.hnsw_config else None
            ),
            optimizers_config=(
                models.OptimizersConfigDiff(**self.optimizers_config)
                if self.optimizers_config
                else None
            ),
            wal_config=(
                models.WalConfigDiff(**self.wal_config) if self.wal_config else None
            ),
            quantization_config=quantization_cfg,
        )

    async def _create_payload_indexes_async(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        payload_indexes: List[Tuple[str, Union[str, Dict[str, Any]]]],
    ) -> None:
        for field_name, field_schema in payload_indexes:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.info(
                "Created payload index '%s' for '%s'", field_name, collection_name
            )

    @staticmethod
    def _to_vector_params(spec: Dict[str, Any]) -> models.VectorParams:
        dist = str(spec.get("distance", "COSINE")).upper()
        return models.VectorParams(
            size=int(spec["size"]),
            distance=getattr(models.Distance, dist),
            on_disk=bool(spec.get("on_disk", True)),
        )

    def _build_vectors_config(
        self, default_size: int
    ) -> Union[models.VectorParams, Dict[str, models.VectorParams]]:
        if isinstance(self.vectors_config, dict) and "size" not in self.vectors_config:
            return {
                name: self._to_vector_params(v)
                for name, v in self.vectors_config.items()
            }

        if isinstance(self.vectors_config, dict) and "size" in self.vectors_config:
            return self._to_vector_params(self.vectors_config)

        return {
            self.dense_vector_name: models.VectorParams(
                size=int(default_size),
                distance=models.Distance.COSINE,
                on_disk=True,
            )
        }

    def _build_sparse_vectors_config(self) -> Dict[str, models.SparseVectorParams]:
        if isinstance(self.sparse_vectors_config, dict) and self.sparse_vectors_config:
            return {
                name: self._parse_sparse_params(params)
                for name, params in self.sparse_vectors_config.items()
            }
        return {self.sparse_vector_name: models.SparseVectorParams()}

    def _parse_sparse_params(self, params: Any) -> models.SparseVectorParams:
        if isinstance(params, models.SparseVectorParams):
            return params

        index = None
        modifier = None
        if isinstance(params, dict):
            idx = params.get("index")
            if isinstance(idx, dict):
                index = models.SparseIndexParams(**idx)
            mod = params.get("modifier")
            if mod is not None:
                mod_upper = str(mod).upper()
                modifier = getattr(models.Modifier, mod_upper, None)
        return models.SparseVectorParams(index=index, modifier=modifier)

    def _create_quantization_config(self) -> Optional[models.ScalarQuantization]:
        q_type_str = str(self.quantization_config.get("type", "int8")).lower()
        q_enum = self.QUANTIZATION_TYPE_MAP.get(q_type_str)
        if q_enum is None:
            return None

        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=q_enum,
                quantile=float(self.quantization_config.get("quantile", 0.99)),
                always_ram=bool(self.quantization_config.get("always_ram", False)),
            ),
        )

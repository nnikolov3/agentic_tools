"""
Purpose:
Manual ingestion pipeline with timeout handling and optimizations.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from qdrant_client.http.models import UpdateStatus
from qdrant_client import models
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

from src.memory.qdrant_client_manager import QdrantClientManager
from src.tools.api_tools import google_documents_api
from src.tools.shell_tools import ShellTools

logger = logging.getLogger(__name__)


def _create_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text for embedding."""
    if not text:
        return []
    chunks: List[str] = []
    start_index: int = 0
    while start_index < len(text):
        end_index: int = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks


class KnowledgeBankIngestor:
    """
    Ingestion with timeout handling and optimizations.
    """

    PROCESSED_HASH_FIELD: str = "processed_content_hash"

    def __init__(self, configuration: Dict[str, Any]):
        self.configuration = configuration
        self.ingestion_config: Dict[str, Any] = self.configuration.get(
            "knowledge_bank_ingestion", {}
        )
        self.memory_config: Dict[str, Any] = self.configuration.get("memory", {})

        # Load from your TOML
        self.source_directory: str = self.ingestion_config.get(
            "source_directory", "knowledge_bank"
        )
        self.supported_extensions: List[str] = self.ingestion_config.get(
            "supported_extensions", [".json", ".md", ".pdf"]
        )
        self.llm_processed_extensions: List[str] = self.ingestion_config.get(
            "llm_processed_extensions", [".pdf"]
        )
        self.prompt: str = self.ingestion_config.get("prompt", "")
        self.model: str = self.ingestion_config.get("model", "gemini-flash-latest")
        self.api_key_name: str = self.ingestion_config.get(
            "google_api_key_name", "GEMINI_API_KEY_KNOWLEDGE_INGESTION"
        )
        self.chunk_size: int = self.ingestion_config.get("chunk_size", 1024)
        self.chunk_overlap: int = self.ingestion_config.get("chunk_overlap", 200)
        self.qdrant_batch_size: int = self.ingestion_config.get(
            "qdrant_batch_size", 32
        )  # Reduced for stability

        self.collection_name: str = self.memory_config.get(
            "knowledge_bank", "knowledge-bank"
        )
        embedding_model_name = self.memory_config.get(
            "embedding_model", "mixedbread-ai/mxbai-embed-large-v1"
        )
        self.device: str = self.memory_config.get("device", "cpu")
        self.embedder = TextEmbedding(
            model_name=embedding_model_name, device=self.device
        )

        try:
            dummy_embeddings = self.embedder.embed(documents=["dummy text"])
            dummy_embedding = list(dummy_embeddings)[0]
            self.embedding_size: int = len(dummy_embedding)
            logger.info(
                f"Initialized embedder with dimension size: {self.embedding_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

        self.qdrant_manager = QdrantClientManager(self.memory_config)
        self.qdrant_client = self.qdrant_manager.get_client()

        self.semaphore = asyncio.Semaphore(1)  # Reduced for stability

        self.shell_tools = ShellTools("knowledge_bank_ingestion", configuration)
        self.files_to_process: List[Path] = self.shell_tools.get_files_by_extensions(
            self.source_directory, self.supported_extensions
        )

    async def run_ingestion(self) -> Dict[str, int]:
        logger.info(f"Starting ingestion from: {self.source_directory}")
        await self.qdrant_manager.ensure_collection_exists(
            collection_name=self.collection_name,
            embedding_size=self.embedding_size,
            payload_indexes=[
                (self.PROCESSED_HASH_FIELD, "keyword"),
                ("raw_file_hash", "keyword"),
            ],
        )

        tasks: List[asyncio.Task[Tuple[str, Path]]] = [
            asyncio.create_task(self._process_file(file_path))
            for file_path in self.files_to_process
        ]
        results: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)

        processed_count = sum(
            1
            for result in results
            if isinstance(result, tuple) and result[0] == "processed"
        )
        skipped_count = sum(
            1
            for result in results
            if isinstance(result, tuple) and result[0] == "skipped"
        )
        failed_count = len(results) - processed_count - skipped_count

        logger.info(
            f"Ingestion complete. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}."
        )

        return {
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    async def _get_text_content_for_embedding(self, file_path: Path) -> str:
        """Process file content for embedding."""
        file_extension = file_path.suffix.lower()
        if file_extension in self.llm_processed_extensions:
            logger.info(f"Processing '{file_path}' using LLM path.")
            return str(
                await google_documents_api(
                    self.model, self.api_key_name, self.prompt, file_path
                )
            )
        else:
            logger.info(f"Processing '{file_path}' using direct path.")
            raw_content = self.shell_tools.read_file_content_for_path(file_path)
            if file_extension == ".json":
                return self._flatten_json_content(raw_content, file_path)
            return raw_content

    def _flatten_json_content(self, raw_content: str, file_path: Path) -> str:
        """Flatten JSON for embedding."""
        try:
            json_data = json.loads(raw_content)
            flattened_items = [f"{key}: {value}" for key, value in json_data.items()]
            return ". ".join(flattened_items)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {file_path}: {e}")
            return raw_content

    async def _is_already_processed(self, raw_file_hash: str) -> bool:
        """Check if file is already processed."""
        try:
            existing_points, _ = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="raw_file_hash",
                            match=models.MatchValue(value=raw_file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return bool(existing_points)
        except Exception as e:
            logger.error(f"Failed to check hash {raw_file_hash}: {e}")
            return False

    def _generate_embeddings(self, chunks: List[str]) -> List[list[float]]:
        """Generate embeddings for chunks."""
        try:
            embeddings_generator = self.embedder.embed(documents=chunks)
            embeddings = [
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                for embedding in embeddings_generator
            ]
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.TimeoutException)),
        reraise=True,
    )
    async def _upsert_batch_with_retry(
        self, batch: List[PointStruct], file_path: Path
    ) -> None:
        """Upsert with retry logic."""
        operation_info = await self.qdrant_client.upsert(
            collection_name=self.collection_name, points=batch, wait=True
        )
        if operation_info.status != UpdateStatus.COMPLETED:
            raise RuntimeError(
                f"Upsert failed for {file_path}: {operation_info.status}"
            )

    async def _process_file(self, file_path: Path) -> Tuple[str, Path]:
        """Process single file."""
        async with self.semaphore:
            try:
                with open(file_path, "rb") as f:
                    raw_file_content = f.read()
                raw_file_hash = hashlib.sha256(raw_file_content).hexdigest()

                if await self._is_already_processed(raw_file_hash):
                    logger.info(f"Skipping processed file: {file_path}")
                    return "skipped", file_path

                text_to_embed = await self._get_text_content_for_embedding(file_path)
                if not text_to_embed or not text_to_embed.strip():
                    logger.warning(f"Empty content: {file_path}")
                    return "skipped", file_path

                chunks = _create_text_chunks(
                    text_to_embed, self.chunk_size, self.chunk_overlap
                )
                if not chunks:
                    logger.warning(f"No chunks for: {file_path}")
                    return "skipped", file_path

                logger.info(f"Generated {len(chunks)} chunks for {file_path}.")

                embeddings = self._generate_embeddings(chunks)

                points_to_upsert: List[PointStruct] = []
                for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                    processed_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    payload = {
                        "text_content": chunk,
                        "original_file_path": str(file_path),
                        "raw_file_hash": raw_file_hash,
                        self.PROCESSED_HASH_FIELD: processed_hash,
                        "chunk_id": i,
                    }
                    points_to_upsert.append(
                        PointStruct(
                            id=str(uuid.uuid4()), vector=vector, payload=payload
                        )
                    )

                for i in range(0, len(points_to_upsert), self.qdrant_batch_size):
                    batch = points_to_upsert[i : i + self.qdrant_batch_size]
                    await self._upsert_batch_with_retry(batch, file_path)

                logger.info(
                    f"Processed {len(points_to_upsert)} chunks for: {file_path}"
                )
                return "processed", file_path
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                return "failed", file_path

"""
Purpose:
This script provides a manual ingestion pipeline for populating the Qdrant "knowledge-bank" collection.
"""

import asyncio
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from src.tools.api_tools import google_documents_api
from src.tools.shell_tools import ShellTools
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeBankIngestor:
    def __init__(self, configuration: Dict[str, Any]):
        self.configuration = configuration
        self.ingestion_config = self.configuration.get("knowledge_bank_ingestion", {})

        self.source_directory: str = self.ingestion_config.get("source_directory", "knowledge_bank")
        self.supported_extensions: List[str] = self.ingestion_config.get("supported_extensions",
                                                                         [".pdf", ".json", ".md"])
        self.llm_processed_extensions: List[str] = self.ingestion_config.get("llm_processed_extensions", [".pdf"])
        self.prompt: str = self.ingestion_config.get(
            "prompt",
            "Provide a concise, structured summary of this document, highlighting key concepts, architecture details, and practical implications for computer organization and RISC-V design."
        )

        self.model: str = self.ingestion_config.get("model", "gemini-2.5-flash")
        self.api_key: str = self.ingestion_config.get("google_api_key_name", os.environ.get("GEMINI_API_KEY_KNOWLEDGE_INGESTION"))
        if not self.api_key:
            logger.info(
                "GEMINI_API_KEY_KNOWLEDGE_INGESTION must be set in environment or configuration.knowledge_bank_ingestion.api_key")

        memory_config = self.configuration.get("memory", {})
        self.qdrant_url: str = memory_config.get("qdrant_url", "http://localhost:6333")
        self.collection_name: str = memory_config.get("knowledge_bank", "knowledge-bank")
        self.qdrant_client = AsyncQdrantClient(url=self.qdrant_url)
        self.embedder = SentenceTransformer(
            memory_config.get("embedding_model", "mixedbread-ai/mxbai-embed-large-v1"))
        self.embedding_size = self.embedder.get_sentence_embedding_dimension()


        self.semaphore = asyncio.Semaphore(3)

        self.shell_tools = ShellTools("knowledge_bank_ingestion", configuration)
        self.files_to_process: List[Path] = self.shell_tools.get_files_by_extensions(
            self.source_directory,
            self.supported_extensions
        )

    async def run_ingestion(self) -> Dict[str, int]:
        logger.info(f"Starting knowledge bank ingestion from: {self.source_directory}")
        await self._ensure_collection_exists()
        logger.info(f"Found {len(self.files_to_process)} files to process.")

        tasks: List[asyncio.Task[Tuple[str, Path]]] = [
            asyncio.create_task(self._process_file(file_path)) for file_path in self.files_to_process
        ]
        results: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)

        processed_count = sum(1 for result in results if isinstance(result, tuple) and result[0] == "processed")
        skipped_count = sum(1 for result in results if isinstance(result, tuple) and result[0] == "skipped")
        failed_count = len(results) - processed_count - skipped_count

        logger.info(
            f"Ingestion complete. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}."
        )
        return {"processed": processed_count, "skipped": skipped_count, "failed": failed_count}

    async def _ensure_collection_exists(self) -> None:
        try:
            collection_exists: bool = await self.qdrant_client.collection_exists(
                collection_name=self.collection_name
            )
            if not collection_exists:
                logger.info(
                    f"Collection '{self.collection_name}' not found. Creating it with embedding size {self.embedding_size}."
                )
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_size, distance=Distance.COSINE
                    ),
                )
                await self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="processed_content_hash",
                    field_schema="keyword",
                )
                logger.info(f"Successfully created collection '{self.collection_name}' and indexed processed_content_hash.")
        except Exception as e:
            logger.error(f"Failed to create or verify Qdrant collection: {e}")
            raise

    def _embed_text(self, text: str) -> List[float]:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        embeddings = self.embedder.encode(sentences)
        vector = np.mean(embeddings, axis=0).tolist()
        return vector

    async def _get_text_content_for_embedding(self, file_path: Path) -> str:
        """
        Determines the processing path based on file extension and returns the text content.
        """
        file_extension = file_path.suffix.lower()

        # Path A: Route to LLM for specified file types
        if file_extension in self.llm_processed_extensions:
            logger.info(f"Processing '{file_path}' using LLM path.")
            return await google_documents_api(
                self.model, self.api_key, self.prompt, file_path
            )

        # Path B: Read raw text directly for all other supported types
        else:
            logger.info(f"Processing '{file_path}' using direct ingestion path.")
            return self.shell_tools.read_file_content_for_path(file_path)

    async def _process_file(self, file_path: Path) -> Tuple[str, Path]:
        async with self.semaphore:
            try:
                with open(file_path, "rb") as f:
                    raw_file_content = f.read()
                raw_file_hash = hashlib.sha256(raw_file_content).hexdigest()

                text_to_embed = await self._get_text_content_for_embedding(file_path)

                if not text_to_embed or not text_to_embed.strip():
                    logger.warning(f"Skipping empty or invalid content from {file_path}.")
                    return "skipped", file_path

                processed_content_hash = hashlib.sha256(text_to_embed.encode("utf-8")).hexdigest()

                # Idempotency strategy:
                # For LLM-processed files (e.g., PDFs), the hash for existence check is based on the *processed text*.
                # This allows re-ingestion if the LLM model or prompt changes, generating new content.
                # For directly ingested files (e.g., MD, JSON), the hash for existence check is based on the raw content,
                # preventing re-ingestion unless the file content itself changes.
                existing_points = await self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="processed_content_hash",
                                match=models.MatchValue(value=processed_content_hash)
                            )
                        ]
                    )
                )
                if existing_points[0]:
                    logger.info(f"Skipping already processed file: {file_path}")
                    return "skipped", file_path

                vector = self._embed_text(text_to_embed)
                if not vector:
                    logger.warning(f"Skipping file due to empty vector: {file_path}")
                    return "skipped", file_path

                payload = {
                    "text_content": text_to_embed,
                    "original_file_path": str(file_path),
                    "raw_file_hash": raw_file_hash,
                    "processed_content_hash": processed_content_hash,
                }
                point_id = str(uuid.uuid4())

                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
                logger.info(f"Successfully processed and stored: {file_path}")
                return "processed", file_path

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
                return "failed", file_path

# src/scripts/ingest_knowledge_bank.py
"""
Purpose:
This module provides a robust and concurrent pipeline for ingesting documents
from a specified knowledge bank directory into a Qdrant vector database.

Core Functionality:
- Discovers files in a source directory based on supported extensions.
- Extracts text content from various file types (PDF, JSON, Markdown).
- For specific file types (e.g., PDF), enhances extracted text with an LLM-generated summary.
- Checks for previously processed files using content hashes to prevent redundant work.
- Splits the processed text into manageable chunks.
- Generates vector embeddings for each text chunk using a FastEmbed model.
- Upserts the chunks, embeddings, and metadata as points into a Qdrant collection.
- Manages concurrency to avoid overwhelming system resources.
- Implements retry logic for robust database operations.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

import httpx
from fastembed import TextEmbedding
from pdfminer.high_level import extract_text
from qdrant_client import models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    )

from src.memory.qdrant_client_manager import QdrantClientManager
from src.tools.api_tools import google_documents_api
from src.tools.shell_tools import ShellTools

# Self-Documenting Code: Dedicated logger for this module.
logger = logging.getLogger(__name__)

# Constants for payload field names to avoid magic strings.
RAW_FILE_HASH_FIELD: Final[str] = "raw_file_hash"
PROCESSED_CONTENT_HASH_FIELD: Final[str] = "processed_content_hash"

# Type alias for content extraction functions.
ContentExtractor = Callable[[Path], Awaitable[str]]


@dataclass(frozen = True)
class IngestionConfig:
    """Configuration for the knowledge bank ingestion process."""

    source_directory: Path
    supported_extensions: tuple[str, ...]
    prompt: str
    model: str
    api_key_name: str
    chunk_size: int
    chunk_overlap: int
    qdrant_batch_size: int
    concurrency_limit: int


@dataclass(frozen = True)
class MemoryConfig:
    """Configuration for the memory (vector database) connection."""

    collection_name: str
    embedding_model_name: str
    device: str
    qdrant_config: dict[str, Any] = field(default_factory = dict)


def create_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits a text string into overlapping chunks.

    Args:
        text: The input text to be chunked.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        # If chunking is disabled, return the whole text as a single chunk.
        return [text]

    # Ensure the step size is at least 1 to prevent infinite loops.
    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: list[str] = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = min(text_length, start_index + chunk_size)
        chunks.append(text[start_index:end_index])
        start_index += step

    return chunks


class KnowledgeBankIngestor:
    """Orchestrates the ingestion of documents into the knowledge bank."""

    def __init__(self, configuration: dict[str, Any]) -> None:
        """Initializes the ingestor with configuration and required services.

        Args:
            configuration: The application configuration dictionary.

        Raises:
            ValueError: If essential configuration sections are missing.
            RuntimeError: If the embedding model fails to initialize.
        """
        ingestion_config_dict = configuration.get("knowledge_bank_ingestion")
        memory_config_dict = configuration.get("memory")

        if not isinstance(ingestion_config_dict, dict) or not isinstance(
            memory_config_dict, dict,
            ):
            raise ValueError(
                "Configuration must contain 'knowledge_bank_ingestion' and 'memory' sections.",
                )

        self.ingestion_config = self._load_ingestion_config(ingestion_config_dict)
        self.memory_config = self._load_memory_config(memory_config_dict)

        self.embedder = TextEmbedding(
            model_name = self.memory_config.embedding_model_name,
            device = self.memory_config.device,
            )
        self.embedding_size = self._get_embedding_size()

        self.qdrant_manager = QdrantClientManager(self.memory_config.qdrant_config)
        self.qdrant_client = self.qdrant_manager.get_client()
        self.shell_tools = ShellTools("knowledge_bank_ingestion", configuration)
        self.semaphore = asyncio.Semaphore(self.ingestion_config.concurrency_limit)

        self._content_extractors: dict[str, ContentExtractor] = {
            ".pdf" : self._extract_and_summarize_pdf,
            ".json": self._flatten_json_content,
            ".md"  : self._read_markdown_content,
            }

    async def run_ingestion(self) -> Counter[str]:
        """Executes the full ingestion workflow.

        Discovers files, processes them concurrently, and reports the results.

        Returns:
            A Counter object with counts for 'processed', 'skipped', and 'failed' files.
        """
        logger.info(
            f"Starting ingestion from: '{self.ingestion_config.source_directory}'",
            )
        await self._ensure_collection_exists()

        files_to_process = self.shell_tools.get_files_by_extensions(
            self.ingestion_config.source_directory,
            list(self.ingestion_config.supported_extensions),
            )

        if not files_to_process:
            logger.warning("No files found to process. Ingestion finished.")
            return Counter()

        tasks = [
            asyncio.create_task(self._process_file(file_path))
            for file_path in files_to_process
            ]
        results = await asyncio.gather(*tasks, return_exceptions = True)

        status_counts: Counter[str] = Counter()
        for result in results:
            if isinstance(result, str):
                status_counts[result] += 1
            else:
                status_counts["failed"] += 1
                if isinstance(result, Exception):
                    logger.error(
                        f"An unexpected error occurred in a task: {result}",
                        exc_info = result,
                        )
        logger.info(f"Ingestion complete. Results: {status_counts}")
        return status_counts

    @staticmethod
    def _load_ingestion_config(config: dict[str, Any]) -> IngestionConfig:
        """Loads and validates the ingestion-specific configuration."""
        return IngestionConfig(
            source_directory = Path(config.get("source_directory", "knowledge_bank")),
            supported_extensions = tuple(
                config.get("supported_extensions", [".json", ".md", ".pdf"]),
                ),
            prompt = config.get("prompt", ""),
            model = config.get("model", "gemini-flash-latest"),
            api_key_name = config.get(
                "google_api_key_name", "GEMINI_API_KEY_KNOWLEDGE_INGESTION",
                ),
            chunk_size = config.get("chunk_size", 1024),
            chunk_overlap = config.get("chunk_overlap", 200),
            qdrant_batch_size = config.get("qdrant_batch_size", 128),
            concurrency_limit = config.get("concurrency_limit", 5),
            )

    @staticmethod
    def _load_memory_config(config: dict[str, Any]) -> MemoryConfig:
        """Loads and validates the memory-specific configuration."""
        return MemoryConfig(
            collection_name = config.get("knowledge_bank", "knowledge-bank"),
            embedding_model_name = config.get(
                "embedding_model", "mixedbread-ai/mxbai-embed-large-v1",
                ),
            device = config.get("device", "cpu"),
            qdrant_config = config,
            )

    def _get_embedding_size(self) -> int:
        """Determines the embedding vector size from the initialized model."""
        try:
            dummy_embeddings = self.embedder.embed(documents = ["test"])
            first_embedding = next(iter(dummy_embeddings))
            size = len(first_embedding)
            logger.info(f"Initialized embedder with dimension size: {size}")
            return size
        except Exception as error:
            logger.exception("Failed to initialize embedding model.")
            raise RuntimeError("Failed to initialize embedding model") from error

    async def _ensure_collection_exists(self) -> None:
        """Ensures the target Qdrant collection exists with the correct schema."""
        await self.qdrant_manager.ensure_collection_exists(
            collection_name = self.memory_config.collection_name,
            embedding_size = self.embedding_size,
            payload_indexes = [
                (PROCESSED_CONTENT_HASH_FIELD, "keyword"),
                (RAW_FILE_HASH_FIELD, "keyword"),
                ],
            )

    async def _is_already_processed(self, raw_file_hash: str) -> bool:
        """Checks if a file with the given hash has already been processed."""
        try:
            result = await self.qdrant_client.count(
                collection_name = self.memory_config.collection_name,
                count_filter = models.Filter(
                    must = [
                        models.FieldCondition(
                            key = RAW_FILE_HASH_FIELD,
                            match = models.MatchValue(value = raw_file_hash),
                            ),
                        ],
                    ),
                exact = False,
                )
            return result.count > 0
        except Exception:
            logger.exception(
                f"Failed to check for existing hash '{raw_file_hash}'. Assuming not processed.",
                )
            return False

    def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of text chunks."""
        try:
            embeddings_generator = self.embedder.embed(documents = chunks)
            return [embedding.tolist() for embedding in embeddings_generator]
        except Exception as error:
            logger.exception("Failed to generate embeddings.")
            raise RuntimeError("Embedding generation failed") from error

    @retry(
        stop = stop_after_attempt(3),
        wait = wait_exponential(min = 4, max = 10),
        retry = retry_if_exception_type((httpx.ConnectTimeout, httpx.TimeoutException)),
        reraise = True,
        )
    async def _upsert_batch_with_retry(
        self, batch: list[models.PointStruct], file_path: Path,
        ) -> None:
        """Upserts a batch of points to Qdrant with an exponential backoff retry."""
        operation_info = await self.qdrant_client.upsert(
            collection_name = self.memory_config.collection_name,
            points = batch,
            wait = True,
            )
        status = getattr(operation_info, "status", None)
        if status and status not in {
            models.UpdateStatus.COMPLETED,
            models.UpdateStatus.ACKNOWLEDGED,
            }:
            raise RuntimeError(
                f"Qdrant upsert failed for a batch from '{file_path}' with status: {status}",
                )

    def _create_points_from_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        raw_file_hash: str,
        file_path: Path,
        ) -> list[models.PointStruct]:
        """Creates a list of Qdrant PointStruct objects from processed data."""
        points: list[models.PointStruct] = []
        vector_name: Optional[str] = getattr(self.qdrant_manager, "vector_name", None)

        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            payload = {
                "text_content"              : chunk,
                "original_file_path"        : str(file_path),
                RAW_FILE_HASH_FIELD         : raw_file_hash,
                PROCESSED_CONTENT_HASH_FIELD: chunk_hash,
                "chunk_id"                  : index,
                }
            # The vector payload structure can vary based on Qdrant configuration
            # (e.g., named vectors). This handles both cases.
            vector_payload: Any = {vector_name: vector} if vector_name else vector
            points.append(
                models.PointStruct(
                    id = str(uuid.uuid4()), vector = vector_payload, payload = payload,
                    ),
                )
        return points

    async def _extract_content_from_file(self, file_path: Path) -> str:
        """Extracts content from a file using the appropriate registered extractor."""
        extension = file_path.suffix.lower()
        extractor = self._content_extractors.get(extension)
        if not extractor:
            logger.warning(f"No content extractor found for extension '{extension}'.")
            return ""
        return await extractor(file_path)

    async def _extract_and_summarize_pdf(self, file_path: Path) -> str:
        """Extracts text from a PDF and prepends an LLM-generated summary."""
        logger.info(f"Extracting text from PDF: {file_path}")
        extracted_text = extract_text(file_path)
        logger.info(f"Generating LLM summary for: {file_path}")
        llm_summary = await google_documents_api(
            self.ingestion_config.model,
            self.ingestion_config.api_key_name,
            self.ingestion_config.prompt,
            file_path,
            )
        # Prepending the summary provides high-level context at the beginning.
        return f"{llm_summary}\n\n{extracted_text}"

    @staticmethod
    async def _flatten_json_content(file_path: Path) -> str:
        """Flattens a JSON object into a single string or returns raw content."""
        try:
            with file_path.open(encoding = "utf-8") as file:
                json_data = json.load(file)
            if isinstance(json_data, dict):
                flattened_items = [f"{key}: {value}" for key, value in
                                   json_data.items()]
                return ". ".join(flattened_items)

            logger.warning(
                f"JSON in '{file_path}' is not a dictionary; returning raw content.",
                )
            return str(json_data)
        except json.JSONDecodeError:
            logger.exception(f"Failed to decode JSON from '{file_path}'.")
            return ""
        except Exception:
            logger.exception(f"Error processing JSON file {file_path}.")
            return ""

    async def _read_markdown_content(self, file_path: Path) -> str:
        """Reads content from a Markdown file."""
        return self.shell_tools.read_file_content(file_path)

    async def _process_file(self, filePath: Path) -> str:
        """Core processing pipeline for a single file.

        Args:
            filePath: The path to the file to process.

        Returns:
            A status string: "processed", "skipped", or "failed".
        """
        try:
            async with self.semaphore:
                logger.info(f"Processing file: {filePath}")
                processed_content = await self._extract_content_from_file(filePath)

                if not processed_content:
                    logger.warning(f"No content extracted from: {filePath}")
                    return "skipped"

                raw_file_hash = hashlib.sha256(
                    processed_content.encode("utf-8"),
                    ).hexdigest()

                if await self._is_already_processed(raw_file_hash):
                    logger.info(f"Skipping already processed file: {filePath}")
                    return "skipped"

                chunks = create_text_chunks(
                    processed_content,
                    self.ingestion_config.chunk_size,
                    self.ingestion_config.chunk_overlap,
                    )
                logger.info(f"Generated {len(chunks)} chunks for {filePath}.")

                if not chunks:
                    logger.warning(f"No chunks generated for: {filePath}")
                    return "skipped"

                embeddings = self._generate_embeddings(chunks)
                points = self._create_points_from_chunks(
                    chunks, embeddings, raw_file_hash, filePath,
                    )

                batch_size = self.ingestion_config.qdrant_batch_size
                for i in range(0, len(points), batch_size):
                    batch = points[i: i + batch_size]
                    await self._upsert_batch_with_retry(batch, filePath)

                logger.info(
                    f"Successfully processed {len(points)} chunks for: {filePath}",
                    )
                return "processed"
        except Exception:
            logger.exception(f"Failed to process file: {filePath}")
            return "failed"

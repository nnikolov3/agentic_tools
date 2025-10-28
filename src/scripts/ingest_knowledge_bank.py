# src/scripts/ingest_knowledge_bank.py
"""Handles the ingestion of documents into the Qdrant vector database.

This script is responsible for discovering documents in a specified source
directory, processing them based on their file type, chunking the content,
generating embeddings, and upserting the resulting vectors into a Qdrant
collection. It is designed to be idempotent, using content hashes to prevent
re-processing of unchanged files.

The core component is the `KnowledgeBankIngestor`, which orchestrates the
entire pipeline. It supports various file formats like PDF, JSON, and Markdown,
employing specific content extraction strategies for each. For PDFs, it uses an
LLM to generate a summary, which is prepended to the extracted text to provide
richer context. The process is asynchronous and uses a semaphore to limit
concurrent file processing, ensuring controlled resource usage.
"""

# Standard Library Imports
import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional, cast

# Third-Party Library Imports
import httpx
from fastembed import TextEmbedding
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pdfminer.high_level import extract_text
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Local Application/Module Imports
from src.memory.qdrant_client_manager import QdrantClientManager
from src.tools.api_tools import google_documents_api
from src.tools.shell_tools import ShellTools

# Initialize logger for this module.
logger = logging.getLogger(__name__)

# Define constants for payload field names to ensure consistency.
RAW_FILE_HASH_FIELD: Final[str] = "raw_file_hash"
PROCESSED_CONTENT_HASH_FIELD: Final[str] = "processed_content_hash"

# Type alias for content extraction functions.
ContentExtractor = Callable[[Path], Awaitable[str]]


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration settings for the ingestion process."""

    source_directory: Path
    supported_extensions: tuple[str, ...]
    prompt: str
    model: str
    api_key_name: str
    chunk_size: int
    chunk_overlap: int
    qdrant_batch_size: int
    concurrency_limit: int


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration settings for the memory/vector database."""

    collection_name: str
    embedding_model_name: str
    device: str
    qdrant_config: dict[str, Any] = field(default_factory=dict)


def create_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits a text string into overlapping chunks using a simple sliding window.

    This function provides a basic, non-semantic chunking mechanism suitable for
    unstructured text where more advanced splitting (e.g., by section) is not
    applicable.

    Args:
        text: The input text to be chunked.
        chunk_size: The target size for each chunk.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        # If chunk_size is not positive, return the whole text as one chunk.
        return [text]

    text_length = len(text)
    chunks: list[str] = []
    start_index = 0
    # Calculate the step size, ensuring it's at least 1 to prevent infinite loops.
    chunk_step = max(1, chunk_size - max(0, chunk_overlap))

    while start_index < text_length:
        end_index = min(text_length, start_index + chunk_size)
        chunks.append(text[start_index:end_index])
        start_index += chunk_step
    return chunks


class KnowledgeBankIngestor:
    """Orchestrates the ingestion of knowledge documents into Qdrant."""

    def __init__(self, configuration: dict[str, Any]) -> None:
        """Initializes the ingestor with configuration and required services.

        Args:
            configuration: The main application configuration dictionary.

        Raises:
            ValueError: If essential configuration sections are missing.
            RuntimeError: If the embedding model fails to initialize.
        """
        ingestion_cfg = configuration.get("knowledge_bank_ingestion")
        memory_cfg = configuration.get("memory")
        if not isinstance(ingestion_cfg, dict) or not isinstance(memory_cfg, dict):
            raise ValueError(
                "Configuration must contain 'knowledge_bank_ingestion' and 'memory' sections."
            )

        self.ingestion_config = self._load_ingestion_config(ingestion_cfg)
        self.memory_config = self._load_memory_config(memory_cfg)

        self.embedder = TextEmbedding(
            model_name=self.memory_config.embedding_model_name,
            device=self.memory_config.device,
        )
        self.embedding_size = self._get_embedding_size()

        self.qdrant_manager = QdrantClientManager(self.memory_config.qdrant_config)
        self.qdrant_client = self.qdrant_manager.get_client()
        self.shell_tools = ShellTools("knowledge_bank_ingestion", configuration)
        self.semaphore = asyncio.Semaphore(self.ingestion_config.concurrency_limit)

        self._content_extractors: dict[str, ContentExtractor] = {
            ".pdf": self._extract_and_summarize_pdf,
            ".json": self._process_json_content,
            ".md": self._read_markdown_content,
        }

        # These will be resolved at runtime by inspecting the Qdrant collection.
        self.kb_dense_name: Optional[str] = None
        self.kb_sparse_name: Optional[str] = None

    async def run_ingestion(self) -> Counter[str]:
        """Executes the end-to-end ingestion workflow.

        This method discovers files, ensures the Qdrant collection exists,
        and processes each file concurrently based on the configured limit.

        Returns:
            A Counter object summarizing the results (e.g., processed, skipped, failed).
        """
        logger.info(
            f"Starting ingestion from: '{self.ingestion_config.source_directory}'"
        )
        await self._ensure_collection_exists()
        await self._resolve_kb_vector_names()

        files = self.shell_tools.get_files_by_extensions(
            self.ingestion_config.source_directory,
            list(self.ingestion_config.supported_extensions),
        )
        if not files:
            logger.warning("No files found to process. Ingestion finished.")
            return Counter()

        tasks = [asyncio.create_task(self._process_file(p)) for p in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        counts: Counter[str] = Counter()
        for result in results:
            if isinstance(result, str):
                counts[result] += 1
            else:
                counts["failed"] += 1
                if isinstance(result, Exception):
                    logger.error("Task failed with an exception", exc_info=result)
        logger.info(f"Ingestion complete. Results: {counts}")
        return counts

    @staticmethod
    def _load_ingestion_config(config: dict[str, Any]) -> IngestionConfig:
        """Loads and validates the ingestion-specific configuration."""
        return IngestionConfig(
            source_directory=Path(config.get("source_directory", "knowledge_bank")),
            supported_extensions=tuple(
                config.get("supported_extensions", [".json", ".md", ".pdf"])
            ),
            prompt=config.get("prompt", ""),
            model=config.get("model", "gemini-flash-latest"),
            api_key_name=config.get(
                "google_api_key_name", "GEMINI_API_KEY_KNOWLEDGE_INGESTION"
            ),
            chunk_size=config.get("chunk_size", 1024),
            chunk_overlap=config.get("chunk_overlap", 200),
            qdrant_batch_size=config.get("qdrant_batch_size", 128),
            concurrency_limit=config.get("concurrency_limit", 5),
        )

    @staticmethod
    def _load_memory_config(config: dict[str, Any]) -> MemoryConfig:
        """Loads and validates the memory-specific configuration."""
        return MemoryConfig(
            collection_name=config.get("knowledge_bank", "knowledge-bank"),
            embedding_model_name=config.get(
                "embedding_model", "mixedbread-ai/mxbai-embed-large-v1"
            ),
            device=config.get("device", "cpu"),
            qdrant_config=config,
        )

    def _get_embedding_size(self) -> int:
        """Determines the embedding vector dimension by running a test embed."""
        try:
            embedding_generator = self.embedder.embed(documents=["test"])
            vector = next(iter(embedding_generator))
            size = len(vector)
            logger.info(f"Embedder dimension successfully determined: {size}")
            return size
        except Exception as error:
            logger.exception("Failed to initialize embedding model.")
            raise RuntimeError("Embedding model initialization failed") from error

    async def _ensure_collection_exists(self) -> None:
        """Creates the Qdrant collection and payload indexes if they don't exist."""
        await self.qdrant_manager.ensure_collection_exists(
            collection_name=self.memory_config.collection_name,
            embedding_size=self.embedding_size,
            payload_indexes=[
                (PROCESSED_CONTENT_HASH_FIELD, "keyword"),
                (RAW_FILE_HASH_FIELD, "keyword"),
            ],
        )

    async def _resolve_kb_vector_names(self) -> None:
        """Inspects the live Qdrant collection to determine its vector names.

        This is crucial for compatibility with collections that use named dense
        vectors or a single unnamed vector. It makes the ingestor robust to
        different collection schemas.
        """
        info = await self.qdrant_client.get_collection(
            collection_name=self.memory_config.collection_name
        )
        params = info.config.params

        # Resolve dense vector name
        if isinstance(params.vectors, dict):
            preferred = getattr(self.qdrant_manager, "vector_name", None)
            if preferred and preferred in params.vectors:
                self.kb_dense_name = preferred
            else:
                self.kb_dense_name = next(iter(params.vectors.keys()))
        else:
            # This indicates a single, unnamed dense vector.
            self.kb_dense_name = ""

        # Resolve sparse vector name (for informational purposes)
        if (
            hasattr(params, "sparse_vectors")
            and isinstance(params.sparse_vectors, dict)
            and params.sparse_vectors
        ):
            self.kb_sparse_name = next(iter(params.sparse_vectors.keys()))
        else:
            self.kb_sparse_name = None

        logger.info(
            "Knowledge bank vectors resolved: "
            f"dense='{self.kb_dense_name}', sparse='{self.kb_sparse_name}'"
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(
            (UnexpectedResponse, TimeoutError, httpx.TimeoutException)
        ),
    )
    async def _exists_by_hash(self, raw_file_hash: str) -> bool:
        """Checks if a document with the given hash already exists in Qdrant."""
        try:
            response = await self.qdrant_client.query_points(
                collection_name=self.memory_config.collection_name,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=RAW_FILE_HASH_FIELD,
                            match=models.MatchValue(value=raw_file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                timeout=60,
            )
            points = getattr(response, "points", response)
            logger.debug(f"Exists check via query result: {points}")
            return bool(points)
        except Exception:
            logger.exception(
                f"Exists check via query failed for hash '{raw_file_hash}'."
            )
            return False

    def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a list of text chunks."""
        try:
            embedding_generator = self.embedder.embed(documents=chunks)
            return [embedding.tolist() for embedding in embedding_generator]
        except Exception as error:
            logger.exception("Embedding generation failed.")
            raise RuntimeError("Embedding generation failed") from error

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(
            (UnexpectedResponse, TimeoutError, httpx.TimeoutException)
        ),
    )
    async def _upsert_batch_with_retry(
        self, batch: list[models.PointStruct], file_path: Path
    ) -> None:
        """Upserts a batch of points to Qdrant with a robust retry mechanism."""
        operation_info = await self.qdrant_client.upsert(
            collection_name=self.memory_config.collection_name,
            points=batch,
            wait=True,
        )
        status = getattr(operation_info, "status", None)
        if status and status not in {
            models.UpdateStatus.COMPLETED,
            models.UpdateStatus.ACKNOWLEDGED,
        }:
            raise RuntimeError(
                f"Qdrant upsert status not OK for '{file_path}': {status}"
            )

    def _create_points_from_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        raw_file_hash: str,
        file_path: Path,
    ) -> list[models.PointStruct]:
        """Constructs Qdrant PointStruct objects from processed data."""
        points: list[models.PointStruct] = []

        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            payload = {
                "text_content": chunk,
                "original_file_path": str(file_path),
                RAW_FILE_HASH_FIELD: raw_file_hash,
                PROCESSED_CONTENT_HASH_FIELD: chunk_hash,
                "chunk_id": index,
            }

            # Build the vector payload based on the resolved name.
            # This handles both named and unnamed dense vector collections.
            if self.kb_dense_name is None:
                # This should not happen if _resolve_kb_vector_names was called.
                vector_payload: Any = vector
            elif self.kb_dense_name == "":
                vector_payload = vector  # For unnamed dense vectors.
            else:
                vector_payload = {self.kb_dense_name: vector}

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()), vector=vector_payload, payload=payload
                )
            )
        return points

    async def _extract_content_from_file(self, file_path: Path) -> str:
        """Dispatches to the correct content extractor based on file extension."""
        extension = file_path.suffix.lower()
        extractor = self._content_extractors.get(extension)
        if not extractor:
            logger.warning(f"No content extractor found for extension '{extension}'.")
            return ""
        return await extractor(file_path)

    async def _get_llm_summary(self, file_path: Path) -> str:
        logger.info(f"Generating LLM summary for: {file_path}")
        llm_summary = await google_documents_api(
            self.ingestion_config.model,
            self.ingestion_config.api_key_name,
            self.ingestion_config.prompt,
            file_path,
        )
        return cast(str, llm_summary)

    async def _extract_and_summarize_pdf(self, file_path: Path) -> str:
        """Extracts text from a PDF and prepends an LLM-generated summary."""
        logger.info(f"Extracting text from PDF: {file_path}")
        extracted_text = extract_text(file_path)
        logger.info(f"Generating LLM summary for: {file_path}")
        llm_summary = await self._get_llm_summary(file_path)
        return f"{llm_summary}\n\n{extracted_text}"

    async def _process_json_content(self, file_path: Path) -> str:
        """Reads a JSON file, optionally flattens its content, and prepends an LLM-generated summary."""
        try:
            with file_path.open(encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, dict):
                content = ". ".join(f"{key}: {value}" for key, value in data.items())
            else:
                content = str(data)
            llm_summary = await self._get_llm_summary(file_path)
            return f"{llm_summary}\n\n{content}"
        except Exception:
            logger.exception(f"Error processing JSON file {file_path}.")
            return ""

    async def _read_markdown_content(self, file_path: Path) -> str:
        """Reads the raw content of a Markdown file."""
        return self.shell_tools.read_file_content(file_path)

    async def _process_file(self, file_path: Path) -> str:
        """Runs the complete processing pipeline for a single file.

        This pipeline includes:
        1. Content extraction based on file type.
        2. Hashing the content for deduplication.
        3. Checking if the content already exists in Qdrant.
        4. Chunking the content (using specialized Markdown or basic splitting).
        5. Generating embeddings for the chunks.
        6. Creating Qdrant points and upserting them in batches.

        Args:
            file_path: The path to the file to process.

        Returns:
            A status string: "processed", "skipped", or "failed".
        """
        try:
            async with self.semaphore:
                logger.info(f"Processing file: {file_path}")

                processed_content = await self._extract_content_from_file(file_path)
                if not processed_content:
                    logger.warning(f"No content extracted from: {file_path}")
                    return "skipped"

                raw_file_hash = hashlib.sha256(
                    processed_content.encode("utf-8")
                ).hexdigest()

                if await self._exists_by_hash(raw_file_hash):
                    logger.info(f"Skipping already processed file: {file_path}")
                    return "skipped"

                if file_path.suffix.lower() == ".md":
                    headers_to_split_on = [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                        ("####", "Header 4"),
                    ]
                    md_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on,
                        strip_headers=False,
                    )
                    header_docs = md_splitter.split_text(processed_content)

                    recursive_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.ingestion_config.chunk_size,
                        chunk_overlap=self.ingestion_config.chunk_overlap,
                        separators=["\n\n", "\n", " ", ""],
                    )
                    sized_docs = recursive_splitter.split_documents(header_docs)
                    chunks = [doc.page_content for doc in sized_docs]
                else:
                    chunks = create_text_chunks(
                        processed_content,
                        self.ingestion_config.chunk_size,
                        self.ingestion_config.chunk_overlap,
                    )

                logger.info(f"Generated {len(chunks)} chunks for {file_path}.")
                if not chunks:
                    logger.warning(f"No chunks generated for: {file_path}")
                    return "skipped"

                embeddings = self._generate_embeddings(chunks)
                points = self._create_points_from_chunks(
                    chunks, embeddings, raw_file_hash, file_path
                )

                batch_size = self.ingestion_config.qdrant_batch_size
                for i in range(0, len(points), batch_size):
                    await self._upsert_batch_with_retry(
                        points[i : i + batch_size], file_path
                    )

                logger.info(
                    f"Successfully processed {len(points)} chunks for: {file_path}"
                )
                return "processed"

        except Exception:
            logger.exception(f"Failed to process file: {file_path}")
            return "failed"

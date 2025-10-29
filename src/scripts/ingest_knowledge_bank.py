# src/scripts/ingest_knowledge_bank.py
"""Handles the ingestion of documents into the Qdrant vector database.

This script is responsible for discovering documents in a specified source
directory, processing them based on their file type, chunking the content,
generating embeddings, and upserting the resulting vectors into a Qdrant
collection. It is designed to be idempotent, using content hashes to prevent
re-processing of unchanged files.

The core component is the `KnowledgeBankIngestor`, which orchestrates the
entire pipeline. It supports various file formats like PDF, JSON, and Markdown,
employing specific content extraction strategies for each. For PDFs and JSON,
it uses an LLM to generate a summary, which is prepended to the extracted text
to provide richer context. The process is asynchronous and uses semaphore to
limit concurrent file processing, ensuring controlled resource usage.
"""

# Standard Library Imports
import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Optional

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
from src.tools.api_tools import google_documents_api, google_text
from src.tools.shell_tools import ShellTools

# Initialize logger for this module.
logger = logging.getLogger(__name__)

# Define constants for payload field names to ensure consistency.
RAW_FILE_HASH_FIELD: Final[str] = "raw_file_hash"
PROCESSED_CONTENT_HASH_FIELD: Final[str] = "processed_content_hash"
TEXT_CONTENT_FIELD: Final[str] = "text_content"
ORIGINAL_FILE_PATH_FIELD: Final[str] = "original_file_path"
CHUNK_ID_FIELD: Final[str] = "chunk_id"

# Text cleaning constants
LIGATURES: Final[dict[str, str]] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
RE_NON_PRINTABLE: Final[re.Pattern[str]] = re.compile(r"[^\x20-\x7E\n\r\t]")
RE_MULTI_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Type alias for content extraction functions.
ContentExtractor = Callable[[Path], Awaitable[str]]


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration settings for the ingestion process."""

    source_directory: Path
    supported_extensions: list[str]
    prompt: str
    model: str
    api_key_name: str
    chunk_size: int
    chunk_overlap: int
    qdrant_batch_size: int
    concurrency_limit: int
    old_file_threshold_days: int


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration settings for the memory/vector database."""

    collection_name: str
    embedding_model_name: str
    device: str
    qdrant_config: dict[str, Any] = field(default_factory=dict)


def create_text_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Splits a text string into overlapping chunks using a simple sliding window.

    This function provides a basic, non-semantic chunking mechanism suitable for
    unstructured text where more advanced splitting (e.g., by section) is not
    applicable.

    Args:
        text: The input text to be chunked.
        chunk_size: The target size for each chunk. Must be a positive integer.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        logger.warning(
            "chunk_size must be positive. Returning the whole text as one chunk."
        )
        return [text]

    text_length = len(text)
    chunks: list[str] = []
    start_index = 0
    # The step size determines how much the window slides for the next chunk.
    # It's clamped to a minimum of 1 to prevent infinite loops if overlap >= chunk_size.
    chunk_step = max(1, chunk_size - chunk_overlap)

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
        # Lazily initialize embedding_size to avoid blocking call in constructor.
        self.embedding_size: Optional[int] = None

        self.qdrant_manager = QdrantClientManager(self.memory_config.qdrant_config)
        self.qdrant_client = self.qdrant_manager.get_client()
        self.shell_tools = ShellTools(
            "knowledge_bank_ingestion", configuration, Path(__file__).parent
        )
        self.semaphore = asyncio.Semaphore(self.ingestion_config.concurrency_limit)

        self._content_extractors: dict[str, ContentExtractor] = {
            ".pdf":  self._extract_and_summarize_pdf,
            ".json": self._process_json_content,
            ".md": self._read_markdown_content,
        }

        # These will be resolved at runtime by inspecting the Qdrant collection.
        self.kb_dense_name: Optional[str] = None
        self.kb_sparse_name: Optional[str] = None

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text by removing ligatures and non-printable characters."""
        if not text:
            return ""

        for ligature, replacement in LIGATURES.items():
            text = text.replace(ligature, replacement)

        text = RE_NON_PRINTABLE.sub("", text)
        text = RE_MULTI_WHITESPACE.sub(" ", text).strip()
        return text

    async def _initialize_embedder(self) -> None:
        """Initializes the embedding model and determines its vector dimension.

        This method is called asynchronously to avoid blocking the event loop
        during the potentially slow model loading process.

        Raises:
            RuntimeError: If the embedding model fails to initialize.
        """
        if self.embedding_size is not None:
            return

        logger.info("Initializing embedding model and determining vector size...")

        def _get_size() -> int:
            """Synchronous helper to perform the blocking embedding call."""
            try:
                # This is a blocking, potentially slow operation (downloads model).
                embedding_generator = self.embedder.embed(documents=["test"])
                vector = next(iter(embedding_generator))
                size = len(vector)
                logger.info("Embedder dimension successfully determined: %d", size)
                return size
            except (RuntimeError, ValueError, StopIteration) as error:
                logger.exception("Failed to initialize embedding model.")
                raise RuntimeError("Embedding generation failed") from error

        self.embedding_size = await asyncio.to_thread(_get_size)

    async def run_ingestion(self) -> Counter[str]:
        """Executes the end-to-end ingestion workflow.

        This method discovers files, ensures the Qdrant collection exists,
        and processes each file concurrently based on the configured limit.

        Returns:
            A Counter object summarizing the results (e.g., processed, skipped, failed).
        """
        logger.info(
            "Starting ingestion from: '%s'", self.ingestion_config.source_directory
        )
        await self._initialize_embedder()
        await self._ensure_collection_exists()
        await self._resolve_kb_vector_names()

        files = await self.shell_tools.get_files_by_extensions(
            self.ingestion_config.source_directory,
            self.ingestion_config.supported_extensions
        )
        logger.info("Found %d files to process.", len(files))
        if not files:
            logger.warning("No files found to process. Ingestion finished.")
            return Counter()

        tasks = [
            asyncio.create_task(self._process_file(file_path)) for file_path in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        counts: Counter[str] = Counter()
        for result in results:
            if isinstance(result, str):
                counts[result] += 1
            elif isinstance(result, Exception):
                counts["failed"] += 1
                logger.error("Task failed with an exception", exc_info=result)
            else:
                counts["failed"] += 1
                logger.error("Task failed with an unknown error: %s", result)

        logger.info("Ingestion complete. Results: %s", counts)
        return counts

    @staticmethod
    def _load_ingestion_config(config: dict[str, Any]) -> IngestionConfig:
        """Loads and validates the ingestion-specific configuration."""
        return IngestionConfig(
            source_directory=Path(config.get("source_directory", "knowledge_bank")),
            supported_extensions=config.get("supported_extensions", [".json", ".md", ".pdf"])
            ,
            prompt=config.get("prompt", ""),
            model=config.get("model", "gemini-flash-latest"),
            api_key_name=config.get(
                "google_api_key_name", "GEMINI_API_KEY_KNOWLEDGE_INGESTION"
            ),
            chunk_size=config.get("chunk_size", 1024),
            chunk_overlap=config.get("chunk_overlap", 256),
            qdrant_batch_size=config.get("qdrant_batch_size", 128),
            concurrency_limit=config.get("concurrency_limit", 2),
            old_file_threshold_days=config.get("old_file_threshold_days", 730),
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

    async def _ensure_collection_exists(self) -> None:
        """Creates the Qdrant collections and payload indexes if they don't exist."""
        if self.embedding_size is None:
            raise RuntimeError(
                "Embedding size must be initialized before creating collections."
            )
        for collection_name in [
            self.memory_config.collection_name,
        ]:
            await self.qdrant_manager.ensure_collection_exists(
                collection_name=collection_name,
                embedding_size=self.embedding_size,
                payload_indexes=[
                    (PROCESSED_CONTENT_HASH_FIELD, "keyword"),
                    (RAW_FILE_HASH_FIELD, "keyword"),
                ],
            )

    async def _resolve_kb_vector_names(self) -> None:
        """Inspects the live Qdrant collection to determine its vector names."""
        info = await self.qdrant_client.get_collection(
            collection_name=self.memory_config.collection_name
        )
        params = info.config.params

        if isinstance(params.vectors, dict):
            preferred = getattr(self.qdrant_manager, "vector_name", None)
            self.kb_dense_name = (
                preferred
                if preferred and preferred in params.vectors
                else next(iter(params.vectors.keys()))
            )
        else:
            self.kb_dense_name = ""  # Single, unnamed dense vector

        if (
            hasattr(params, "sparse_vectors")
            and isinstance(params.sparse_vectors, dict)
            and params.sparse_vectors
        ):
            self.kb_sparse_name = next(iter(params.sparse_vectors.keys()))
        else:
            self.kb_sparse_name = None

        logger.info(
            "Knowledge bank vectors resolved: dense='%s', sparse='%s'",
            self.kb_dense_name,
            self.kb_sparse_name,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(
            (UnexpectedResponse, httpx.TimeoutException, httpx.RequestError)
        ),
    )
    async def _exists_by_hash(self, raw_file_hash: str) -> bool:
        """Checks if a document with the given hash already exists in Qdrant."""
        try:
            response, _ = await self.qdrant_client.scroll(
                collection_name=self.memory_config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=RAW_FILE_HASH_FIELD,
                            match=models.MatchValue(value=raw_file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            logger.debug("Exists check via scroll result: %s", response)
            return bool(response)
        except (UnexpectedResponse, httpx.RequestError) as error:
            logger.warning(
                "Exists check via query failed for hash '%s'. Assuming not present. Error: %s",
                raw_file_hash,
                error,
            )
            return False

    async def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a list of text chunks."""
        if not chunks:
            return []

        def _embed() -> list[list[float]]:
            """Synchronous helper to perform the blocking embedding call."""
            try:
                embedding_generator = self.embedder.embed(documents=chunks)
                return [embedding.tolist() for embedding in embedding_generator]
            except (RuntimeError, ValueError) as error:
                logger.exception("Embedding generation failed.")
                raise RuntimeError("Embedding generation failed") from error

        return await asyncio.to_thread(_embed)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(
            (UnexpectedResponse, httpx.TimeoutException, httpx.RequestError)
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

    async def _upsert_summary(self, summary: str, file_path: Path) -> None:
        """Upserts a summary to the summaries collection."""
        if not summary:
            return

        embeddings = await self._generate_embeddings([summary])
        if not embeddings:
            logger.warning("Failed to generate embedding for summary of %s", file_path)
            return
        embedding = embeddings[0]

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={self.kb_dense_name: embedding} if self.kb_dense_name else embedding,
            payload={
                TEXT_CONTENT_FIELD: summary,
                ORIGINAL_FILE_PATH_FIELD: str(file_path),
                "ingestion_date": datetime.now(timezone.utc).isoformat(),
                "modification_date": datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            },
        )
        await self.qdrant_client.upsert(
            collection_name=self.memory_config.collection_name,
            points=[point],
            wait=True,
        )

    def _create_points_from_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        raw_file_hash: str,
        file_path: Path,
    ) -> list[models.PointStruct]:
        """Constructs Qdrant PointStruct objects from processed data."""
        if self.kb_dense_name is None:
            raise RuntimeError("Dense vector name not resolved before creating points.")

        points: list[models.PointStruct] = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            payload = {
                TEXT_CONTENT_FIELD: chunk,
                ORIGINAL_FILE_PATH_FIELD: str(file_path),
                RAW_FILE_HASH_FIELD: raw_file_hash,
                PROCESSED_CONTENT_HASH_FIELD: chunk_hash,
                CHUNK_ID_FIELD: index,
                "file_extension": file_path.suffix,
                "ingestion_date": datetime.now(timezone.utc).isoformat(),
                "modification_date": datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }

            vector_payload: models.VectorStruct = (
                {self.kb_dense_name: vector} if self.kb_dense_name else vector
            )

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
            logger.warning("No content extractor found for extension '%s'.", extension)
            return ""
        return await extractor(file_path)

    async def _get_llm_summary_for_pdf(self, file_path: Path) -> str:
        """Invokes the LLM API to summarize a PDF document."""
        logger.info("Generating LLM summary for PDF: %s", file_path)
        return await google_documents_api(
            self.ingestion_config.model,
            self.ingestion_config.api_key_name,
            self.ingestion_config.prompt,
            str(file_path),
        )

    async def _get_llm_summary_for_text(self, text: str, file_path: Path) -> str:
        """Invokes the LLM API to summarize a string of text."""
        logger.info("Generating LLM summary for text from: %s", file_path)
        return await google_text(
            self.ingestion_config.model,
            self.ingestion_config.api_key_name,
            self.ingestion_config.prompt,
            text,
        )

    async def _extract_and_summarize_pdf(self, file_path: Path) -> str:
        """Extracts text from a PDF and prepends an LLM-generated summary."""
        logger.info("Extracting text from PDF: %s", file_path)
        extracted_text = await asyncio.to_thread(extract_text, file_path)
        cleaned_text = self.clean_text(str(extracted_text))

        llm_summary = await self._get_llm_summary_for_pdf(file_path)
        cleaned_summary = self.clean_text(str(llm_summary))
        await self._upsert_summary(cleaned_summary, file_path)
        return f"{cleaned_summary}\n\n{cleaned_text}"

    async def _process_json_content(self, file_path: Path) -> str:
        """Reads JSON, flattens it, and prepends an LLM-generated summary."""

        def _read_json_sync(path: Path) -> Any:
            """Synchronous helper to read and parse a JSON file."""
            with path.open(encoding="utf-8") as file_handle:
                return json.load(file_handle)

        try:
            data = await asyncio.to_thread(_read_json_sync, file_path)

            content = (
                ". ".join(f"{key}: {value}" for key, value in data.items())
                if isinstance(data, dict)
                else str(data)
            )
            cleaned_content = self.clean_text(str(content))

            llm_summary = await self._get_llm_summary_for_text(
                cleaned_content, file_path
            )
            cleaned_summary = self.clean_text(str(llm_summary))

            await self._upsert_summary(cleaned_summary, file_path)

            return f"{cleaned_summary}\n\n{cleaned_content}"
        except (OSError, json.JSONDecodeError) as error:
            logger.exception("Error processing JSON file %s.", file_path)
            raise RuntimeError(f"Failed to process JSON file {file_path}") from error

    async def _read_markdown_content(self, file_path: Path) -> str:
        """Reads and cleans the raw content of a Markdown file."""
        raw_content = await self.shell_tools.read_file_content(file_path)
        return self.clean_text(str(raw_content))

    async def _chunk_document(self, content: str, file_extension: str) -> list[str]:
        """Selects the appropriate strategy to chunk document content."""
        if file_extension == ".md":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
            header_docs = await asyncio.to_thread(md_splitter.split_text, content)

            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.ingestion_config.chunk_size,
                chunk_overlap=self.ingestion_config.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            sized_docs = await asyncio.to_thread(
                recursive_splitter.split_documents, header_docs
            )
            return [doc.page_content for doc in sized_docs]

        return create_text_chunks(
            content,
            self.ingestion_config.chunk_size,
            self.ingestion_config.chunk_overlap,
        )

    async def _process_file(self, file_path: Path) -> str:
        """Runs the complete processing pipeline for a single file."""
        async with self.semaphore:
            logger.info("Processing file: %s", file_path)

            # Check if the file is old
            file_stat = await asyncio.to_thread(file_path.stat)
            modification_time = datetime.fromtimestamp(
                file_stat.st_mtime, tz=timezone.utc
            )
            age = datetime.now(timezone.utc) - modification_time
            if age > timedelta(days=self.ingestion_config.old_file_threshold_days):
                warning = (
                    f"Warning: This document is from {modification_time.year} and may be outdated. "
                    "Use for informational and research purposes only.\n\n"
                )
            else:
                warning = ""

            # 1. Extract and clean content based on file type.
            processed_content = await self._extract_content_from_file(file_path)
            if not processed_content:
                logger.warning("No content extracted from: %s", file_path)
                return "skipped"

            processed_content = warning + processed_content

            # 2. Hash content to check for duplicates.
            raw_file_hash = hashlib.sha256(
                processed_content.encode("utf-8")
            ).hexdigest()
            if await self._exists_by_hash(raw_file_hash):
                logger.info("Skipping already processed file: %s", file_path)
                return "skipped"

            # 3. Chunk the content using the appropriate strategy.
            chunks = await self._chunk_document(
                processed_content, file_path.suffix.lower()
            )
            logger.info("Generated %d chunks for %s.", len(chunks), file_path)
            if not chunks:
                logger.warning("No chunks generated for: %s", file_path)
                return "skipped"

            # 4. Generate embeddings for the chunks.
            embeddings = await self._generate_embeddings(chunks)

            # 5. Create Qdrant points from chunks and embeddings.
            points = self._create_points_from_chunks(
                chunks, embeddings, raw_file_hash, file_path
            )

            # 6. Upsert points to Qdrant in batches.
            batch_size = self.ingestion_config.qdrant_batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await self._upsert_batch_with_retry(batch, file_path)

            logger.info(
                "Successfully processed %d chunks for: %s", len(points), file_path
            )
            return "processed"

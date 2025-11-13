"""
This module defines the DocumentProcessor for processing files and storing
them in the semantic memory layer.
"""

import logging
from typing import Any, Dict

from src.memory.memory import Memory
from src.utils.content_extractor import ContentExtractor
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles file ingestion by extracting text, cleaning it, chunking it,
    and storing it in the semantic memory layer.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        memory: Memory,
        providers: Dict[str, Any],  # Added providers dictionary
        content_extractor: ContentExtractor,
    ) -> None:
        """
        Initializes the processor with dependencies.
        """
        self.config = config
        self.memory = memory
        self.providers = providers  # Store providers
        self.content_extractor = content_extractor
        # Read chunk_size from the document_processor section of the config
        self.chunk_size = config.get("document_processor", {}).get("chunk_size", 500)

    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Processes a document into the semantic memory layer.
        """
        logger.info("Processing document: %s", document_path)

        raw_text = await self.content_extractor.extract_text_from_file(document_path)

        if not raw_text:
            logger.error("Text extraction failed or returned empty. Aborting.")
            return {"error": "Text extraction failed.", "source": document_path}

        cleaned_text = clean_text(raw_text)
        logger.info(
            "Extracted and cleaned text. Original length: %d, Cleaned length: %d",
            len(raw_text),
            len(cleaned_text),
        )

        chunks = self._chunk_text(cleaned_text, self.chunk_size)
        logger.info("Created %d chunks from document.", len(chunks))

        memory_ids = []
        for index, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": index,
                "source": document_path,
                "type": "document",
            }

            memory_id = await self.memory.add(
                text_content=chunk,
                collection=self.memory.semantic_collection,
                metadata=metadata,
            )
            memory_ids.append(memory_id)

        logger.info("Stored %d chunks to semantic memory collection.", len(memory_ids))
        return {
            "source": document_path,
            "characters_processed": len(cleaned_text),
            "chunks_created": len(chunks),
            "memory_ids": memory_ids,
        }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """
        Chunks a given text into smaller segments based on word count.
        """
        words = text.split()
        chunks = []
        current_chunk_words = []

        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) >= chunk_size:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks

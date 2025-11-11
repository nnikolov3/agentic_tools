"""Document ingestion orchestrator."""

import logging
from typing import Any, Dict

from src.apis.mistral_client import MistralClient
from src.memory.memory import Memory

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """PDF ingestion: OCR → chunk → embed → semantic storage."""

    def __init__(self, config: Dict[str, Any], memory: Memory) -> None:
        """Initialize with config and memory."""
        self.config = config
        self.memory = memory
        self.mistral = MistralClient(config)
        self.chunk_size = (
            config.get("agents", {}).get("ingestor", {}).get("chunk_size", 2000)
        )

    async def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Ingest PDF to semantic memory."""
        logger.info("Ingesting PDF: %s", pdf_path)

        result = self.mistral.extract_text(pdf_path)
        text = result["text"]
        logger.info("Extracted %d characters", len(text))

        chunks = self._chunk_text(text, self.chunk_size)
        logger.info("Created %d chunks", len(chunks))

        memory_ids = []
        for index, chunk in enumerate(chunks):
            memory_id = await self.memory.add_to_semantic(
                chunk,
                metadata={
                    "chunk_index": index,
                    "source": pdf_path,
                    "type": "document",
                },
            )
            memory_ids.append(memory_id)

        logger.info("Stored %d chunks to semantic", len(memory_ids))
        return {
            "source": pdf_path,
            "chars": len(text),
            "chunks": len(chunks),
            "memory_ids": memory_ids,
        }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Chunk text by words."""
        words = text.split()
        chunks = []
        current = []
        size = 0

        for word in words:
            current.append(word)
            size += len(word) + 1

            if size >= chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                size = 0

        if current:
            chunks.append(" ".join(current))

        return chunks

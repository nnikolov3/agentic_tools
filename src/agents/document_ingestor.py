"""Document ingestion orchestrator."""

import logging
from typing import Any, Dict

from src.apis.mistral_client import MistralClient
from src.memory.memory import Memory

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Ingest PDFs: OCR → chunk → embed → semantic storage."""

    def __init__(self, config: Dict[str, Any], memory: Memory):
        agent_cfg = config.get("agents", {}).get("ingestor", {})
        self.memory = memory
        self.mistral = MistralClient(
            api_key_env=agent_cfg.get("mistral_api_key_env", "MISTRAL_API_KEY")
        )

    async def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Ingest PDF: OCR → chunk → embed → semantic."""
        logger.info(f"Ingesting: {pdf_path}")

        text = self.mistral.extract_text(pdf_path)
        logger.info(f"Extracted {len(text)} chars")

        chunks = self._chunk_text(text, chunk_size=2000)
        logger.info(f"Created {len(chunks)} chunks")

        memory_ids = []
        for i, chunk in enumerate(chunks):
            memory_id = await self.memory.add_to_semantic(
                chunk,
                metadata={
                    "chunk_index": i,
                    "source": pdf_path,
                    "type": "document",
                },
            )
            memory_ids.append(memory_id)

        logger.info(f"Stored {len(memory_ids)} chunks to semantic")
        return {
            "source": pdf_path,
            "chars": len(text),
            "chunks": len(chunks),
            "memory_ids": memory_ids,
        }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
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

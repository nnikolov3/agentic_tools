import logging
from typing import Any, Dict, Optional, List
from enum import Enum

from pydantic import BaseModel, Field
from mistralai.extra import response_format_from_pydantic_model

from src.memory.memory import Memory
from src.utils.content_extractor import ContentExtractor
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)


# Pydantic models for Mistral OCR annotations
class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class Image(BaseModel):
    image_type: ImageType = Field(
        ..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'."
    )
    description: str = Field(..., description="A description of the image.")


class Document(BaseModel):
    language: str = Field(
        ..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr')."
    )
    summary: str = Field(..., description="A summary of the document.")
    authors: list[str] = Field(..., description="A list of authors who contributed to the document.")


def _strip_image_data(data: Any) -> Any:
    """
    Recursively removes 'image_base64' fields from the annotation data.
    """
    if isinstance(data, dict):
        return {
            key: _strip_image_data(value)
            for key, value in data.items()
            if key != "image_base64"
        }
    if isinstance(data, list):
        return [_strip_image_data(item) for item in data]
    return data


class KnowledgeAugmentor:
    """
    Encapsulates the knowledge augmentation workflow.
    It processes agent responses for memory storage and knowledge augmentation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        memory: Memory,
        providers: Dict[str, Any],
        content_extractor: ContentExtractor,
    ) -> None:
        """
        Initializes the augmentor with its configuration, memory, and providers.
        """
        self.config = config
        self.memory = memory
        self.providers = providers
        self.content_extractor = content_extractor

        # Read configuration values
        augmentation_config = self.config.get("knowledge_augmentor", {})
        self.augmentation_provider_name = augmentation_config.get(
            "augmentation_provider"
        )
        self.augmentation_model_name = augmentation_config.get("augmentation_model")
        self.augmentation_system_prompt = augmentation_config.get(
            "augmentation_system_prompt"
        )
        self.augmentation_user_prompt = augmentation_config.get(
            "augmentation_user_prompt"
        )
        self.chunk_size = augmentation_config.get("chunk_size", 500)
        self.async_timeout: float = float(self.config.get("async_timeout", 30.0))
        self.temperature = augmentation_config.get("temperature", 0.3)

    async def run(self, document_path: str) -> Dict[str, Any]:
        """
        Executes the knowledge augmentation workflow for a given document.
        This method is for processing documents directly, not agent responses.
        """
        # 1. Define annotation formats
        doc_annotation_format = response_format_from_pydantic_model(Document)
        bbox_annotation_format = response_format_from_pydantic_model(Image)

        # 2. Extract text and annotations
        logger.info("Extracting text and annotations from document: %s", document_path)
        raw_text, annotations = await self.content_extractor.extract_text_from_file(
            document_path,
            document_annotation_format=doc_annotation_format,
            bbox_annotation_format=bbox_annotation_format,
        )

        if not raw_text:
            logger.error("Text extraction failed. Aborting.")
            exit(1)

        original_text = clean_text(raw_text)
        logger.info(
            "Extracted and cleaned text. Original length: %d, Cleaned length: %d",
            len(raw_text),
            len(original_text),
        )

        # 3. Generate alternate version using the configured provider
        if (
            self.augmentation_provider_name
            and self.augmentation_provider_name in self.providers
        ):
            provider = self.providers[self.augmentation_provider_name]
            logger.info(
                "Generating alternate version with examples using provider: %s",
                self.augmentation_provider_name,
            )

            system_instruction = self.augmentation_system_prompt
            user_content = self.augmentation_user_prompt.format(
                original_text=original_text
            )

            augmented_text = await provider.generate_text(
                model_name=self.augmentation_model_name,
                system_instruction=system_instruction,
                user_content=user_content,
                temperature=self.temperature,
            )
            augmented_text = clean_text(augmented_text)
        else:
            logger.warning(
                "No augmentation provider configured or available. Skipping text augmentation."
            )
            augmented_text = original_text

        if not original_text and not augmented_text:
            logger.error("No text to store. Exiting.")
            exit(1)

        # 4. Store in memory with annotations
        logger.info("Storing content in memory...")
        await self._store_document_content_in_memory(
            document_path, original_text, augmented_text, annotations
        )

        logger.info("Successfully augmented and stored document.")
        return {
            "original_text_length": len(original_text),
            "augmented_text_length": len(augmented_text),
        }

    async def process_agent_response(
        self, agent_name: str, agent_response: str, chat: Optional[str] = None
    ) -> None:
        """
        Processes an agent's response and stores it in the appropriate memory collections.
        """
        logger.info(
            "Processing response from agent '%s' for memory storage.", agent_name
        )

        await self.memory.add(
            text_content=agent_response,
            collection=self.memory.episodic_collection,
            metadata={
                "source_agent": agent_name,
                "type": "agent_response",
                "chat": chat,
            },
        )

        await self.memory.add(
            text_content=agent_response,
            collection=self.memory.working_collection,
            metadata={
                "source_agent": agent_name,
                "type": "agent_response_working",
                "chat": chat,
            },
        )

        response_chunks = self._chunk_text(agent_response)
        for i, chunk in enumerate(response_chunks):
            await self.memory.add(
                text_content=chunk,
                collection=self.memory.semantic_collection,
                metadata={
                    "source_agent": agent_name,
                    "chunk_index": i,
                    "type": "agent_response_semantic",
                    "chat": chat,
                },
            )
        logger.info("Agent response from '%s' stored in memory.", agent_name)

    async def _store_document_content_in_memory(
        self,
        document_path: str,
        original_text: str,
        augmented_text: str,
        annotations: Optional[Dict[str, Any]],
    ) -> None:
        """
        Helper to store document content (original and augmented) in memory.
        """
        clean_annotations = _strip_image_data(annotations)
        base_metadata = {"source": document_path}
        if clean_annotations:
            base_metadata.update(clean_annotations)

        # Chunk both original and augmented text to avoid payload size issues
        original_chunks = self._chunk_text(original_text)
        augmented_chunks = self._chunk_text(augmented_text)

        # Store augmented text in working memory (chunked)
        for i, chunk in enumerate(augmented_chunks):
            await self.memory.add(
                text_content=chunk,
                collection=self.memory.working_collection,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "type": "augmented_chunk",
                },
            )

        # Store original and augmented chunks in semantic memory
        for i, chunk in enumerate(original_chunks):
            await self.memory.add(
                text_content=chunk,
                collection=self.memory.semantic_collection,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "source_type": "original",
                },
            )

        for i, chunk in enumerate(augmented_chunks):
            await self.memory.add(
                text_content=chunk,
                collection=self.memory.semantic_collection,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "source_type": "generated_example",
                },
            )

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunks text by word count.
        """
        words = text.split()
        chunks = []
        current_chunk_words = []
        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) >= self.chunk_size:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        return chunks

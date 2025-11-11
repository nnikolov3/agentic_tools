"""Mistral OCR API with annotations."""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict

from mistralai import Mistral, DocumentURLChunk
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Document-level annotation schema."""

    title: str = Field(..., description="Document title")
    summary: str = Field(..., description="Brief document summary")
    key_topics: list[str] = Field(..., description="Main topics covered")


class ImageAnnotation(BaseModel):
    """BBox annotation schema for images."""

    image_type: str = Field(
        ..., description="Type of image (chart, diagram, photo, etc)"
    )
    short_description: str = Field(..., description="Brief description of the image")
    summary: str = Field(..., description="Detailed summary of the image content")


class MistralClient:
    """Mistral OCR client with annotations support."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize from config - reads API key and model settings."""
        ingestor_configuration = config.get("agents", {}).get("ingestor", {})
        self.model = ingestor_configuration.get("ocr_model", "mistral-ocr-latest")
        api_key_environment_variable = ingestor_configuration.get(
            "api_key_env", "MISTRAL_API_KEY"
        )

        api_key = os.getenv(api_key_environment_variable)
        if not api_key:
            raise ValueError(
                f"Missing environment variable: {api_key_environment_variable}"
            )

        self.client = Mistral(api_key=api_key)

    def extract_text(
        self,
        pdf_path: str,
        include_bbox_annotations: bool = False,
        include_document_annotation: bool = False,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """Extract text from PDF using Mistral OCR with optional annotations."""
        pdf_bytes = Path(pdf_path).read_bytes()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        kwargs = {
            "model": self.model,
            "document": DocumentURLChunk(
                document_url=f"data:application/pdf;base64,{base64_pdf}"
            ),
            "include_image_base64": include_images,
        }

        if include_bbox_annotations:
            kwargs["bbox_annotation_format"] = str(
                response_format_from_pydantic_model(ImageAnnotation)
            )

        if include_document_annotation:
            kwargs["document_annotation_format"] = str(
                response_format_from_pydantic_model(DocumentMetadata)
            )

        ocr_response = self.client.ocr.process(**kwargs)

        text = ""
        if hasattr(ocr_response, "pages"):
            for page in ocr_response.pages:
                if hasattr(page, "content"):
                    text += page.content + "\n"

        result = {"text": text}

        if include_bbox_annotations and hasattr(ocr_response, "pages"):
            bbox_annotations = []
            for page in ocr_response.pages:
                if hasattr(page, "bboxes"):
                    for bbox in page.bboxes:
                        if hasattr(bbox, "annotation"):
                            bbox_annotations.append(
                                {
                                    "page": (
                                        page.page_number
                                        if hasattr(page, "page_number")
                                        else None
                                    ),
                                    "annotation": bbox.annotation,
                                    "image_base64": (
                                        bbox.image_base64
                                        if include_images
                                        and hasattr(bbox, "image_base64")
                                        else None
                                    ),
                                }
                            )
            result["bbox_annotations"] = str(bbox_annotations)

        if include_document_annotation and hasattr(ocr_response, "document_annotation"):
            result["document_annotation"] = ocr_response.document_annotation

        return result

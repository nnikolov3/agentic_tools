"""Mistral API client: OCR only."""

import base64
import logging
import os
from pathlib import Path

from mistralai import Mistral

logger = logging.getLogger(__name__)


class MistralClient:
    """Mistral API client for document OCR."""

    def __init__(self, api_key_env: str = "MISTRAL_API_KEY"):
        """Initialize Mistral client."""
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {api_key_env}")

        self.client = Mistral(api_key=api_key)
        self.model = "mistral-ocr-latest"
        logger.info("MistralClient initialized")

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using Mistral OCR."""
        pdf_bytes = Path(pdf_path).read_bytes()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        response = self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            include_image_base64=False,
        )

        text = ""
        if hasattr(response, "pages"):
            for page in response.pages:
                if hasattr(page, "content"):
                    text += page.content + "\n"

        logger.info(f"Extracted {len(text)} chars from {pdf_path}")
        return text

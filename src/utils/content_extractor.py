"""
This module provides a ContentExtractor utility for intelligently extracting
text from different file types.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported plain text extensions
PLAIN_TEXT_EXTENSIONS: List[str] = [
    ".txt",
    ".md",
    ".py",
    ".go",
    ".rs",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".toml",
    ".sh",
]


class ContentExtractor:
    """
    A utility that selects the correct strategy for extracting text from a
    file based on its extension.
    """

    def __init__(self, config: Dict[str, Any], providers: Dict[str, Any]):
        """
        Initializes the extractor with configuration and available providers.
        """
        self.config = config
        self.providers = providers
        # Corrected access: config already represents the [content_extractor] section
        self.ocr_provider_name = self.config.get("ocr_provider")
        self.ocr_model_name = self.config.get("ocr_model")

    async def extract_text_from_file(
        self,
        file_path: str,
        document_annotation_format: Optional[Dict[str, Any]] = None,
        bbox_annotation_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Extracts text from a file, using OCR for PDFs and direct reads
        for plain text files. It returns the extracted text and any
        structured annotation data.

        Why: This strategy pattern ensures the right tool is used for the job,
        making extraction efficient and accurate. It is easily extensible to
        support new file types in a single location.
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if not path.exists():
            logger.error("File not found: %s", file_path)
            return "", None

        if extension == ".pdf":
            if self.ocr_provider_name and self.ocr_provider_name in self.providers:
                ocr_provider = self.providers[self.ocr_provider_name]
                logger.info(
                    "Using OCR provider '%s' to extract from PDF: %s",
                    self.ocr_provider_name,
                    file_path,
                )
                result = await ocr_provider.extract_text_from_pdf(
                    file_path,
                    model_name=self.ocr_model_name,
                    document_annotation_format=document_annotation_format,
                    bbox_annotation_format=bbox_annotation_format,
                )
                
                logger.debug("Raw OCR response: %s", json.dumps(result, indent=2))

                if document_annotation_format or bbox_annotation_format:
                    # When annotations are requested, return the full JSON response
                    # and extract the markdown text for the text content.
                    text_content = " ".join(
                        page.get("markdown", "")
                        for page in result.get("pages", [])
                        if "markdown" in page
                    )
                    return text_content, result

                # For basic OCR, extract the markdown text.
                if result and "pages" in result and isinstance(result["pages"], list):
                    all_text = [
                        page["markdown"]
                        for page in result["pages"]
                        if "markdown" in page
                    ]
                    return " ".join(all_text), None
                
                logger.warning("OCR response did not contain the expected 'pages' structure.")
                return "", None
            else:
                logger.warning(
                    "No OCR provider configured or available for PDF extraction. Skipping OCR."
                )
                return "", None

        if extension in PLAIN_TEXT_EXTENSIONS:
            logger.info("Reading plain text from file: %s", file_path)
            try:
                return path.read_text(encoding="utf-8"), None
            except Exception as e:
                logger.error("Failed to read text file %s: %s", file_path, e)
                return "", None

        logger.warning(
            "Unsupported file type '%s'. Attempting plain text read.", extension
        )
        try:
            return path.read_text(encoding="utf-8"), None
        except Exception as e:
            logger.error(
                "Fallback plain text read failed for file %s: %s", file_path, e
            )
            exit(1)

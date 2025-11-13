"""
This module consolidates all external API provider logic into a single,
authoritative location. It defines provider classes (e.g., GoogleProvider,
MistralProvider, CerebrasProvider) that encapsulate all interaction details
for their respective services, including text generation, embeddings, and OCR.

This centralized design ensures that provider-specific logic is not scattered
across the application, adhering to the Single Responsibility Principle and
making the system more maintainable and extensible.
"""

import asyncio
import base64
import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import httpx
from google.generativeai import types
from mistralai import Mistral
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
MAX_PAGES_PER_CHUNK = 8


class GoogleProvider:
    """
    A provider for all Google Generative AI services (Gemini).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        api_key_env = self.config.get("google_api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {api_key_env}")
        genai.configure(api_key=api_key)
        self.client = genai
        self._semaphore = asyncio.Semaphore(config.get("concurrent_requests", 2))

    async def generate_text(
        self,
        model_name: str,
        user_content: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generates text using a Gemini model."""
        async with self._semaphore:
            try:
                model = self.client.GenerativeModel(
                    model_name,
                    system_instruction=(
                        system_instruction if system_instruction else None
                    ),
                )
                contents = [{"role": "user", "parts": [{"text": user_content}]}]
                response = await model.generate_content_async(
                    contents,
                    generation_config=types.GenerationConfig(temperature=temperature),
                )
                return response.text
            except Exception as e:
                logger.error("GoogleProvider failed to generate text: %s", e)
                return ""

    async def generate_from_document(
        self, prompt: str, file_path: str, model_name: str
    ) -> str:
        """Generates text from a document (e.g., PDF)."""
        async with self._semaphore:
            try:
                resolved_path = pathlib.Path(file_path).resolve()
                response = await asyncio.to_thread(
                    self.client.GenerativeModel(model_name).generate_content,
                    contents=[
                        types.Part.from_uri(
                            uri=resolved_path.as_uri(), mime_type="application/pdf"
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                )
                return response.text
            except Exception as e:
                logger.error("GoogleProvider failed to generate from document: %s", e)
                return ""

    def embed(
        self,
        text: str,
        model_name: str,
        task: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: Optional[int] = None,
    ) -> List[float]:
        """Embeds a single string of text."""
        if not text:
            return []
        try:
            config = (
                {"output_dimensionality": output_dimensionality}
                if output_dimensionality
                else {}
            )
            result = self.client.embed_content(
                model=model_name, content=text, task_type=task, **config
            )
            return result["embedding"]
        except Exception as e:
            logger.error("GoogleProvider failed to embed text: %s", e)
            return []

    def embed_batch(
        self,
        texts: List[str],
        model_name: str,
        task: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        """Embeds a batch of text strings."""
        if not texts:
            return []
        try:
            config = (
                {"output_dimensionality": output_dimensionality}
                if output_dimensionality
                else {}
            )
            result = self.client.embed_content(
                model=model_name, content=texts, task_type=task, **config
            )
            return result["embedding"]
        except Exception as e:
            logger.error("GoogleProvider failed to embed batch of texts: %s", e)
            return [[] for _ in texts]


class MistralProvider:
    """
    A provider for all Mistral AI services, including OCR.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key_env = self.config.get("mistral_api_key_env", "MISTRAL_API_KEY")
        self.api_key = os.getenv(self.api_key_env)
        self.client = Mistral(api_key=self.api_key)

    @staticmethod
    def encode_pdf(pdf_path: str) -> str:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")

    @staticmethod
    def get_pdf_page_count(pdf_path: str) -> int:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)

    async def _process_chunk(
        self,
        pdf_path: str,
        model_name: str,
        page_chunk: List[int],
        document_annotation_format: Optional[Dict[str, Any]],
        bbox_annotation_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base64_pdf = self.encode_pdf(pdf_path)
        params = {
            "model": model_name,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            "pages": page_chunk,
            "include_image_base64": True,
        }
        if document_annotation_format:
            params["document_annotation_format"] = document_annotation_format
        if bbox_annotation_format:
            params["bbox_annotation_format"] = bbox_annotation_format

        ocr_response = self.client.ocr.process(**params)
        return json.loads(ocr_response.model_dump_json())

    async def extract_text_from_pdf(
        self,
        pdf_path: str,
        model_name: str,
        document_annotation_format: Optional[Dict[str, Any]] = None,
        bbox_annotation_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts text from a PDF using the Mistral OCR API, handling page limits
        for document annotations by chunking requests.
        """
        try:
            num_pages = self.get_pdf_page_count(pdf_path)

            if not document_annotation_format or num_pages <= MAX_PAGES_PER_CHUNK:
                page_list = list(range(num_pages))
                return await self._process_chunk(
                    pdf_path,
                    model_name,
                    page_list,
                    document_annotation_format,
                    bbox_annotation_format,
                )

            # Chunking logic for document annotations
            all_pages = list(range(num_pages))
            tasks = []
            for i in range(0, num_pages, MAX_PAGES_PER_CHUNK):
                chunk = all_pages[i : i + MAX_PAGES_PER_CHUNK]
                tasks.append(
                    self._process_chunk(
                        pdf_path,
                        model_name,
                        chunk,
                        document_annotation_format,
                        bbox_annotation_format,
                    )
                )

            chunk_results = await asyncio.gather(*tasks)

            # Merge the results
            merged_response = {
                "document_annotation": [],
                "pages": [],
            }
            for result in chunk_results:
                if "document_annotation" in result and result["document_annotation"]:
                    merged_response["document_annotation"].extend(result["document_annotation"])
                if "pages" in result and result["pages"]:
                    merged_response["pages"].extend(result["pages"])
            
            # If there's only one document annotation, flatten the list
            if len(merged_response["document_annotation"]) == 1:
                merged_response["document_annotation"] = merged_response["document_annotation"][0]

            return merged_response

        except Exception as e:
            logger.error("MistralProvider failed to extract text from PDF: %s", e)
            exit(1)


class CerebrasProvider:
    """
    A provider for Cerebras AI services.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        api_key_env = self.config.get("cerebras_api_key_env", "CEREBRAS_API_KEY")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key environment variable: {api_key_env}")

    async def generate_text(
        self,
        model_name: str,
        user_content: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generates text using a Cerebras model."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_content})
        payload = {
            "model": model_name,
            "stream": False,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": -1,
            "seed": 0,
            "top_p": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    CEREBRAS_API_URL, headers=headers, json=payload
                )
                response.raise_for_status()
            data = response.json()
            if data and "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            logger.warning("Cerebras received an empty response: %s", data)
            return ""
        except Exception as e:
            logger.error("Cerebras failed to generate text: %s", e)
            return ""

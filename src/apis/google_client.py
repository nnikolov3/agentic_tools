"""Google API client: embeddings only."""

import logging
import os
from typing import List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GoogleClient:
    """Google API client for embeddings (gemini-embedding-001)."""

    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        embedding_model: str = "gemini-embedding-001",
        embedding_size: int = 3072,
    ):
        """Initialize Google client."""
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {api_key_env}")

        self.client = genai.Client(api_key=api_key)
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        logger.info(f"GoogleClient initialized: {embedding_model}, {embedding_size}D")

    def embed(self, text: str) -> List[float]:
        """Embed text."""
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.embedding_size,
            ),
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

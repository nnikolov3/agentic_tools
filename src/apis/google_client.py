"""Google Gemini embeddings API (config-driven)."""

import logging
import os
from typing import Dict, Any, List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GoogleClient:
    """Gemini embeddings from config."""

    def __init__(self, config: Dict[str, Any]):
        mem_cfg = config.get("memory", {})
        api_key_env = mem_cfg.get("google_api_key_env", "GEMINI_API_KEY")
        self.model = mem_cfg.get("embedding_model", "gemini-embedding-001")
        self.size = int(mem_cfg.get("embedding_size", 3072))

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing: {api_key_env}")
        self.client = genai.Client(api_key=api_key)

    def embed(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.size),
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(model=self.model, contents=texts)
        return [list(e.values) for e in result.embeddings]

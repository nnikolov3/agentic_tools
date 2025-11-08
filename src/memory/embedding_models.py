from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import google.generativeai as genai
from fastembed import TextEmbedding
from mistralai.client import MistralClient


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def embedding_size(self) -> int: ...


import numpy as np

class GoogleEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = config.get("model", "models/embedding-001")
        # Prioritize 'embedding_size' from config, then 'output_dimensionality', then default to 3072
        self._embedding_size = config.get("embedding_size", config.get("output_dimensionality", 3072))

    def embed(self, text: str, task_type: Optional[str] = None) -> list[float]:
        embedding = genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type,
            output_dimensionality=self._embedding_size
        )["embedding"]
        
        # Normalize if not 3072 (as per Gemini API docs)
        if self._embedding_size != 3072:
            embedding = self._normalize_embedding(embedding)
            
        return embedding

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()


class MistralEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = MistralClient(api_key=api_key)
        self.model = config.get("model", "mistral-embed")
        self._cached_embedding_size: Optional[int] = None

    def embed(self, text: str) -> list[float]:
        return self.client.embeddings(model=self.model, input=[text]).data[0].embedding

    @property
    def embedding_size(self) -> int:
        if self._cached_embedding_size is None:
            # Determine embedding size with a dummy call, and cache it
            self._cached_embedding_size = len(self.embed("test"))
        return self._cached_embedding_size


class FastEmbedEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        self.model = TextEmbedding(
            model_name=config.get("model", "BAAI/bge-small-en-v1.5")
        )

    def embed(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()

    @property
    def embedding_size(self) -> int:
        # This is a mock value for testing purposes.
        # In a real scenario, you would get this from the model properties.
        return 384


def create_embedder(config: dict[str, Any]) -> Embedder:
    provider = config.get("provider")
    if provider == "google":
        return GoogleEmbedder(config)
    elif provider == "mistral":
        return MistralEmbedder(config)
    elif provider == "fastembed":
        return FastEmbedEmbedder(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

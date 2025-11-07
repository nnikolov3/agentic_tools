from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import google.generativeai as genai
from fastembed import TextEmbedding
from mistralai.client import MistralClient


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...


class GoogleEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = config.get("model", "models/embedding-001")

    def embed(self, text: str) -> list[float]:
        return genai.embed_content(model=self.model, content=text)["embedding"]


class MistralEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = MistralClient(api_key=api_key)
        self.model = config.get("model", "mistral-embed")

    def embed(self, text: str) -> list[float]:
        return self.client.embeddings(model=self.model, input=[text]).data[0].embedding


class FastEmbedEmbedder(Embedder):
    def __init__(self, config: dict[str, Any]) -> None:
        self.model = TextEmbedding(model_name=config.get("model", "BAAI/bge-small-en-v1.5"))

    def embed(self, text: str) -> list[float]:
        return self.model.embed(text)


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

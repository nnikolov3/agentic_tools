"""
Reranker component for enhancing search result relevance.

This module provides a Reranker class that uses a cross-encoder model to
re-score and sort documents based on their semantic relevance to a query.
By isolating this functionality, we keep the QdrantMemory class focused on
retrieval and storage, adhering to the Single Responsibility Principle.
"""
from __future__ import annotations

import logging
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    A wrapper for a cross-encoder model to rerank search results.
    
    This class is initialized with a configuration dictionary and lazily loads
    the model on the first use to optimize startup time.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initializes the Reranker with configuration.

        Args:
            config: A dictionary containing the reranker settings, typically
                    from the [reranker] section of the main config file.
        """
        self.enabled = config.get("enabled", True)
        self.model_name = config.get(
            "model_name", "jinaai/jina-reranker-v2-base-multilingual"
        )
        self.device = config.get("device", "cpu")
        self._model: TextCrossEncoder | None = None

    def _lazy_load_model(self) -> None:
        """
        Lazily loads the cross-encoder model on the first call to rerank.
        
        Why: This prevents loading a potentially large model into memory if
        reranking is disabled or not used, improving application startup
        performance and resource efficiency.
        """
        if self._model is None:
            try:
                self._model = TextCrossEncoder(
                    model_name=self.model_name, device=self.device
                )
                logger.info("Reranker model '%s' loaded on device '%s'.", self.model_name, self.device)
            except Exception as e:
                logger.error("Failed to load reranker model: %s", e)
                # Disable reranking if the model fails to load
                self.enabled = False

    def rerank(self, query: str, documents: list[str]) -> list[str]:
        """
        Reranks a list of documents based on their relevance to a query.

        If reranking is disabled or the model fails to load, it returns the
        original list of documents without modification.

        Args:
            query: The search query string.
            documents: A list of document strings to be reranked.

        Returns:
            A sorted list of documents, with the most relevant first.
        """
        if not self.enabled or not documents:
            return documents

        self._lazy_load_model()
        if self._model is None:
            return documents

        try:
            # The rerank method returns scores, not sorted documents.
            scores = self._model.rerank(query, documents)
            
            # Combine documents with their scores and sort
            ranked_pairs = sorted(
                zip(documents, scores), key=lambda item: item[1], reverse=True
            )
            
            # Return only the sorted documents
            return [doc for doc, _ in ranked_pairs]
        except Exception as e:
            logger.warning(
                "Reranking failed due to an error, returning original order: %s", e
            )
            return documents

"""
Qdrant integration for storing approver decisions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from qdrant_client import QdrantClient, models

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = object  # type: ignore
    models = object  # type: ignore


logger = logging.getLogger(__name__)

# Default vector size for all-MiniLM-L6-v2, used as a final fallback if size cannot be determined dynamically.
DEFAULT_VECTOR_SIZE = 384


class QdrantIntegration:
    """
    Handles integration with Qdrant for storing approver decisions.
    """

    def __init__(
        self,
        local_path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "approver_decisions",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_sizes: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize Qdrant integration.

        Args:
            local_path: Path to local Qdrant storage
            url: Qdrant server URL (for remote)
            api_key: API key for remote Qdrant server
            collection_name: Name of the collection to store decisions
            embedding_model: Embedding model to use
        """
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client is required for Qdrant integration. "
                "Install with: pip install qdrant-client[fastembed]"
            )

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model_sizes = model_sizes or {}

        # Initialize Qdrant client with gRPC
        if local_path:
            self.client = QdrantClient(path=local_path, prefer_grpc=True)
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
        else:
            # Default to in-memory for testing if no path or URL provided
            self.client = QdrantClient(":memory:", prefer_grpc=True)

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _get_vector_size(self) -> int:
        """
        Determine the vector size based on the embedding model.
        This addresses the 'No Magic Numbers' principle by dynamically
        determining the vector size from the configured model.
        """
        try:
            # Try to get the model's embedding dimension dynamically
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model)
            # Get the embedding dimension directly from the model
            # Note: Some models may not have this method, so we have a fallback
            embedding_size = model.get_sentence_embedding_dimension()
            if embedding_size is not None:
                return embedding_size

            # If the method doesn't work, return a default value
            return DEFAULT_VECTOR_SIZE
        except Exception:
            # If all else fails, return a default value
            # Use configured model sizes if available
            if self.model_sizes and self.embedding_model in self.model_sizes:
                return self.model_sizes[self.embedding_model]

            # Fallback to default value if we can't determine size
            return DEFAULT_VECTOR_SIZE

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [
                collection.name for collection in collections.collections
            ]

            if self.collection_name not in collection_names:
                # Create collection with vector configuration based on the embedding model
                vector_size = self._get_vector_size()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
                logger.info(
                    f"Created Qdrant collection: {self.collection_name} with vector size: {vector_size}"
                )
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
        except Exception as collection_error:
            logger.error(f"Error ensuring Qdrant collection: {collection_error}")
            raise

    def _create_decision_payload(
        self, decision_data: Dict[str, Any], combined_text: str
    ) -> Dict[str, Any]:
        """
        Helper method to create the payload for storing approver decisions in Qdrant.
        Addresses DRY principle by centralizing payload creation.
        """
        return {
            "decision": decision_data.get("decision", ""),
            "summary": decision_data.get("summary", ""),
            "positive_points": decision_data.get("positive_points", []),
            "negative_points": decision_data.get("negative_points", []),
            "required_actions": decision_data.get("required_actions", []),
            "decision_full_data": decision_data,
            "timestamp": decision_data.get("timestamp", ""),
            "user_chat": decision_data.get("user_chat", ""),
            "combined_text": combined_text,
        }

    def _create_patch_payload(
        self, patch_data: Dict[str, Any], combined_text: str
    ) -> Dict[str, Any]:
        """
        Helper method to create the payload for storing patches in Qdrant.
        """
        return {
            "patch_type": patch_data.get("patch_type", ""),
            "content": patch_data.get("content", ""),
            "description": patch_data.get("description", ""),
            "timestamp": patch_data.get("timestamp", ""),
            "user_chat": patch_data.get("user_chat", ""),
            "agent_name": patch_data.get("agent_name", ""),
            "patch_full_data": patch_data,
            "combined_text": combined_text,
        }

    def _store_document_from_content(
        self,
        data: Dict[str, Any],
        doc_id: str,
        create_payload_func,
        content_for_embedding: str,
    ) -> bool:
        """
        Private helper method to store a document in Qdrant with shared embedding logic using pre-formatted content.

        Args:
            data: The data to store
            doc_id: Unique identifier for this document
            create_payload_func: Function to create the appropriate payload
            content_for_embedding: Content string to use for embedding

        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Try local inference with the all-MiniLM-L6-v2 model
            try:
                from qdrant_client.models import Document

                # Create a document using the local embedding model
                documents = [
                    Document(text=content_for_embedding, model=self.embedding_model)
                ]

                # Upload the documents to Qdrant using local inference
                payload = create_payload_func(data, content_for_embedding)
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=documents,
                    ids=[doc_id],  # Use the original UUID string ID directly
                    payload=[payload],
                )
                logger.info(
                    f"Successfully stored document in Qdrant with local inference: {doc_id}"
                )
                return True
            except Exception as inference_error:
                logger.warning(
                    f"Local inference failed, attempting with pre-computed embeddings: {inference_error}"
                )

                # Fallback: try with manual embedding
                try:
                    # Import sentence transformers to compute embedding manually
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.embedding_model)
                    embedding = model.encode([content_for_embedding])[
                        0
                    ]  # Get the embedding as a list
                    embedding_list = embedding.tolist()

                    from qdrant_client.models import PointStruct

                    payload = create_payload_func(data, content_for_embedding)
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=doc_id,  # Use the original UUID string ID directly
                                vector=embedding_list,
                                payload=payload,
                            )
                        ],
                    )
                    logger.info(
                        f"Successfully stored document in Qdrant with manual embedding: {doc_id}"
                    )
                    return True
                except ImportError:
                    logger.error(
                        "sentence-transformers not available for manual embedding computation"
                    )
                    return False
                except Exception as manual_embedding_error:
                    logger.error(
                        f"Failed to store document using manual embedding: {manual_embedding_error}"
                    )
                    return False

        except Exception as storage_error:
            logger.error(f"Error storing document in Qdrant: {storage_error}")
            return False

    def store_approver_decision(
        self,
        decision_data: Dict[str, Any],
        decision_id: str,
        content_for_embedding: str,
    ) -> bool:
        """
        Store an approver decision in Qdrant.

        Args:
            decision_data: The decision data from the approver tool
            decision_id: Unique identifier for this decision (will be converted to int if needed)
            content_for_embedding: Content string to use for embedding
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Use the shared document storage logic
            return self._store_document_from_content(
                decision_data,
                decision_id,
                self._create_decision_payload,
                content_for_embedding,
            )

        except Exception as decision_storage_error:
            logger.error(
                f"Error preparing to store approver decision in Qdrant: {decision_storage_error}"
            )
            return False

    def store_patch(
        self, patch_data: Dict[str, Any], patch_id: str, content_for_embedding: str
    ) -> bool:
        """
        Store a patch in Qdrant.

        Args:
            patch_data: The patch data to store
            patch_id: Unique identifier for this patch (will be converted to int if needed)
            content_for_embedding: Content string to use for embedding

        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Use the shared document storage logic
            return self._store_document_from_content(
                patch_data, patch_id, self._create_patch_payload, content_for_embedding
            )

        except Exception as patch_storage_error:
            logger.error(
                f"Error preparing to store patch in Qdrant: {patch_storage_error}"
            )
            return False

    def _extract_payloads_from_result(self, search_result: Any) -> list:
        """
        Helper method to extract payloads from different types of search results.

        Args:
            search_result: Result from Qdrant search (can be QueryResponse or list)

        Returns:
            List of payloads from the search results
        """
        if hasattr(search_result, "points"):
            # It's a QueryResponse object with points attribute
            return [hit.payload for hit in search_result.points]
        elif isinstance(search_result, list):
            # It's a list of ScoredPoint objects
            result_list = []
            for hit in search_result:
                if hasattr(hit, "payload"):
                    result_list.append(hit.payload)
            return result_list
        else:
            # Unknown format
            return []

    def search_similar_decisions(self, query: str, limit: int = 5) -> list:
        """
        Search for similar decisions based on the query.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of similar decision payloads
        """
        try:
            from qdrant_client.models import Document

            query_doc = Document(text=query, model=self.embedding_model)

            search_result = self.client.query_points(
                collection_name=self.collection_name, query=query_doc, limit=limit
            )

            return [hit.payload for hit in search_result.points]

        except Exception as search_error:
            logger.error(
                f"Error searching for similar decisions in Qdrant: {search_error}"
            )
            # Fallback: try with manual embedding
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.embedding_model)
                query_embedding = model.encode([query])[0].tolist()

                # Use search method which returns different types depending on client version
                search_result_any: Any = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                )

                # Extract payloads from the result using helper method
                return self._extract_payloads_from_result(search_result_any)

            except ImportError:
                logger.error(
                    "sentence-transformers not available for manual embedding computation in search"
                )
                return []
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []

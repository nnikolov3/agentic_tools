# STEP 8: CORE QDRANT MEMORY CLASS
# File: src/memory/qdrant_memory.py

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import models
from src.memory.qdrant_client_manager import QdrantClientManager
from src.memory.embedding_models import create_embedder
from src.memory.rfm_calculator import RFMCalculator
from src.memory.models import (
    Memory, EpisodicMemory, SemanticMemory,
    MemoryMetadata, MemoryType, EventType
)

logger = logging.getLogger(__name__)


class QdrantMemory:
    """
    Core memory system implementing bio-inspired episodic/semantic memory.
    
    Based on:
    - MIT 2017: Simultaneous hippocampal-cortical encoding
    - Squire & Alvarez 1995: Systems consolidation
    - Cowan 2001: 4-chunk working memory capacity
    - Ebbinghaus 1885: Exponential forgetting curve
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_manager: QdrantClientManager
    ) -> None:
        self.config = config
        self.qdrant_manager = qdrant_manager
        self.client = qdrant_manager.client
        self.collection_name = qdrant_manager.collection_name
        
        # Initialize embedder
        embedding_config = config.get("memory", {}).get("embedding_model", {})
        self.embedder = create_embedder(embedding_config)
        
        # Initialize RFM calculator
        rfm_config = config.get("memory", {}).get("rfm_config", {})
        self.rfm_calculator = RFMCalculator(
            recency_half_life_days=rfm_config.get("recency_half_life_days", 30.0),
            frequency_max_accesses=rfm_config.get("frequency_max_accesses", 100),
            frequency_log_base=rfm_config.get("frequency_log_base", 10.0)
        )
        
        # Working memory buffer (4 chunks)
        working_memory_config = config.get("memory", {}).get("working_memory", {})
        self.working_memory_capacity = working_memory_config.get("capacity", 4)
        self.working_memory_buffer = []
        
        logger.info(f"QdrantMemory initialized with collection: {self.collection_name}")
    
    @classmethod
    async def create(
        cls,
        config: Dict[str, Any],
        qdrant_manager: QdrantClientManager
    ) -> "QdrantMemory":
        """
        Async factory method to create QdrantMemory.
        ALWAYS use this instead of __init__ directly.
        """
        await qdrant_manager.initialize()
        await qdrant_manager.create_collection_if_not_exists(qdrant_manager.collection_name)
        return cls(config, qdrant_manager)
    
    async def add_memory(self, memory: Memory) -> str:
        """
        Add memory to Qdrant with full RFM metadata.
        
        Steps:
        1. Calculate RFM scores (recency=1.0 for new memory)
        2. Generate dense embedding
        3. Build complete payload (20+ fields)
        4. Upsert to Qdrant
        5. Add to working memory buffer if capacity allows
        
        Args:
            memory: Memory object (EpisodicMemory or SemanticMemory)
        
        Returns:
            Memory ID (UUID string)
        """
        # Calculate RFM scores
        memory.metadata.recency_score = self.rfm_calculator.calculate_recency_score(
            created_at=memory.created_at
        )
        memory.metadata.frequency_score = self.rfm_calculator.calculate_frequency_score(
            access_count=memory.metadata.access_count
        )
        # importance_score already set by user or defaults to 0.5
        
        # Generate embedding
        embedding = self.embedder.embed(memory.text_content)
        
        # Build payload (20+ fields for complete memory trace)
        payload = {
            "memory_id": memory.id,
            "memory_type": memory.memory_type.value,
            "text_content": memory.text_content,
            "tags": memory.tags,
            "parent_memory_id": memory.parent_memory_id,
            "created_at": memory.created_at.timestamp(),
            "day_of_week": memory.created_at.strftime("%A"),
            "hour_of_day": memory.created_at.hour,
            
            # RFM scores
            "recency_score": memory.metadata.recency_score,
            "frequency_score": memory.metadata.frequency_score,
            "importance_score": memory.metadata.importance_score,
            "priority_score": memory.metadata.priority_score,
            
            # Access tracking
            "access_count": memory.metadata.access_count,
            "last_accessed_at": memory.metadata.last_accessed_at.timestamp() if memory.metadata.last_accessed_at else None,
        }
        
        # Type-specific fields
        if isinstance(memory, EpisodicMemory):
            payload.update({
                "event_type": memory.event_type.value,
                "context": memory.context,
                "agent_name": memory.agent_name,
            })
        elif isinstance(memory, SemanticMemory):
            payload.update({
                "domain": memory.domain,
                "confidence_score": memory.confidence_score,
                "source_memory_ids": memory.source_memory_ids,
            })
        
        # Upsert to Qdrant (single dense vector)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory.id,
                    vector={"dense": embedding},  # Named vector
                    payload=payload
                )
            ],
            wait=True
        )
        
        # Add to working memory buffer (FIFO, max 4 chunks)
        self._add_to_working_memory(memory)
        
        logger.info(f"Added memory {memory.id} (type={memory.memory_type}, priority={memory.metadata.priority_score:.3f})")
        return memory.id
    
    def _add_to_working_memory(self, memory: Memory) -> None:
        """
        Add memory to working memory buffer.
        Eviction policy: FIFO (oldest out when capacity=4 exceeded)
        Based on Cowan 2001: 4-chunk capacity limit.
        """
        self.working_memory_buffer.append(memory)
        if len(self.working_memory_buffer) > self.working_memory_capacity:
            evicted = self.working_memory_buffer.pop(0)
            logger.debug(f"Evicted {evicted.id} from working memory (FIFO)")
    
    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        min_priority_score: float = 0.0
    ) -> str:
        """
        Retrieve relevant memories for query.
        
        Steps:
        1. Generate query embedding
        2. Search Qdrant (cosine similarity)
        3. Filter by min_priority_score
        4. Sort by priority (RFM score)
        5. Format as context string
        
        Args:
            query: Search query text
            limit: Maximum memories to retrieve
            min_priority_score: Filter threshold (0.0 to 1.0)
        
        Returns:
            Formatted context string
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search Qdrant
        search_result = await self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_embedding),
            limit=limit * 2,  # Oversample for filtering
            with_payload=True
        )
        
        # Filter by priority score
        filtered_results = [
            point for point in search_result
            if point.payload.get("priority_score", 0.0) >= min_priority_score
        ]
        
        # Sort by priority (highest first)
        sorted_results = sorted(
            filtered_results,
            key=lambda p: p.payload.get("priority_score", 0.0),
            reverse=True
        )[:limit]
        
        # Format context
        context_lines = ["--- Retrieved Memories (RFM-Prioritized) ---"]
        for i, point in enumerate(sorted_results, 1):
            priority = point.payload.get("priority_score", 0.0)
            memory_type = point.payload.get("memory_type", "unknown")
            text = point.payload.get("text_content", "")
            context_lines.append(f"{i}. [{memory_type}] (priority={priority:.3f}): {text}")
        
        return "\n".join(context_lines)
    
    async def update_memory_access(self, memory_id: str) -> None:
        """
        Update access count and recalculate frequency score.
        Called automatically when memory is retrieved.
        
        Steps:
        1. Increment access_count
        2. Update last_accessed_at to now
        3. Recalculate frequency_score (logarithmic)
        4. Update payload in Qdrant
        
        Args:
            memory_id: Memory UUID to update
        """
        # Retrieve current payload
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id],
            with_payload=True
        )
        
        if not points:
            logger.warning(f"Memory {memory_id} not found for access update")
            return
        
        point = points[0]
        payload = point.payload
        
        # Increment access count
        access_count = payload.get("access_count", 0) + 1
        last_accessed_at = datetime.now(timezone.utc)
        
        # Recalculate frequency score
        frequency_score = self.rfm_calculator.calculate_frequency_score(access_count)
        
        # Recalculate priority score
        recency_score = payload.get("recency_score", 1.0)
        importance_score = payload.get("importance_score", 0.5)
        priority_score = self.rfm_calculator.calculate_priority_score(
            recency_score, frequency_score, importance_score
        )
        
        # Update payload
        payload.update({
            "access_count": access_count,
            "last_accessed_at": last_accessed_at.timestamp(),
            "frequency_score": frequency_score,
            "priority_score": priority_score
        })
        
        # Update in Qdrant
        await self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[memory_id],
            wait=True
        )
        
        logger.debug(f"Updated access for {memory_id}: count={access_count}, freq={frequency_score:.3f}, priority={priority_score:.3f}")
    
    async def get_memory_children(
        self,
        parent_id: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Retrieve all child memories of a parent.
        Enables hierarchical memory graphs.
        
        Args:
            parent_id: Parent memory ID
            limit: Max children to retrieve
        
        Returns:
            List of Memory objects
        """
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="parent_memory_id",
                    match=models.MatchValue(value=parent_id)
                )
            ]
        )
        
        scroll_result = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        memories = []
        for point in scroll_result[0]:
            mem = self._point_to_memory(point)
            if mem:
                memories.append(mem)
        
        return memories
    
    async def get_memory_tree(
        self,
        memory_id: str,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Build hierarchical tree from root memory.
        
        Args:
            memory_id: Root memory ID
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth (internal)
        
        Returns:
            Dictionary tree structure
        """
        # Retrieve root memory
        root_point = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False
        )
        
        if not root_point:
            return {}
        
        root_mem = self._point_to_memory(root_point[0])
        
        tree = {
            "memory_id": memory_id,
            "text_content": root_mem.text_content if root_mem else "",
            "memory_type": root_mem.memory_type.value if root_mem else "",
            "priority_score": root_mem.metadata.priority_score if root_mem else 0.0,
            "children": []
        }
        
        # Recursively get children
        if current_depth < max_depth:
            children = await self.get_memory_children(memory_id, limit=20)
            for child in children:
                subtree = await self.get_memory_tree(
                    child.id,
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
                if subtree:
                    tree["children"].append(subtree)
        
        return tree
    
    def _point_to_memory(self, point) -> Optional[Memory]:
        """Convert Qdrant point to Memory object"""
        if not point or not point.payload:
            return None
        
        payload = point.payload
        memory_type = payload.get("memory_type", "episodic")
        
        try:
            # Reconstruct metadata
            metadata = MemoryMetadata(
                recency_score=payload.get("recency_score", 1.0),
                frequency_score=payload.get("frequency_score", 0.0),
                importance_score=payload.get("importance_score", 0.5),
                access_count=payload.get("access_count", 0),
                last_accessed_at=datetime.fromtimestamp(payload.get("last_accessed_at", 0), tz=timezone.utc) if payload.get("last_accessed_at") else None
            )
            
            # Base fields
            base_args = {
                "id": payload.get("memory_id", str(point.id)),
                "text_content": payload.get("text_content", ""),
                "tags": payload.get("tags", []),
                "parent_memory_id": payload.get("parent_memory_id"),
                "created_at": datetime.fromtimestamp(payload.get("created_at", 0), tz=timezone.utc),
                "metadata": metadata
            }
            
            # Type-specific reconstruction
            if memory_type == "episodic":
                return EpisodicMemory(
                    **base_args,
                    event_type=EventType(payload.get("event_type", "user_interaction")),
                    context=payload.get("context", {}),
                    agent_name=payload.get("agent_name")
                )
            elif memory_type == "semantic":
                return SemanticMemory(
                    **base_args,
                    domain=payload.get("domain", ""),
                    confidence_score=payload.get("confidence_score", 0.5),
                    source_memory_ids=payload.get("source_memory_ids", [])
                )
            else:
                return Memory(
                    **base_args,
                    memory_type=MemoryType(memory_type)
                )
        except Exception as e:
            logger.error(f"Failed to convert point to memory: {e}")
            return None

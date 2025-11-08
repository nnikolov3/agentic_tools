# STEP 9: MEMORY PRUNING MODULE
# File: src/memory/prune_memories.py

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)


async def prune_memories(
    client: AsyncQdrantClient,
    collection_name: str,
    prune_days: int = 365,
    min_priority_score: float = 0.2,
    batch_size: int = 100,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Prune old, low-priority memories based on age + priority threshold.
    
    Scientific basis:
    - Ebbinghaus forgetting curve: Older memories decay
    - RFM scoring: Low-priority memories less valuable
    - Cost optimization: Prevents unbounded storage growth
    
    Args:
        client: Qdrant async client
        collection_name: Collection to prune
        prune_days: Delete memories older than this (default: 365)
        min_priority_score: Delete if priority < this (default: 0.2)
        batch_size: Points to delete per batch (default: 100)
        dry_run: If True, only count without deleting
    
    Returns:
        Dictionary with pruning statistics
    """
    cutoff_timestamp = (
        datetime.now(timezone.utc) - timedelta(days=prune_days)
    ).timestamp()
    
    # Build filter: old AND low priority
    prune_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="created_at",
                range=models.Range(lt=cutoff_timestamp)
            ),
            models.FieldCondition(
                key="priority_score",
                range=models.Range(lt=min_priority_score)
            )
        ]
    )
    
    # Count matching points
    count_result = await client.count(
        collection_name=collection_name,
        count_filter=prune_filter,
        exact=True
    )
    
    total_to_prune = count_result.count
    logger.info(
        f"Found {total_to_prune} memories to prune "
        f"(older than {prune_days} days, priority < {min_priority_score})"
    )
    
    if dry_run or total_to_prune == 0:
        return {
            "dry_run": dry_run,
            "total_to_prune": total_to_prune,
            "deleted": 0,
            "cutoff_days": prune_days,
            "min_priority": min_priority_score
        }
    
    # Scroll through matching points and delete in batches
    deleted_count = 0
    offset = None
    
    while True:
        scroll_result = await client.scroll(
            collection_name=collection_name,
            scroll_filter=prune_filter,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )
        
        points = scroll_result[0]
        offset = scroll_result[1]
        
        if not points:
            break
        
        # Extract IDs
        point_ids = [p.id for p in points]
        
        # Delete batch
        await client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True
        )
        
        deleted_count += len(point_ids)
        logger.info(f"Deleted batch of {len(point_ids)} memories (total: {deleted_count})")
        
        if offset is None:
            break
    
    logger.info(f"Pruning complete. Deleted {deleted_count} memories.")
    
    return {
        "dry_run": False,
        "total_to_prune": total_to_prune,
        "deleted": deleted_count,
        "cutoff_days": prune_days,
        "min_priority": min_priority_score
    }

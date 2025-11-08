from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class EventType(str, Enum):
    USER_INTERACTION = "user_interaction"
    TOOL_EXECUTION = "tool_execution"
    ERROR_EVENT = "error_event"
    SYSTEM_EVENT = "system_event"
    AGENT_DECISION = "agent_decision"


class MemoryMetadata(BaseModel):
    recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: Optional[datetime] = None

    @property
    def priority_score(self) -> float:
        return (
            self.recency_score * 0.3
            + self.frequency_score * 0.2
            + self.importance_score * 0.5
        )


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    text_content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    tags: list[str] = Field(default_factory=list)
    agent_name: Optional[str] = None

    def to_qdrant_payload(self) -> dict[str, Any]:
        payload = {
            "memory_id": str(self.id),
            "memory_type": self.memory_type.value,
            "text_content": self.text_content,
            "created_at": self.created_at.timestamp(),
            "updated_at": self.updated_at.timestamp(),
            "last_accessed_at": (
                self.metadata.last_accessed_at.timestamp()
                if self.metadata.last_accessed_at
                else None
            ),
            "recency_score": self.metadata.recency_score,
            "frequency_score": self.metadata.frequency_score,
            "importance_score": self.metadata.importance_score,
            "priority_score": self.metadata.priority_score,
            "access_count": self.metadata.access_count,
            "tags": self.tags,
            "agent_name": self.agent_name,
            "day_of_week": self.created_at.strftime("%A"),
            "hour_of_day": self.created_at.hour,
        }
        return payload


class EpisodicMemory(Memory):
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC, frozen=True)
    event_type: EventType = Field(default=EventType.USER_INTERACTION)
    context: dict[str, Any] = Field(default_factory=dict)
    parent_memory_id: Optional[str] = None

    def to_qdrant_payload(self) -> dict[str, Any]:
        payload = super().to_qdrant_payload()
        payload.update(
            {
                "event_type": self.event_type.value,
                "context": self.context,
                "parent_memory_id": self.parent_memory_id,
            }
        )
        return payload


class SemanticMemory(Memory):
    memory_type: MemoryType = Field(default=MemoryType.SEMANTIC, frozen=True)
    source_memory_ids: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    domain: Optional[str] = None

    def to_qdrant_payload(self) -> dict[str, Any]:
        payload = super().to_qdrant_payload()
        payload.update(
            {
                "source_memory_ids": self.source_memory_ids,
                "confidence_score": self.confidence_score,
                "domain": self.domain,
            }
        )
        return payload


class MemoryQuery(BaseModel):
    query_text: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    memory_types: list[MemoryType] = Field(default_factory=list)
    min_priority_score: Optional[float] = None
    tags: Optional[list[str]] = None
    agent_name: Optional[str] = None
    time_range_hours: Optional[int] = None

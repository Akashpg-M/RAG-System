from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


Metadata = Dict[str, Any]
Vector = List[float]


@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass
class ChildChunk:
    chunk_id: str
    document_id: str
    parent_id: str
    text: str
    token_count: int
    content_hash: str
    metadata: Metadata = field(default_factory=dict)

@dataclass
class RawDocument:
    document_id: str
    filename: str
    raw_text: str

    @classmethod
    def from_text(cls, filename: str, raw_text: str) -> "RawDocument":
        digest = hashlib.sha256(f"{filename}\0{raw_text}".encode("utf-8")).hexdigest()[:16]
        return cls(document_id=digest, filename=filename, raw_text=raw_text)


class IndexingStatus(str, Enum):
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING_DENSE = "INDEXING_DENSE"
    INDEXING_SPARSE = "INDEXING_SPARSE"
    INDEXING_GRAPH = "INDEXING_GRAPH"
    READY = "READY"
    DELETE_PENDING = "DELETE_PENDING"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETED = "DELETED"


@dataclass(frozen=True)
class DocumentUploadEvent:
    document_id: str
    source_uri: str
    version_id: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class DocumentDeletionEvent:
    document_id: str
    source_uri: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    version_id: str
    source_uri: str
    content_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Metadata = field(default_factory=dict)


@dataclass
class IngestionTask:
    task_id: str
    source_uri: str
    document_id: Optional[str] = None
    version_id: Optional[str] = None
    status: IndexingStatus = IndexingStatus.PENDING
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[IndexingStatus] = field(default_factory=list)


@dataclass
class RetrievalCandidate:
    chunk_id: str
    text: str
    parent_id: str = ""
    metadata: Metadata = field(default_factory=dict)
    retriever: str = ""
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    graph_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class SourceCitation:
    chunk_id: str
    source_uri: str
    document_id: str = ""
    version_id: str = ""
    parent_id: str = ""
    title: Optional[str] = None


@dataclass
class QueryResult:
    query: str
    answer: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    citations: List[SourceCitation] = field(default_factory=list)


@dataclass(frozen=True)
class QueryRepresentations:
    original_query: str
    rewritten_query: str
    hyde_document: str

    def as_dict(self) -> Dict[str, str]:
        return vars(self).copy()

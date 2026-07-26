"""Backward-compatible imports for Stage 0 consumers."""

from src.core.contracts import (
    ChildChunk,
    DocumentDeletionEvent,
    DocumentUploadEvent,
    DocumentVersion,
    IndexingStatus,
    IngestionTask,
    ParentChunk,
    QueryRepresentations,
    QueryResult,
    RawDocument,
    RetrievalCandidate,
    SourceCitation,
)
from src.infrastructure.cache import SQLiteEmbeddingCache

EmbeddingCache = SQLiteEmbeddingCache

__all__ = [
    "ChildChunk", "DocumentDeletionEvent", "DocumentUploadEvent", "DocumentVersion",
    "EmbeddingCache", "IndexingStatus", "IngestionTask", "ParentChunk", "QueryRepresentations",
    "QueryResult", "RawDocument", "RetrievalCandidate", "SourceCitation", "SQLiteEmbeddingCache",
]

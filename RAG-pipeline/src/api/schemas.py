from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.contracts import IndexingStatus


class RetrievalMode(str, Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: Optional[str] = Field(default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def bounded_metadata(cls, value: Dict[str, str]) -> Dict[str, str]:
        if len(value) > 8:
            raise ValueError("too many metadata filters")
        for item in value.values():
            if not item or len(item) > 128:
                raise ValueError("metadata filter values must contain 1 to 128 characters")
        return value


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    filters: Optional[QueryFilters] = None
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    conversation_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    stream: bool = False

    @field_validator("query")
    @classmethod
    def non_whitespace_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class SourceResponse(BaseModel):
    document_id: str
    version_id: str
    chunk_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    excerpt: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    graph_score: Optional[float] = None
    rrf_score: float
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_strategy: str
    sources: List[SourceResponse]
    model_version: str
    configuration_version: str
    trace_id: str
    empty_context: bool
    refused: bool
    publication_revision: Optional[int] = None
    graph_index_required: bool = False
    publication_degraded: bool = False


class UploadResponse(BaseModel):
    document_id: str
    version_id: str
    task_id: str
    status: IndexingStatus
    status_url: str
    upload: Optional[Dict[str, str]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    document_id: str
    version_id: str
    status: IndexingStatus
    status_history: List[IndexingStatus]
    error_code: Optional[str] = None


class DeleteResponse(BaseModel):
    document_id: str
    version_id: str
    status: IndexingStatus
    deleted_chunks: int
    already_deleted: bool


class HealthResponse(BaseModel):
    status: Literal["alive"] = "alive"


class DependencyStatus(BaseModel):
    name: str
    ready: bool
    required: bool


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: List[DependencyStatus]


class ErrorResponse(BaseModel):
    error: str
    message: str
    trace_id: str

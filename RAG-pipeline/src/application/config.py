from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Profile(str, Enum):
    TEST = "test"
    LOCAL = "local"
    BENCHMARK = "benchmark"
    AWS_DEMO = "aws_demo"


class ProviderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    dense_index: str = "qdrant"
    sparse_index: str = "sqlite"
    graph_index: str = "postgres"
    embeddings: str = "sentence_transformer"
    reranker: str = "cross_encoder"
    query_processor: str = "groq"
    answer_generator: str = "groq"
    cache: str = "sqlite"
    queue: str = "redis"
    object_storage: str = "local"
    document_repository: str = "postgres"
    task_repository: str = "postgres"


class StorageSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    qdrant_path: str = "./qdrant_local_data"
    qdrant_url: str = "http://127.0.0.1:6333"
    sparse_db_path: str = "./sparse_index.db"
    graph_db_path: str = "./graph_store.db"
    cache_db_path: str = "./embedding_cache.db"
    collection_name: str = "enterprise_kb"
    upload_path: str = "./uploaded_documents"
    control_db_path: str = "./control_plane.db"
    control_database_url: str = "postgresql://rag:rag-local-only@127.0.0.1:5432/rag_control"


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""


class PipelineSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_size: int = Field(default=256, gt=0)
    chunk_overlap: int = Field(default=30, ge=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    vector_batch_size: int = Field(default=100, gt=0)
    retrieval_top_k: int = Field(default=10, gt=0)
    rerank_pool_multiplier: int = Field(default=3, gt=0)
    graph_max_hops: int = Field(default=2, ge=0)
    provider_timeout_seconds: float = Field(default=30.0, gt=0)
    queue_wait_timeout_seconds: float = Field(default=300.0, gt=0)
    enable_hyde: bool = True
    enable_graph_extraction: bool = True
    enable_generation: bool = True
    graph_index_required: bool = False
    pipeline_version: str = "stage-4"
    parser_version: str = "docling-2"
    chunker_config_version: str = "semantic-v1"
    index_schema_version: str = "multi-index-v1"

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "PipelineSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key: str = ""
    auth_enabled: bool = True
    max_query_length: int = Field(default=4000, gt=0)
    max_top_k: int = Field(default=20, gt=0)
    default_top_k: int = Field(default=5, gt=0)
    max_request_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)
    max_extracted_characters: int = Field(default=2_000_000, gt=0)
    max_archive_entries: int = Field(default=2000, gt=0)
    max_archive_uncompressed_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    excerpt_characters: int = Field(default=500, gt=0, le=2000)
    rate_limit_requests: int = Field(default=120, gt=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)
    allowed_extensions: tuple[str, ...] = (".md", ".txt", ".pdf", ".docx")
    allowed_metadata_filters: tuple[str, ...] = ("category", "department", "language")
    config_version: str = "stage-2"

    @model_validator(mode="after")
    def validate_query_limits(self) -> "ApiSettings":
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")
        if self.max_top_k > 20:
            raise ValueError("max_top_k must not exceed the public API contract limit of 20")
        return self


class QueueSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: str = "redis"
    redis_url: str = "redis://127.0.0.1:6379/0"
    stream_name: str = "rag:ingestion"
    consumer_group: str = "rag-workers"
    worker_id: str = "worker-1"
    max_concurrency: int = Field(default=2, gt=0, le=64)
    poll_timeout_seconds: float = Field(default=5, gt=0, le=20)
    lease_duration_seconds: float = Field(default=120, gt=1)
    heartbeat_interval_seconds: float = Field(default=30, gt=0)
    visibility_timeout_seconds: int = Field(default=120, gt=1, le=43200)
    visibility_heartbeat_seconds: float = Field(default=30, gt=0)
    max_attempts: int = Field(default=5, gt=0)
    retry_min_seconds: float = Field(default=1, ge=0)
    retry_max_seconds: float = Field(default=300, gt=0)
    capacity: int = Field(default=10000, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)
    sqs_queue_url: str = ""
    sqs_dlq_url: str = ""

    @model_validator(mode="after")
    def validate_safety(self) -> "QueueSettings":
        if self.heartbeat_interval_seconds >= self.lease_duration_seconds:
            raise ValueError("lease heartbeat must be smaller than lease duration")
        if self.visibility_heartbeat_seconds >= self.visibility_timeout_seconds:
            raise ValueError("visibility heartbeat must be smaller than visibility timeout")
        if self.retry_min_seconds > self.retry_max_seconds:
            raise ValueError("retry minimum must not exceed retry maximum")
        if self.backend == "sqs" and (not self.sqs_queue_url or not self.sqs_dlq_url):
            raise ValueError("SQS queue URL and DLQ URL are required")
        return self


class PublicationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    retention_versions: int = Field(default=2, ge=1, le=100)
    namespace: str = "default"
    reconciliation_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    staging_abandon_seconds: int = Field(default=3600, gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile: Profile = Profile.LOCAL
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    publication: PublicationSettings = Field(default_factory=PublicationSettings)


def profile_config(profile: Profile | str, root: Optional[Path] = None) -> AppConfig:
    selected = Profile(profile)
    base = root or Path(".")
    if selected is Profile.TEST:
        return AppConfig(
            profile=selected,
            providers=ProviderSettings(
                dense_index="memory", sparse_index="memory", graph_index="memory",
                embeddings="deterministic", reranker="identity", query_processor="identity",
                answer_generator="deterministic", cache="memory", queue="memory",
                object_storage="local", document_repository="memory", task_repository="memory",
            ),
            storage=StorageSettings(
                qdrant_path=str(base / "test-qdrant"), sparse_db_path=str(base / "test-sparse.db"),
                graph_db_path=str(base / "test-graph.db"), cache_db_path=str(base / "test-cache.db"),
                collection_name="test",
                upload_path=str(base / "uploads"), control_db_path=str(base / "control.db"),
            ),
            pipeline=PipelineSettings(enable_hyde=False, enable_graph_extraction=False),
            api=ApiSettings(api_key="test-api-key", max_upload_bytes=1024 * 1024, rate_limit_requests=1000),
            queue=QueueSettings(backend="memory"),
        )
    if selected is Profile.BENCHMARK:
        return AppConfig(
            profile=selected,
            pipeline=PipelineSettings(
                embedding_batch_size=128, vector_batch_size=256, retrieval_top_k=20,
                enable_hyde=True, enable_graph_extraction=False, enable_generation=False,
            ),
        )
    if selected is Profile.AWS_DEMO:
        return AppConfig(
            profile=selected,
            providers=ProviderSettings(queue="aws_future", cache="memory"),
            queue=QueueSettings(backend="sqs", sqs_queue_url="https://sqs.invalid/queue",
                                sqs_dlq_url="https://sqs.invalid/dlq"),
            pipeline=PipelineSettings(provider_timeout_seconds=60.0),
        )
    return AppConfig(profile=selected)


def load_config(profile: Profile | str = Profile.LOCAL, environ: Optional[Mapping[str, str]] = None) -> AppConfig:
    """Read environment only at the application boundary."""
    load_dotenv()
    values = environ or os.environ
    base = profile_config(profile)
    storage = StorageSettings(**{
            **base.storage.model_dump(),
            "qdrant_path": values.get("QDRANT_STORAGE_PATH", base.storage.qdrant_path),
            "qdrant_url": values.get("QDRANT_URL", base.storage.qdrant_url),
            "sparse_db_path": values.get("SPARSE_DB_PATH", base.storage.sparse_db_path),
            "graph_db_path": values.get("GRAPH_DB_PATH", base.storage.graph_db_path),
            "cache_db_path": values.get("EMBEDDING_CACHE_PATH", base.storage.cache_db_path),
            "upload_path": values.get("UPLOAD_PATH", base.storage.upload_path),
            "control_db_path": values.get("CONTROL_DB_PATH", base.storage.control_db_path),
            "control_database_url": values.get("CONTROL_DATABASE_URL", base.storage.control_database_url),
        })
    models = ModelSettings(**{
            **base.models.model_dump(),
            "embedding_model": values.get("EMBEDDING_MODEL_NAME", base.models.embedding_model),
            "reranker_model": values.get("RERANKER_MODEL_NAME", base.models.reranker_model),
            "groq_model": values.get("GROQ_MODEL_NAME", base.models.groq_model),
            "groq_api_key": values.get("GROQ_API_KEY", base.models.groq_api_key),
        })
    pipeline = PipelineSettings(**{
            **base.pipeline.model_dump(),
            "chunk_size": int(values.get("CHUNK_SIZE", base.pipeline.chunk_size)),
            "chunk_overlap": int(values.get("CHUNK_OVERLAP", base.pipeline.chunk_overlap)),
            "graph_index_required": values.get(
                "GRAPH_INDEX_REQUIRED", str(base.pipeline.graph_index_required)
            ).lower() in ("1", "true", "yes"),
        })
    api = ApiSettings(**{
        **base.api.model_dump(),
        "api_key": values.get("RAG_API_KEY", base.api.api_key),
        "auth_enabled": values.get("AUTH_ENABLED", str(base.api.auth_enabled)).lower() in ("1", "true", "yes"),
        "max_upload_bytes": int(values.get("MAX_UPLOAD_BYTES", base.api.max_upload_bytes)),
        "rate_limit_requests": int(values.get("RATE_LIMIT_REQUESTS", base.api.rate_limit_requests)),
    })
    queue = QueueSettings(**{
        **base.queue.model_dump(),
        "backend": values.get("INGESTION_QUEUE_BACKEND", base.queue.backend),
        "redis_url": values.get("REDIS_URL", base.queue.redis_url),
        "stream_name": values.get("INGESTION_STREAM", base.queue.stream_name),
        "consumer_group": values.get("INGESTION_CONSUMER_GROUP", base.queue.consumer_group),
        "worker_id": values.get("INGESTION_WORKER_ID", base.queue.worker_id),
        "max_concurrency": int(values.get("INGESTION_MAX_CONCURRENCY", base.queue.max_concurrency)),
        "lease_duration_seconds": float(values.get("INGESTION_LEASE_SECONDS", base.queue.lease_duration_seconds)),
        "heartbeat_interval_seconds": float(values.get("INGESTION_HEARTBEAT_SECONDS", base.queue.heartbeat_interval_seconds)),
        "visibility_timeout_seconds": int(values.get("SQS_VISIBILITY_TIMEOUT", base.queue.visibility_timeout_seconds)),
        "visibility_heartbeat_seconds": float(values.get("SQS_VISIBILITY_HEARTBEAT", base.queue.visibility_heartbeat_seconds)),
        "max_attempts": int(values.get("INGESTION_MAX_ATTEMPTS", base.queue.max_attempts)),
        "capacity": int(values.get("INGESTION_QUEUE_CAPACITY", base.queue.capacity)),
        "sqs_queue_url": values.get("SQS_QUEUE_URL", base.queue.sqs_queue_url),
        "sqs_dlq_url": values.get("SQS_DLQ_URL", base.queue.sqs_dlq_url),
    })
    publication = PublicationSettings(**{
        **base.publication.model_dump(),
        "retention_versions": int(values.get(
            "PUBLICATION_RETENTION_VERSIONS", base.publication.retention_versions
        )),
        "namespace": values.get("PUBLICATION_NAMESPACE", base.publication.namespace),
        "staging_abandon_seconds": int(values.get(
            "STAGING_ABANDON_SECONDS", base.publication.staging_abandon_seconds
        )),
    })
    return AppConfig(
        profile=base.profile, providers=base.providers, storage=storage, models=models, pipeline=pipeline, api=api,
        queue=queue,
        publication=publication,
    )

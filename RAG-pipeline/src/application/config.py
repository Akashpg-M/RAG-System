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
    graph_index: str = "sqlite"
    embeddings: str = "sentence_transformer"
    reranker: str = "cross_encoder"
    query_processor: str = "groq"
    answer_generator: str = "groq"
    cache: str = "sqlite"
    queue: str = "thread"
    object_storage: str = "local"
    document_repository: str = "sqlite"
    task_repository: str = "sqlite"


class StorageSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    qdrant_path: str = "./qdrant_local_data"
    sparse_db_path: str = "./sparse_index.db"
    graph_db_path: str = "./graph_store.db"
    cache_db_path: str = "./embedding_cache.db"
    collection_name: str = "enterprise_kb"
    upload_path: str = "./uploaded_documents"
    control_db_path: str = "./control_plane.db"


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


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile: Profile = Profile.LOCAL
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


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
            "sparse_db_path": values.get("SPARSE_DB_PATH", base.storage.sparse_db_path),
            "graph_db_path": values.get("GRAPH_DB_PATH", base.storage.graph_db_path),
            "cache_db_path": values.get("EMBEDDING_CACHE_PATH", base.storage.cache_db_path),
            "upload_path": values.get("UPLOAD_PATH", base.storage.upload_path),
            "control_db_path": values.get("CONTROL_DB_PATH", base.storage.control_db_path),
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
        })
    api = ApiSettings(**{
        **base.api.model_dump(),
        "api_key": values.get("RAG_API_KEY", base.api.api_key),
        "auth_enabled": values.get("AUTH_ENABLED", str(base.api.auth_enabled)).lower() in ("1", "true", "yes"),
        "max_upload_bytes": int(values.get("MAX_UPLOAD_BYTES", base.api.max_upload_bytes)),
        "rate_limit_requests": int(values.get("RATE_LIMIT_REQUESTS", base.api.rate_limit_requests)),
    })
    return AppConfig(
        profile=base.profile, providers=base.providers, storage=storage, models=models, pipeline=pipeline, api=api
    )

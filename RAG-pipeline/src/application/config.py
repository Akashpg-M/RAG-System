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


class StorageSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    qdrant_path: str = "./qdrant_local_data"
    sparse_db_path: str = "./sparse_index.db"
    graph_db_path: str = "./graph_store.db"
    cache_db_path: str = "./embedding_cache.db"
    collection_name: str = "enterprise_kb"


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


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile: Profile = Profile.LOCAL
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


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
            ),
            storage=StorageSettings(
                qdrant_path=str(base / "test-qdrant"), sparse_db_path=str(base / "test-sparse.db"),
                graph_db_path=str(base / "test-graph.db"), cache_db_path=str(base / "test-cache.db"),
                collection_name="test",
            ),
            pipeline=PipelineSettings(enable_hyde=False, enable_graph_extraction=False),
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
    return AppConfig(
        profile=base.profile, providers=base.providers, storage=storage, models=models, pipeline=pipeline
    )

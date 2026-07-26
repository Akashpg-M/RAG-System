import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.application.composition import build_application, build_in_memory_application
from src.application.config import PipelineSettings, Profile, load_config, profile_config
from src.infrastructure.cache import SQLiteEmbeddingCache
from src.infrastructure.memory import (
    DeterministicAnswerGenerator,
    DeterministicEmbeddingProvider,
    IdentityQueryProcessor,
    InMemoryCache,
    InMemoryDenseIndex,
    InMemoryGraphIndex,
    InMemoryQueue,
    InMemorySparseIndex,
    TokenOverlapReranker,
)
from src.infrastructure.repositories import LocalFileObjectStorage, SQLiteDocumentRepository, SQLiteTaskRepository


def test_core_has_no_infrastructure_specific_imports():
    forbidden = {"qdrant_client", "sqlite3", "groq", "sentence_transformers", "docling", "nltk"}
    for path in Path("src/core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert not imports & forbidden, f"{path} imports provider modules: {imports & forbidden}"


def test_profiles_are_validated_and_select_expected_providers(tmp_path):
    assert profile_config(Profile.TEST, tmp_path).providers.dense_index == "memory"
    assert profile_config(Profile.LOCAL).providers.dense_index == "qdrant"
    assert not profile_config(Profile.BENCHMARK).pipeline.enable_generation
    assert profile_config(Profile.AWS_DEMO).providers.queue == "aws_future"
    with pytest.raises(ValidationError):
        PipelineSettings(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValidationError):
        load_config(Profile.TEST, {"CHUNK_SIZE": "4", "CHUNK_OVERLAP": "4"})


def test_offline_ingest_query_reingest_and_delete(tmp_path):
    source = tmp_path / "document.md"
    source.write_text("Python services deploy with Kubernetes for high availability.", encoding="utf-8")
    application = build_application(profile_config(Profile.TEST, tmp_path))

    application.ingest(str(source))
    first = application.query("How do services deploy?", top_k=2)
    assert first.candidates
    assert "Kubernetes" in first.answer
    assert first.citations[0].source_uri == str(source.resolve())

    application.ingest(str(source))
    assert len(application.retrieval.manager.sparse.store.documents) == 1
    assert application.delete(str(source)) == 1
    assert application.query("Kubernetes").candidates == []


def test_provider_overrides_are_exchanged_by_dependency_injection(tmp_path):
    class RecordingEmbedder(DeterministicEmbeddingProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_embeddings_batched(self, texts, batch_size=64):
            self.calls += 1
            return super().get_embeddings_batched(texts, batch_size)

    embedder = RecordingEmbedder()
    application = build_in_memory_application(
        profile_config(Profile.TEST, tmp_path), embedding_provider=embedder,
        query_processor=IdentityQueryProcessor(), reranker=TokenOverlapReranker(),
        answer_generator=DeterministicAnswerGenerator(),
    )
    source = tmp_path / "source.txt"
    source.write_text("exchangeable provider", encoding="utf-8")
    application.ingest(str(source))
    application.query("provider")
    assert embedder.calls >= 2


def test_local_and_memory_adapters_expose_core_port_operations(tmp_path):
    adapters = {
        InMemoryCache(): ("get", "set"),
        InMemoryDenseIndex(): ("upsert_chunks_bulk", "search_similar", "delete_document"),
        InMemorySparseIndex(): ("add_documents", "search", "get_by_chunk_or_parent", "delete_document"),
        InMemoryGraphIndex(): ("add_triples_bulk", "traverse_graph_hops", "delete_document"),
        InMemoryQueue(): ("submit_task", "get_task_status"),
        SQLiteEmbeddingCache(str(tmp_path / "cache.db")): ("get", "set"),
        LocalFileObjectStorage(): ("exists", "read_bytes", "delete"),
        SQLiteDocumentRepository(str(tmp_path / "documents.db")): ("save", "get_latest", "delete"),
        SQLiteTaskRepository(str(tmp_path / "tasks.db")): ("save", "get"),
    }
    for adapter, operations in adapters.items():
        assert all(callable(getattr(adapter, operation, None)) for operation in operations)

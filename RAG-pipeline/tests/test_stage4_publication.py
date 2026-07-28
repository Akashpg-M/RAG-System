import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

import pytest

from src.api.metrics import ApiMetrics
from src.api.schemas import QueryRequest
from src.api.services import MetadataChunker, QueryApplicationService
from src.application.config import Profile, profile_config
from src.application.publication_services import CleanupService, ExternalS3EventRegistrar, ReconciliationService
from src.core.contracts import DocumentVersion
from src.core.publication import (
    IndexStageResult, ManifestEntry, PublicationValidationError, StaleFencingToken, manifest_checksum,
)
from src.infrastructure.memory import PlainTextChunker
from src.infrastructure.publication import SQLitePublicationRepository
from src.infrastructure.repositories import SQLiteDocumentRepository, SQLiteLeaseRepository, SQLiteTaskRepository


def control_plane(tmp_path):
    path = str(tmp_path / "control.db")
    SQLiteTaskRepository(path)
    return path, SQLitePublicationRepository(path), SQLiteLeaseRepository(path)


def entries(version="v1"):
    return [
        ManifestEntry(f"document#{version}#c0", f"document#{version}#p0", "hash-0", 0),
        ManifestEntry(f"document#{version}#c1", f"document#{version}#p0", "hash-1", 1),
    ]


def stage(repo, version, name, outcome="SUCCESS", values=None):
    values = values or entries(version)
    repo.record_stage("document", version, IndexStageResult(
        name, outcome, len(values) if outcome == "SUCCESS" else 0, manifest_checksum(values)
    ))


def acquire(leases, version, token=None):
    token = token or f"token-{version}"
    fence = leases.acquire(f"document:{version}", "worker", token, time.time(), 60)
    return token, fence


def publish(repo, leases, version, required=("dense", "sparse")):
    repo.register_version("document", version)
    repo.save_manifest("document", version, entries(version))
    for name in required:
        stage(repo, version, name)
    token, fence = acquire(leases, version)
    return repo.activate("document", version, f"document:{version}", token, fence, required)


def test_immutable_document_versions(tmp_path):
    repository = SQLiteDocumentRepository(str(tmp_path / "documents.db"))
    version = DocumentVersion("document", "v1", "/source", "hash")
    repository.save(version)
    repository.save(version)
    with pytest.raises(ValueError):
        repository.save(DocumentVersion("document", "v1", "/different", "hash"))


def test_versioned_chunk_provenance_is_complete(tmp_path):
    source = tmp_path / "document.md"
    source.write_text("one two three", encoding="utf-8")
    metadata = {
        "version_id": "v1", "parser_version": "parser", "chunker_config_version": "chunker",
        "embedding_model_version": "embedding", "index_schema_version": "schema", "namespace": "tenant",
    }
    _, children = MetadataChunker(PlainTextChunker(10, 0), metadata).process_file(str(source), "document")
    assert children[0].chunk_id == "document#v1#c0"
    assert children[0].parent_id == "document#v1#p0"
    assert set(metadata) <= set(children[0].metadata)


def test_dense_or_sparse_failure_cannot_replace_active_version(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    publish(repo, leases, "v1")
    repo.register_version("document", "v2")
    repo.save_manifest("document", "v2", entries("v2"))
    stage(repo, "v2", "dense", "FAILED")
    stage(repo, "v2", "sparse")
    token, fence = acquire(leases, "v2")
    with pytest.raises(PublicationValidationError):
        repo.activate("document", "v2", "document:v2", token, fence, ("dense", "sparse"))
    assert repo.snapshot().active_versions == {"document": "v1"}


def test_required_graph_failure_blocks_optional_graph_failure_degrades(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    repo.register_version("document", "v1")
    repo.save_manifest("document", "v1", entries())
    stage(repo, "v1", "dense")
    stage(repo, "v1", "sparse")
    stage(repo, "v1", "graph", "FAILED")
    token, fence = acquire(leases, "v1")
    with pytest.raises(PublicationValidationError):
        repo.activate("document", "v1", "document:v1", token, fence, ("dense", "sparse", "graph"))
    revision, activated = repo.activate("document", "v1", "document:v1", token, fence, ("dense", "sparse"))
    assert activated and revision == 1
    with sqlite3.connect(repo.db_path) as connection:
        assert connection.execute("SELECT degraded FROM version_publications").fetchone()[0] == 1


def test_retry_activation_is_idempotent_and_stale_fence_is_rejected(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    repo.register_version("document", "v1")
    repo.save_manifest("document", "v1", entries())
    stage(repo, "v1", "dense")
    stage(repo, "v1", "sparse")
    old_token, old_fence = acquire(leases, "v1")
    assert leases.release("document:v1", old_token, old_fence)
    token, fence = acquire(leases, "v1")
    with pytest.raises(StaleFencingToken):
        repo.activate("document", "v1", "document:v1", old_token, old_fence, ("dense", "sparse"))
    assert repo.activate("document", "v1", "document:v1", token, fence, ("dense", "sparse"))[1]
    assert repo.activate("document", "v1", "document:v1", token, fence, ("dense", "sparse"))[1] is False
    assert repo.snapshot().revision == 1


def test_concurrent_activation_keeps_single_active_reference(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    results = []
    for version in ("v1", "v2"):
        repo.register_version("document", version)
        repo.save_manifest("document", version, entries(version))
        stage(repo, version, "dense")
        stage(repo, version, "sparse")
    ownership = {version: acquire(leases, version) for version in ("v1", "v2")}

    def activate(version):
        token, fence = ownership[version]
        results.append(repo.activate("document", version, f"document:{version}", token, fence,
                                     ("dense", "sparse")))

    threads = [threading.Thread(target=activate, args=(version,)) for version in ("v1", "v2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(repo.snapshot().active_versions) == 1
    assert len(results) == 2


def test_tombstone_excludes_immediately_and_blocks_rollback(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    publish(repo, leases, "v1")
    assert repo.tombstone("document")[1]
    snapshot = repo.snapshot()
    assert "document" in snapshot.tombstones and "document" not in snapshot.active_versions
    with pytest.raises(PublicationValidationError):
        repo.rollback("document", "v1")
    assert repo.tombstone("document")[1] is False


def test_physical_deletion_job_is_idempotent(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    publish(repo, leases, "v1")
    repo.tombstone("document")

    class Ingestion:
        calls = 0

        def delete_document_by_id(self, document_id):
            self.calls += 1

    class Documents:
        def list_versions(self, document_id):
            return []

    class Storage:
        def delete(self, uri):
            raise AssertionError("no source versions")

    ingestion = Ingestion()
    cleanup = CleanupService(repo, Documents(), Storage(), ingestion)
    assert cleanup.run_once() == 1
    assert cleanup.run_once() == 0
    assert ingestion.calls == 1


def test_rollback_to_validated_version_and_reject_staging(tmp_path):
    _, repo, leases = control_plane(tmp_path)
    publish(repo, leases, "v1")
    publish(repo, leases, "v2")
    assert repo.rollback("document", "v1")[1]
    assert repo.snapshot().active_versions["document"] == "v1"
    repo.register_version("document", "v3")
    with pytest.raises(PublicationValidationError):
        repo.rollback("document", "v3")


class Inspector:
    def __init__(self, values):
        self.values = values

    def version_chunks(self, document_id, version_id):
        return self.values


def test_reconciliation_finds_missing_orphan_count_and_checksum(tmp_path):
    _, repo, _ = control_plane(tmp_path)
    repo.register_version("document", "v1")
    repo.save_manifest("document", "v1", entries())
    actual = {"document#v1#c0": "wrong", "document#v1#orphan": "orphan"}
    report = ReconciliationService(repo, {"dense": Inspector(actual)}).inspect_version("document", "v1")
    assert {item.kind for item in report.discrepancies} == {"missing", "orphaned", "checksum"}


def test_reconciliation_detects_abandoned_staging(tmp_path):
    _, repo, _ = control_plane(tmp_path)
    repo.register_version("document", "v1")
    with sqlite3.connect(repo.db_path) as connection:
        connection.execute("UPDATE version_publications SET started_at=?", (time.time() - 100,))
        connection.commit()
    report = ReconciliationService(repo, {}).inspect_control_plane(10)
    assert [(item.kind, item.count) for item in report.discrepancies] == [("abandoned_staging", 1)]


def test_external_s3_event_registers_control_plane_once(tmp_path):
    path = str(tmp_path / "control.db")
    tasks = SQLiteTaskRepository(path)
    documents = SQLiteDocumentRepository(path)
    publication = SQLitePublicationRepository(path)
    registrar = ExternalS3EventRegistrar(documents, tasks, publication)
    body = json.dumps({"Records": [{"eventName": "ObjectCreated:Put", "s3": {
        "bucket": {"name": "docs"}, "object": {"key": "incoming%2Ffile.pdf", "eTag": "etag-1"}
    }}]})
    first, second = registrar.register(body)[0], registrar.register(body)[0]
    assert first.task_id == second.task_id
    assert documents.get_version(first.document_id, first.version_id) is not None
    assert len(tasks.pending_outbox()) == 1


def test_publication_snapshot_filters_query_candidates_consistently(tmp_path):
    _, publication, leases = control_plane(tmp_path)
    publish(publication, leases, "v1")

    class Retrieval:
        def retrieve_context(self, *args, **kwargs):
            return [
                {"chunk_id": "document#v2#c0", "text": "new incomplete", "metadata": {
                    "document_id": "document", "version_id": "v2"}, "rrf_score": 2.0},
                {"chunk_id": "document#v1#c0", "text": "old complete", "metadata": {
                    "document_id": "document", "version_id": "v1"}, "rrf_score": 1.0},
            ]

    class Generator:
        def generate_stream(self, query, context):
            yield context[0]["text"]

    class App:
        retrieval = Retrieval()
        generator = Generator()

    documents = SQLiteDocumentRepository(str(tmp_path / "documents.db"))
    documents.save(DocumentVersion("document", "v1", "/old", "hash", datetime.now(timezone.utc), {"filename": "old"}))
    service = QueryApplicationService(App(), profile_config(Profile.TEST, tmp_path), documents, ApiMetrics(lambda: 0),
                                      publication)
    result = service.execute(QueryRequest(query="which"), "trace")
    assert result.answer == "old complete"
    assert [source.version_id for source in result.sources] == ["v1"]
    assert result.publication_revision == 1

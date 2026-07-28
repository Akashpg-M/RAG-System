import os
import uuid
from datetime import datetime, timezone

import pytest

from src.core.contracts import DocumentVersion, IndexingStatus, IngestionTask
from src.core.publication import IndexStageResult, ManifestEntry, manifest_checksum
from src.infrastructure.postgres import PostgresControlPlane


DATABASE_URL = os.getenv(
    "RAG_POSTGRES_TEST_URL", "postgresql://rag:rag-local-only@127.0.0.1:5432/rag_control"
)


def live_control():
    control = PostgresControlPlane(DATABASE_URL)
    try:
        control.is_ready()
    except Exception as error:
        pytest.skip(f"real PostgreSQL unavailable: {type(error).__name__}")
    return control


@pytest.mark.integration
def test_real_postgres_migration_schema_and_transactional_publication():
    control = live_control()
    suffix = uuid.uuid4().hex
    document_id, version_id, task_id = f"doc-{suffix}", f"v-{suffix}", f"task-{suffix}"
    now = datetime.now(timezone.utc)
    version = DocumentVersion(document_id, version_id, f"/objects/{suffix}.md", f"hash-{suffix}", now, {
        "namespace": "integration", "source_version": f"etag-{suffix}", "pipeline_version": "stage-4",
        "parser_version": "test", "chunker_config_version": "test", "embedding_model_version": "test",
        "index_schema_version": "test",
    })
    control.save(version)
    task = IngestionTask(task_id, version.source_uri, document_id, version_id, IndexingStatus.QUEUED,
                         created_at=now, updated_at=now, history=[IndexingStatus.QUEUED],
                         idempotency_key=f"claim-{suffix}")
    event_id = f"event-{suffix}"
    persisted, created = control.create_with_outbox(task, event_id, '{"schema_version":"test"}')
    assert created and persisted.task_id == task_id
    assert control.get_by_idempotency_key(task.idempotency_key).task_id == task_id
    assert any(row[0] == event_id for row in control.pending_outbox())
    control.mark_published(event_id)

    manifest = [ManifestEntry(f"{document_id}#{version_id}#c0", f"{document_id}#{version_id}#p0", "chunk-hash", 0)]
    checksum = control.save_manifest(document_id, version_id, manifest)
    for index in ("dense", "sparse"):
        control.record_stage(document_id, version_id, IndexStageResult(index, "SUCCESS", 1, checksum))
    resource, token = f"{document_id}:{version_id}", f"token-{suffix}"
    fence = control.acquire(resource, "integration-worker", token, now.timestamp(), 60)
    revision, activated = control.activate(document_id, version_id, resource, token, fence, ("dense", "sparse"))
    assert activated and revision > 0
    assert control.snapshot().active_versions[document_id] == version_id
    assert manifest_checksum(control.manifest(document_id, version_id)) == checksum


@pytest.mark.integration
def test_real_postgres_monotonic_fencing_and_tombstone():
    control = live_control()
    resource = f"fence-{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc).timestamp()
    first = control.acquire(resource, "one", "token-one", now, 30)
    assert control.release(resource, "token-one", first)
    second = control.acquire(resource, "two", "token-two", now + 1, 30)
    assert second == first + 1

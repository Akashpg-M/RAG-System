from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.contracts import DocumentVersion, IndexingStatus, IngestionTask
from src.core.events import IngestionEvent, idempotency_key, parse_s3_notification


class RollbackService:
    def __init__(self, publication: Any):
        self.publication = publication

    def rollback(self, document_id: str, version_id: str) -> tuple[int, bool]:
        return self.publication.rollback(document_id, version_id)


class CleanupService:
    def __init__(self, publication: Any, documents: Any, storage: Any, ingestion: Any):
        self.publication, self.documents, self.storage, self.ingestion = publication, documents, storage, ingestion

    def run_once(self) -> int:
        completed = 0
        for job_id, document_id, version_id, job_type in self.publication.pending_cleanup():
            try:
                if job_type == "RETIRE_VERSION" and version_id:
                    for index in (self.ingestion.vector_store, self.ingestion.sparse_store, self.ingestion.graph_store):
                        delete = getattr(index, "delete_version", None)
                        if delete:
                            delete(document_id, version_id)
                elif job_type == "DELETE_DOCUMENT":
                    self.ingestion.delete_document_by_id(document_id)
                    list_versions = getattr(self.documents, "list_versions", None)
                    for version in list_versions(document_id) if list_versions else []:
                        self.storage.delete(version.source_uri)
                self.publication.complete_cleanup(job_id, True)
                completed += 1
            except Exception:
                self.publication.complete_cleanup(job_id, False, "physical_cleanup_failed")
        return completed


@dataclass
class ReconciliationDiscrepancy:
    index_name: str
    kind: str
    count: int


@dataclass
class ReconciliationReport:
    discrepancies: list[ReconciliationDiscrepancy] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.discrepancies


class ReconciliationService:
    """Read-only by default; inspectors expose deterministic version chunk IDs."""
    def __init__(self, publication: Any, inspectors: dict[str, Any]):
        self.publication, self.inspectors = publication, inspectors

    def inspect_version(self, document_id: str, version_id: str) -> ReconciliationReport:
        expected = {entry.chunk_id: entry.content_hash for entry in self.publication.manifest(document_id, version_id)}
        report = ReconciliationReport()
        for name, inspector in self.inspectors.items():
            actual = inspector.version_chunks(document_id, version_id)
            actual_ids = set(actual)
            missing, unexpected = set(expected) - actual_ids, actual_ids - set(expected)
            mismatched = {chunk_id for chunk_id in set(expected) & actual_ids if actual[chunk_id] != expected[chunk_id]}
            for kind, values in (("missing", missing), ("orphaned", unexpected), ("checksum", mismatched)):
                if values:
                    report.discrepancies.append(ReconciliationDiscrepancy(name, kind, len(values)))
            if len(actual) != len(expected):
                report.discrepancies.append(ReconciliationDiscrepancy(name, "count", abs(len(actual)-len(expected))))
        return report

    def inspect_control_plane(self, staging_age_seconds: float) -> ReconciliationReport:
        report = ReconciliationReport()
        abandoned = self.publication.abandoned_staging(staging_age_seconds)
        if abandoned:
            report.discrepancies.append(ReconciliationDiscrepancy("control_plane", "abandoned_staging",
                                                                  len(abandoned)))
        retired = self.publication.retired_awaiting_cleanup()
        if retired:
            report.discrepancies.append(ReconciliationDiscrepancy("control_plane", "retired_cleanup", retired))
        return report


class ExternalS3EventRegistrar:
    """Create durable control-plane records for notifications not produced by the API outbox."""
    def __init__(self, documents: Any, tasks: Any, publication: Any, pipeline_version: str = "stage-4",
                 namespace: str = "default"):
        self.documents, self.tasks, self.publication = documents, tasks, publication
        self.pipeline_version, self.namespace = pipeline_version, namespace

    def register(self, body: str | bytes) -> list[IngestionTask]:
        registered = []
        for item in parse_s3_notification(body, self.pipeline_version):
            bucket, key = str(item["namespace"]), str(item["object_key"])
            source_version, source_uri = str(item["object_version"]), str(item["source_uri"])
            claim = idempotency_key(bucket, key, source_version, self.pipeline_version)
            existing = self.tasks.get_by_idempotency_key(claim)
            if existing:
                registered.append(existing)
                continue
            document_id = hashlib.sha256(f"{bucket}\0{key}".encode()).hexdigest()[:24]
            version_id = hashlib.sha256(f"{source_version}\0{self.pipeline_version}".encode()).hexdigest()[:24]
            task_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc)
            metadata = {
                "namespace": self.namespace, "storage_namespace": bucket, "object_key": key,
                "source_version": source_version, "pipeline_version": self.pipeline_version,
                "parser_version": "external", "chunker_config_version": "configured",
                "embedding_model_version": "configured", "index_schema_version": "multi-index-v1",
            }
            version = DocumentVersion(document_id, version_id, source_uri, source_version, now, metadata)
            self.documents.save(version)
            self.publication.register_version(document_id, version_id)
            task = IngestionTask(task_id, source_uri, document_id, version_id, IndexingStatus.QUEUED,
                                 created_at=now, updated_at=now, history=[IndexingStatus.QUEUED],
                                 idempotency_key=claim)
            event_id = uuid.uuid4().hex
            event = IngestionEvent(event_id, task_id, document_id, version_id, bucket, key, source_version,
                                   self.pipeline_version, source_uri, metadata=metadata)
            persisted, _ = self.tasks.create_with_outbox(task, event_id, event.to_json())
            registered.append(persisted)
        return registered

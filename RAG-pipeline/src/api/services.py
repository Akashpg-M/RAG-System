from __future__ import annotations

import hashlib
import re
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.api.errors import ApiError
from src.api.metrics import ApiMetrics
from src.api.schemas import (
    DeleteResponse,
    DependencyStatus,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    SourceResponse,
    TaskStatusResponse,
    UploadResponse,
)
from src.api.validation import UploadValidator
from src.application.composition import RagApplication
from src.application.config import AppConfig
from src.core.contracts import DocumentVersion, IndexingStatus, IngestionTask
from src.core.ports import DocumentChunker, DocumentRepository, ObjectStorage, TaskRepository
from src.infrastructure.task_queue import BackgroundWorkQueue


ALLOWED_TRANSITIONS = {
    IndexingStatus.UPLOADING: {IndexingStatus.QUEUED, IndexingStatus.FAILED_PERMANENT},
    IndexingStatus.QUEUED: {IndexingStatus.PARSING, IndexingStatus.DELETE_PENDING, IndexingStatus.FAILED_RETRYABLE},
    IndexingStatus.PARSING: {IndexingStatus.CHUNKING, IndexingStatus.DELETE_PENDING, IndexingStatus.FAILED_PERMANENT},
    IndexingStatus.CHUNKING: {IndexingStatus.EMBEDDING, IndexingStatus.DELETE_PENDING, IndexingStatus.FAILED_PERMANENT},
    IndexingStatus.EMBEDDING: {
        IndexingStatus.INDEXING_DENSE, IndexingStatus.READY, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.INDEXING_DENSE: {
        IndexingStatus.INDEXING_SPARSE, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.INDEXING_SPARSE: {
        IndexingStatus.INDEXING_GRAPH, IndexingStatus.READY, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.INDEXING_GRAPH: {
        IndexingStatus.READY, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.READY: {IndexingStatus.DELETE_PENDING},
    IndexingStatus.DELETE_PENDING: {
        IndexingStatus.DELETED, IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.FAILED_RETRYABLE: {IndexingStatus.QUEUED, IndexingStatus.DELETE_PENDING},
    IndexingStatus.FAILED_PERMANENT: {IndexingStatus.DELETE_PENDING},
    IndexingStatus.DELETED: {IndexingStatus.DELETED},
}


class DeletionRequested(Exception):
    pass


class LifecycleController:
    def __init__(self, repository: TaskRepository):
        self.repository = repository
        self._lock = threading.Lock()

    def transition(self, task_id: str, status: IndexingStatus, error_code: Optional[str] = None) -> IngestionTask:
        with self._lock:
            current = self.repository.get(task_id)
            if current is None:
                raise KeyError(task_id)
            if current.status != status and status not in ALLOWED_TRANSITIONS.get(current.status, set()):
                raise RuntimeError(f"Invalid task transition from {current.status.value} to {status.value}")
            history = [*current.history]
            if not history or history[-1] != status:
                history.append(status)
            updated = replace(
                current, status=status, error=error_code, updated_at=datetime.now(timezone.utc), history=history
            )
            self.repository.save(updated)
            return updated


class MetadataChunker:
    def __init__(self, wrapped: DocumentChunker, metadata: Dict[str, Any]):
        self.wrapped = wrapped
        self.metadata = metadata

    def process_file(self, file_path: str, document_id: str):
        parents, children = self.wrapped.process_file(file_path, document_id)
        for parent in parents:
            parent.metadata.update(self.metadata)
        for child in children:
            child.metadata.update(self.metadata)
        return parents, children


class DocumentControlService:
    def __init__(
        self,
        application: RagApplication,
        config: AppConfig,
        storage: ObjectStorage,
        documents: DocumentRepository,
        tasks: TaskRepository,
        queue: BackgroundWorkQueue,
        metrics: ApiMetrics,
    ):
        self.application = application
        self.config = config
        self.storage = storage
        self.documents = documents
        self.tasks = tasks
        self.queue = queue
        self.metrics = metrics
        self.validator = UploadValidator(config.api)
        self.lifecycle = LifecycleController(tasks)
        self._document_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

    def accept_upload(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        status_url_template: str,
        requested_document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResponse:
        safe_name = self.validator.validate(filename, content_type, data)
        document_id = requested_document_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", document_id):
            raise ApiError(422, "invalid_document_id", "document_id must contain 8 to 128 safe characters")
        content_hash = hashlib.sha256(data).hexdigest()
        version_id = hashlib.sha256(f"{document_id}\0{content_hash}".encode("utf-8")).hexdigest()[:16]
        source_uri = str(Path(self.config.storage.upload_path).resolve() / document_id / version_id / safe_name)
        task_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        task = IngestionTask(
            task_id=task_id, source_uri=source_uri, document_id=document_id, version_id=version_id,
            status=IndexingStatus.UPLOADING, created_at=now, updated_at=now,
            history=[IndexingStatus.UPLOADING],
        )
        self.tasks.save(task)
        try:
            self.storage.put_bytes(source_uri, data, content_type)
        except Exception as error:
            self.lifecycle.transition(task_id, IndexingStatus.FAILED_PERMANENT, "storage_write_failed")
            raise ApiError(503, "storage_unavailable", "Document storage is unavailable") from error

        version_metadata: Dict[str, Any] = {
            "filename": safe_name, "content_type": content_type, "status": IndexingStatus.QUEUED.value,
            **(metadata or {}),
        }
        version = DocumentVersion(document_id, version_id, source_uri, content_hash, now, version_metadata)
        self.documents.save(version)
        self.lifecycle.transition(task_id, IndexingStatus.QUEUED)
        self.queue.enqueue(lambda: self._process(task_id, version))
        self.metrics.increment("uploads_total")
        self.metrics.increment("ingestion_tasks_total")
        return UploadResponse(
            document_id=document_id, version_id=version_id, task_id=task_id,
            status=IndexingStatus.QUEUED,
            status_url=status_url_template.format(task_id=task_id), upload=None,
        )

    def _process(self, task_id: str, version: DocumentVersion) -> None:
        try:
            with self._document_locks[version.document_id]:
                latest_before_start = self.documents.get_latest(version.document_id)
                if latest_before_start and latest_before_start.metadata.get("status") in (
                    IndexingStatus.DELETE_PENDING.value, IndexingStatus.DELETED.value,
                ):
                    return

                def progress(value: str) -> None:
                    current = self.tasks.get(task_id)
                    if current and current.status in (IndexingStatus.DELETE_PENDING, IndexingStatus.DELETED):
                        raise DeletionRequested()
                    self.lifecycle.transition(task_id, IndexingStatus(value))

                chunk_metadata = {
                    "document_id": version.document_id,
                    "version_id": version.version_id,
                    **{key: value for key, value in version.metadata.items() if key not in ("status", "content_type")},
                }
                chunker = MetadataChunker(self.application.chunker, chunk_metadata)
                self.application.ingestion.ingest_document(
                    version.source_uri, chunker, document_id=version.document_id,
                    version_id=version.version_id, progress_callback=progress,
                )
                latest = self.documents.get_latest(version.document_id)
                if latest and latest.metadata.get("status") == IndexingStatus.DELETE_PENDING.value:
                    self.application.ingestion.delete_document_by_id(version.document_id)
                    self.lifecycle.transition(task_id, IndexingStatus.DELETE_PENDING)
                    self.lifecycle.transition(task_id, IndexingStatus.DELETED)
                    return
                ready = replace(version, metadata={**version.metadata, "status": IndexingStatus.READY.value})
                self.documents.save(ready)
                self.lifecycle.transition(task_id, IndexingStatus.READY)
        except DeletionRequested:
            return
        except Exception as error:
            retryable = isinstance(error, (ConnectionError, TimeoutError))
            status = IndexingStatus.FAILED_RETRYABLE if retryable else IndexingStatus.FAILED_PERMANENT
            code = "ingestion_retryable" if retryable else "ingestion_failed"
            try:
                self.lifecycle.transition(task_id, status, code)
            except Exception:
                # Preserve the original controlled failure state when an adapter
                # fails during persistence of the status itself.
                pass
            latest = self.documents.get_latest(version.document_id)
            if latest:
                self.documents.save(replace(latest, metadata={**latest.metadata, "status": status.value}))

    def get_status(self, task_id: str) -> TaskStatusResponse:
        task = self.tasks.get(task_id)
        if task is None or not task.document_id or not task.version_id:
            raise ApiError(404, "task_not_found", "Ingestion task was not found")
        return TaskStatusResponse(
            task_id=task.task_id, document_id=task.document_id, version_id=task.version_id,
            status=task.status, status_history=task.history, error_code=task.error,
        )

    def delete(self, document_id: str) -> DeleteResponse:
        version = self.documents.get_latest(document_id)
        if version is None:
            raise ApiError(404, "document_not_found", "Document was not found")
        if version.metadata.get("status") == IndexingStatus.DELETED.value:
            return DeleteResponse(
                document_id=document_id, version_id=version.version_id, status=IndexingStatus.DELETED,
                deleted_chunks=0, already_deleted=True,
            )
        pending = replace(version, metadata={**version.metadata, "status": IndexingStatus.DELETE_PENDING.value})
        self.documents.save(pending)
        task = self.tasks.get_latest_for_document(document_id)
        if task and task.status != IndexingStatus.DELETE_PENDING:
            self.lifecycle.transition(task.task_id, IndexingStatus.DELETE_PENDING)
        try:
            with self._document_locks[document_id]:
                deleted = self.application.ingestion.delete_document_by_id(document_id)
                self.storage.delete(version.source_uri)
                tombstone = replace(version, metadata={**version.metadata, "status": IndexingStatus.DELETED.value})
                self.documents.save(tombstone)
                if task:
                    self.lifecycle.transition(task.task_id, IndexingStatus.DELETED)
        except Exception as error:
            if task:
                try:
                    self.lifecycle.transition(task.task_id, IndexingStatus.FAILED_RETRYABLE, "deletion_failed")
                except Exception:
                    pass
            raise ApiError(503, "deletion_failed", "Document deletion could not be completed") from error
        return DeleteResponse(
            document_id=document_id, version_id=version.version_id, status=IndexingStatus.DELETED,
            deleted_chunks=deleted, already_deleted=False,
        )


class QueryApplicationService:
    def __init__(self, application: RagApplication, config: AppConfig, documents: DocumentRepository, metrics: ApiMetrics):
        self.application = application
        self.config = config
        self.documents = documents
        self.metrics = metrics

    def execute(self, request: QueryRequest, trace_id: str) -> QueryResponse:
        if request.stream:
            raise ApiError(422, "streaming_not_supported", "Streaming is not available in this API version")
        if len(request.query) > self.config.api.max_query_length:
            raise ApiError(422, "query_too_long", "Query exceeds the configured length limit")
        if request.top_k > self.config.api.max_top_k:
            raise ApiError(422, "top_k_too_large", "top_k exceeds the server-controlled limit")
        filters: Dict[str, Any] = {}
        if request.filters:
            if request.filters.document_id:
                filters["document_id"] = request.filters.document_id
            invalid = set(request.filters.metadata) - set(self.config.api.allowed_metadata_filters)
            if invalid:
                raise ApiError(422, "filter_not_allowed", "One or more metadata filter fields are not allowed")
            filters.update(request.filters.metadata)
        self.metrics.increment("queries_total")
        try:
            context = self.application.retrieval.retrieve_context(
                request.query, top_k=request.top_k, filters=filters or None, mode=request.retrieval_mode.value
            )
        except Exception as error:
            self.metrics.increment("query_failures_total")
            raise ApiError(503, "query_failed", "Query processing is temporarily unavailable") from error
        if not context:
            self.metrics.increment("empty_context_total")
            return QueryResponse(
                answer="I cannot answer because no matching indexed context was found.",
                retrieval_strategy=request.retrieval_mode.value, sources=[], model_version=self.config.models.groq_model,
                configuration_version=self.config.api.config_version, trace_id=trace_id,
                empty_context=True, refused=True,
            )
        try:
            answer = "".join(self.application.generator.generate_stream(request.query, context))
        except Exception as error:
            self.metrics.increment("query_failures_total")
            raise ApiError(503, "generation_failed", "Answer generation is temporarily unavailable") from error
        sources = [self._source(candidate) for candidate in context]
        return QueryResponse(
            answer=answer, retrieval_strategy=request.retrieval_mode.value, sources=sources,
            model_version=self.config.models.groq_model, configuration_version=self.config.api.config_version,
            trace_id=trace_id, empty_context=False, refused=False,
        )

    def _source(self, candidate: Dict[str, Any]) -> SourceResponse:
        metadata = candidate.get("metadata", {})
        document_id = metadata.get("document_id") or str(candidate["chunk_id"]).split("#", 1)[0]
        version = self.documents.get_latest(str(document_id))
        version_id = metadata.get("version_id") or (version.version_id if version else "unversioned")
        filename = version.metadata.get("filename", "indexed-document") if version else "indexed-document"
        excerpt = str(candidate["text"])[:self.config.api.excerpt_characters]
        page_value = metadata.get("page") or metadata.get("page_no")
        page = int(page_value) if isinstance(page_value, int) or str(page_value).isdigit() else None
        return SourceResponse(
            document_id=str(document_id), version_id=str(version_id), chunk_id=str(candidate["chunk_id"]),
            source=str(filename), page=page, section=metadata.get("section") or metadata.get("title"), excerpt=excerpt,
            dense_score=candidate.get("dense_score"), sparse_score=candidate.get("sparse_score"),
            graph_score=candidate.get("graph_score"), rrf_score=float(candidate.get("rrf_score", 0.0)),
            rerank_score=candidate.get("rerank_score"),
        )


class ReadinessService:
    def __init__(self, probes: Iterable[Tuple[str, Any, bool]], timeout_seconds: float):
        self.probes = list(probes)
        self.timeout_seconds = timeout_seconds

    def check(self) -> ReadinessResponse:
        statuses: List[DependencyStatus] = []
        for name, target, required in self.probes:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(target.is_ready)
            try:
                ready = bool(future.result(timeout=self.timeout_seconds))
            except Exception:
                ready = False
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            statuses.append(DependencyStatus(name=name, ready=ready, required=required))
        overall = all(status.ready for status in statuses if status.required)
        return ReadinessResponse(status="ready" if overall else "not_ready", dependencies=statuses)

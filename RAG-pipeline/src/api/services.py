from __future__ import annotations

import hashlib
import logging
import re
import contextvars
import threading
import time
import uuid
from collections import defaultdict
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
from src.core.events import IngestionEvent, idempotency_key
from src.core.ports import DocumentChunker, DocumentRepository, ObjectStorage, TaskRepository
from src.application.ingestion_runtime import OutboxDispatcher
from src.application.snapshot_cache import PublicationSnapshotCache
from src.core.cache_keys import retrieval_key
from src.core.performance import BoundedExecutor, Bulkhead, CapacityExhausted, Deadline, DeadlineExceeded
from src.core.queue import IngestionQueue

logger = logging.getLogger(__name__)


ALLOWED_TRANSITIONS = {
    IndexingStatus.UPLOADING: {IndexingStatus.QUEUED, IndexingStatus.FAILED_PERMANENT},
    IndexingStatus.QUEUED: {IndexingStatus.PARSING, IndexingStatus.DELETE_PENDING, IndexingStatus.FAILED_RETRYABLE},
    IndexingStatus.PARSING: {
        IndexingStatus.CHUNKING, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
    IndexingStatus.CHUNKING: {
        IndexingStatus.EMBEDDING, IndexingStatus.DELETE_PENDING,
        IndexingStatus.FAILED_RETRYABLE, IndexingStatus.FAILED_PERMANENT,
    },
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
        version_id = self.metadata.get("version_id")
        if version_id:
            parent_ids = {parent.parent_id: f"{document_id}#{version_id}#{parent.parent_id.split('#')[-1]}"
                          for parent in parents}
            for parent in parents:
                parent.parent_id = parent_ids[parent.parent_id]
            for child in children:
                child.chunk_id = f"{document_id}#{version_id}#{child.chunk_id.split('#')[-1]}"
                child.parent_id = parent_ids.get(child.parent_id, child.parent_id)
        for parent in parents:
            parent.metadata.update(self.metadata)
        for child in children:
            child.metadata.update(self.metadata)
            child.metadata["parent_id"] = child.parent_id
        return parents, children


class DocumentControlService:
    def __init__(
        self,
        application: RagApplication,
        config: AppConfig,
        storage: ObjectStorage,
        documents: DocumentRepository,
        tasks: TaskRepository,
        queue: IngestionQueue,
        metrics: ApiMetrics,
        outbox: Optional[OutboxDispatcher] = None,
        publication: Any = None,
    ):
        self.application = application
        self.config = config
        self.storage = storage
        self.documents = documents
        self.tasks = tasks
        self.queue = queue
        self.metrics = metrics
        self.outbox = outbox
        self.publication = publication
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
        with self.metrics.observability.span("ingestion.upload_acceptance", {
            "rag.content_bytes": len(data), "rag.content_type": content_type,
        }):
            safe_name = self.validator.validate(filename, content_type, data)
        document_id = requested_document_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", document_id):
            raise ApiError(422, "invalid_document_id", "document_id must contain 8 to 128 safe characters")
        content_hash = hashlib.sha256(data).hexdigest()
        version_id = hashlib.sha256(
            f"{document_id}\0{content_hash}\0{self.config.pipeline.pipeline_version}".encode("utf-8")
        ).hexdigest()[:16]
        source_uri = str(Path(self.config.storage.upload_path).resolve() / document_id / version_id / safe_name)
        namespace = "local"
        pipeline_version = self.config.pipeline.pipeline_version
        key = idempotency_key(namespace, source_uri, content_hash, pipeline_version)
        find_existing = getattr(self.tasks, "get_by_idempotency_key", None)
        existing = find_existing(key) if find_existing else None
        if existing and existing.document_id and existing.version_id:
            return UploadResponse(
                document_id=existing.document_id, version_id=existing.version_id, task_id=existing.task_id,
                status=existing.status, status_url=status_url_template.format(task_id=existing.task_id), upload=None,
            )
        task_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        task = IngestionTask(
            task_id=task_id, source_uri=source_uri, document_id=document_id, version_id=version_id,
            status=IndexingStatus.UPLOADING, created_at=now, updated_at=now,
            history=[IndexingStatus.UPLOADING],
        )
        try:
            self.storage.put_bytes(source_uri, data, content_type)
        except Exception as error:
            self.tasks.save(task)
            self.lifecycle.transition(task_id, IndexingStatus.FAILED_PERMANENT, "storage_write_failed")
            raise ApiError(503, "storage_unavailable", "Document storage is unavailable") from error

        version_metadata: Dict[str, Any] = {
            "filename": safe_name, "content_type": content_type, "status": IndexingStatus.QUEUED.value,
            "namespace": self.config.publication.namespace,
            "source_version": content_hash,
            "pipeline_version": self.config.pipeline.pipeline_version,
            "parser_version": self.config.pipeline.parser_version,
            "chunker_config_version": (
                f"{self.config.pipeline.chunker_config_version}:"
                f"{self.config.pipeline.chunk_size}:{self.config.pipeline.chunk_overlap}"
            ),
            "embedding_model_version": self.config.models.embedding_model,
            "index_schema_version": self.config.pipeline.index_schema_version,
            **(metadata or {}),
        }
        version = DocumentVersion(document_id, version_id, source_uri, content_hash, now, version_metadata)
        self.documents.save(version)
        if self.publication:
            self.publication.register_version(document_id, version_id)
        task = replace(task, status=IndexingStatus.QUEUED, history=[IndexingStatus.UPLOADING, IndexingStatus.QUEUED],
                       updated_at=datetime.now(timezone.utc))
        task = replace(task, idempotency_key=key)
        event_id = uuid.uuid4().hex
        event = IngestionEvent(
            event_id, task_id, document_id, version_id, namespace, source_uri, content_hash,
            pipeline_version, source_uri, metadata=version_metadata,
            trace_context=self.metrics.observability.inject(),
        )
        create = getattr(self.tasks, "create_with_outbox", None)
        if not create:
            raise RuntimeError("task repository does not support durable publication")
        with self.metrics.observability.span("ingestion.durable_task_outbox"):
            persisted, created = create(task, event_id, event.to_json())
        if not created:
            task_id = persisted.task_id
            document_id = persisted.document_id or document_id
            version_id = persisted.version_id or version_id
        if self.outbox:
            self.outbox.dispatch_once()
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
        if self.publication:
            _, created = self.publication.tombstone(document_id)
            task = self.tasks.get_latest_for_document(document_id)
            if task and task.status not in (IndexingStatus.DELETE_PENDING, IndexingStatus.DELETED):
                self.lifecycle.transition(task.task_id, IndexingStatus.DELETE_PENDING)
                self.lifecycle.transition(task.task_id, IndexingStatus.DELETED)
            deleted = 0
            if self.config.profile.value == "test":
                deleted = self.application.ingestion.delete_document_by_id(document_id)
                self.storage.delete(version.source_uri)
            return DeleteResponse(
                document_id=document_id, version_id=version.version_id, status=IndexingStatus.DELETED,
                deleted_chunks=deleted, already_deleted=not created,
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
    def __init__(self, application: RagApplication, config: AppConfig, documents: DocumentRepository, metrics: ApiMetrics,
                 publication: Any = None):
        self.application = application
        self.config = config
        self.documents = documents
        self.metrics = metrics
        self.publication = publication
        self.snapshot_cache = PublicationSnapshotCache(
            config.performance.snapshot_cache_entries, config.performance.snapshot_cache_ttl_seconds
        )
        self.generation_bulkhead = Bulkhead(config.performance.generation_concurrency, "generation")
        self.generation_executor = BoundedExecutor(
            config.performance.generation_concurrency,
            max(config.performance.generation_concurrency, config.performance.query_max_concurrency),
            "rag-generation",
        )

    def execute(self, request: QueryRequest, trace_id: str, authorization_scope: str = "anonymous") -> QueryResponse:
        telemetry = self.metrics.telemetry
        query_started = time.perf_counter()
        deadline = Deadline.after(self.config.performance.query_timeout_seconds)
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
        snapshot = None
        try:
            snapshot_started = time.perf_counter()
            with self.metrics.observability.span("query.publication_snapshot"):
                snapshot = self.snapshot_cache.load(self.publication) if self.publication else None
            telemetry.labels(telemetry.snapshot_duration).observe(time.perf_counter() - snapshot_started)
            telemetry.labels(telemetry.snapshot_documents).observe(len(snapshot.active_versions) if snapshot else 0)
            execution_mode, adaptive_decision, adaptive_reason = self._execution_mode(
                request.query, request.retrieval_mode.value
            )
            telemetry.labels(
                telemetry.adaptive_decisions, decision=adaptive_decision, reason=adaptive_reason,
                mode=self.config.performance.adaptive_retrieval_mode,
            ).inc()
            candidate_count = min(
                self.config.performance.refill_candidate_cap,
                max(request.top_k, request.top_k * 2 if snapshot else request.top_k),
            )
            rounds = 0
            approved: list[Dict[str, Any]] = []
            degradation_reasons: list[str] = []
            seen: set[str] = set()
            while True:
                deadline.require(0.001)
                cache_key = retrieval_key(
                    request.query, snapshot.revision if snapshot else 0,
                    {"mode": execution_mode, "reranker_cap": self.config.performance.reranker_candidate_cap},
                    filters, self.config.publication.namespace, authorization_scope, candidate_count,
                    self.config.pipeline.index_schema_version,
                )

                def retrieve() -> list[Dict[str, Any]]:
                    with self.metrics.observability.span("query.retrieval", {
                        "rag.strategy": execution_mode, "rag.requested_top_k": candidate_count,
                    }):
                        result = self.application.retrieval.retrieve_context(
                            request.query, top_k=candidate_count, filters=filters or None,
                            mode=execution_mode, deadline=deadline,
                        )
                    degradation_reasons.extend(getattr(result, "degraded_reasons", ()))
                    return list(result)

                retrieval_cache = getattr(self.application, "retrieval_cache", None)
                raw = retrieval_cache.get_or_compute(cache_key, retrieve) if retrieval_cache else retrieve()
                filtering_started = time.perf_counter()
                with self.metrics.observability.span("query.publication_filter"):
                    filtered = self._filter_candidates(raw, snapshot, filters)
                telemetry.labels(telemetry.filter_duration).observe(time.perf_counter() - filtering_started)
                for candidate in filtered:
                    identifier = str(candidate.get("chunk_id", ""))
                    if identifier not in seen:
                        approved.append(candidate)
                        seen.add(identifier)
                if len(approved) >= request.top_k or not snapshot:
                    break
                if rounds >= self.config.performance.refill_max_rounds or deadline.remaining <= 0.01:
                    break
                next_count = min(self.config.performance.refill_candidate_cap, candidate_count * 2)
                if next_count == candidate_count:
                    break
                rounds += 1
                candidate_count = next_count
            context = approved[:request.top_k]
            telemetry.labels(telemetry.candidates_filtered).observe(len(context))
            telemetry.labels(telemetry.refill_rounds).observe(rounds)
            if len(context) < request.top_k:
                degradation_reasons.append("candidate_shortfall")
                logger.info("publication_filter_shortfall", extra={
                    "component": "publication_filter", "outcome": "empty" if not context else "degraded",
                })
        except DeadlineExceeded as error:
            telemetry.labels(telemetry.query_errors, error_type="provider_timeout").inc()
            telemetry.labels(telemetry.query_requests, outcome="failure", strategy=request.retrieval_mode.value).inc()
            telemetry.labels(telemetry.query_duration).observe(time.perf_counter() - query_started)
            raise ApiError(503, "query_timeout", "Query deadline was exceeded", {"Retry-After": "1"}) from error
        except Exception as error:
            telemetry.labels(telemetry.query_errors, error_type="retrieval").inc()
            telemetry.labels(telemetry.query_requests, outcome="failure",
                             strategy=request.retrieval_mode.value).inc()
            telemetry.labels(telemetry.query_duration).observe(time.perf_counter() - query_started)
            raise ApiError(503, "query_failed", "Query processing is temporarily unavailable") from error
        if not context:
            telemetry.labels(telemetry.empty_context).inc()
            telemetry.labels(telemetry.query_requests, outcome="empty", strategy=request.retrieval_mode.value).inc()
            telemetry.labels(telemetry.query_duration).observe(time.perf_counter() - query_started)
            return QueryResponse(
                answer="I cannot answer because no matching indexed context was found.",
                retrieval_strategy=request.retrieval_mode.value, sources=[], model_version=self.config.models.groq_model,
                configuration_version=self.config.api.config_version, trace_id=trace_id,
                empty_context=True, refused=True,
                publication_revision=snapshot.revision if snapshot else None,
                graph_index_required=self.config.pipeline.graph_index_required,
                execution_degraded=bool(degradation_reasons),
                degradation_reasons=sorted(set(degradation_reasons)),
                adaptive_route=adaptive_decision,
            )
        try:
            if not self.config.pipeline.enable_generation or deadline.remaining < 0.05:
                answer = "Generation was skipped; verified sources are returned."
                degradation_reasons.append("generation_deadline")
            else:
                generation_started = time.perf_counter()
                with self.metrics.observability.span("query.generation"):
                    future = self.generation_executor.submit(
                        contextvars.copy_context().run,
                        self.generation_bulkhead.run,
                        lambda: "".join(self.application.generator.generate_stream(request.query, context)),
                    )
                    try:
                        answer = future.result(timeout=deadline.remaining)
                    except TimeoutError:
                        future.cancel()
                        answer = "Generation exceeded the query deadline; verified sources are returned."
                        degradation_reasons.append("generation_timeout")
                telemetry.labels(telemetry.generation_duration).observe(time.perf_counter() - generation_started)
        except CapacityExhausted:
            answer = "Generation capacity is temporarily unavailable; verified sources are returned."
            degradation_reasons.append("generation_capacity")
        except Exception as error:
            telemetry.labels(telemetry.query_errors, error_type="generation").inc()
            telemetry.labels(telemetry.query_requests, outcome="failure",
                             strategy=request.retrieval_mode.value).inc()
            telemetry.labels(telemetry.query_duration).observe(time.perf_counter() - query_started)
            raise ApiError(503, "generation_failed", "Answer generation is temporarily unavailable") from error
        sources = [self._source(candidate) for candidate in context]
        degraded = bool(snapshot and any(
            (source.document_id, source.version_id) in snapshot.degraded_versions for source in sources
        ))
        telemetry.labels(telemetry.query_requests, outcome="success", strategy=request.retrieval_mode.value).inc()
        telemetry.labels(telemetry.query_duration).observe(time.perf_counter() - query_started)
        return QueryResponse(
            answer=answer, retrieval_strategy=request.retrieval_mode.value, sources=sources,
            model_version=self.config.models.groq_model, configuration_version=self.config.api.config_version,
            trace_id=trace_id, empty_context=False, refused=False,
            publication_revision=snapshot.revision if snapshot else None,
            graph_index_required=self.config.pipeline.graph_index_required,
            publication_degraded=degraded,
            execution_degraded=bool(degradation_reasons),
            degradation_reasons=sorted(set(degradation_reasons)),
            adaptive_route=adaptive_decision,
        )

    def close(self) -> None:
        self.generation_executor.shutdown(wait=False)

    def _filter_candidates(self, context: Iterable[Dict[str, Any]], snapshot: Any,
                           filters: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not snapshot:
            return list(context)
        approved = []
        for candidate in context:
            metadata = candidate.get("metadata", {})
            document_id = str(metadata.get("document_id") or candidate.get("document_id") or
                              str(candidate.get("chunk_id", "")).split("#", 1)[0])
            version_id = str(metadata.get("version_id", ""))
            namespace = str(metadata.get("namespace", "default"))
            reason = None
            if document_id in snapshot.tombstones:
                reason = "tombstoned"
            elif filters.get("document_id") and document_id != filters["document_id"]:
                reason = "filter"
            elif any(metadata.get(key) != value for key, value in filters.items() if key != "document_id"):
                reason = "filter"
            elif namespace != self.config.publication.namespace:
                reason = "namespace"
            elif document_id not in snapshot.active_versions:
                reason = "orphaned"
            elif snapshot.active_versions[document_id] != version_id:
                reason = "inactive"
            if reason:
                self.metrics.telemetry.labels(self.metrics.telemetry.discarded, reason=reason).inc()
            else:
                approved.append(candidate)
        return approved

    def _execution_mode(self, query: str, requested: str) -> tuple[str, str, str]:
        configured = self.config.performance.adaptive_retrieval_mode
        if requested != "hybrid" or configured == "off":
            return requested, "full", "default_fast_path"
        lowered = query.casefold()
        entity_terms = ("relationship", "depends on", "connected", "between", "entity", "calls")
        if any(term in lowered for term in entity_terms):
            decision, reason, route = "graph", "entity_query", "adaptive_graph"
        elif len(query.split()) >= 24:
            decision, reason, route = "hyde", "complex_query", "adaptive_hyde"
        else:
            decision, reason, route = "fast", "default_fast_path", "fast"
        return ("hybrid" if configured == "shadow" else route), decision, reason

    def _source(self, candidate: Dict[str, Any]) -> SourceResponse:
        metadata = candidate.get("metadata", {})
        document_id = metadata.get("document_id") or str(candidate["chunk_id"]).split("#", 1)[0]
        get_version = getattr(self.documents, "get_version", None)
        version = get_version(str(document_id), str(metadata.get("version_id"))) if get_version and metadata.get("version_id") \
            else self.documents.get_latest(str(document_id))
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
        workers = max(1, min(8, len(self.probes)))
        self.executor = BoundedExecutor(workers, workers, "rag-readiness")

    def check(self) -> ReadinessResponse:
        statuses: List[DependencyStatus] = []
        for name, target, required in self.probes:
            try:
                future = self.executor.submit(target.is_ready)
                ready = bool(future.result(timeout=self.timeout_seconds))
            except Exception:
                ready = False
            statuses.append(DependencyStatus(name=name, ready=ready, required=required))
        overall = all(status.ready for status in statuses if status.required)
        return ReadinessResponse(status="ready" if overall else "not_ready", dependencies=statuses)

    def close(self) -> None:
        self.executor.shutdown(wait=False)

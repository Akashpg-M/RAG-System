from __future__ import annotations

import logging
import random
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from src.core.contracts import IndexingStatus
from src.core.events import IngestionEvent
from src.core.publication import IndexStageResult, chunk_manifest
from src.core.queue import IngestionQueue, QueueMessage, QueueSaturated


logger = logging.getLogger(__name__)


class PermanentIngestionError(Exception):
    pass


class PoisonMessageError(Exception):
    pass


def classify_failure(error: BaseException, stage: str = "unknown") -> tuple[bool, str]:
    if isinstance(error, PoisonMessageError):
        return False, "invalid_event_envelope"
    if isinstance(error, (ValueError, UnicodeError)):
        return False, "ingestion_failed"
    if isinstance(error, (NotImplementedError, PermanentIngestionError)):
        return False, f"unsupported_{stage}"
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True, f"temporary_{stage}_failure"
    return True, "unexpected_ingestion_failure"


class OutboxDispatcher:
    def __init__(self, repository: Any, queue: IngestionQueue):
        self.repository, self.queue = repository, queue

    def dispatch_once(self, limit: int = 100) -> int:
        published = 0
        for event_id, _, payload in self.repository.pending_outbox(limit):
            try:
                self.queue.publish(payload, event_id)
                self.repository.mark_published(event_id)
                published += 1
            except QueueSaturated:
                break
            except Exception:
                self.repository.mark_publish_failed(event_id)
        return published

    def reconcile_queued(self, documents: Any, pipeline_version: str, limit: int = 100) -> int:
        repaired = 0
        for task in self.repository.queued_without_event(limit):
            if not task.document_id or not task.version_id:
                continue
            version = documents.get_latest(task.document_id)
            if version is None or version.version_id != task.version_id:
                continue
            event_id = uuid.uuid4().hex
            event = IngestionEvent(
                event_id, task.task_id, task.document_id, task.version_id, "local", task.source_uri,
                version.content_hash, pipeline_version, task.source_uri, metadata=version.metadata,
            )
            repaired += int(self.repository.add_outbox(event_id, task.task_id, event.to_json()))
        return repaired


class IngestionWorker:
    """Bounded, lease-protected at-least-once ingestion consumer."""
    def __init__(self, queue: IngestionQueue, tasks: Any, documents: Any, leases: Any, storage: Any,
                 application: Any, worker_id: str, max_concurrency: int = 2, poll_timeout: float = 5,
                 lease_duration: float = 120, heartbeat_interval: float = 30, max_attempts: int = 5,
                 retry_min: float = 1, retry_max: float = 300, shutdown_timeout: float = 30,
                 publication: Any = None, graph_required: bool = False, maintenance: Any = None):
        if heartbeat_interval >= lease_duration:
            raise ValueError("heartbeat interval must be smaller than lease duration")
        self.queue, self.tasks, self.documents, self.leases, self.storage = queue, tasks, documents, leases, storage
        self.application, self.worker_id = application, worker_id
        self.max_concurrency, self.poll_timeout = max_concurrency, poll_timeout
        self.lease_duration, self.heartbeat_interval = lease_duration, heartbeat_interval
        self.max_attempts, self.retry_min, self.retry_max = max_attempts, retry_min, retry_max
        self.shutdown_timeout = shutdown_timeout
        self.publication, self.graph_required = publication, graph_required
        self.maintenance = maintenance
        self._stop = threading.Event()
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._slots.acquire(timeout=min(0.2, self.poll_timeout)):
                continue
            message = self.queue.receive(self.poll_timeout)
            if message is None:
                self._slots.release()
                if self.maintenance:
                    try:
                        self.maintenance()
                    except Exception:
                        logger.exception("worker maintenance pass failed")
                continue
            thread = threading.Thread(target=self._process_guarded, args=(message,), daemon=False)
            with self._lock:
                self._threads.add(thread)
            thread.start()
        self._wait_for_active()

    def stop(self) -> None:
        self._stop.set()

    def _wait_for_active(self) -> None:
        deadline = time.monotonic() + self.shutdown_timeout
        while time.monotonic() < deadline:
            with self._lock:
                active = list(self._threads)
            if not active:
                return
            for thread in active:
                thread.join(timeout=min(0.1, max(0, deadline - time.monotonic())))

    def _process_guarded(self, message: QueueMessage) -> None:
        try:
            self.process_message(message)
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())
            self._slots.release()

    def process_message(self, message: QueueMessage) -> None:
        try:
            event = IngestionEvent.from_json(message.body)
        except Exception as error:
            if message.attempts >= self.max_attempts:
                self.queue.dead_letter(message, "invalid_event_envelope")
            else:
                self.queue.retry(message, self._backoff(message.attempts))
            logger.warning("poison ingestion message: %s", type(error).__name__)
            return
        task = self.tasks.get(event.task_id)
        if task is None:
            self.queue.dead_letter(message, "task_not_found")
            return
        if task.status is IndexingStatus.READY:
            self.queue.acknowledge(message)
            return
        resource_id = f"{event.document_id}:{event.version_id}"
        ownership = uuid.uuid4().hex
        now = time.time()
        fencing = self.leases.acquire(resource_id, self.worker_id, ownership, now, self.lease_duration)
        if fencing is None:
            self.queue.retry(message, self._backoff(message.attempts))
            return
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(resource_id, ownership, fencing, message,
                                                                    heartbeat_stop), daemon=True)
        heartbeat.start()
        temporary = tempfile.mkdtemp(prefix=f"rag-ingest-{event.task_id[:8]}-")
        stage = "storage"
        try:
            data = self.storage.read_bytes(event.source_uri)
            suffix = Path(event.object_key).suffix
            local_path = Path(temporary) / f"document{suffix}"
            local_path.write_bytes(data)
            get_version = getattr(self.documents, "get_version", None)
            version = get_version(event.document_id, event.version_id) if get_version else self.documents.get_latest(event.document_id)
            if version is None or version.version_id != event.version_id:
                raise PermanentIngestionError("document version unavailable")
            task = replace(task, attempt_count=max(task.attempt_count + 1, message.attempts), fencing_token=fencing)
            self.tasks.save(task)

            def progress(value: str) -> None:
                nonlocal stage
                stage = value.lower()
                if not self.leases.owns(resource_id, ownership, fencing, time.time()):
                    raise ConnectionError("lease ownership lost")
                self._transition(event.task_id, IndexingStatus(value), fencing=fencing)

            from src.api.services import MetadataChunker
            metadata = {"document_id": event.document_id, "version_id": event.version_id,
                        **{k: v for k, v in version.metadata.items() if k not in ("status", "content_type")}}
            if self.publication:
                prepared = self.application.ingestion.prepare_document(
                    str(local_path), MetadataChunker(self.application.chunker, metadata),
                    event.document_id, event.version_id, progress,
                )
                entries, checksum = chunk_manifest(prepared.children)
                self.publication.save_manifest(event.document_id, event.version_id, entries)
                self.application.ingestion.prepare_embeddings(prepared, progress)
                self._run_index_stage("dense", event, prepared, checksum,
                                      lambda: self.application.ingestion.write_dense(prepared, progress))
                self._run_index_stage("sparse", event, prepared, checksum,
                                      lambda: self.application.ingestion.write_sparse(prepared, progress))
                try:
                    self._run_index_stage("graph", event, prepared, checksum,
                                          lambda: self.application.ingestion.write_graph(prepared, progress))
                except Exception:
                    if self.graph_required:
                        raise
                required = ["dense", "sparse", *( ["graph"] if self.graph_required else [])]
                self.publication.activate(event.document_id, event.version_id, resource_id, ownership, fencing, required)
            else:
                self.application.ingestion.ingest_document(
                    str(local_path), MetadataChunker(self.application.chunker, metadata),
                    event.document_id, event.version_id, progress,
                )
            if not self.leases.owns(resource_id, ownership, fencing, time.time()):
                raise ConnectionError("lease ownership lost")
            if not self.publication:
                self.documents.save(replace(version, metadata={**version.metadata, "status": IndexingStatus.READY.value}))
            self._transition(event.task_id, IndexingStatus.READY, fencing=fencing)
            self.queue.acknowledge(message)
        except Exception as error:
            retryable, code = classify_failure(error, stage)
            exhausted = message.attempts >= self.max_attempts
            final = IndexingStatus.FAILED_PERMANENT if exhausted or not retryable else IndexingStatus.FAILED_RETRYABLE
            try:
                self._transition(event.task_id, final, code, fencing=fencing)
                version = self.documents.get_latest(event.document_id)
                if version and not self.publication:
                    self.documents.save(replace(version, metadata={**version.metadata, "status": final.value}))
                if retryable and not exhausted:
                    self._transition(event.task_id, IndexingStatus.QUEUED, fencing=fencing)
                    self.queue.retry(message, self._backoff(message.attempts))
                else:
                    self.queue.dead_letter(message, code)
            except Exception:
                # No acknowledgement: the transport will redeliver after visibility/claim expiry.
                logger.exception("failed to persist controlled ingestion failure")
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.heartbeat_interval + 0.1)
            self.leases.release(resource_id, ownership, fencing)
            shutil.rmtree(temporary, ignore_errors=True)

    def _run_index_stage(self, name: str, event: IngestionEvent, prepared: Any, checksum: str,
                         operation: Any) -> None:
        started = time.perf_counter()
        try:
            operation()
        except Exception:
            self.publication.record_stage(event.document_id, event.version_id, IndexStageResult(
                name, "FAILED", 0, checksum, time.perf_counter() - started, f"{name}_index_failed"
            ))
            raise
        self.publication.record_stage(event.document_id, event.version_id, IndexStageResult(
            name, "SUCCESS", len(prepared.children), checksum, time.perf_counter() - started
        ))

    def _transition(self, task_id: str, status: IndexingStatus, error: Optional[str] = None,
                    fencing: Optional[int] = None) -> None:
        from src.api.services import LifecycleController
        current = self.tasks.get(task_id)
        if fencing is not None and current and current.fencing_token not in (0, fencing):
            raise RuntimeError("stale fencing token")
        LifecycleController(self.tasks).transition(task_id, status, error)

    def _heartbeat(self, resource: str, ownership: str, fencing: int, message: QueueMessage,
                   stopped: threading.Event) -> None:
        while not stopped.wait(self.heartbeat_interval):
            if not self.leases.renew(resource, ownership, fencing, time.time(), self.lease_duration):
                return
            self.queue.heartbeat(message)

    def _backoff(self, attempt: int) -> float:
        upper = min(self.retry_max, self.retry_min * (2 ** max(0, attempt - 1)))
        return random.uniform(self.retry_min, max(self.retry_min, upper))

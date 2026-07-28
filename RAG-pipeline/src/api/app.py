from __future__ import annotations

import time
import uuid
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Iterable, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.errors import ApiError
from src.api.metrics import ApiMetrics
from src.api.routes import router
from src.api.security import ApiSecurity
from src.api.services import DocumentControlService, QueryApplicationService, ReadinessService
from src.application.composition import RagApplication, build_application
from src.application.config import AppConfig, Profile, load_config
from src.application.ingestion_runtime import IngestionWorker, OutboxDispatcher
from src.infrastructure.ingestion_queues import InMemoryIngestionQueue, RedisStreamsQueue, SQSQueue
from src.infrastructure.repositories import (
    LocalFileObjectStorage, SQLiteDocumentRepository, SQLiteLeaseRepository, SQLiteTaskRepository,
)
from src.infrastructure.publication import SQLitePublicationRepository


def create_api(
    config: Optional[AppConfig] = None,
    rag_application: Optional[RagApplication] = None,
    readiness_probes: Optional[Iterable[Tuple[str, Any, bool]]] = None,
) -> FastAPI:
    settings = config or load_config(os.getenv("RAG_PROFILE", Profile.LOCAL.value))
    rag = rag_application or build_application(settings, include_queue=False)
    if settings.queue.backend == "memory":
        work_queue = InMemoryIngestionQueue(settings.queue.capacity)
    elif settings.queue.backend == "redis":
        try:
            import redis
        except ImportError as error:
            raise RuntimeError("Redis queue selected; install the 'redis' package") from error
        work_queue = RedisStreamsQueue(redis.Redis.from_url(settings.queue.redis_url), settings.queue.stream_name,
                                       settings.queue.consumer_group, f"api-{os.getpid()}", settings.queue.capacity)
    elif settings.queue.backend == "sqs":
        try:
            import boto3
        except ImportError as error:
            raise RuntimeError("SQS queue selected; install the 'boto3' package") from error
        work_queue = SQSQueue(boto3.client("sqs"), settings.queue.sqs_queue_url, settings.queue.sqs_dlq_url,
                              settings.queue.visibility_timeout_seconds, settings.queue.capacity)
    else:
        raise ValueError(f"Unsupported ingestion queue backend: {settings.queue.backend}")
    storage = LocalFileObjectStorage()
    if settings.profile is Profile.TEST or settings.providers.task_repository == "sqlite":
        documents = SQLiteDocumentRepository(settings.storage.control_db_path)
        tasks = SQLiteTaskRepository(settings.storage.control_db_path)
        leases = SQLiteLeaseRepository(settings.storage.control_db_path)
        publication = SQLitePublicationRepository(
            settings.storage.control_db_path, settings.publication.retention_versions
        )
    else:
        from src.infrastructure.postgres import PostgresControlPlane
        control = PostgresControlPlane(
            settings.storage.control_database_url, settings.publication.retention_versions
        )
        documents = tasks = leases = publication = control
    metrics = ApiMetrics(lambda: work_queue.stats().depth, work_queue.stats, publication.stats)
    dispatcher = OutboxDispatcher(tasks, work_queue)
    document_service = DocumentControlService(
        rag, settings, storage, documents, tasks, work_queue, metrics, dispatcher, publication
    )
    query_service = QueryApplicationService(rag, settings, documents, metrics, publication)
    probes = list(readiness_probes) if readiness_probes is not None else [
        ("dense_index", rag.ingestion.vector_store, True),
        ("sparse_index", rag.ingestion.sparse_store, True),
        ("graph_index", rag.ingestion.graph_store, settings.pipeline.enable_graph_extraction),
        ("task_repository", tasks, True), ("object_storage", storage, True), ("ingestion_queue", work_queue, True),
    ]
    readiness_service = ReadinessService(probes, min(settings.pipeline.provider_timeout_seconds, 5.0))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stop_dispatch = threading.Event()
        def dispatch_loop():
            while not stop_dispatch.wait(1.0):
                dispatcher.reconcile_queued(documents, settings.pipeline.pipeline_version)
                dispatcher.dispatch_once()
        dispatcher.reconcile_queued(documents, settings.pipeline.pipeline_version)
        dispatcher.dispatch_once()
        dispatch_thread = threading.Thread(target=dispatch_loop, daemon=True, name="outbox-dispatcher")
        dispatch_thread.start()
        embedded_worker = None
        worker_thread = None
        if settings.profile is Profile.TEST:
            embedded_worker = IngestionWorker(
                work_queue, tasks, documents, leases, storage, rag, "test-worker", max_concurrency=2,
                poll_timeout=0.05, lease_duration=5, heartbeat_interval=1, max_attempts=2,
                retry_min=0, retry_max=0, shutdown_timeout=2,
                publication=publication, graph_required=settings.pipeline.graph_index_required,
            )
            worker_thread = threading.Thread(target=embedded_worker.run, daemon=True, name="test-ingestion-worker")
            worker_thread.start()
        yield
        stop_dispatch.set()
        dispatch_thread.join(timeout=2)
        if embedded_worker:
            embedded_worker.stop()
        if worker_thread:
            worker_thread.join(timeout=3)
        work_queue.close()
        if rag.queue and hasattr(rag.queue, "shutdown"):
            rag.queue.shutdown()

    app = FastAPI(
        title="Multi-Index RAG API",
        summary="Versioned query and document-control API for the modular RAG core.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.config = settings
    app.state.rag = rag
    app.state.metrics = metrics
    app.state.security = ApiSecurity(settings.api)
    app.state.document_service = document_service
    app.state.query_service = query_service
    app.state.readiness_service = readiness_service

    @app.middleware("http")
    async def operational_middleware(request: Request, call_next):
        started = time.perf_counter()
        request.state.trace_id = uuid.uuid4().hex
        limit = (
            settings.api.max_upload_bytes
            if request.url.path == "/api/v1/documents/upload"
            else settings.api.max_request_bytes
        )
        content_length = request.headers.get("content-length")
        multipart_overhead = 64 * 1024 if request.url.path == "/api/v1/documents/upload" else 0
        if content_length and content_length.isdigit() and int(content_length) > limit + multipart_overhead:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large", "message": "Request exceeds the configured size limit",
                    "trace_id": request.state.trace_id,
                },
            )
        else:
            response = await call_next(request)
        response.headers["X-Trace-ID"] = request.state.trace_id
        metrics.record_request(request.url.path, request.method, response.status_code, time.perf_counter() - started)
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": error.code, "message": error.message, "trace_id": request.state.trace_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, _: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_failed", "message": "Request validation failed",
                "trace_id": request.state.trace_id,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": "http_error", "message": "Request could not be completed", "trace_id": request.state.trace_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, _: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "An internal error occurred", "trace_id": request.state.trace_id},
        )

    app.include_router(router)
    return app

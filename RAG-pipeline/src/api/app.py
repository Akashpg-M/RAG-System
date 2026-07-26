from __future__ import annotations

import time
import uuid
import os
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
from src.infrastructure.memory import InMemoryDocumentRepository, InMemoryTaskRepository
from src.infrastructure.repositories import LocalFileObjectStorage, SQLiteDocumentRepository, SQLiteTaskRepository
from src.infrastructure.task_queue import BackgroundWorkQueue


def create_api(
    config: Optional[AppConfig] = None,
    rag_application: Optional[RagApplication] = None,
    readiness_probes: Optional[Iterable[Tuple[str, Any, bool]]] = None,
) -> FastAPI:
    settings = config or load_config(os.getenv("RAG_PROFILE", Profile.LOCAL.value))
    rag = rag_application or build_application(settings, include_queue=False)
    work_queue = BackgroundWorkQueue()
    metrics = ApiMetrics(work_queue.depth)
    storage = LocalFileObjectStorage()
    if settings.profile is Profile.TEST:
        documents = InMemoryDocumentRepository()
        tasks = InMemoryTaskRepository()
    else:
        documents = SQLiteDocumentRepository(settings.storage.control_db_path)
        tasks = SQLiteTaskRepository(settings.storage.control_db_path)
    document_service = DocumentControlService(rag, settings, storage, documents, tasks, work_queue, metrics)
    query_service = QueryApplicationService(rag, settings, documents, metrics)
    probes = list(readiness_probes) if readiness_probes is not None else [
        ("dense_index", rag.ingestion.vector_store, True),
        ("sparse_index", rag.ingestion.sparse_store, True),
        ("graph_index", rag.ingestion.graph_store, settings.pipeline.enable_graph_extraction),
    ]
    readiness_service = ReadinessService(probes, min(settings.pipeline.provider_timeout_seconds, 5.0))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        work_queue.shutdown()
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

import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, Response, Security, UploadFile, status
from fastapi.security import APIKeyHeader

from src.api.errors import ApiError
from src.api.schemas import (
    DeleteResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    TaskStatusResponse,
    UploadResponse,
)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def protected(request: Request, _: Optional[str] = Security(api_key_header)) -> None:
    with request.app.state.observability.span("api.authentication"):
        request.app.state.security.authorize(request)


router = APIRouter()


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={401: {"description": "Invalid API key"}, 422: {"description": "Invalid query"}},
    dependencies=[Depends(protected)],
    tags=["query"],
)
def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    return request.app.state.query_service.execute(request_body, request.state.trace_id)


@router.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(protected)],
    tags=["documents"],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(default=None),
    category: Optional[str] = Form(default=None, max_length=128),
    department: Optional[str] = Form(default=None, max_length=128),
    language: Optional[str] = Form(default=None, max_length=128),
) -> UploadResponse:
    limit = request.app.state.config.api.max_upload_bytes
    data = await file.read(limit + 1)
    await file.close()
    if len(data) > limit:
        raise ApiError(413, "upload_too_large", "Uploaded document exceeds the configured size limit")
    metadata = {
        key: value for key, value in {
            "category": category, "department": department, "language": language,
        }.items() if value is not None
    }
    return request.app.state.document_service.accept_upload(
        filename=file.filename or "",
        content_type=file.content_type or "application/octet-stream",
        data=data,
        status_url_template=str(request.base_url).rstrip("/") + "/api/v1/documents/{task_id}/status",
        requested_document_id=document_id,
        metadata=metadata,
    )


@router.get(
    "/api/v1/documents/{task_id}/status",
    response_model=TaskStatusResponse,
    dependencies=[Depends(protected)],
    tags=["documents"],
)
def document_status(task_id: str, request: Request) -> TaskStatusResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", task_id):
        raise ApiError(422, "invalid_task_id", "Task ID is invalid")
    return request.app.state.document_service.get_status(task_id)


@router.delete(
    "/api/v1/documents/{document_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(protected)],
    tags=["documents"],
)
def delete_document(document_id: str, request: Request) -> DeleteResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", document_id):
        raise ApiError(422, "invalid_document_id", "Document ID is invalid")
    return request.app.state.document_service.delete(document_id)


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse, tags=["operations"])
def ready(request: Request, response: Response) -> ReadinessResponse:
    result = request.app.state.readiness_service.check()
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/metrics", response_class=Response, tags=["operations"])
def metrics(request: Request) -> Response:
    return Response(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

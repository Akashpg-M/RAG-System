import time

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_api
from src.application.composition import build_application
from src.application.config import ApiSettings, Profile, profile_config
from src.core.contracts import IndexingStatus


TERMINAL = {
    IndexingStatus.READY.value,
    IndexingStatus.DELETED.value,
    IndexingStatus.FAILED_RETRYABLE.value,
    IndexingStatus.FAILED_PERMANENT.value,
}


@pytest.fixture
def api_config(tmp_path):
    config = profile_config(Profile.TEST, tmp_path)
    return config.model_copy(update={
        "api": config.api.model_copy(update={
            "api_key": "secret-key", "max_upload_bytes": 1024,
            "max_request_bytes": 2048, "rate_limit_requests": 1000,
        })
    })


@pytest.fixture
def client(api_config):
    with TestClient(create_api(api_config)) as test_client:
        yield test_client


def auth():
    return {"X-API-Key": "secret-key"}


def upload(client, text, filename="document.md", document_id=None, **fields):
    data = {**fields}
    if document_id:
        data["document_id"] = document_id
    return client.post(
        "/api/v1/documents/upload", headers=auth(), data=data,
        files={"file": (filename, text.encode("utf-8"), "text/markdown")},
    )


def wait_for_status(client, task_id, expected=None):
    last = None
    for _ in range(100):
        response = client.get(f"/api/v1/documents/{task_id}/status", headers=auth())
        assert response.status_code == 200
        last = response.json()
        if expected and last["status"] == expected:
            return last
        if not expected and last["status"] in TERMINAL:
            return last
        time.sleep(0.01)
    raise AssertionError(f"task did not reach expected state; last={last}")


def test_openapi_and_required_endpoints(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set((path, method) for path, methods in paths.items() for method in methods) >= {
        ("/api/v1/query", "post"),
        ("/api/v1/documents/upload", "post"),
        ("/api/v1/documents/{task_id}/status", "get"),
        ("/api/v1/documents/{document_id}", "delete"),
        ("/health", "get"), ("/ready", "get"), ("/metrics", "get"),
    }
    assert schema["info"]["version"] == "2.0.0"
    assert "APIKeyHeader" in schema["components"]["securitySchemes"]


def test_authentication_and_controlled_query_validation(client):
    assert client.post("/api/v1/query", json={"query": "hello"}).status_code == 401
    assert client.post(
        "/api/v1/query", headers={"X-API-Key": "wrong"}, json={"query": "hello"}
    ).status_code == 401
    assert client.post("/api/v1/query", headers=auth(), json={"query": "hello"}).status_code == 200
    response = client.post("/api/v1/query", headers=auth(), json={"query": "   "})
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_failed"
    assert "input" not in response.json()
    assert client.post(
        "/api/v1/query", headers=auth(), json={"query": "hello", "stream": True}
    ).json()["error"] == "streaming_not_supported"
    assert client.post(
        "/api/v1/query", headers=auth(),
        json={"query": "hello", "filters": {"metadata": {"internal_field": "value"}}},
    ).json()["error"] == "filter_not_allowed"


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected"),
    [
        ("malware.exe", "application/octet-stream", b"data", "unsupported_extension"),
        ("document.md", "application/pdf", b"text", "mime_mismatch"),
        ("../unsafe.md", "text/markdown", b"text", "unsafe_filename"),
        ("broken.pdf", "application/pdf", b"%PDF-not-valid", "malformed_document"),
    ],
)
def test_upload_validation(client, filename, content_type, content, expected):
    response = client.post(
        "/api/v1/documents/upload", headers=auth(), files={"file": (filename, content, content_type)}
    )
    assert response.status_code in (400, 415)
    assert response.json()["error"] == expected


def test_oversized_upload_rejected(client):
    response = client.post(
        "/api/v1/documents/upload", headers=auth(),
        files={"file": ("large.md", b"x" * 1025, "text/markdown")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "upload_too_large"

    oversized_json = client.post(
        "/api/v1/query", headers=auth(), json={"query": "x" * 3000}
    )
    assert oversized_json.status_code == 413
    assert oversized_json.json()["error"] == "request_too_large"


def test_upload_status_query_filter_and_idempotent_delete(client):
    first = upload(
        client, "Kubernetes schedules the production service.", "first.md",
        document_id="document-one", category="platform",
    )
    assert first.status_code == 202
    accepted = first.json()
    assert accepted["status"] == "QUEUED"
    assert accepted["status_url"].endswith(f"/{accepted['task_id']}/status")
    ready = wait_for_status(client, accepted["task_id"], IndexingStatus.READY.value)
    assert ready["document_id"] == "document-one"
    assert ready["status_history"][0] == "UPLOADING"
    assert {"QUEUED", "PARSING", "CHUNKING", "EMBEDDING", "INDEXING_DENSE", "INDEXING_SPARSE", "READY"} <= set(
        ready["status_history"]
    )

    second = upload(
        client, "PostgreSQL stores the accounting records.", "second.md",
        document_id="document-two", category="finance",
    ).json()
    wait_for_status(client, second["task_id"], IndexingStatus.READY.value)

    query = client.post(
        "/api/v1/query", headers=auth(),
        json={
            "query": "Which system schedules the service?", "top_k": 3,
            "filters": {"document_id": "document-one", "metadata": {"category": "platform"}},
        },
    )
    assert query.status_code == 200
    payload = query.json()
    assert not payload["empty_context"]
    assert payload["sources"]
    assert {source["document_id"] for source in payload["sources"]} == {"document-one"}
    assert all(source["version_id"] == accepted["version_id"] for source in payload["sources"])
    assert all(source["chunk_id"].startswith("document-one#") for source in payload["sources"])
    assert "Kubernetes" in payload["answer"]
    assert payload["trace_id"]

    deleted = client.delete("/api/v1/documents/document-one", headers=auth())
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "DELETED"
    assert deleted.json()["deleted_chunks"] > 0
    deleted_task = client.get(accepted["status_url"], headers=auth()).json()
    assert deleted_task["status"] == "DELETED"
    assert deleted_task["status_history"][-2:] == ["DELETE_PENDING", "DELETED"]
    repeated = client.delete("/api/v1/documents/document-one", headers=auth())
    assert repeated.status_code == 200
    assert repeated.json()["already_deleted"] is True
    empty = client.post(
        "/api/v1/query", headers=auth(),
        json={"query": "Kubernetes", "filters": {"document_id": "document-one"}},
    ).json()
    assert empty["empty_context"] is True
    assert empty["refused"] is True
    assert client.delete("/api/v1/documents/unknown-doc", headers=auth()).status_code == 404


def test_failed_ingestion_has_controlled_state(api_config):
    rag = build_application(api_config)

    class BrokenChunker:
        def process_file(self, file_path, document_id):
            raise ValueError("sensitive parser detail")

    rag.chunker = BrokenChunker()
    with TestClient(create_api(api_config, rag_application=rag)) as test_client:
        accepted = upload(test_client, "valid text", document_id="broken-document").json()
        failed = wait_for_status(test_client, accepted["task_id"])
        assert failed["status"] == "FAILED_PERMANENT"
        assert failed["error_code"] == "ingestion_failed"
        assert "sensitive" not in str(failed)


def test_health_readiness_and_metrics_are_safe(api_config):
    class FailedProbe:
        def is_ready(self):
            return False

    rag = build_application(api_config)
    with TestClient(create_api(api_config, rag_application=rag, readiness_probes=[("required", FailedProbe(), True)])) as test_client:
        assert test_client.get("/health").json() == {"status": "alive"}
        readiness = test_client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["status"] == "not_ready"
        secret_query = "do-not-put-this-query-in-metrics"
        document_id = "sensitive-document-id"
        accepted = upload(test_client, "metrics content", document_id=document_id).json()
        wait_for_status(test_client, accepted["task_id"])
        test_client.post("/api/v1/query", headers=auth(), json={"query": secret_query})
        test_client.get(f"/api/v1/documents/{accepted['task_id']}/status", headers=auth())
        metrics = test_client.get("/metrics")
        assert metrics.status_code == 200
        assert "rag_http_requests_total" in metrics.text
        assert "rag_ingestion_queue_depth" in metrics.text
        assert "/api/v1/documents/{task_id}/status" in metrics.text
        assert secret_query not in metrics.text
        assert document_id not in metrics.text
        assert accepted["task_id"] not in metrics.text


def test_rate_limit_is_configurable(tmp_path):
    config = profile_config(Profile.TEST, tmp_path)
    config = config.model_copy(update={
        "api": ApiSettings(api_key="secret-key", rate_limit_requests=1, rate_limit_window_seconds=60)
    })
    with TestClient(create_api(config)) as test_client:
        assert test_client.post("/api/v1/query", headers=auth(), json={"query": "first"}).status_code == 200
        response = test_client.post("/api/v1/query", headers=auth(), json={"query": "second"})
        assert response.status_code == 429
        assert response.json()["error"] == "rate_limit_exceeded"

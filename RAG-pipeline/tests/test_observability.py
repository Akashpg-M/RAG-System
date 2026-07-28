import io
import json
import logging
import time

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.api.app import create_api
from src.api.metrics import ApiMetrics
from src.application.config import Profile, profile_config
from src.core.events import IngestionEvent
from src.core.retrieval import RetrieverManager
from src.infrastructure.ingestion_queues import SQSQueue
from src.observability import JsonLogFormatter, Observability, set_observability


def make_observability(exporter=None):
    return Observability("rag-api", "test", "5.0.0", "stage-5", span_exporter=exporter)


def test_json_logs_are_correlated_and_sensitive_content_is_centrally_suppressed():
    exporter = InMemorySpanExporter()
    observability = make_observability(exporter)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter("rag-api", "test"))
    logger = logging.getLogger("telemetry-test")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    secret_query = "show me payroll for Alice"
    with observability.span("query.execute"):
        logger.info(secret_query, extra={"operation": "query", "error_code": "validation"})
    payload = json.loads(output.getvalue())
    assert payload["event"] == "redacted_log_event"
    assert payload["trace_id"] and payload["span_id"]
    assert secret_query not in output.getvalue()
    assert {"timestamp", "severity", "service", "environment", "request_id"} <= payload.keys()
    observability.shutdown()


def test_private_metric_registry_has_buckets_bounded_routes_and_no_sensitive_values():
    first, second = make_observability(), make_observability()
    first.metrics.labels(first.metrics.query_duration).observe(0.12)
    first.metrics.labels(first.metrics.http_requests, route="unmatched", method="GET", status_class="2xx").inc()
    rendered = first.metrics.render()
    assert 'rag_query_duration_seconds_bucket{le="0.25",service="rag-api"}' in rendered
    assert 'route="unmatched"' in rendered
    assert "documents/private-id" not in rendered
    assert second.metrics.render()
    first.shutdown()
    second.shutdown()


def test_publication_consistency_gauges_are_sampled_outside_scrapes():
    observability = make_observability()
    metrics = ApiMetrics(lambda: 0, publication_stats=lambda: {
        "tombstones": 2, "retired": 3, "staging_age": 17.5, "rollbacks": 1,
    }, observability=observability)
    metrics.sample_dependencies()
    rendered = metrics.render()
    assert 'rag_tombstoned_documents{service="rag-api"} 2.0' in rendered
    assert 'rag_retired_versions_awaiting_cleanup{service="rag-api"} 3.0' in rendered
    assert 'rag_staging_version_age_seconds{service="rag-api"} 17.5' in rendered
    observability.shutdown()


class SlowRetriever:
    def retrieve(self, query, top_k, filters=None):
        time.sleep(0.05)
        return [{"chunk_id": query, "text": "safe", "metadata": {}}]


def test_concurrent_retrievers_are_overlapping_sibling_spans():
    exporter = InMemorySpanExporter()
    observability = make_observability(exporter)
    set_observability(observability)
    retriever = SlowRetriever()
    manager = RetrieverManager(retriever, retriever, retriever, enable_hyde=False)
    with observability.span("query.retrieval"):
        manager.execute_routing({
            "original_query": "opaque", "rewritten_query": "opaque", "hyde_document": "opaque"
        }, mode="hybrid")
    spans = [span for span in exporter.get_finished_spans() if span.name.startswith("query.retrieve.")]
    assert len(spans) == 3
    assert len({span.parent.span_id for span in spans}) == 1
    assert max(span.start_time for span in spans) < min(span.end_time for span in spans)
    observability.shutdown()


class StubSqs:
    def __init__(self, body):
        self.body, self.sent = body, None

    def get_queue_attributes(self, **kwargs):
        return {"Attributes": {"ApproximateNumberOfMessages": "0"}}

    def send_message(self, **kwargs):
        self.sent = kwargs
        return {"MessageId": "sent"}

    def receive_message(self, **kwargs):
        return {"Messages": [{
            "MessageId": "m1", "ReceiptHandle": "r1", "Body": self.body,
            "Attributes": {"ApproximateReceiveCount": "1"},
            "MessageAttributes": {"traceparent": {"StringValue": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}},
        }]}


def test_event_and_sqs_preserve_w3c_trace_context():
    event = IngestionEvent(
        "event", "task", "document", "version", "local", "object", "hash", "stage-5", "local://object",
        trace_context={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"},
    )
    assert IngestionEvent.from_json(event.to_json()).trace_context == event.trace_context
    client = StubSqs(event.to_json())
    queue = SQSQueue(client, "queue", "dlq")
    queue.publish(event.to_json(), event.event_id)
    assert client.sent["MessageAttributes"]["traceparent"]["StringValue"] == event.trace_context["traceparent"]
    received = IngestionEvent.from_json(queue.receive(1).body)
    assert received.trace_context["traceparent"].startswith("00-1111")


def test_api_query_has_root_snapshot_filter_and_retrieval_spans(tmp_path):
    exporter = InMemorySpanExporter()
    observability = make_observability(exporter)
    config = profile_config(Profile.TEST, tmp_path)
    with TestClient(create_api(
        config, observability=observability, dispatcher_observability=observability
    )) as client:
        response = client.post("/api/v1/query", headers={"X-API-Key": "test-api-key"}, json={"query": "opaque"})
        assert response.status_code == 200
    names = {span.name for span in exporter.get_finished_spans()}
    assert {"api.request", "api.authentication", "query.publication_snapshot", "query.retrieval"} <= names


def test_ingestion_trace_contains_durable_queue_and_publication_stages(tmp_path):
    exporter = InMemorySpanExporter()
    observability = make_observability(exporter)
    config = profile_config(Profile.TEST, tmp_path)
    with TestClient(create_api(
        config, observability=observability, dispatcher_observability=observability
    )) as client:
        response = client.post(
            "/api/v1/documents/upload", headers={"X-API-Key": "test-api-key"},
            files={"file": ("opaque.md", b"A safe document for tracing.", "text/markdown")},
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        for _ in range(100):
            status = client.get(
                f"/api/v1/documents/{task_id}/status", headers={"X-API-Key": "test-api-key"}
            ).json()["status"]
            if status == "READY":
                break
            time.sleep(0.01)
        assert status == "READY"
    names = {span.name for span in exporter.get_finished_spans()}
    required = {
        "ingestion.upload_acceptance", "ingestion.durable_task_outbox", "outbox.publish",
        "ingestion.process", "ingestion.lease_acquire", "ingestion.storage_download", "ingestion.parsing",
        "ingestion.chunking", "ingestion.manifest_persist", "ingestion.embedding", "ingestion.index.dense",
        "ingestion.index.sparse", "ingestion.index.graph", "ingestion.atomic_activation", "queue.acknowledge",
        "ingestion.temporary_cleanup",
    }
    assert required <= names


def test_exporter_failure_does_not_break_span_context():
    class BrokenExporter:
        def export(self, spans):
            raise RuntimeError("backend unavailable")

        def shutdown(self):
            pass

    observability = make_observability(BrokenExporter())
    with observability.span("query.execute"):
        completed = True
    assert completed
    observability.shutdown()

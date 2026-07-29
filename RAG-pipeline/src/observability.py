from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Mapping, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector, generate_latest


LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
INGESTION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
COUNT_BUCKETS = (0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

SAFE_ERROR_TYPES = {
    "validation", "authentication", "retrieval", "generation", "provider_timeout", "storage",
    "dense_index", "sparse_index", "graph_index", "publication_validation", "fencing", "unexpected",
}
SAFE_STAGES = {
    "acceptance", "outbox", "queue", "lease", "storage", "parsing", "chunking", "manifest",
    "embedding", "dense", "sparse", "graph", "validation", "activation", "acknowledgement", "cleanup",
}
SAFE_RETRIEVERS = {"dense", "dense_rewrite", "dense_hyde", "sparse", "graph"}
SAFE_OUTCOMES = {"success", "failure", "retryable", "permanent", "degraded", "empty", "refused"}
SAFE_DISCARD_REASONS = {"inactive", "retired", "staging", "orphaned", "tombstoned", "namespace", "filter"}

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_sensitive_key = re.compile(r"(authorization|api[_-]?key|credential|secret|password|token)", re.I)
_bearer = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_path = re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+")
_event_name = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")


def opaque_identifier(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def sanitize(value: Any, key: str = "") -> Any:
    if _sensitive_key.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, key) for item in value[:50]]
    if isinstance(value, str):
        cleaned = _bearer.sub("[REDACTED]", value)
        cleaned = _path.sub("[REDACTED_PATH]", cleaned)
        return cleaned[:512]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return type(value).__name__


class JsonLogFormatter(logging.Formatter):
    def __init__(self, service: str, environment: str):
        super().__init__()
        self.service, self.environment = service, environment

    def format(self, record: logging.LogRecord) -> str:
        trace_id, span_id = current_trace_ids()
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "severity": record.levelname,
            "service": self.service,
            "environment": self.environment,
            # Operational event names are a closed-form token. Free-form messages
            # are suppressed centrally so an accidental query/content log cannot leak.
            "event": record.getMessage() if _event_name.fullmatch(record.getMessage()) else "redacted_log_event",
            "trace_id": trace_id,
            "span_id": span_id,
            "request_id": _request_id.get(),
        }
        for key in ("operation", "component", "lifecycle_stage", "outcome", "retry_attempt",
                    "duration_seconds", "error_code"):
            if hasattr(record, key):
                payload[key] = sanitize(getattr(record, key), key)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configure_json_logging(service: str, environment: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter(service, environment))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def set_request_id(value: str):
    return _request_id.set(value)


def reset_request_id(token: Any) -> None:
    _request_id.reset(token)


def current_trace_ids() -> tuple[str, str]:
    try:
        from opentelemetry import trace
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return f"{context.trace_id:032x}", f"{context.span_id:016x}"
    except Exception:
        pass
    return "", ""


class TelemetryMetrics:
    """One private registry per service instance prevents duplicate registration in tests."""
    def __init__(self, service: str):
        self.registry = CollectorRegistry(auto_describe=True)
        self.service = service
        ProcessCollector(registry=self.registry)
        common = ["service"]
        self.http_requests = Counter("rag_http_requests_total", "HTTP requests by normalized route.",
                                     common + ["route", "method", "status_class"], registry=self.registry)
        self.http_duration = Histogram("rag_http_request_duration_seconds", "HTTP request latency.",
                                       common + ["route", "method"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.query_requests = Counter("rag_query_requests_total", "Query requests by outcome.",
                                      common + ["outcome", "strategy"], registry=self.registry)
        self.query_errors = Counter("rag_query_errors_total", "Query errors by bounded category.",
                                    common + ["error_type"], registry=self.registry)
        self.admission_rejections = Counter("rag_admission_rejections_total", "Fail-fast admission rejections.",
                                            common + ["reason"], registry=self.registry)
        self.circuit_transitions = Counter("rag_circuit_transitions_total", "Circuit state transitions.",
                                           common + ["dependency", "state"], registry=self.registry)
        self.circuit_state = Gauge("rag_circuit_state", "Current circuit state as one-hot gauge.",
                                   common + ["dependency", "state"], registry=self.registry)
        self.pool_exhaustions = Counter("rag_connection_pool_exhaustions_total", "Connection pool exhaustion.",
                                        common + ["pool"], registry=self.registry)
        self.pool_checked_out = Gauge("rag_connection_pool_checked_out", "Checked-out connections.",
                                      common + ["pool"], registry=self.registry)
        self.pool_wait = Histogram("rag_connection_pool_wait_seconds", "Connection pool wait latency.",
                                   common + ["pool"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.adaptive_decisions = Counter("rag_adaptive_route_decisions_total", "Adaptive route decisions.",
                                          common + ["decision", "reason", "mode"], registry=self.registry)
        self.query_duration = Histogram("rag_query_duration_seconds", "End-to-end query latency.", common,
                                        buckets=LATENCY_BUCKETS, registry=self.registry)
        self.retrieval_duration = Histogram("rag_retrieval_duration_seconds", "Retriever latency.",
                                            common + ["retriever"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.fusion_duration = Histogram("rag_fusion_duration_seconds", "Fusion latency.", common,
                                         buckets=LATENCY_BUCKETS, registry=self.registry)
        self.filter_duration = Histogram("rag_publication_filter_duration_seconds", "Publication filtering latency.",
                                         common, buckets=LATENCY_BUCKETS, registry=self.registry)
        self.rerank_duration = Histogram("rag_rerank_duration_seconds", "Reranking latency.", common,
                                         buckets=LATENCY_BUCKETS, registry=self.registry)
        self.generation_duration = Histogram("rag_generation_duration_seconds", "Generation latency.", common,
                                             buckets=LATENCY_BUCKETS, registry=self.registry)
        self.ttft = Histogram("rag_time_to_first_token_seconds", "Provider-observed streamed time to first token.", common,
                              buckets=LATENCY_BUCKETS, registry=self.registry)
        self.snapshot_duration = Histogram("rag_publication_snapshot_duration_seconds", "Snapshot acquisition latency.",
                                           common, buckets=LATENCY_BUCKETS, registry=self.registry)
        self.ingestion_duration = Histogram("rag_ingestion_duration_seconds", "Ingestion stage latency.",
                                            common + ["stage"], buckets=INGESTION_BUCKETS, registry=self.registry)
        self.embedding_batch_duration = Histogram("rag_embedding_batch_duration_seconds", "Embedding batch latency.",
                                                  common, buckets=INGESTION_BUCKETS, registry=self.registry)
        self.index_publication_duration = Histogram("rag_index_publication_duration_seconds", "Index write latency.",
                                                    common + ["index"], buckets=INGESTION_BUCKETS,
                                                    registry=self.registry)
        self.cleanup_duration = Histogram("rag_cleanup_duration_seconds", "Cleanup job latency.", common,
                                          buckets=INGESTION_BUCKETS, registry=self.registry)
        self.ingestion_tasks = Counter("rag_ingestion_tasks_total", "Ingestion task transitions.",
                                       common + ["status"], registry=self.registry)
        self.ingestion_failures = Counter("rag_ingestion_failures_total", "Ingestion failures.",
                                          common + ["stage", "error_type"], registry=self.registry)
        self.cache_requests = Counter("rag_cache_requests_total", "Cache requests.",
                                      common + ["cache", "result"], registry=self.registry)
        self.empty_context = Counter("rag_empty_context_total", "Queries without approved context.", common,
                                     registry=self.registry)
        self.retries = Counter("rag_ingestion_retries_total", "Ingestion retries.", common + ["stage"],
                               registry=self.registry)
        self.dlq = Counter("rag_dlq_messages_total", "Messages sent to DLQ.", common, registry=self.registry)
        self.llm_tokens = Counter("rag_llm_tokens_total", "Provider-reported or tokenizer-derived LLM tokens.",
                                  common + ["direction"], registry=self.registry)
        self.publications = Counter("rag_publication_attempts_total", "Publication attempts.",
                                    common + ["outcome"], registry=self.registry)
        self.publication_degraded = Counter("rag_publication_degraded_total", "Degraded publications.", common,
                                            registry=self.registry)
        self.cleanup_jobs = Counter("rag_cleanup_jobs_total", "Cleanup job outcomes.", common + ["outcome"],
                                    registry=self.registry)
        self.reconciliation = Counter("rag_reconciliation_discrepancies_total", "Reconciliation discrepancies.",
                                      common + ["type"], registry=self.registry)
        self.discarded = Counter("rag_candidate_discarded_total", "Candidates rejected by publication filtering.",
                                 common + ["reason"], registry=self.registry)
        self.queue_depth = Gauge("rag_queue_depth", "Cached queue depth.", common, registry=self.registry)
        self.legacy_queue_depth = Gauge("rag_ingestion_queue_depth", "Deprecated alias for rag_queue_depth.", common,
                                        registry=self.registry)
        self.queue_pending = Gauge("rag_queue_pending_messages", "Cached in-flight queue messages.", common,
                                   registry=self.registry)
        self.queue_oldest = Gauge("rag_queue_oldest_message_age_seconds", "Cached oldest message age.", common,
                                  registry=self.registry)
        self.active_workers = Gauge("rag_active_workers", "Active worker processes.", common, registry=self.registry)
        self.active_ingestions = Gauge("rag_active_ingestions", "Active ingestion operations.", common,
                                       registry=self.registry)
        self.active_queries = Gauge("rag_active_queries", "Active admitted query operations.", common,
                                    registry=self.registry)
        self.tombstoned_documents = Gauge("rag_tombstoned_documents", "Current control-plane tombstones.", common,
                                          registry=self.registry)
        self.retired_cleanup = Gauge("rag_retired_versions_awaiting_cleanup",
                                     "Retired versions with pending or failed cleanup.", common,
                                     registry=self.registry)
        self.staging_age = Gauge("rag_staging_version_age_seconds", "Age of the oldest staging version.", common,
                                 registry=self.registry)
        self.rollback_operations = Gauge("rag_rollback_operations", "Durable rollback revision count.", common,
                                         registry=self.registry)
        self.candidates_returned = Histogram("rag_retrieval_candidates", "Candidates returned by retriever.",
                                             common + ["retriever"], buckets=COUNT_BUCKETS, registry=self.registry)
        self.candidates_filtered = Histogram("rag_publication_candidates", "Candidates remaining after filtering.",
                                             common, buckets=COUNT_BUCKETS, registry=self.registry)
        self.refill_rounds = Histogram("rag_candidate_refill_rounds", "Candidate refill rounds.", common,
                                       buckets=(0, 1, 2, 3, 4, 5), registry=self.registry)
        self.embedding_batch_size = Histogram("rag_embedding_batch_size", "Embedding batch size.", common,
                                              buckets=BATCH_BUCKETS, registry=self.registry)
        self.snapshot_documents = Histogram("rag_publication_snapshot_documents", "Documents in a snapshot.", common,
                                            buckets=COUNT_BUCKETS, registry=self.registry)

    def labels(self, metric: Any, **labels: str):
        allowed = {
            "route": {"/api/v1/query", "/api/v1/documents/upload", "/api/v1/documents/{task_id}/status",
                      "/api/v1/documents/{document_id}", "/health", "/ready", "/metrics", "unmatched"},
            "method": {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"},
            "status_class": {"1xx", "2xx", "3xx", "4xx", "5xx"},
            "retriever": SAFE_RETRIEVERS,
            "stage": SAFE_STAGES,
            "error_type": SAFE_ERROR_TYPES,
            "outcome": SAFE_OUTCOMES,
            "reason": SAFE_DISCARD_REASONS | {"query_capacity", "executor_capacity", "entity_query",
                                               "default_fast_path", "low_confidence", "complex_query", "deadline"},
            "strategy": {"hybrid", "dense", "sparse", "graph"},
            "cache": {"embedding", "query", "retrieval", "representation", "graph"},
            "result": {"hit", "miss", "write", "error", "bypass"},
            "direction": {"input", "output"},
            "index": {"dense", "sparse", "graph"},
            "status": {"accepted", "queued", "parsing", "chunking", "embedding", "ready",
                       "indexing_dense", "indexing_sparse", "indexing_graph", "failed_retryable",
                       "failed_permanent", "delete_pending", "deleted"},
            "type": {"missing", "orphaned", "checksum", "count", "abandoned_staging", "retired_cleanup",
                     "tombstoned_entries", "premature_cleanup"},
            "dependency": {"groq", "qdrant"},
            "state": {"closed", "open", "half_open"},
            "pool": {"postgres", "redis", "qdrant", "groq"},
            "decision": {"fast", "graph", "hyde", "full"},
            "mode": {"off", "adaptive", "shadow"},
        }
        bounded: Dict[str, str] = {}
        for key, value in labels.items():
            value = str(value)
            bounded[key] = value if key not in allowed or value in allowed[key] else (
                "unmatched" if key == "route" else "unexpected"
            )
        return metric.labels(service=self.service, **bounded)

    def render(self) -> str:
        return generate_latest(self.registry).decode("utf-8")


class Observability:
    def __init__(self, service: str, environment: str, service_version: str, pipeline_version: str,
                 endpoint: str = "", sample_ratio: float = 1.0, span_exporter: Any = None):
        self.service, self.environment = service, environment
        self.metrics = TelemetryMetrics(service)
        self.provider = None
        self._tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
            resource = Resource.create({
                "service.name": service, "service.version": service_version,
                "deployment.environment": environment, "rag.pipeline.version": pipeline_version,
                "service.instance.id": os.getenv("HOSTNAME", f"{service}-{os.getpid()}"),
            })
            self.provider = TracerProvider(resource=resource,
                                           sampler=ParentBased(TraceIdRatioBased(max(0.0, min(1.0, sample_ratio)))))
            if span_exporter is not None:
                self.provider.add_span_processor(SimpleSpanProcessor(span_exporter))
            elif endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"), timeout=2)
                self.provider.add_span_processor(BatchSpanProcessor(
                    exporter, max_queue_size=2048, max_export_batch_size=256, schedule_delay_millis=1000,
                    export_timeout_millis=2000,
                ))
            self._tracer = self.provider.get_tracer("rag-pipeline", service_version)
            if endpoint:
                self._instrument_clients()
        except Exception:
            self.provider = None
            self._tracer = trace.get_tracer("rag-pipeline") if "trace" in locals() else None

    def _instrument_clients(self) -> None:
        """Best-effort auto-instrumentation; never affects application startup."""
        for module_name, class_name in (
            ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
            ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
            ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
            ("opentelemetry.instrumentation.psycopg", "PsycopgInstrumentor"),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                instrumentor = getattr(module, class_name)()
                if not instrumentor.is_instrumented_by_opentelemetry:
                    instrumentor.instrument(tracer_provider=self.provider)
            except Exception:
                continue

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None, context: Any = None) -> Iterator[Any]:
        if not self._tracer:
            yield None
            return
        try:
            manager = self._tracer.start_as_current_span(name, context=context, attributes=sanitize(attributes or {}))
        except Exception:
            yield None
            return
        with manager as span:
            yield span

    def inject(self) -> Dict[str, str]:
        carrier: Dict[str, str] = {}
        try:
            from opentelemetry.propagate import inject
            inject(carrier)
        except Exception:
            pass
        return carrier

    def extract(self, carrier: Mapping[str, str]):
        try:
            from opentelemetry.propagate import extract
            return extract(dict(carrier))
        except Exception:
            return None

    def shutdown(self) -> None:
        try:
            if self.provider:
                self.provider.shutdown()
        except Exception:
            pass


_default = Observability("rag-unknown", "test", "0", "unknown", sample_ratio=0)


def get_observability() -> Observability:
    return _default


def set_observability(value: Observability) -> None:
    global _default
    _default = value

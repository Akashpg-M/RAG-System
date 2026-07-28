from __future__ import annotations

import re
from typing import Any, Callable, Optional

from src.observability import Observability


def normalized_route(path: str) -> str:
    if re.fullmatch(r"/api/v1/documents/[^/]+/status", path):
        return "/api/v1/documents/{task_id}/status"
    if re.fullmatch(r"/api/v1/documents/[^/]+", path):
        return "/api/v1/documents/{document_id}"
    allowed = {"/api/v1/query", "/api/v1/documents/upload", "/health", "/ready", "/metrics"}
    return path if path in allowed else "unmatched"


class ApiMetrics:
    """Compatibility facade over the single Stage 5 Prometheus registry."""
    def __init__(self, queue_depth: Callable[[], int], queue_stats: Optional[Callable[[], Any]] = None,
                 publication_stats: Optional[Callable[[], Any]] = None,
                 observability: Optional[Observability] = None):
        self.observability = observability or Observability("rag-api-test", "test", "5.0.0", "stage-5",
                                                            sample_ratio=0)
        self.telemetry = self.observability.metrics
        self._queue_depth, self._queue_stats, self._publication_stats = queue_depth, queue_stats, publication_stats

    def record_request(self, route: str, method: str, status_code: int, seconds: float) -> None:
        route = normalized_route(route)
        labels = {"route": route, "method": method, "status_class": f"{status_code // 100}xx"}
        self.telemetry.labels(self.telemetry.http_requests, **labels).inc()
        self.telemetry.labels(self.telemetry.http_duration, route=route, method=method).observe(seconds)

    def increment(self, name: str) -> None:
        mapping = {
            "queries_total": lambda: self.telemetry.labels(
                self.telemetry.query_requests, outcome="success", strategy="hybrid"
            ).inc(),
            "query_failures_total": lambda: self.telemetry.labels(
                self.telemetry.query_errors, error_type="unexpected"
            ).inc(),
            "uploads_total": lambda: self.telemetry.labels(
                self.telemetry.ingestion_tasks, status="accepted"
            ).inc(),
            "ingestion_tasks_total": lambda: self.telemetry.labels(
                self.telemetry.ingestion_tasks, status="queued"
            ).inc(),
            "empty_context_total": lambda: self.telemetry.labels(self.telemetry.empty_context).inc(),
        }
        operation = mapping.get(name)
        if operation:
            operation()

    def sample_dependencies(self) -> None:
        """Called in a background loop; scrapes only read cached gauges."""
        try:
            stats = self._queue_stats() if self._queue_stats else None
            depth = getattr(stats, "depth", self._queue_depth())
            pending = getattr(stats, "pending", 0)
            oldest = getattr(stats, "oldest_age_seconds", None) or 0
            self.telemetry.labels(self.telemetry.queue_depth).set(depth)
            self.telemetry.labels(self.telemetry.legacy_queue_depth).set(depth)
            self.telemetry.labels(self.telemetry.queue_pending).set(pending)
            self.telemetry.labels(self.telemetry.queue_oldest).set(oldest)
        except Exception:
            pass
        try:
            stats = self._publication_stats() if self._publication_stats else {}
            self.telemetry.labels(self.telemetry.tombstoned_documents).set(int(stats.get("tombstones", 0)))
            self.telemetry.labels(self.telemetry.retired_cleanup).set(int(stats.get("retired", 0)))
            self.telemetry.labels(self.telemetry.staging_age).set(float(stats.get("staging_age", 0)))
            self.telemetry.labels(self.telemetry.rollback_operations).set(int(stats.get("rollbacks", 0)))
        except Exception:
            pass

    def render(self) -> str:
        return self.telemetry.render()

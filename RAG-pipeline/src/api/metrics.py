from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Optional, Tuple


def normalized_route(path: str) -> str:
    if re.fullmatch(r"/api/v1/documents/[^/]+/status", path):
        return "/api/v1/documents/{task_id}/status"
    if re.fullmatch(r"/api/v1/documents/[^/]+", path):
        return "/api/v1/documents/{document_id}"
    allowed = {"/api/v1/query", "/api/v1/documents/upload", "/health", "/ready", "/metrics"}
    return path if path in allowed else "unmatched"


class ApiMetrics:
    def __init__(self, queue_depth: Callable[[], int], queue_stats: Optional[Callable[[], Any]] = None):
        self._lock = threading.Lock()
        self._request_counts: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
        self._request_latency_sum: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        self._request_latency_count: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._queue_depth = queue_depth
        self._queue_stats = queue_stats

    def record_request(self, route: str, method: str, status_code: int, seconds: float) -> None:
        route = normalized_route(route)
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._request_counts[(route, method, status_class)] += 1
            self._request_latency_sum[(route, method)] += seconds
            self._request_latency_count[(route, method)] += 1

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def render(self) -> str:
        lines = [
            "# HELP rag_http_requests_total HTTP requests by normalized route, method, and status class.",
            "# TYPE rag_http_requests_total counter",
        ]
        with self._lock:
            for (route, method, status_class), count in sorted(self._request_counts.items()):
                lines.append(
                    f'rag_http_requests_total{{route="{route}",method="{method}",status_class="{status_class}"}} {count}'
                )
            lines.extend([
                "# HELP rag_http_request_duration_seconds Request latency summary.",
                "# TYPE rag_http_request_duration_seconds summary",
            ])
            for (route, method), total in sorted(self._request_latency_sum.items()):
                labels = f'route="{route}",method="{method}"'
                lines.append(f"rag_http_request_duration_seconds_sum{{{labels}}} {total:.9f}")
                lines.append(
                    f"rag_http_request_duration_seconds_count{{{labels}}} {self._request_latency_count[(route, method)]}"
                )
            counter_help = {
                "queries_total": "Accepted query operations.",
                "query_failures_total": "Failed query operations.",
                "uploads_total": "Accepted document uploads.",
                "ingestion_tasks_total": "Queued ingestion tasks.",
                "empty_context_total": "Query responses without retrieved context.",
            }
            for name, help_text in counter_help.items():
                lines.append(f"# HELP rag_{name} {help_text}")
                lines.append(f"# TYPE rag_{name} counter")
                lines.append(f"rag_{name} {self._counters[name]}")
        stats = self._queue_stats() if self._queue_stats else None
        pending = getattr(stats, "pending", 0)
        oldest = getattr(stats, "oldest_age_seconds", None) or 0
        retries = getattr(stats, "retries", 0)
        dlq = getattr(stats, "dlq", 0)
        lines.extend([
            "# HELP rag_ingestion_queue_depth Current durable ingestion queue depth.",
            "# TYPE rag_ingestion_queue_depth gauge",
            f"rag_ingestion_queue_depth {self._queue_depth()}",
            "# HELP rag_ingestion_pending_messages Current in-flight ingestion messages.",
            "# TYPE rag_ingestion_pending_messages gauge",
            f"rag_ingestion_pending_messages {pending}",
            "# HELP rag_ingestion_oldest_message_age_seconds Age of the oldest queued message when available.",
            "# TYPE rag_ingestion_oldest_message_age_seconds gauge",
            f"rag_ingestion_oldest_message_age_seconds {oldest}",
            "# HELP rag_ingestion_attempts_total Ingestion attempts by bounded outcome.",
            "# TYPE rag_ingestion_attempts_total counter",
            'rag_ingestion_attempts_total{outcome="success"} 0',
            'rag_ingestion_attempts_total{outcome="retryable"} 0',
            'rag_ingestion_attempts_total{outcome="permanent"} 0',
            "# HELP rag_ingestion_retries_total Ingestion retries.",
            "# TYPE rag_ingestion_retries_total counter",
            f"rag_ingestion_retries_total {retries}",
            "# HELP rag_ingestion_dlq_total Messages routed to the ingestion DLQ.",
            "# TYPE rag_ingestion_dlq_total counter",
            f"rag_ingestion_dlq_total {dlq}",
            "# HELP rag_ingestion_lease_conflicts_total Lease acquisition conflicts.",
            "# TYPE rag_ingestion_lease_conflicts_total counter",
            "rag_ingestion_lease_conflicts_total 0",
            "# HELP rag_ingestion_lease_recoveries_total Expired lease recoveries.",
            "# TYPE rag_ingestion_lease_recoveries_total counter",
            "rag_ingestion_lease_recoveries_total 0",
            "# HELP rag_ingestion_duration_seconds Ingestion duration summary.",
            "# TYPE rag_ingestion_duration_seconds summary",
            "rag_ingestion_duration_seconds_sum 0",
            "rag_ingestion_duration_seconds_count 0",
            "# HELP rag_ingestion_active Current active ingestion count.",
            "# TYPE rag_ingestion_active gauge",
            "rag_ingestion_active 0",
        ])
        return "\n".join(lines) + "\n"

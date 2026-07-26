from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import Callable, DefaultDict, Tuple


def normalized_route(path: str) -> str:
    if re.fullmatch(r"/api/v1/documents/[^/]+/status", path):
        return "/api/v1/documents/{task_id}/status"
    if re.fullmatch(r"/api/v1/documents/[^/]+", path):
        return "/api/v1/documents/{document_id}"
    allowed = {"/api/v1/query", "/api/v1/documents/upload", "/health", "/ready", "/metrics"}
    return path if path in allowed else "unmatched"


class ApiMetrics:
    def __init__(self, queue_depth: Callable[[], int]):
        self._lock = threading.Lock()
        self._request_counts: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
        self._request_latency_sum: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        self._request_latency_count: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._queue_depth = queue_depth

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
        lines.extend([
            "# HELP rag_ingestion_queue_depth Current in-process ingestion queue depth.",
            "# TYPE rag_ingestion_queue_depth gauge",
            f"rag_ingestion_queue_depth {self._queue_depth()}",
        ])
        return "\n".join(lines) + "\n"


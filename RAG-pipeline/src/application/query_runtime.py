from __future__ import annotations

import asyncio
import contextvars
from functools import partial
from typing import Any, Callable

from src.api.errors import ApiError
from src.core.performance import BoundedExecutor, Bulkhead, CapacityExhausted
from src.observability import Observability


class QueryRuntime:
    """Async boundary around bounded blocking query execution."""
    def __init__(self, max_concurrency: int, max_pending: int, observability: Observability):
        self.bulkhead = Bulkhead(max_concurrency, "query")
        self.executor = BoundedExecutor(max_concurrency, max_pending, "rag-query")
        self.observability = observability
        self._stopping = False

    async def execute(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._stopping:
            raise ApiError(503, "service_stopping", "Query service is shutting down", {"Retry-After": "1"})
        try:
            self.bulkhead.acquire()
        except CapacityExhausted as error:
            self.observability.metrics.labels(
                self.observability.metrics.admission_rejections, reason="query_capacity"
            ).inc()
            raise ApiError(503, "query_capacity_exhausted", "Query capacity is temporarily exhausted",
                           {"Retry-After": "1"}) from error
        active = self.observability.metrics.labels(self.observability.metrics.active_queries)
        active.inc()
        try:
            context = contextvars.copy_context()
            future = self.executor.submit(context.run, partial(operation, *args, **kwargs))
            released = False

            def release_capacity(_: Any) -> None:
                nonlocal released
                if not released:
                    released = True
                    active.dec()
                    self.bulkhead.release()

            future.add_done_callback(release_capacity)
            return await asyncio.wrap_future(future)
        except CapacityExhausted as error:
            active.dec()
            self.bulkhead.release()
            self.observability.metrics.labels(
                self.observability.metrics.admission_rejections, reason="executor_capacity"
            ).inc()
            raise ApiError(503, "query_capacity_exhausted", "Query capacity is temporarily exhausted",
                           {"Retry-After": "1"}) from error

    def shutdown(self) -> None:
        self._stopping = True
        self.executor.shutdown(wait=False)

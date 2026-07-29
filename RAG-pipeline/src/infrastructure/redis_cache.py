from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable, Optional, TypeVar

from src.observability import get_observability


T = TypeVar("T")


class RedisVersionedCache:
    """Fail-open JSON cache with bounded values and local single-flight computation."""
    def __init__(self, client: Any, namespace: str, ttl_seconds: int = 3600,
                 max_value_bytes: int = 1_000_000, lock_seconds: int = 10):
        self.client, self.namespace = client, namespace
        self.ttl_seconds, self.max_value_bytes, self.lock_seconds = ttl_seconds, max_value_bytes, lock_seconds
        self._flights: dict[str, threading.Condition] = {}
        self._flight_lock = threading.Lock()
        self.records_metrics = True

    def _key(self, key: str) -> str:
        return f"rag:cache:{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        metric = get_observability().metrics
        try:
            raw = self.client.get(self._key(key))
            if raw is None or len(raw) > self.max_value_bytes:
                metric.labels(metric.cache_requests, cache=self.namespace, result="miss").inc()
                return None
            payload = json.loads(raw)
            if payload.get("schema") != "stage6-v1":
                metric.labels(metric.cache_requests, cache=self.namespace, result="bypass").inc()
                return None
            metric.labels(metric.cache_requests, cache=self.namespace, result="hit").inc()
            return payload.get("value")
        except Exception:
            metric.labels(metric.cache_requests, cache=self.namespace, result="error").inc()
            return None

    def set(self, key: str, value: Any) -> bool:
        try:
            raw = json.dumps({"schema": "stage6-v1", "value": value}, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
            if len(raw) > self.max_value_bytes:
                get_observability().metrics.labels(
                    get_observability().metrics.cache_requests, cache=self.namespace, result="bypass"
                ).inc()
                return False
            self.client.set(self._key(key), raw, ex=self.ttl_seconds)
            get_observability().metrics.labels(
                get_observability().metrics.cache_requests, cache=self.namespace, result="write"
            ).inc()
            return True
        except Exception:
            get_observability().metrics.labels(
                get_observability().metrics.cache_requests, cache=self.namespace, result="error"
            ).inc()
            return False

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        owner = False
        with self._flight_lock:
            condition = self._flights.get(key)
            if condition is None:
                condition = threading.Condition(self._flight_lock)
                self._flights[key] = condition
                owner = True
            if not owner:
                deadline = time.monotonic() + self.lock_seconds
                while key in self._flights and time.monotonic() < deadline:
                    condition.wait(timeout=max(0.0, deadline - time.monotonic()))
                cached = self.get(key)
                if cached is not None:
                    return cached
        if not owner:
            return compute()
        lock_key = self._key(f"lock:{key}")
        lock_token = uuid.uuid4().hex
        distributed_owner = True
        try:
            try:
                distributed_owner = bool(self.client.set(lock_key, lock_token, nx=True, ex=self.lock_seconds))
            except Exception:
                distributed_owner = True
            if not distributed_owner:
                deadline = time.monotonic() + self.lock_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                    cached = self.get(key)
                    if cached is not None:
                        return cached
            value = compute()
            self.set(key, value)
            return value
        finally:
            if distributed_owner:
                try:
                    current = self.client.get(lock_key)
                    current_text = current.decode("utf-8") if isinstance(current, bytes) else current
                    if current_text == lock_token:
                        self.client.delete(lock_key)
                except Exception:
                    pass
            with self._flight_lock:
                flight = self._flights.pop(key, None)
                if flight:
                    flight.notify_all()

    def delete_prefix_async(self, prefix: str) -> None:
        def remove() -> None:
            try:
                cursor = 0
                pattern = self._key(prefix) + "*"
                while True:
                    cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        self.client.delete(*keys)
                    if cursor == 0:
                        return
            except Exception:
                return
        threading.Thread(target=remove, daemon=True, name="cache-cleanup").start()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()

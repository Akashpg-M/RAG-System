from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class DeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.monotonic() + max(0.0, seconds))

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    def require(self, minimum_seconds: float = 0.0) -> float:
        remaining = self.remaining
        if remaining <= minimum_seconds:
            raise DeadlineExceeded("query deadline exhausted")
        return remaining


class CapacityExhausted(RuntimeError):
    pass


class Bulkhead:
    """Fail-fast bounded concurrency without an in-memory waiting queue."""
    def __init__(self, capacity: int, name: str):
        if capacity < 1:
            raise ValueError("bulkhead capacity must be positive")
        self.name = name
        self.capacity = capacity
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 0.0) -> None:
        if not self._semaphore.acquire(timeout=max(0.0, timeout)):
            raise CapacityExhausted(f"{self.name} capacity exhausted")
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1
        self._semaphore.release()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def run(self, operation: Callable[..., T], *args: Any, timeout: float = 0.0, **kwargs: Any) -> T:
        self.acquire(timeout)
        try:
            return operation(*args, **kwargs)
        finally:
            self.release()


class BoundedExecutor:
    """Thread pool with a semaphore bounding running plus queued operations."""
    def __init__(self, max_workers: int, max_pending: int, prefix: str):
        if max_pending < max_workers:
            raise ValueError("max_pending must be at least max_workers")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=prefix)
        self._slots = threading.BoundedSemaphore(max_pending)
        self.max_workers, self.max_pending = max_workers, max_pending

    def submit(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        if not self._slots.acquire(blocking=False):
            raise CapacityExhausted("executor queue capacity exhausted")
        try:
            future = self._executor.submit(operation, *args, **kwargs)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(ConnectionError):
    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 15,
                 half_open_probes: int = 1, clock: Callable[[], float] = time.monotonic):
        if failure_threshold < 1 or recovery_timeout <= 0 or half_open_probes < 1:
            raise ValueError("unsafe circuit breaker configuration")
        self.name, self.failure_threshold = name, failure_threshold
        self.recovery_timeout, self.half_open_probes = recovery_timeout, half_open_probes
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._probes = 0
            return self._state

    def _allow(self) -> None:
        state = self.state
        with self._lock:
            if state is CircuitState.OPEN:
                raise CircuitOpen(f"{self.name} circuit open")
            if state is CircuitState.HALF_OPEN:
                if self._probes >= self.half_open_probes:
                    raise CircuitOpen(f"{self.name} half-open probe capacity exhausted")
                self._probes += 1

    def success(self) -> None:
        with self._lock:
            self._state, self._failures, self._probes = CircuitState.CLOSED, 0, 0

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state, self._opened_at, self._probes = CircuitState.OPEN, self._clock(), 0

    def call(self, operation: Callable[..., T], *args: Any,
             transient: Callable[[BaseException], bool] | None = None, **kwargs: Any) -> T:
        self._allow()
        try:
            result = operation(*args, **kwargs)
        except BaseException as error:
            should_count = transient(error) if transient else isinstance(error, Exception) and not isinstance(
                error, (CapacityExhausted, DeadlineExceeded, CircuitOpen)
            )
            if should_count:
                self.failure()
            raise
        self.success()
        return result

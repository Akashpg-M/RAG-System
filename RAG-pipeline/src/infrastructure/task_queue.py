import queue
import threading
from typing import Callable, Optional


class BackgroundWorkQueue:
    """Single-process FIFO adapter for short request acceptance and background work."""

    def __init__(self):
        self._queue: queue.Queue[object] = queue.Queue()
        self._sentinel = object()
        self._thread = threading.Thread(target=self._run, daemon=True, name="document-ingestion-worker")
        self._thread.start()

    def enqueue(self, work: Callable[[], None]) -> None:
        self._queue.put(work)

    def depth(self) -> int:
        return self._queue.qsize()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.put(self._sentinel)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item: Optional[object] = None
            try:
                item = self._queue.get()
                if item is self._sentinel:
                    return
                if callable(item):
                    item()
            finally:
                if item is not None:
                    self._queue.task_done()


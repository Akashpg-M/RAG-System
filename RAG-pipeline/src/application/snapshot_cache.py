from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class PublicationSnapshotCache:
    """Immutable, revision-keyed snapshot cache; revision is checked in PostgreSQL first."""
    def __init__(self, max_entries: int = 8, ttl_seconds: float = 300):
        self.max_entries, self.ttl_seconds = max_entries, ttl_seconds
        self._entries: OrderedDict[int, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, revision: int) -> Any:
        with self._lock:
            value = self._entries.get(revision)
            if not value:
                return None
            created, snapshot = value
            if time.monotonic() - created > self.ttl_seconds:
                self._entries.pop(revision, None)
                return None
            self._entries.move_to_end(revision)
            return snapshot

    def put(self, revision: int, snapshot: Any) -> None:
        with self._lock:
            self._entries[revision] = (time.monotonic(), snapshot)
            self._entries.move_to_end(revision)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def load(self, publication: Any) -> Any:
        revision_reader = getattr(publication, "current_revision", None)
        if not revision_reader:
            snapshot = publication.snapshot()
            self.put(snapshot.revision, snapshot)
            return snapshot
        revision = int(revision_reader())
        cached = self.get(revision)
        if cached is not None:
            return cached
        snapshot = publication.snapshot()
        self.put(snapshot.revision, snapshot)
        return snapshot

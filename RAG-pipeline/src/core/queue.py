from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class QueueSaturated(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    body: str
    receipt: str
    attempts: int = 1
    enqueued_at: Optional[float] = None


@dataclass(frozen=True)
class QueueStats:
    depth: int = 0
    pending: int = 0
    retries: int = 0
    dlq: int = 0
    oldest_age_seconds: Optional[float] = None


class IngestionQueue(Protocol):
    def publish(self, body: str, event_id: str) -> str: ...
    def receive(self, timeout_seconds: float) -> Optional[QueueMessage]: ...
    def acknowledge(self, message: QueueMessage) -> None: ...
    def retry(self, message: QueueMessage, delay_seconds: float) -> None: ...
    def dead_letter(self, message: QueueMessage, reason: str) -> None: ...
    def heartbeat(self, message: QueueMessage) -> None: ...
    def stats(self) -> QueueStats: ...
    def is_ready(self) -> bool: ...
    def close(self) -> None: ...

from __future__ import annotations

import heapq
import json
import threading
import time
from collections import deque
from typing import Any, Optional

from src.core.queue import IngestionQueue, QueueMessage, QueueSaturated, QueueStats


class InMemoryIngestionQueue(IngestionQueue):
    """Deterministic transport for tests; acknowledgements are explicit."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._ready: deque[QueueMessage] = deque()
        self._pending: dict[str, QueueMessage] = {}
        self._delayed: list[tuple[float, str, QueueMessage]] = []
        self._dlq: list[QueueMessage] = []
        self._published: dict[str, str] = {}
        self._condition = threading.Condition()

    def publish(self, body: str, event_id: str) -> str:
        with self._condition:
            if event_id in self._published:
                return self._published[event_id]
            if len(self._ready) + len(self._pending) + len(self._delayed) >= self.capacity:
                raise QueueSaturated("ingestion queue capacity reached")
            message_id = event_id
            self._published[event_id] = message_id
            self._ready.append(QueueMessage(message_id, body, message_id, 1, time.time()))
            self._condition.notify()
            return message_id

    def receive(self, timeout_seconds: float) -> Optional[QueueMessage]:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                now = time.time()
                while self._delayed and self._delayed[0][0] <= now:
                    _, _, message = heapq.heappop(self._delayed)
                    self._ready.append(message)
                if self._ready:
                    message = self._ready.popleft()
                    self._pending[message.receipt] = message
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.05))

    def acknowledge(self, message: QueueMessage) -> None:
        with self._condition:
            self._pending.pop(message.receipt, None)

    def retry(self, message: QueueMessage, delay_seconds: float) -> None:
        with self._condition:
            self._pending.pop(message.receipt, None)
            retry = QueueMessage(message.message_id, message.body, message.receipt, message.attempts + 1,
                                 message.enqueued_at)
            heapq.heappush(self._delayed, (time.time() + delay_seconds, message.message_id, retry))
            self._condition.notify()

    def dead_letter(self, message: QueueMessage, reason: str) -> None:
        with self._condition:
            self._pending.pop(message.receipt, None)
            self._dlq.append(message)

    def heartbeat(self, message: QueueMessage) -> None:
        return None

    def stats(self) -> QueueStats:
        with self._condition:
            ages = [time.time() - item.enqueued_at for item in self._ready if item.enqueued_at]
            return QueueStats(len(self._ready) + len(self._delayed), len(self._pending),
                              sum(item.attempts - 1 for item in self._pending.values()), len(self._dlq),
                              max(ages) if ages else None)

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        with self._condition:
            self._condition.notify_all()


class RedisStreamsQueue(IngestionQueue):
    """Redis Streams consumer-group adapter with a sorted-set delayed retry schedule."""
    def __init__(self, client: Any, stream: str = "rag:ingestion", group: str = "rag-workers",
                 consumer: str = "worker", capacity: int = 10000, claim_idle_ms: int = 60000,
                 dlq_stream: Optional[str] = None, retry_set: Optional[str] = None):
        self.client, self.stream, self.group, self.consumer = client, stream, group, consumer
        self.capacity, self.claim_idle_ms = capacity, claim_idle_ms
        self.dlq_stream = dlq_stream or f"{stream}:dlq"
        self.retry_set = retry_set or f"{stream}:retries"
        self.dedup = f"{stream}:published"
        try:
            client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as error:
            if "BUSYGROUP" not in str(error):
                raise

    def publish(self, body: str, event_id: str) -> str:
        existing = self.client.hget(self.dedup, event_id)
        if existing:
            return self._text(existing)
        stats = self.stats()
        if stats.depth + stats.pending >= self.capacity:
            raise QueueSaturated("ingestion queue capacity reached")
        message_id = self._text(self.client.xadd(self.stream, {"body": body, "attempts": "1", "event_id": event_id}))
        self.client.hset(self.dedup, event_id, message_id)
        return message_id

    def _dispatch_due(self, limit: int = 100) -> None:
        now = time.time()
        values = self.client.zrangebyscore(self.retry_set, 0, now, start=0, num=limit)
        for raw in values:
            encoded = self._text(raw)
            if self.client.zrem(self.retry_set, raw):
                data = json.loads(encoded)
                self.client.xadd(self.stream, {"body": data["body"], "attempts": str(data["attempts"]),
                                               "event_id": data["event_id"]})

    def receive(self, timeout_seconds: float) -> Optional[QueueMessage]:
        self._dispatch_due()
        try:
            claimed = self.client.xautoclaim(self.stream, self.group, self.consumer, self.claim_idle_ms, "0-0", count=1)
            entries = claimed[1] if claimed and len(claimed) > 1 else []
            if entries:
                return self._message(entries[0])
        except Exception:
            pass
        result = self.client.xreadgroup(self.group, self.consumer, {self.stream: ">"}, count=1,
                                        block=max(1, int(timeout_seconds * 1000)))
        if not result:
            return None
        return self._message(result[0][1][0])

    def _message(self, entry: Any) -> QueueMessage:
        message_id, fields = entry
        normalized = {self._text(k): self._text(v) for k, v in fields.items()}
        timestamp = int(self._text(message_id).split("-", 1)[0]) / 1000
        attempts = int(normalized.get("attempts", "1"))
        try:
            pending = self.client.xpending_range(self.stream, self.group, self._text(message_id),
                                                 self._text(message_id), 1)
            if pending:
                delivery = pending[0].get("times_delivered", pending[0].get(b"times_delivered", attempts))
                attempts = max(attempts, int(delivery))
        except Exception:
            pass
        return QueueMessage(self._text(message_id), normalized["body"], self._text(message_id), attempts, timestamp)

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def acknowledge(self, message: QueueMessage) -> None:
        self.client.xack(self.stream, self.group, message.receipt)

    def retry(self, message: QueueMessage, delay_seconds: float) -> None:
        event_id = message.message_id
        try:
            event_id = IngestionEvent.from_json(message.body).event_id
        except Exception:
            pass
        payload = json.dumps({"body": message.body, "attempts": message.attempts + 1, "event_id": event_id},
                             sort_keys=True)
        self.client.zadd(self.retry_set, {payload: time.time() + delay_seconds})
        self.acknowledge(message)

    def dead_letter(self, message: QueueMessage, reason: str) -> None:
        self.client.xadd(self.dlq_stream, {"body": message.body, "attempts": str(message.attempts),
                                          "reason": reason})
        self.acknowledge(message)

    def heartbeat(self, message: QueueMessage) -> None:
        try:
            self.client.xclaim(self.stream, self.group, self.consumer, 0, [message.receipt], justid=True)
        except TypeError:
            self.client.xclaim(self.stream, self.group, self.consumer, 0, [message.receipt])

    def stats(self) -> QueueStats:
        info = self.client.xinfo_stream(self.stream)
        groups = self.client.xinfo_groups(self.stream)
        group = next((g for g in groups if self._text(g.get("name", g.get(b"name", ""))) == self.group), {})
        length = int(info.get("length", info.get(b"length", 0)))
        pending = int(group.get("pending", group.get(b"pending", 0)))
        dlq = int(self.client.xlen(self.dlq_stream))
        retries = int(self.client.zcard(self.retry_set))
        return QueueStats(max(0, length - pending), pending, retries, dlq, None)

    def is_ready(self) -> bool:
        return bool(self.client.ping())

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


class SQSQueue(IngestionQueue):
    def __init__(self, client: Any, queue_url: str, dlq_url: str, visibility_timeout: int = 120,
                 capacity: int = 10000, wait_time_seconds: int = 20):
        if not queue_url or not dlq_url:
            raise ValueError("SQS queue and DLQ URLs are required")
        self.client, self.queue_url, self.dlq_url = client, queue_url, dlq_url
        self.visibility_timeout, self.capacity = visibility_timeout, capacity
        self.wait_time_seconds = min(20, max(0, wait_time_seconds))

    def publish(self, body: str, event_id: str) -> str:
        if self.stats().depth >= self.capacity:
            raise QueueSaturated("ingestion queue capacity reached")
        attributes = {"event_id": {"DataType": "String", "StringValue": event_id}}
        try:
            carrier = IngestionEvent.from_json(body).trace_context
            for key in ("traceparent", "tracestate"):
                if carrier.get(key):
                    attributes[key] = {"DataType": "String", "StringValue": carrier[key]}
        except Exception:
            pass
        response = self.client.send_message(QueueUrl=self.queue_url, MessageBody=body,
                                            MessageAttributes=attributes)
        return response["MessageId"]

    def receive(self, timeout_seconds: float) -> Optional[QueueMessage]:
        response = self.client.receive_message(
            QueueUrl=self.queue_url, MaxNumberOfMessages=1,
            WaitTimeSeconds=min(self.wait_time_seconds, int(max(0, timeout_seconds))),
            VisibilityTimeout=self.visibility_timeout,
            AttributeNames=["ApproximateReceiveCount", "SentTimestamp"],
            MessageAttributeNames=["All"],
        )
        if not response.get("Messages"):
            return None
        raw = response["Messages"][0]
        attributes = raw.get("Attributes", {})
        body = raw["Body"]
        try:
            event = IngestionEvent.from_json(body)
            carrier = dict(event.trace_context)
            for key in ("traceparent", "tracestate"):
                value = raw.get("MessageAttributes", {}).get(key, {}).get("StringValue")
                if value:
                    carrier[key] = value
            if carrier != event.trace_context:
                from dataclasses import replace
                body = replace(event, trace_context=carrier).to_json()
        except Exception:
            pass
        return QueueMessage(raw["MessageId"], body, raw["ReceiptHandle"],
                            int(attributes.get("ApproximateReceiveCount", "1")),
                            float(attributes.get("SentTimestamp", "0")) / 1000 or None)

    def acknowledge(self, message: QueueMessage) -> None:
        self.client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=message.receipt)

    def retry(self, message: QueueMessage, delay_seconds: float) -> None:
        self.client.change_message_visibility(QueueUrl=self.queue_url, ReceiptHandle=message.receipt,
                                              VisibilityTimeout=min(43200, max(0, int(delay_seconds))))

    def dead_letter(self, message: QueueMessage, reason: str) -> None:
        attributes = {"failure_code": {"DataType": "String", "StringValue": reason}}
        try:
            for key, value in IngestionEvent.from_json(message.body).trace_context.items():
                if key in ("traceparent", "tracestate") and value:
                    attributes[key] = {"DataType": "String", "StringValue": value}
        except Exception:
            pass
        self.client.send_message(QueueUrl=self.dlq_url, MessageBody=message.body,
                                 MessageAttributes=attributes)
        self.acknowledge(message)

    def heartbeat(self, message: QueueMessage) -> None:
        self.client.change_message_visibility(QueueUrl=self.queue_url, ReceiptHandle=message.receipt,
                                              VisibilityTimeout=self.visibility_timeout)

    def stats(self) -> QueueStats:
        response = self.client.get_queue_attributes(QueueUrl=self.queue_url, AttributeNames=[
            "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"
        ])
        attrs = response.get("Attributes", {})
        return QueueStats(int(attrs.get("ApproximateNumberOfMessages", 0)),
                          int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)))

    def is_ready(self) -> bool:
        self.client.get_queue_attributes(QueueUrl=self.queue_url, AttributeNames=["QueueArn"])
        return True

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


# Delayed import avoids coupling the queue contracts to event serialization.
from src.core.events import IngestionEvent  # noqa: E402

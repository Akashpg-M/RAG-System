import os
import uuid

import pytest

from src.core.events import IngestionEvent
from src.infrastructure.ingestion_queues import RedisStreamsQueue


redis = pytest.importorskip("redis")


def live_redis():
    url = os.getenv("RAG_REDIS_TEST_URL", "redis://127.0.0.1:6379/15")
    client = redis.Redis.from_url(url)
    try:
        client.ping()
    except Exception as error:
        pytest.skip(f"real Redis unavailable: {type(error).__name__}")
    return client


@pytest.mark.integration
def test_real_redis_stream_ack_duplicate_retry_and_dlq():
    client = live_redis()
    suffix = uuid.uuid4().hex
    stream = f"test:rag:{suffix}"
    queue = RedisStreamsQueue(client, stream, f"group-{suffix}", f"consumer-{suffix}", capacity=10,
                              claim_idle_ms=1)
    event = IngestionEvent("event-1", "task-1", "document-1", "version-1", "local", "doc.md", "etag",
                           "pipeline", "/doc.md")
    first = queue.publish(event.to_json(), event.event_id)
    assert queue.publish(event.to_json(), event.event_id) == first
    message = queue.receive(1)
    assert message is not None
    queue.retry(message, 0)
    retried = queue.receive(1)
    assert retried is not None and retried.attempts >= 2
    queue.dead_letter(retried, "integration_test")
    assert queue.stats().dlq == 1
    client.delete(stream, f"{stream}:dlq", f"{stream}:retries", f"{stream}:published")

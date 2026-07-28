import json
import pytest

from src.application.ingestion_runtime import OutboxDispatcher, classify_failure
from src.core.contracts import IndexingStatus, IngestionTask
from src.core.events import IngestionEvent, idempotency_key, parse_s3_notification
from src.core.queue import QueueSaturated
from src.infrastructure.ingestion_queues import InMemoryIngestionQueue, SQSQueue
from src.infrastructure.repositories import SQLiteLeaseRepository, SQLiteTaskRepository


def event(event_id="event-1"):
    return IngestionEvent(
        event_id, "task-1", "document-1", "version-1", "local", "object.md", "etag-1", "pipeline-1",
        "/objects/object.md",
    )


def test_event_round_trip_version_and_stable_idempotency():
    original = event()
    assert IngestionEvent.from_json(original.to_json()) == original
    assert original.idempotency_key == idempotency_key("local", "object.md", "etag-1", "pipeline-1")
    assert original.idempotency_key != idempotency_key("local", "object.md", "etag-2", "pipeline-1")
    payload = json.loads(original.to_json())
    payload["schema_version"] = "99"
    with pytest.raises(ValueError):
        IngestionEvent.from_json(json.dumps(payload))


def test_s3_object_created_parsing_decodes_key_and_ignores_other_events():
    body = {"Records": [
        {"eventName": "ObjectCreated:Put", "s3": {"bucket": {"name": "docs"}, "object": {
            "key": "incoming%2Fhello+world.pdf", "versionId": "v3", "eTag": "ignored"}}},
        {"eventName": "ObjectRemoved:Delete", "s3": {"bucket": {"name": "docs"}, "object": {"key": "x"}}},
    ]}
    assert parse_s3_notification(json.dumps(body)) == [{
        "namespace": "docs", "object_key": "incoming/hello world.pdf", "object_version": "v3",
        "pipeline_version": "stage-3", "source_uri": "s3://docs/incoming/hello world.pdf",
    }]


def test_outbox_republication_is_safe_and_survives_repository_restart(tmp_path):
    path = str(tmp_path / "control.db")
    repository = SQLiteTaskRepository(path)
    task = IngestionTask("task-1", "source", "document-1", "version-1", IndexingStatus.QUEUED,
                         history=[IndexingStatus.QUEUED], idempotency_key=event().idempotency_key)
    repository.create_with_outbox(task, event().event_id, event().to_json())
    queue = InMemoryIngestionQueue()
    assert OutboxDispatcher(SQLiteTaskRepository(path), queue).dispatch_once() == 1
    # Marking is idempotent and a duplicate transport publication uses event_id de-duplication.
    repository.mark_published(event().event_id)
    assert OutboxDispatcher(repository, queue).dispatch_once() == 0
    assert queue.stats().depth == 1


def test_lease_acquire_renew_expire_fence_and_safe_release(tmp_path):
    leases = SQLiteLeaseRepository(str(tmp_path / "leases.db"))
    assert leases.acquire("doc:v1", "one", "token-one", 100, 10) == 1
    assert leases.acquire("doc:v1", "two", "token-two", 105, 10) is None
    assert leases.renew("doc:v1", "token-one", 1, 106, 10)
    assert leases.acquire("doc:v1", "two", "token-two", 117, 10) == 2
    assert not leases.renew("doc:v1", "token-one", 1, 118, 10)
    assert not leases.release("doc:v1", "token-one", 1)
    assert leases.owns("doc:v1", "token-two", 2, 118)
    assert leases.release("doc:v1", "token-two", 2)


def test_memory_queue_ack_retry_dlq_and_capacity():
    queue = InMemoryIngestionQueue(capacity=1)
    queue.publish(event().to_json(), event().event_id)
    assert queue.publish(event().to_json(), event().event_id) == event().event_id
    with pytest.raises(QueueSaturated):
        queue.publish(event("event-2").to_json(), "event-2")
    message = queue.receive(0.01)
    queue.retry(message, 0)
    retried = queue.receive(0.1)
    assert retried.attempts == 2
    queue.dead_letter(retried, "permanent")
    assert queue.stats().dlq == 1


class StubSqs:
    def __init__(self):
        self.calls = []

    def get_queue_attributes(self, **kwargs):
        self.calls.append(("attributes", kwargs))
        return {"Attributes": {"ApproximateNumberOfMessages": "1", "ApproximateNumberOfMessagesNotVisible": "1"}}

    def receive_message(self, **kwargs):
        self.calls.append(("receive", kwargs))
        return {"Messages": [{"MessageId": "m1", "ReceiptHandle": "r1", "Body": event().to_json(),
                              "Attributes": {"ApproximateReceiveCount": "2", "SentTimestamp": "1000"}}]}

    def delete_message(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def change_message_visibility(self, **kwargs):
        self.calls.append(("visibility", kwargs))

    def send_message(self, **kwargs):
        self.calls.append(("send", kwargs))
        return {"MessageId": "sent"}


def test_sqs_manual_ack_visibility_heartbeat_retry_and_dlq():
    client = StubSqs()
    queue = SQSQueue(client, "queue-url", "dlq-url", visibility_timeout=60)
    message = queue.receive(20)
    assert message.attempts == 2
    queue.heartbeat(message)
    queue.retry(message, 7)
    queue.dead_letter(message, "bad_document")
    operations = [call[0] for call in client.calls]
    assert operations.count("visibility") == 2
    assert operations[-2:] == ["send", "delete"]


@pytest.mark.parametrize(
    ("error", "retryable"),
    [(TimeoutError(), True), (ConnectionError(), True), (ValueError(), False), (RuntimeError(), True)],
)
def test_failure_classification(error, retryable):
    assert classify_failure(error, "parsing")[0] is retryable

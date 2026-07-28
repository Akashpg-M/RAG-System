import multiprocessing
import os
import uuid
from pathlib import Path

import pytest


DATABASE_URL = os.getenv(
    "RAG_POSTGRES_TEST_URL", "postgresql://rag:rag-local-only@127.0.0.1:5432/rag_control"
)
REDIS_URL = os.getenv("RAG_REDIS_TEST_URL", "redis://127.0.0.1:6379/14")


def api_process(root, stream, document_id, output):
    from fastapi.testclient import TestClient

    from src.api.app import create_api
    from src.application.composition import build_in_memory_application
    from src.application.config import Profile, profile_config
    from src.infrastructure.postgres import PostgresControlPlane

    test_config = profile_config(Profile.TEST, Path(root))
    config = test_config.model_copy(update={
        "profile": Profile.LOCAL,
        "providers": test_config.providers.model_copy(update={
            "task_repository": "postgres", "document_repository": "postgres",
        }),
        "queue": test_config.queue.model_copy(update={
            "backend": "redis", "redis_url": REDIS_URL, "stream_name": stream,
            "consumer_group": f"group:{stream}",
        }),
        "storage": test_config.storage.model_copy(update={"control_database_url": DATABASE_URL}),
    })
    application = build_in_memory_application(test_config)
    control = PostgresControlPlane(DATABASE_URL)
    for event_id, _, _ in control.pending_outbox(1000):
        control.mark_published(event_id)
    with TestClient(create_api(config, rag_application=application)) as client:
        response = client.post(
            "/api/v1/documents/upload", headers={"X-API-Key": config.api.api_key},
            data={"document_id": document_id},
            files={"file": ("distributed.md", b"distributed publication content", "text/markdown")},
        )
        response.raise_for_status()
        result = response.json()
        output.put((result["task_id"], result["version_id"]))


def worker_process(root, stream):
    import redis

    from src.application.composition import build_in_memory_application
    from src.application.config import Profile, profile_config
    from src.application.ingestion_runtime import IngestionWorker
    from src.infrastructure.ingestion_queues import RedisStreamsQueue
    from src.infrastructure.postgres import PostgresControlPlane
    from src.infrastructure.repositories import LocalFileObjectStorage

    config = profile_config(Profile.TEST, Path(root))
    control = PostgresControlPlane(DATABASE_URL)
    queue = RedisStreamsQueue(redis.Redis.from_url(REDIS_URL), stream, f"group:{stream}", "worker", 100)
    message = queue.receive(10)
    if message is None:
        raise RuntimeError("worker did not receive published ingestion event")
    worker = IngestionWorker(queue, control, control, control, LocalFileObjectStorage(),
                             build_in_memory_application(config), "distributed-worker", poll_timeout=1,
                             lease_duration=30, heartbeat_interval=5, publication=control)
    worker.process_message(message)


def services_available():
    try:
        import redis
        from src.infrastructure.postgres import PostgresControlPlane
        assert redis.Redis.from_url(REDIS_URL).ping()
        assert PostgresControlPlane(DATABASE_URL).is_ready()
    except Exception as error:
        pytest.skip(f"distributed services unavailable: {type(error).__name__}")


@pytest.mark.integration
def test_api_and_worker_separate_processes_share_postgres_and_redis(tmp_path):
    services_available()
    suffix = uuid.uuid4().hex
    stream, document_id = f"test:distributed:{suffix}", f"document-{suffix}"
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    api = context.Process(target=api_process, args=(str(tmp_path), stream, document_id, output))
    api.start()
    api.join(30)
    assert api.exitcode == 0
    task_id, version_id = output.get(timeout=2)
    worker = context.Process(target=worker_process, args=(str(tmp_path), stream))
    worker.start()
    worker.join(30)
    assert worker.exitcode == 0

    from src.core.contracts import IndexingStatus
    from src.infrastructure.postgres import PostgresControlPlane
    control = PostgresControlPlane(DATABASE_URL)
    assert control.get(task_id).status is IndexingStatus.READY
    assert control.snapshot().active_versions[document_id] == version_id

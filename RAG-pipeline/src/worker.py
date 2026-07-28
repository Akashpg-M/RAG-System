from __future__ import annotations

import logging
import os
import signal
import threading

from src.application.composition import build_application
from src.application.config import Profile, load_config
from src.application.ingestion_runtime import IngestionWorker
from src.infrastructure.ingestion_queues import RedisStreamsQueue, SQSQueue
from src.infrastructure.repositories import (
    LocalFileObjectStorage, SQLiteDocumentRepository, SQLiteLeaseRepository, SQLiteTaskRepository,
)


def build_worker() -> IngestionWorker:
    config = load_config(os.getenv("RAG_PROFILE", Profile.LOCAL.value))
    application = build_application(config, include_queue=False)
    if config.queue.backend == "redis":
        import redis
        queue = RedisStreamsQueue(redis.Redis.from_url(config.queue.redis_url), config.queue.stream_name,
                                  config.queue.consumer_group, config.queue.worker_id, config.queue.capacity,
                                  int(config.queue.lease_duration_seconds * 1000))
    elif config.queue.backend == "sqs":
        import boto3
        queue = SQSQueue(boto3.client("sqs"), config.queue.sqs_queue_url, config.queue.sqs_dlq_url,
                         config.queue.visibility_timeout_seconds, config.queue.capacity)
    else:
        raise ValueError("standalone worker requires redis or sqs queue backend")
    tasks = SQLiteTaskRepository(config.storage.control_db_path)
    return IngestionWorker(
        queue, tasks, SQLiteDocumentRepository(config.storage.control_db_path),
        SQLiteLeaseRepository(config.storage.control_db_path), LocalFileObjectStorage(), application,
        config.queue.worker_id, config.queue.max_concurrency, config.queue.poll_timeout_seconds,
        config.queue.lease_duration_seconds,
        min(config.queue.heartbeat_interval_seconds, config.queue.visibility_heartbeat_seconds)
        if config.queue.backend == "sqs" else config.queue.heartbeat_interval_seconds,
        config.queue.max_attempts, config.queue.retry_min_seconds, config.queue.retry_max_seconds,
        config.queue.shutdown_timeout_seconds,
    )


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    worker = build_worker()
    stopping = threading.Event()

    def stop(*_: object) -> None:
        if not stopping.is_set():
            stopping.set()
            worker.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        worker.run()
    finally:
        worker.queue.close()


if __name__ == "__main__":
    main()

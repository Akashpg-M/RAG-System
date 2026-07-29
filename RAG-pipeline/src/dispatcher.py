from __future__ import annotations

import os
import signal
import threading

from src.application.config import Profile, load_config
from src.application.ingestion_runtime import OutboxDispatcher
from src.infrastructure.ingestion_queues import RedisStreamsQueue, SQSQueue
from src.infrastructure.repositories import SQLiteDocumentRepository, SQLiteTaskRepository
from src.observability import Observability, configure_json_logging, set_observability


def main() -> None:
    config = load_config(os.getenv("RAG_PROFILE", Profile.LOCAL.value))
    telemetry = Observability(
        "rag-outbox-dispatcher", config.profile.value, config.observability.service_version,
        config.pipeline.pipeline_version,
        config.observability.otlp_endpoint if config.observability.enabled else "",
        config.observability.sample_ratio,
    )
    set_observability(telemetry)
    configure_json_logging("rag-outbox-dispatcher", config.profile.value, os.getenv("LOG_LEVEL", "INFO"))
    if config.providers.task_repository == "sqlite":
        tasks = SQLiteTaskRepository(config.storage.control_db_path)
        documents = SQLiteDocumentRepository(config.storage.control_db_path)
    else:
        from src.infrastructure.postgres import PostgresControlPlane
        tasks = documents = PostgresControlPlane(
            config.storage.control_database_url, config.publication.retention_versions,
            config.performance.postgres_pool_min, config.performance.postgres_pool_max,
        )
    if config.queue.backend == "redis":
        import redis
        queue = RedisStreamsQueue(
            redis.Redis.from_url(config.queue.redis_url), config.queue.stream_name,
            config.queue.consumer_group, f"dispatcher-{os.getpid()}", config.queue.capacity,
        )
    elif config.queue.backend == "sqs":
        import boto3
        queue = SQSQueue(
            boto3.client("sqs"), config.queue.sqs_queue_url, config.queue.sqs_dlq_url,
            config.queue.visibility_timeout_seconds, config.queue.capacity,
        )
    else:
        raise ValueError("standalone dispatcher requires redis or sqs queue backend")
    dispatcher = OutboxDispatcher(tasks, queue, telemetry)
    stopping = threading.Event()

    def stop(*_: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server = None
    if config.observability.dispatcher_metrics_port:
        from prometheus_client import start_http_server
        server, _ = start_http_server(
            config.observability.dispatcher_metrics_port, registry=telemetry.metrics.registry
        )
    try:
        while not stopping.is_set():
            dispatcher.reconcile_queued(documents, config.pipeline.pipeline_version)
            dispatcher.dispatch_once()
            stopping.wait(config.observability.queue_sample_interval_seconds)
    finally:
        queue.close()
        if server:
            server.shutdown()
        telemetry.shutdown()


if __name__ == "__main__":
    main()

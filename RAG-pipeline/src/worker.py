from __future__ import annotations

import os
import signal
import threading

from src.application.composition import build_application
from src.application.config import Profile, load_config
from src.application.ingestion_runtime import IngestionWorker
from src.application.publication_services import CleanupService
from src.infrastructure.ingestion_queues import RedisStreamsQueue, SQSQueue
from src.infrastructure.repositories import (
    LocalFileObjectStorage, SQLiteDocumentRepository, SQLiteLeaseRepository, SQLiteTaskRepository,
)
from src.infrastructure.publication import SQLitePublicationRepository
from src.observability import Observability, configure_json_logging, set_observability


def build_worker() -> IngestionWorker:
    config = load_config(os.getenv("RAG_PROFILE", Profile.LOCAL.value))
    observability = Observability(
        "rag-worker", config.profile.value, config.observability.service_version,
        config.pipeline.pipeline_version,
        config.observability.otlp_endpoint if config.observability.enabled else "",
        config.observability.sample_ratio,
    )
    set_observability(observability)
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
    if config.providers.task_repository == "sqlite":
        tasks = SQLiteTaskRepository(config.storage.control_db_path)
        documents = SQLiteDocumentRepository(config.storage.control_db_path)
        leases = SQLiteLeaseRepository(config.storage.control_db_path)
        publication = SQLitePublicationRepository(
            config.storage.control_db_path, config.publication.retention_versions
        )
    else:
        from src.infrastructure.postgres import PostgresControlPlane
        tasks = documents = leases = publication = PostgresControlPlane(
            config.storage.control_database_url, config.publication.retention_versions
        )
    storage = LocalFileObjectStorage()
    cleanup = CleanupService(publication, documents, storage, application.ingestion)
    worker = IngestionWorker(
        queue, tasks, documents, leases, storage, application,
        config.queue.worker_id, config.queue.max_concurrency, config.queue.poll_timeout_seconds,
        config.queue.lease_duration_seconds,
        min(config.queue.heartbeat_interval_seconds, config.queue.visibility_heartbeat_seconds)
        if config.queue.backend == "sqs" else config.queue.heartbeat_interval_seconds,
        config.queue.max_attempts, config.queue.retry_min_seconds, config.queue.retry_max_seconds,
        config.queue.shutdown_timeout_seconds,
        publication, config.pipeline.graph_index_required, cleanup.run_once, observability,
    )
    worker.metrics_port = config.observability.worker_metrics_port
    return worker


def main() -> None:
    configure_json_logging("rag-worker", os.getenv("RAG_PROFILE", Profile.LOCAL.value),
                           os.getenv("LOG_LEVEL", "INFO"))
    worker = build_worker()
    if worker.metrics_port:
        from prometheus_client import start_http_server
        start_http_server(worker.metrics_port, registry=worker.observability.metrics.registry)
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
        worker.observability.shutdown()


if __name__ == "__main__":
    main()

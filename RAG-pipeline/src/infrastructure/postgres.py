from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Tuple

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from src.core.contracts import DocumentVersion, IndexingStatus, IngestionTask
from src.core.publication import (
    IndexStageResult, ManifestEntry, PublicationSnapshot, PublicationValidationError, StaleFencingToken,
    manifest_checksum,
)
from src.observability import get_observability


_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def shared_pool(database_url: str, minimum: int = 1, maximum: int = 12) -> ConnectionPool:
    normalized = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with _pools_lock:
        pool = _pools.get(normalized)
        if pool is None:
            pool = ConnectionPool(normalized, min_size=minimum, max_size=maximum,
                                  kwargs={"row_factory": dict_row}, timeout=5, open=True)
            _pools[normalized] = pool
        return pool


class PostgresGraphIndex:
    def __init__(self, database_url: str, namespace: str = "default"):
        self.control = PostgresControlPlane(database_url)
        self.namespace = namespace

    def add_triples_bulk(self, triples: list[dict[str, Any]], chunk_id: Optional[str] = None) -> None:
        with self.control._connection() as connection:
            for triple in triples:
                target_chunk = str(triple.get("chunk_id") or chunk_id or "")
                parts = target_chunk.split("#")
                if len(parts) < 3:
                    continue
                document_id, version_id = parts[0], parts[1]
                source = str(triple.get("source", "")).strip().lower()
                target = str(triple.get("target", "")).strip().lower()
                relation = str(triple.get("relation", "")).strip().lower()
                if not source or not target or not relation:
                    continue
                source_id = connection.execute(
                    "INSERT INTO graph_entities(namespace,normalized_name,document_id,version_id) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT(namespace,normalized_name,document_id,version_id) DO UPDATE SET normalized_name=excluded.normalized_name "
                    "RETURNING entity_id", (self.namespace, source, document_id, version_id),
                ).fetchone()["entity_id"]
                target_id = connection.execute(
                    "INSERT INTO graph_entities(namespace,normalized_name,document_id,version_id) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT(namespace,normalized_name,document_id,version_id) DO UPDATE SET normalized_name=excluded.normalized_name "
                    "RETURNING entity_id", (self.namespace, target, document_id, version_id),
                ).fetchone()["entity_id"]
                connection.execute(
                    "INSERT INTO graph_relationships(document_id,version_id,source_entity_id,target_entity_id,relation,chunk_id) "
                    "SELECT %s,%s,%s,%s,%s,%s WHERE NOT EXISTS (SELECT 1 FROM graph_relationships WHERE "
                    "document_id=%s AND version_id=%s AND source_entity_id=%s AND target_entity_id=%s AND relation=%s AND chunk_id=%s)",
                    (document_id, version_id, source_id, target_id, relation, target_chunk,
                     document_id, version_id, source_id, target_id, relation, target_chunk),
                )

    def traverse_graph_hops(self, seed_entities: list[str], max_hops: int = 1) -> list[dict[str, Any]]:
        if not seed_entities:
            return []
        with self.control._connection() as connection:
            rows = connection.execute(
                "SELECT s.normalized_name source,r.relation,t.normalized_name target,r.chunk_id "
                "FROM graph_relationships r JOIN graph_entities s ON s.entity_id=r.source_entity_id "
                "JOIN graph_entities t ON t.entity_id=r.target_entity_id WHERE s.namespace=%s AND s.normalized_name=ANY(%s)",
                (self.namespace, [value.strip().lower() for value in seed_entities]),
            ).fetchall()
        return [{**row, "hop_level": 1} for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self.control._connection() as connection:
            connection.execute("DELETE FROM graph_relationships WHERE document_id=%s", (document_id,))
            connection.execute("DELETE FROM graph_entities WHERE document_id=%s", (document_id,))

    def delete_version(self, document_id: str, version_id: str) -> None:
        with self.control._connection() as connection:
            connection.execute("DELETE FROM graph_relationships WHERE document_id=%s AND version_id=%s",
                               (document_id, version_id))
            connection.execute("DELETE FROM graph_entities WHERE document_id=%s AND version_id=%s",
                               (document_id, version_id))

    def is_ready(self) -> bool:
        return self.control.is_ready()


class PostgresControlPlane:
    """Shared adapter for existing repository ports and Stage 4 publication transactions."""
    def __init__(self, database_url: str, retention_versions: int = 2,
                 pool_min: int = 1, pool_max: int = 12):
        self.database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        self.retention_versions = retention_versions
        self.pool = shared_pool(self.database_url, pool_min, pool_max)

    def _connection(self):
        @contextmanager
        def acquire():
            started = time.perf_counter()
            metrics = get_observability().metrics
            try:
                with self.pool.connection() as connection:
                    metrics.labels(metrics.pool_wait, pool="postgres").observe(time.perf_counter() - started)
                    stats = self.pool.get_stats()
                    checked_out = max(0, int(stats.get("pool_size", 0)) - int(stats.get("pool_available", 0)))
                    metrics.labels(metrics.pool_checked_out, pool="postgres").set(checked_out)
                    yield connection
            except Exception as error:
                if error.__class__.__name__ == "PoolTimeout":
                    metrics.labels(metrics.pool_exhaustions, pool="postgres").inc()
                raise
            finally:
                stats = self.pool.get_stats()
                checked_out = max(0, int(stats.get("pool_size", 0)) - int(stats.get("pool_available", 0)))
                metrics.labels(metrics.pool_checked_out, pool="postgres").set(checked_out)

        return acquire()

    def pool_stats(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.pool.get_stats().items()}

    def is_ready(self) -> bool:
        with self._connection() as connection:
            connection.execute("SELECT 1")
        return True

    def save(self, value: DocumentVersion | IngestionTask) -> None:
        if isinstance(value, DocumentVersion):
            self._save_version(value)
        else:
            self._save_task(value)

    def _save_version(self, version: DocumentVersion) -> None:
        metadata = version.metadata
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO documents(document_id,namespace) VALUES (%s,%s) ON CONFLICT(document_id) DO NOTHING",
                (version.document_id, metadata.get("namespace", "default")),
            )
            row = connection.execute(
                "SELECT source_uri,content_hash,metadata FROM document_versions WHERE document_id=%s AND version_id=%s",
                (version.document_id, version.version_id),
            ).fetchone()
            if row:
                if (row["source_uri"] != version.source_uri or row["content_hash"] != version.content_hash or
                        dict(row["metadata"]) != metadata):
                    raise ValueError("document versions are immutable")
                return
            connection.execute(
                "INSERT INTO document_versions VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (version.document_id, version.version_id, version.source_uri,
                 metadata.get("source_version", version.content_hash), version.content_hash,
                 metadata.get("pipeline_version", "stage-4"), metadata.get("parser_version", "unknown"),
                 metadata.get("chunker_config_version", "unknown"),
                 metadata.get("embedding_model_version", "unknown"),
                 metadata.get("index_schema_version", "unknown"), Jsonb(metadata), version.created_at),
            )
            connection.execute(
                "INSERT INTO version_publications(document_id,version_id,status) VALUES (%s,%s,'STAGING')",
                (version.document_id, version.version_id),
            )

    def get_latest(self, document_id: str) -> Optional[DocumentVersion]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT document_id,version_id,source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=%s ORDER BY created_at DESC LIMIT 1", (document_id,),
            ).fetchone()
        return self._version(row)

    def get_version(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT document_id,version_id,source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=%s AND version_id=%s", (document_id, version_id),
            ).fetchone()
        return self._version(row)

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT document_id,version_id,source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=%s ORDER BY created_at", (document_id,),
            ).fetchall()
        return [version for row in rows if (version := self._version(row)) is not None]

    @staticmethod
    def _version(row: Optional[dict[str, Any]]) -> Optional[DocumentVersion]:
        if not row:
            return None
        return DocumentVersion(row["document_id"], row["version_id"], row["source_uri"], row["content_hash"],
                               row["created_at"], dict(row["metadata"]))

    def delete(self, document_id: str) -> None:
        self.tombstone(document_id)

    def _save_task(self, task: IngestionTask) -> None:
        with self._connection() as connection:
            previous = connection.execute("SELECT status FROM ingestion_tasks WHERE task_id=%s", (task.task_id,)).fetchone()
            connection.execute(
                "INSERT INTO ingestion_tasks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,error_code=excluded.error_code,"
                "updated_at=excluded.updated_at,attempt_count=excluded.attempt_count,"
                "idempotency_key=excluded.idempotency_key,fencing_token=excluded.fencing_token",
                (task.task_id, task.source_uri, task.document_id, task.version_id, task.status.value, task.error,
                 task.created_at, task.updated_at, task.attempt_count, task.idempotency_key, task.fencing_token),
            )
            if not previous or previous["status"] != task.status.value:
                connection.execute(
                    "INSERT INTO ingestion_task_history(task_id,status,error_code,occurred_at) VALUES (%s,%s,%s,%s)",
                    (task.task_id, task.status.value, task.error, task.updated_at),
                )

    def get(self, task_id: str) -> Optional[IngestionTask]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM ingestion_tasks WHERE task_id=%s", (task_id,)).fetchone()
            if not row:
                return None
            history = connection.execute(
                "SELECT status FROM ingestion_task_history WHERE task_id=%s ORDER BY sequence", (task_id,)
            ).fetchall()
        return IngestionTask(task_id=row["task_id"], source_uri=row["source_uri"], document_id=row["document_id"],
                             version_id=row["version_id"], status=IndexingStatus(row["status"]),
                             error=row["error_code"], created_at=row["created_at"], updated_at=row["updated_at"],
                             history=[IndexingStatus(item["status"]) for item in history],
                             attempt_count=row["attempt_count"], idempotency_key=row["idempotency_key"],
                             fencing_token=row["fencing_token"])

    def get_latest_for_document(self, document_id: str) -> Optional[IngestionTask]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT task_id FROM ingestion_tasks WHERE document_id=%s ORDER BY updated_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        return self.get(row["task_id"]) if row else None

    def get_by_idempotency_key(self, key: str) -> Optional[IngestionTask]:
        with self._connection() as connection:
            row = connection.execute("SELECT task_id FROM idempotency_claims WHERE idempotency_key=%s", (key,)).fetchone()
        return self.get(row["task_id"]) if row else None

    def create_with_outbox(self, task: IngestionTask, event_id: str, payload: str) -> Tuple[IngestionTask, bool]:
        with self._connection() as connection:
            if task.idempotency_key:
                existing = connection.execute(
                    "SELECT task_id FROM idempotency_claims WHERE idempotency_key=%s FOR UPDATE",
                    (task.idempotency_key,),
                ).fetchone()
                if existing:
                    connection.rollback()
                    loaded = self.get(existing["task_id"])
                    if loaded is None:
                        raise RuntimeError("idempotency claim refers to missing task")
                    return loaded, False
            self._save_task_in(connection, task)
            connection.execute("INSERT INTO outbox_events(event_id,task_id,payload) VALUES (%s,%s,%s)",
                               (event_id, task.task_id, Jsonb(json.loads(payload))))
            if task.idempotency_key:
                connection.execute("INSERT INTO idempotency_claims VALUES (%s,%s,%s,%s,now())",
                                   (task.idempotency_key, task.document_id, task.version_id, task.task_id))
        return task, True

    @staticmethod
    def _save_task_in(connection: Any, task: IngestionTask) -> None:
        connection.execute(
            "INSERT INTO ingestion_tasks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (task.task_id, task.source_uri, task.document_id, task.version_id, task.status.value, task.error,
             task.created_at, task.updated_at, task.attempt_count, task.idempotency_key, task.fencing_token),
        )
        for status in task.history:
            connection.execute("INSERT INTO ingestion_task_history(task_id,status,occurred_at) VALUES (%s,%s,%s)",
                               (task.task_id, status.value, task.updated_at))

    def pending_outbox(self, limit: int = 100) -> list[tuple[str, str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id,task_id,payload FROM outbox_events WHERE published_at IS NULL ORDER BY created_at LIMIT %s",
                (limit,),
            ).fetchall()
        return [(row["event_id"], row["task_id"], json.dumps(row["payload"])) for row in rows]

    def mark_published(self, event_id: str) -> None:
        with self._connection() as connection:
            connection.execute("UPDATE outbox_events SET published_at=now(),publish_attempts=publish_attempts+1,"
                               "last_error_code=NULL WHERE event_id=%s", (event_id,))

    def mark_publish_failed(self, event_id: str) -> None:
        with self._connection() as connection:
            connection.execute("UPDATE outbox_events SET publish_attempts=publish_attempts+1,"
                               "last_error_code='queue_publish_failed' WHERE event_id=%s", (event_id,))

    def queued_without_event(self, limit: int = 100) -> list[IngestionTask]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT task_id FROM ingestion_tasks t WHERE status='QUEUED' AND NOT EXISTS "
                "(SELECT 1 FROM outbox_events o WHERE o.task_id=t.task_id) LIMIT %s", (limit,),
            ).fetchall()
        return [task for row in rows if (task := self.get(row["task_id"]))]

    def add_outbox(self, event_id: str, task_id: str, payload: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                "INSERT INTO outbox_events(event_id,task_id,payload) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (event_id, task_id, Jsonb(json.loads(payload))),
            )
            return result.rowcount == 1

    def acquire(self, resource_id: str, worker_id: str, token: str, now: float, duration: float) -> Optional[int]:
        instant = datetime.fromtimestamp(now, timezone.utc)
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM worker_leases WHERE resource_id=%s FOR UPDATE", (resource_id,)).fetchone()
            if row and row["expires_at"] > instant:
                return None
            fence = (row["fencing_token"] if row else 0) + 1
            connection.execute(
                "INSERT INTO worker_leases VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(resource_id) DO UPDATE SET "
                "worker_id=excluded.worker_id,ownership_token=excluded.ownership_token,fencing_token=excluded.fencing_token,"
                "expires_at=excluded.expires_at,updated_at=excluded.updated_at",
                (resource_id, worker_id, token, fence, instant + __import__("datetime").timedelta(seconds=duration), instant),
            )
            return fence

    def renew(self, resource_id: str, token: str, fencing: int, now: float, duration: float) -> bool:
        instant = datetime.fromtimestamp(now, timezone.utc)
        with self._connection() as connection:
            result = connection.execute(
                "UPDATE worker_leases SET expires_at=%s,updated_at=%s WHERE resource_id=%s AND ownership_token=%s "
                "AND fencing_token=%s AND expires_at>%s",
                (instant + __import__("datetime").timedelta(seconds=duration), instant, resource_id, token, fencing, instant),
            )
            return result.rowcount == 1

    def owns(self, resource_id: str, token: str, fencing: int, now: float) -> bool:
        with self._connection() as connection:
            return bool(connection.execute(
                "SELECT 1 FROM worker_leases WHERE resource_id=%s AND ownership_token=%s AND fencing_token=%s "
                "AND expires_at>%s", (resource_id, token, fencing, datetime.fromtimestamp(now, timezone.utc)),
            ).fetchone())

    def release(self, resource_id: str, token: str, fencing: int) -> bool:
        with self._connection() as connection:
            return connection.execute("UPDATE worker_leases SET expires_at=to_timestamp(0),updated_at=now() "
                                      "WHERE resource_id=%s AND ownership_token=%s AND fencing_token=%s",
                                      (resource_id, token, fencing)).rowcount == 1

    def register_version(self, document_id: str, version_id: str) -> None:
        return None  # document save creates the publication row transactionally

    def save_manifest(self, document_id: str, version_id: str, entries: Iterable[ManifestEntry]) -> str:
        values, checksum = list(entries), ""
        checksum = manifest_checksum(values)
        with self._connection() as connection:
            row = connection.execute("SELECT expected_chunk_count,manifest_checksum FROM version_publications "
                                     "WHERE document_id=%s AND version_id=%s FOR UPDATE", (document_id, version_id)).fetchone()
            if row["manifest_checksum"]:
                if row["expected_chunk_count"] != len(values) or row["manifest_checksum"] != checksum:
                    raise PublicationValidationError("immutable manifest mismatch")
                return checksum
            with connection.cursor() as cursor:
                cursor.executemany("INSERT INTO chunk_manifests VALUES (%s,%s,%s,%s,%s,%s,%s)", [
                    (document_id, version_id, e.chunk_id, e.parent_id, e.content_hash, e.ordinal, Jsonb(e.metadata))
                    for e in values
                ])
            connection.execute("UPDATE version_publications SET expected_chunk_count=%s,manifest_checksum=%s "
                               "WHERE document_id=%s AND version_id=%s", (len(values), checksum, document_id, version_id))
        return checksum

    def record_stage(self, document_id: str, version_id: str, result: IndexStageResult) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO index_stage_results VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT(document_id,version_id,index_name) DO UPDATE SET outcome=excluded.outcome,"
                "chunk_count=excluded.chunk_count,checksum=excluded.checksum,duration_seconds=excluded.duration_seconds,"
                "error_code=excluded.error_code,completed_at=now()",
                (document_id, version_id, result.index_name, result.outcome, result.chunk_count, result.checksum,
                 result.duration_seconds, result.error_code),
            )

    def activate(self, document_id: str, version_id: str, resource_id: str, ownership_token: str,
                 fencing_token: int, required_indexes: Iterable[str]) -> tuple[int, bool]:
        required = set(required_indexes)
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (document_id,))
            if connection.execute("SELECT 1 FROM deletion_tombstones WHERE document_id=%s", (document_id,)).fetchone():
                raise PublicationValidationError("tombstoned document")
            lease = connection.execute("SELECT * FROM worker_leases WHERE resource_id=%s FOR UPDATE", (resource_id,)).fetchone()
            if not lease or lease["ownership_token"] != ownership_token or lease["fencing_token"] != fencing_token \
                    or lease["expires_at"] <= datetime.now(timezone.utc):
                raise StaleFencingToken("stale publication worker")
            current = connection.execute("SELECT * FROM active_versions WHERE document_id=%s FOR UPDATE",
                                         (document_id,)).fetchone()
            if current and current["version_id"] == version_id:
                return current["revision"], False
            publication = connection.execute("SELECT * FROM version_publications WHERE document_id=%s AND version_id=%s",
                                             (document_id, version_id)).fetchone()
            stages = {row["index_name"]: row for row in connection.execute(
                "SELECT * FROM index_stage_results WHERE document_id=%s AND version_id=%s", (document_id, version_id)
            ).fetchall()}
            for name in required:
                result = stages.get(name)
                if not publication or not result or result["outcome"] != "SUCCESS" or \
                        result["chunk_count"] != publication["expected_chunk_count"] or \
                        result["checksum"] != publication["manifest_checksum"]:
                    raise PublicationValidationError(f"mandatory stage incomplete: {name}")
            revision = connection.execute("UPDATE corpus_publication SET revision=revision+1 RETURNING revision").fetchone()["revision"]
            connection.execute("INSERT INTO publication_revisions(revision,action,document_id,version_id) "
                               "VALUES (%s,'ACTIVATE',%s,%s)", (revision, document_id, version_id))
            previous = current["version_id"] if current else None
            connection.execute("INSERT INTO active_versions VALUES (%s,%s,%s,now()) ON CONFLICT(document_id) DO UPDATE "
                               "SET version_id=excluded.version_id,revision=excluded.revision,activated_at=now()",
                               (document_id, version_id, revision))
            degraded = any(row["outcome"] != "SUCCESS" for name, row in stages.items() if name not in required)
            connection.execute("UPDATE version_publications SET status='ACTIVE',degraded=%s,completed_at=COALESCE(completed_at,now()),"
                               "activated_at=now(),activation_fence=%s WHERE document_id=%s AND version_id=%s",
                               (degraded, fencing_token, document_id, version_id))
            if previous and previous != version_id:
                connection.execute("UPDATE version_publications SET status='RETIRING',retired_at=now() "
                                   "WHERE document_id=%s AND version_id=%s", (document_id, previous))
                connection.execute("INSERT INTO cleanup_jobs(job_id,document_id,version_id,job_type,status) "
                                   "VALUES (%s,%s,%s,'RETIRE_VERSION','HELD') ON CONFLICT DO NOTHING",
                                   (uuid.uuid4().hex, document_id, previous))
                stale = connection.execute(
                    "SELECT version_id FROM version_publications WHERE document_id=%s AND status IN "
                    "('ACTIVE','RETIRING','RETIRED') ORDER BY activated_at DESC NULLS LAST OFFSET %s",
                    (document_id, self.retention_versions),
                ).fetchall()
                for row in stale:
                    connection.execute("UPDATE cleanup_jobs SET status='PENDING',updated_at=now() WHERE document_id=%s "
                                       "AND version_id=%s AND job_type='RETIRE_VERSION'", (document_id, row["version_id"]))
                    connection.execute("UPDATE version_publications SET status='RETIRED' WHERE document_id=%s AND version_id=%s",
                                       (document_id, row["version_id"]))
            return revision, True

    def snapshot(self) -> PublicationSnapshot:
        with self._connection() as connection:
            revision = connection.execute("SELECT revision FROM corpus_publication").fetchone()["revision"]
            active = {row["document_id"]: row["version_id"] for row in connection.execute(
                "SELECT document_id,version_id FROM active_versions WHERE revision<=%s", (revision,)).fetchall()}
            tombstones = frozenset(row["document_id"] for row in connection.execute(
                "SELECT document_id FROM deletion_tombstones WHERE revision<=%s", (revision,)).fetchall())
            degraded = frozenset((row["document_id"], row["version_id"]) for row in connection.execute(
                "SELECT document_id,version_id FROM version_publications WHERE degraded=true AND status='ACTIVE'"
            ).fetchall())
        return PublicationSnapshot(revision, active, tombstones, degraded)

    def current_revision(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT revision FROM corpus_publication").fetchone()["revision"])

    def tombstone(self, document_id: str) -> tuple[int, bool]:
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (document_id,))
            row = connection.execute("SELECT revision FROM deletion_tombstones WHERE document_id=%s", (document_id,)).fetchone()
            if row:
                return row["revision"], False
            revision = connection.execute("UPDATE corpus_publication SET revision=revision+1 RETURNING revision").fetchone()["revision"]
            connection.execute("INSERT INTO publication_revisions(revision,action,document_id) "
                               "VALUES (%s,'TOMBSTONE',%s)", (revision, document_id))
            connection.execute("INSERT INTO deletion_tombstones VALUES (%s,%s,now(),NULL)", (document_id, revision))
            connection.execute("DELETE FROM active_versions WHERE document_id=%s", (document_id,))
            connection.execute("UPDATE version_publications SET status='TOMBSTONED' WHERE document_id=%s", (document_id,))
            connection.execute("INSERT INTO cleanup_jobs(job_id,document_id,job_type,status) "
                               "VALUES (%s,%s,'DELETE_DOCUMENT','PENDING') ON CONFLICT DO NOTHING",
                               (uuid.uuid4().hex, document_id))
            return revision, True

    def rollback(self, document_id: str, version_id: str) -> tuple[int, bool]:
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (document_id,))
            if connection.execute("SELECT 1 FROM deletion_tombstones WHERE document_id=%s", (document_id,)).fetchone():
                raise PublicationValidationError("tombstoned document")
            target = connection.execute("SELECT status FROM version_publications WHERE document_id=%s AND version_id=%s",
                                        (document_id, version_id)).fetchone()
            if not target or target["status"] not in ("ACTIVE", "RETIRING", "RETIRED"):
                raise PublicationValidationError("invalid rollback target")
            current = connection.execute("SELECT * FROM active_versions WHERE document_id=%s FOR UPDATE", (document_id,)).fetchone()
            if current and current["version_id"] == version_id:
                return current["revision"], False
            revision = connection.execute("UPDATE corpus_publication SET revision=revision+1 RETURNING revision").fetchone()["revision"]
            connection.execute("INSERT INTO publication_revisions(revision,action,document_id,version_id) "
                               "VALUES (%s,'ROLLBACK',%s,%s)", (revision, document_id, version_id))
            connection.execute("INSERT INTO active_versions VALUES (%s,%s,%s,now()) ON CONFLICT(document_id) DO UPDATE "
                               "SET version_id=excluded.version_id,revision=excluded.revision,activated_at=now()",
                               (document_id, version_id, revision))
            connection.execute("UPDATE version_publications SET status='ACTIVE',activated_at=now() "
                               "WHERE document_id=%s AND version_id=%s", (document_id, version_id))
            if current:
                connection.execute("UPDATE version_publications SET status='RETIRING',retired_at=now() "
                                   "WHERE document_id=%s AND version_id=%s", (document_id, current["version_id"]))
            return revision, True

    def manifest(self, document_id: str, version_id: str) -> list[ManifestEntry]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM chunk_manifests WHERE document_id=%s AND version_id=%s ORDER BY ordinal",
                                      (document_id, version_id)).fetchall()
        return [ManifestEntry(row["chunk_id"], row["parent_id"], row["content_hash"], row["ordinal"],
                              dict(row["metadata"])) for row in rows]

    def pending_cleanup(self) -> list[tuple[str, str, Optional[str], str]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT job_id,document_id,version_id,job_type FROM cleanup_jobs "
                                      "WHERE status IN ('PENDING','FAILED')").fetchall()
        return [(row["job_id"], row["document_id"], row["version_id"], row["job_type"]) for row in rows]

    def claim_cleanup(self, job_id: str) -> bool:
        """Claim only conclusively stale content and make cleanup mutually exclusive with rollback."""
        with self._connection() as connection:
            job = connection.execute(
                "SELECT * FROM cleanup_jobs WHERE job_id=%s FOR UPDATE", (job_id,)
            ).fetchone()
            if not job or job["status"] not in ("PENDING", "FAILED"):
                return False
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (job["document_id"],))
            if job["job_type"] == "RETIRE_VERSION":
                active = connection.execute(
                    "SELECT 1 FROM active_versions WHERE document_id=%s AND version_id=%s",
                    (job["document_id"], job["version_id"]),
                ).fetchone()
                version = connection.execute(
                    "SELECT status FROM version_publications WHERE document_id=%s AND version_id=%s FOR UPDATE",
                    (job["document_id"], job["version_id"]),
                ).fetchone()
                if active or not version or version["status"] != "RETIRED":
                    return False
                connection.execute(
                    "UPDATE version_publications SET status='CLEANING' WHERE document_id=%s AND version_id=%s",
                    (job["document_id"], job["version_id"]),
                )
            elif not connection.execute(
                "SELECT 1 FROM deletion_tombstones WHERE document_id=%s", (job["document_id"],)
            ).fetchone():
                return False
            connection.execute("UPDATE cleanup_jobs SET status='RUNNING',updated_at=now() WHERE job_id=%s", (job_id,))
            return True

    def abandoned_staging(self, age_seconds: float) -> list[tuple[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT p.document_id,p.version_id FROM version_publications p JOIN document_versions v USING(document_id,version_id) "
                "WHERE p.status='STAGING' AND v.created_at < now()-(%s * interval '1 second')", (age_seconds,),
            ).fetchall()
        return [(row["document_id"], row["version_id"]) for row in rows]

    def retired_awaiting_cleanup(self) -> int:
        with self._connection() as connection:
            return connection.execute("SELECT COUNT(*) count FROM cleanup_jobs WHERE job_type='RETIRE_VERSION' "
                                      "AND status IN ('PENDING','FAILED')").fetchone()["count"]

    def complete_cleanup(self, job_id: str, success: bool, error_code: Optional[str] = None) -> None:
        with self._connection() as connection:
            job = connection.execute("SELECT document_id,version_id,job_type FROM cleanup_jobs WHERE job_id=%s",
                                     (job_id,)).fetchone()
            connection.execute("UPDATE cleanup_jobs SET status=%s,attempts=attempts+1,error_code=%s,updated_at=now() "
                               "WHERE job_id=%s", ("COMPLETE" if success else "FAILED", error_code, job_id))
            if success and job and job["job_type"] == "RETIRE_VERSION" and job["version_id"]:
                connection.execute("UPDATE version_publications SET status='REMOVED' WHERE document_id=%s AND version_id=%s",
                                   (job["document_id"], job["version_id"]))
            elif not success and job and job["job_type"] == "RETIRE_VERSION" and job["version_id"]:
                connection.execute("UPDATE version_publications SET status='RETIRED' WHERE document_id=%s AND version_id=%s",
                                   (job["document_id"], job["version_id"]))

    def stats(self) -> dict[str, object]:
        with self._connection() as connection:
            tombstones = connection.execute("SELECT COUNT(*) count FROM deletion_tombstones").fetchone()["count"]
            retired = connection.execute("SELECT COUNT(*) count FROM cleanup_jobs WHERE job_type='RETIRE_VERSION' "
                                         "AND status IN ('PENDING','FAILED')").fetchone()["count"]
            rollbacks = connection.execute("SELECT COUNT(*) count FROM publication_revisions WHERE action='ROLLBACK'").fetchone()["count"]
            activations = connection.execute("SELECT COUNT(*) count FROM publication_revisions WHERE action='ACTIVATE'").fetchone()["count"]
            staging_age = connection.execute("SELECT COALESCE(EXTRACT(EPOCH FROM now()-MIN(v.created_at)),0) age "
                                             "FROM version_publications p JOIN document_versions v USING(document_id,version_id) "
                                             "WHERE p.status='STAGING'").fetchone()["age"]
            durations = {row["index_name"]: float(row["duration"]) for row in connection.execute(
                "SELECT index_name,COALESCE(SUM(duration_seconds),0) duration FROM index_stage_results GROUP BY index_name"
            ).fetchall()}
        return {"tombstones": tombstones, "retired": retired, "rollbacks": rollbacks, "activations": activations,
                "staging_age": float(staging_age), "stage_durations": durations}

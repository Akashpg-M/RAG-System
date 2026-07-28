from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.core.contracts import DocumentVersion, IndexingStatus, IngestionTask


class LocalFileObjectStorage:
    def put_bytes(self, uri: str, data: bytes, content_type: str) -> None:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, uri: str) -> bool:
        return Path(uri).is_file()

    def read_bytes(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    def delete(self, uri: str) -> None:
        Path(uri).unlink(missing_ok=True)

    def is_ready(self) -> bool:
        return True


class SQLiteDocumentRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT,
                    version_id TEXT,
                    source_uri TEXT,
                    content_hash TEXT,
                    created_at TEXT,
                    metadata TEXT,
                    PRIMARY KEY (document_id, version_id)
                )
            """)

    def save(self, version: DocumentVersion) -> None:
        with sqlite3.connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=? AND version_id=?", (version.document_id, version.version_id),
            ).fetchone()
            if existing:
                expected = (version.source_uri, version.content_hash, version.created_at.isoformat(),
                            json.dumps(version.metadata))
                if existing != expected:
                    raise ValueError("document versions are immutable")
                return
            connection.execute(
                "INSERT INTO document_versions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    version.document_id, version.version_id, version.source_uri, version.content_hash,
                    version.created_at.isoformat(), json.dumps(version.metadata),
                ),
            )
            connection.commit()

    def get_latest(self, document_id: str) -> Optional[DocumentVersion]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT document_id, version_id, source_uri, content_hash, created_at, metadata "
                "FROM document_versions WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        if not row:
            return None
        return DocumentVersion(row[0], row[1], row[2], row[3], datetime.fromisoformat(row[4]), json.loads(row[5]))

    def get_version(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT document_id,version_id,source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=? AND version_id=?", (document_id, version_id),
            ).fetchone()
        return DocumentVersion(row[0], row[1], row[2], row[3], datetime.fromisoformat(row[4]), json.loads(row[5])) \
            if row else None

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT document_id,version_id,source_uri,content_hash,created_at,metadata FROM document_versions "
                "WHERE document_id=? ORDER BY created_at", (document_id,),
            ).fetchall()
        return [DocumentVersion(row[0], row[1], row[2], row[3], datetime.fromisoformat(row[4]), json.loads(row[5]))
                for row in rows]

    def delete(self, document_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM document_versions WHERE document_id = ?", (document_id,))
            connection.commit()

    def is_ready(self) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("SELECT 1")
        return True


class SQLiteTaskRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_tasks (
                    task_id TEXT PRIMARY KEY,
                    source_uri TEXT,
                    document_id TEXT,
                    version_id TEXT,
                    status TEXT,
                    error TEXT,
                    created_at TEXT,
                    updated_at TEXT
                    ,history TEXT
                )
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(ingestion_tasks)")}
            if "version_id" not in columns:
                connection.execute("ALTER TABLE ingestion_tasks ADD COLUMN version_id TEXT")
            if "history" not in columns:
                connection.execute("ALTER TABLE ingestion_tasks ADD COLUMN history TEXT DEFAULT '[]'")
            for name, declaration in (
                ("attempt_count", "INTEGER DEFAULT 0"), ("idempotency_key", "TEXT"),
                ("fencing_token", "INTEGER DEFAULT 0"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE ingestion_tasks ADD COLUMN {name} {declaration}")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS ingestion_outbox (
                    event_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL, published_at TEXT, publish_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS ingestion_idempotency (
                    idempotency_key TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT NOT NULL,
                    task_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingestion_leases (
                    resource_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, ownership_token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL, expires_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_pending ON ingestion_outbox(published_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON ingestion_tasks(status, updated_at);
            """)

    def save(self, task: IngestionTask) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO ingestion_tasks "
                "(task_id, source_uri, document_id, version_id, status, error, created_at, updated_at, history, "
                "attempt_count, idempotency_key, fencing_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.task_id, task.source_uri, task.document_id, task.version_id, task.status.value, task.error,
                    task.created_at.isoformat(), task.updated_at.isoformat(),
                    json.dumps([status.value for status in task.history]),
                    task.attempt_count, task.idempotency_key, task.fencing_token,
                ),
            )
            connection.commit()

    def get(self, task_id: str) -> Optional[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT task_id, source_uri, document_id, version_id, status, error, created_at, updated_at, history, "
                "attempt_count, idempotency_key, fencing_token "
                "FROM ingestion_tasks WHERE task_id = ?", (task_id,),
            ).fetchone()
        if not row:
            return None
        return IngestionTask(
            task_id=row[0], source_uri=row[1], document_id=row[2], version_id=row[3],
            status=IndexingStatus(row[4]), error=row[5], created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]), history=[IndexingStatus(value) for value in json.loads(row[8] or "[]")],
            attempt_count=row[9] or 0, idempotency_key=row[10], fencing_token=row[11] or 0,
        )

    def get_latest_for_document(self, document_id: str) -> Optional[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT task_id FROM ingestion_tasks WHERE document_id = ? ORDER BY updated_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        return self.get(row[0]) if row else None

    def get_by_idempotency_key(self, key: str) -> Optional[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT task_id FROM ingestion_idempotency WHERE idempotency_key=?", (key,)
            ).fetchone()
        return self.get(row[0]) if row else None

    def is_ready(self) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("SELECT 1")
        return True

    def create_with_outbox(self, task: IngestionTask, event_id: str, payload: str) -> Tuple[IngestionTask, bool]:
        """Atomically persist task, idempotency claim, and publication intent."""
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if task.idempotency_key:
                existing = connection.execute(
                    "SELECT task_id FROM ingestion_idempotency WHERE idempotency_key = ?", (task.idempotency_key,)
                ).fetchone()
                if existing:
                    connection.rollback()
                    loaded = self.get(existing[0])
                    if loaded is None:
                        raise RuntimeError("idempotency record refers to missing task")
                    return loaded, False
            self._save_connection(connection, task)
            connection.execute(
                "INSERT INTO ingestion_outbox(event_id, task_id, payload, created_at) VALUES (?, ?, ?, ?)",
                (event_id, task.task_id, payload, datetime.now().astimezone().isoformat()),
            )
            if task.idempotency_key:
                connection.execute(
                    "INSERT INTO ingestion_idempotency VALUES (?, ?, ?, ?, ?)",
                    (task.idempotency_key, task.document_id, task.version_id, task.task_id,
                     datetime.now().astimezone().isoformat()),
                )
            connection.commit()
        return task, True

    def _save_connection(self, connection: sqlite3.Connection, task: IngestionTask) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO ingestion_tasks (task_id, source_uri, document_id, version_id, status, error, "
            "created_at, updated_at, history, attempt_count, idempotency_key, fencing_token) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task.task_id, task.source_uri, task.document_id, task.version_id, task.status.value, task.error,
             task.created_at.isoformat(), task.updated_at.isoformat(), json.dumps([s.value for s in task.history]),
             task.attempt_count, task.idempotency_key, task.fencing_token),
        )

    def pending_outbox(self, limit: int = 100) -> list[tuple[str, str, str]]:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT event_id, task_id, payload FROM ingestion_outbox WHERE published_at IS NULL "
                "ORDER BY created_at LIMIT ?", (limit,),
            ).fetchall()

    def mark_published(self, event_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE ingestion_outbox SET published_at = ?, publish_attempts = publish_attempts + 1, "
                "last_error_code = NULL WHERE event_id = ?",
                (datetime.now().astimezone().isoformat(), event_id),
            )
            connection.commit()

    def mark_publish_failed(self, event_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE ingestion_outbox SET publish_attempts = publish_attempts + 1, last_error_code = ? "
                "WHERE event_id = ?", ("queue_publish_failed", event_id),
            )
            connection.commit()

    def queued_without_event(self, limit: int = 100) -> list[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT task_id FROM ingestion_tasks t WHERE status = 'QUEUED' AND NOT EXISTS "
                "(SELECT 1 FROM ingestion_outbox o WHERE o.task_id=t.task_id) LIMIT ?", (limit,),
            ).fetchall()
        return [task for row in rows if (task := self.get(row[0])) is not None]

    def add_outbox(self, event_id: str, task_id: str, payload: str) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO ingestion_outbox(event_id, task_id, payload, created_at) VALUES (?, ?, ?, ?)",
                (event_id, task_id, payload, datetime.now().astimezone().isoformat()),
            )
            connection.commit()
            return cursor.rowcount == 1


class SQLiteLeaseRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        SQLiteTaskRepository(db_path)

    def acquire(self, resource_id: str, worker_id: str, token: str, now: float, duration: float) -> Optional[int]:
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT ownership_token, fencing_token, expires_at FROM ingestion_leases WHERE resource_id=?",
                (resource_id,),
            ).fetchone()
            if row and row[2] > now:
                connection.rollback()
                return None
            fencing = (row[1] if row else 0) + 1
            connection.execute(
                "INSERT OR REPLACE INTO ingestion_leases VALUES (?, ?, ?, ?, ?, ?)",
                (resource_id, worker_id, token, fencing, now + duration, now),
            )
            connection.commit()
            return fencing

    def renew(self, resource_id: str, token: str, fencing: int, now: float, duration: float) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE ingestion_leases SET expires_at=?, updated_at=? WHERE resource_id=? AND ownership_token=? "
                "AND fencing_token=? AND expires_at>?", (now + duration, now, resource_id, token, fencing, now),
            )
            connection.commit()
            return cursor.rowcount == 1

    def owns(self, resource_id: str, token: str, fencing: int, now: float) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT 1 FROM ingestion_leases WHERE resource_id=? AND ownership_token=? AND fencing_token=? "
                "AND expires_at>?", (resource_id, token, fencing, now),
            ).fetchone()
        return bool(row)

    def release(self, resource_id: str, token: str, fencing: int) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE ingestion_leases SET expires_at=0,updated_at=? WHERE resource_id=? AND ownership_token=? "
                "AND fencing_token=?", (time.time(), resource_id, token, fencing),
            )
            connection.commit()
            return cursor.rowcount == 1

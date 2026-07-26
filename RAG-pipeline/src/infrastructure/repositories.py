from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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
            connection.execute(
                "INSERT OR REPLACE INTO document_versions VALUES (?, ?, ?, ?, ?, ?)",
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

    def delete(self, document_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM document_versions WHERE document_id = ?", (document_id,))
            connection.commit()


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

    def save(self, task: IngestionTask) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO ingestion_tasks "
                "(task_id, source_uri, document_id, version_id, status, error, created_at, updated_at, history) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.task_id, task.source_uri, task.document_id, task.version_id, task.status.value, task.error,
                    task.created_at.isoformat(), task.updated_at.isoformat(),
                    json.dumps([status.value for status in task.history]),
                ),
            )
            connection.commit()

    def get(self, task_id: str) -> Optional[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT task_id, source_uri, document_id, version_id, status, error, created_at, updated_at, history "
                "FROM ingestion_tasks WHERE task_id = ?", (task_id,),
            ).fetchone()
        if not row:
            return None
        return IngestionTask(
            task_id=row[0], source_uri=row[1], document_id=row[2], version_id=row[3],
            status=IndexingStatus(row[4]), error=row[5], created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]), history=[IndexingStatus(value) for value in json.loads(row[8] or "[]")],
        )

    def get_latest_for_document(self, document_id: str) -> Optional[IngestionTask]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT task_id FROM ingestion_tasks WHERE document_id = ? ORDER BY updated_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        return self.get(row[0]) if row else None

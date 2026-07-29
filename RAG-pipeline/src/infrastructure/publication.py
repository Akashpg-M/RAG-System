from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Iterable, Optional

from src.core.publication import (
    IndexStageResult, ManifestEntry, PublicationSnapshot, PublicationValidationError, StaleFencingToken,
    manifest_checksum,
)


class SQLitePublicationRepository:
    """Compatibility publication boundary mirroring PostgreSQL transaction semantics."""
    def __init__(self, db_path: str, retention_versions: int = 2):
        self.db_path = db_path
        self.retention_versions = retention_versions
        with sqlite3.connect(db_path) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS publication_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO publication_state VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS publication_revisions (
                    revision INTEGER PRIMARY KEY, action TEXT NOT NULL, document_id TEXT,
                    version_id TEXT, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_versions (
                    document_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    activated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deletion_tombstones (
                    document_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, deleted_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunk_manifests (
                    document_id TEXT NOT NULL, version_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL, content_hash TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    metadata TEXT NOT NULL, PRIMARY KEY(document_id, version_id, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS version_manifest_summary (
                    document_id TEXT NOT NULL, version_id TEXT NOT NULL, expected_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(document_id, version_id)
                );
                CREATE TABLE IF NOT EXISTS index_stage_results (
                    document_id TEXT NOT NULL, version_id TEXT NOT NULL, index_name TEXT NOT NULL,
                    outcome TEXT NOT NULL, chunk_count INTEGER NOT NULL, checksum TEXT NOT NULL,
                    duration_seconds REAL NOT NULL, error_code TEXT, completed_at REAL NOT NULL,
                    PRIMARY KEY(document_id, version_id, index_name)
                );
                CREATE TABLE IF NOT EXISTS version_publications (
                    document_id TEXT NOT NULL, version_id TEXT NOT NULL, status TEXT NOT NULL,
                    degraded INTEGER NOT NULL DEFAULT 0, completed_at REAL, activated_at REAL,
                    retired_at REAL, failed_at REAL, activation_fence INTEGER, started_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(document_id, version_id)
                );
                CREATE TABLE IF NOT EXISTS cleanup_jobs (
                    job_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT,
                    job_type TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    UNIQUE(document_id, version_id, job_type)
                );
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(version_publications)")}
            if "started_at" not in columns:
                connection.execute("ALTER TABLE version_publications ADD COLUMN started_at REAL NOT NULL DEFAULT 0")

    def register_version(self, document_id: str, version_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO version_publications(document_id,version_id,status,started_at) "
                "VALUES (?,?,'STAGING',?)", (document_id, version_id, time.time()),
            )
            connection.commit()

    def save_manifest(self, document_id: str, version_id: str, entries: Iterable[ManifestEntry]) -> str:
        values = list(entries)
        checksum = manifest_checksum(values)
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT expected_count,checksum FROM version_manifest_summary WHERE document_id=? AND version_id=?",
                (document_id, version_id),
            ).fetchone()
            if existing:
                if existing != (len(values), checksum):
                    raise PublicationValidationError("immutable manifest differs from persisted manifest")
                connection.rollback()
                return checksum
            connection.executemany(
                "INSERT INTO chunk_manifests VALUES (?,?,?,?,?,?,?)",
                [(document_id, version_id, e.chunk_id, e.parent_id, e.content_hash, e.ordinal,
                  json.dumps(e.metadata, sort_keys=True)) for e in values],
            )
            connection.execute(
                "INSERT INTO version_manifest_summary VALUES (?,?,?,?,?)",
                (document_id, version_id, len(values), checksum, time.time()),
            )
            connection.commit()
        return checksum

    def record_stage(self, document_id: str, version_id: str, result: IndexStageResult) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO index_stage_results VALUES (?,?,?,?,?,?,?,?,?)",
                (document_id, version_id, result.index_name, result.outcome, result.chunk_count, result.checksum,
                 result.duration_seconds, result.error_code, time.time()),
            )
            connection.commit()

    def activate(self, document_id: str, version_id: str, resource_id: str, ownership_token: str,
                 fencing_token: int, required_indexes: Iterable[str]) -> tuple[int, bool]:
        required = set(required_indexes)
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            tombstone = connection.execute(
                "SELECT 1 FROM deletion_tombstones WHERE document_id=?", (document_id,)
            ).fetchone()
            if tombstone:
                raise PublicationValidationError("tombstoned documents cannot be activated")
            lease = connection.execute(
                "SELECT ownership_token,fencing_token,expires_at FROM ingestion_leases WHERE resource_id=?",
                (resource_id,),
            ).fetchone()
            if not lease or lease[0] != ownership_token or lease[1] != fencing_token or lease[2] <= time.time():
                raise StaleFencingToken("worker no longer owns publication lease")
            current = connection.execute(
                "SELECT version_id,revision FROM active_versions WHERE document_id=?", (document_id,)
            ).fetchone()
            if current and current[0] == version_id:
                connection.rollback()
                return current[1], False
            summary = connection.execute(
                "SELECT expected_count,checksum FROM version_manifest_summary WHERE document_id=? AND version_id=?",
                (document_id, version_id),
            ).fetchone()
            if not summary:
                raise PublicationValidationError("chunk manifest is missing")
            stages = {row[0]: row[1:] for row in connection.execute(
                "SELECT index_name,outcome,chunk_count,checksum FROM index_stage_results "
                "WHERE document_id=? AND version_id=?", (document_id, version_id),
            )}
            for name in required:
                result = stages.get(name)
                if not result or result[0] != "SUCCESS" or result[1] != summary[0] or result[2] != summary[1]:
                    raise PublicationValidationError(f"mandatory index stage is incomplete: {name}")
            revision = connection.execute("SELECT revision FROM publication_state WHERE singleton=1").fetchone()[0] + 1
            previous = current[0] if current else None
            connection.execute("UPDATE publication_state SET revision=? WHERE singleton=1", (revision,))
            connection.execute("INSERT INTO publication_revisions VALUES (?,?,?,?,?)",
                               (revision, "ACTIVATE", document_id, version_id, time.time()))
            connection.execute(
                "INSERT OR REPLACE INTO active_versions VALUES (?,?,?,?)",
                (document_id, version_id, revision, time.time()),
            )
            degraded = any(value[0] != "SUCCESS" for name, value in stages.items() if name not in required)
            connection.execute(
                "UPDATE version_publications SET status='ACTIVE',degraded=?,completed_at=COALESCE(completed_at,?),"
                "activated_at=?,activation_fence=? WHERE document_id=? AND version_id=?",
                (int(degraded), time.time(), time.time(), fencing_token, document_id, version_id),
            )
            if previous and previous != version_id:
                connection.execute(
                    "UPDATE version_publications SET status='RETIRING',retired_at=? WHERE document_id=? AND version_id=?",
                    (time.time(), document_id, previous),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO cleanup_jobs VALUES (?,?,?,?,?,0,NULL,?,?)",
                    (uuid.uuid4().hex, document_id, previous, "RETIRE_VERSION", "HELD", time.time(), time.time()),
                )
                retained = connection.execute(
                    "SELECT version_id FROM version_publications WHERE document_id=? AND status IN "
                    "('ACTIVE','RETIRING','RETIRED') ORDER BY COALESCE(activated_at,0) DESC",
                    (document_id,),
                ).fetchall()
                for stale in retained[self.retention_versions:]:
                    connection.execute(
                        "UPDATE cleanup_jobs SET status='PENDING',updated_at=? WHERE document_id=? AND version_id=? "
                        "AND job_type='RETIRE_VERSION'", (time.time(), document_id, stale[0]),
                    )
                    connection.execute(
                        "UPDATE version_publications SET status='RETIRED' WHERE document_id=? AND version_id=?",
                        (document_id, stale[0]),
                    )
            connection.commit()
            return revision, True

    def snapshot(self) -> PublicationSnapshot:
        with sqlite3.connect(self.db_path) as connection:
            revision = connection.execute("SELECT revision FROM publication_state WHERE singleton=1").fetchone()[0]
            active = dict(connection.execute("SELECT document_id,version_id FROM active_versions WHERE revision<=?",
                                             (revision,)).fetchall())
            tombstones = frozenset(row[0] for row in connection.execute(
                "SELECT document_id FROM deletion_tombstones WHERE revision<=?", (revision,)
            ))
            degraded = frozenset(connection.execute(
                "SELECT document_id,version_id FROM version_publications WHERE degraded=1 AND status='ACTIVE'"
            ).fetchall())
        return PublicationSnapshot(revision, active, tombstones, degraded)

    def current_revision(self) -> int:
        with sqlite3.connect(self.db_path) as connection:
            return int(connection.execute(
                "SELECT revision FROM publication_state WHERE singleton=1"
            ).fetchone()[0])

    def tombstone(self, document_id: str) -> tuple[int, bool]:
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM deletion_tombstones WHERE document_id=?", (document_id,)).fetchone():
                revision = connection.execute("SELECT revision FROM publication_state WHERE singleton=1").fetchone()[0]
                connection.rollback()
                return revision, False
            revision = connection.execute("SELECT revision FROM publication_state WHERE singleton=1").fetchone()[0] + 1
            connection.execute("UPDATE publication_state SET revision=? WHERE singleton=1", (revision,))
            connection.execute("INSERT INTO publication_revisions VALUES (?,?,?,?,?)",
                               (revision, "TOMBSTONE", document_id, None, time.time()))
            connection.execute("INSERT INTO deletion_tombstones VALUES (?,?,?)", (document_id, revision, time.time()))
            connection.execute("DELETE FROM active_versions WHERE document_id=?", (document_id,))
            connection.execute(
                "INSERT OR IGNORE INTO cleanup_jobs VALUES (?,?,?,?,?,0,NULL,?,?)",
                (uuid.uuid4().hex, document_id, None, "DELETE_DOCUMENT", "PENDING", time.time(), time.time()),
            )
            connection.execute("UPDATE version_publications SET status='TOMBSTONED' WHERE document_id=?", (document_id,))
            connection.commit()
            return revision, True

    def rollback(self, document_id: str, version_id: str) -> tuple[int, bool]:
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM deletion_tombstones WHERE document_id=?", (document_id,)).fetchone():
                raise PublicationValidationError("cannot roll back a tombstoned document")
            target = connection.execute(
                "SELECT status FROM version_publications WHERE document_id=? AND version_id=?",
                (document_id, version_id),
            ).fetchone()
            if not target or target[0] not in ("ACTIVE", "RETIRING", "RETIRED"):
                raise PublicationValidationError("rollback target is not a validated version")
            current = connection.execute("SELECT version_id,revision FROM active_versions WHERE document_id=?",
                                         (document_id,)).fetchone()
            if current and current[0] == version_id:
                connection.rollback()
                return current[1], False
            revision = connection.execute("SELECT revision FROM publication_state WHERE singleton=1").fetchone()[0] + 1
            connection.execute("UPDATE publication_state SET revision=? WHERE singleton=1", (revision,))
            connection.execute("INSERT INTO publication_revisions VALUES (?,?,?,?,?)",
                               (revision, "ROLLBACK", document_id, version_id, time.time()))
            connection.execute("INSERT OR REPLACE INTO active_versions VALUES (?,?,?,?)",
                               (document_id, version_id, revision, time.time()))
            if current:
                connection.execute("UPDATE version_publications SET status='RETIRING',retired_at=? "
                                   "WHERE document_id=? AND version_id=?", (time.time(), document_id, current[0]))
            connection.execute("UPDATE version_publications SET status='ACTIVE',activated_at=? "
                               "WHERE document_id=? AND version_id=?", (time.time(), document_id, version_id))
            connection.commit()
            return revision, True

    def manifest(self, document_id: str, version_id: str) -> list[ManifestEntry]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT chunk_id,parent_id,content_hash,ordinal,metadata FROM chunk_manifests "
                "WHERE document_id=? AND version_id=? ORDER BY ordinal", (document_id, version_id),
            ).fetchall()
        return [ManifestEntry(row[0], row[1], row[2], row[3], json.loads(row[4])) for row in rows]

    def pending_cleanup(self) -> list[tuple[str, str, Optional[str], str]]:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT job_id,document_id,version_id,job_type FROM cleanup_jobs WHERE status IN ('PENDING','FAILED')"
            ).fetchall()

    def claim_cleanup(self, job_id: str) -> bool:
        """Atomically fence rollback/activation from a physical cleanup in progress."""
        with sqlite3.connect(self.db_path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT document_id,version_id,job_type,status FROM cleanup_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not job or job[3] not in ("PENDING", "FAILED"):
                connection.rollback()
                return False
            if job[2] == "RETIRE_VERSION":
                active = connection.execute(
                    "SELECT 1 FROM active_versions WHERE document_id=? AND version_id=?", (job[0], job[1])
                ).fetchone()
                status = connection.execute(
                    "SELECT status FROM version_publications WHERE document_id=? AND version_id=?", (job[0], job[1])
                ).fetchone()
                if active or not status or status[0] != "RETIRED":
                    connection.rollback()
                    return False
                connection.execute(
                    "UPDATE version_publications SET status='CLEANING' WHERE document_id=? AND version_id=?",
                    (job[0], job[1]),
                )
            elif not connection.execute(
                "SELECT 1 FROM deletion_tombstones WHERE document_id=?", (job[0],)
            ).fetchone():
                connection.rollback()
                return False
            connection.execute("UPDATE cleanup_jobs SET status='RUNNING',updated_at=? WHERE job_id=?", (time.time(), job_id))
            connection.commit()
            return True

    def abandoned_staging(self, age_seconds: float) -> list[tuple[str, str]]:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT document_id,version_id FROM version_publications WHERE status='STAGING' AND started_at<?",
                (time.time() - age_seconds,),
            ).fetchall()

    def retired_awaiting_cleanup(self) -> int:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM cleanup_jobs WHERE job_type='RETIRE_VERSION' AND status IN ('PENDING','FAILED')"
            ).fetchone()[0]

    def complete_cleanup(self, job_id: str, success: bool, error_code: Optional[str] = None) -> None:
        with sqlite3.connect(self.db_path) as connection:
            job = connection.execute("SELECT document_id,version_id,job_type FROM cleanup_jobs WHERE job_id=?",
                                     (job_id,)).fetchone()
            connection.execute(
                "UPDATE cleanup_jobs SET status=?,attempts=attempts+1,error_code=?,updated_at=? WHERE job_id=?",
                ("COMPLETE" if success else "FAILED", error_code, time.time(), job_id),
            )
            if success and job and job[2] == "RETIRE_VERSION" and job[1]:
                connection.execute("UPDATE version_publications SET status='REMOVED' WHERE document_id=? AND version_id=?",
                                   (job[0], job[1]))
            elif not success and job and job[2] == "RETIRE_VERSION" and job[1]:
                connection.execute("UPDATE version_publications SET status='RETIRED' WHERE document_id=? AND version_id=?",
                                   (job[0], job[1]))
            connection.commit()

    def stats(self) -> dict[str, object]:
        with sqlite3.connect(self.db_path) as connection:
            tombstones = connection.execute("SELECT COUNT(*) FROM deletion_tombstones").fetchone()[0]
            retired = connection.execute("SELECT COUNT(*) FROM cleanup_jobs WHERE job_type='RETIRE_VERSION' "
                                         "AND status IN ('PENDING','FAILED')").fetchone()[0]
            rollbacks = connection.execute("SELECT COUNT(*) FROM publication_revisions WHERE action='ROLLBACK'").fetchone()[0]
            activations = connection.execute("SELECT COUNT(*) FROM publication_revisions WHERE action='ACTIVATE'").fetchone()[0]
            oldest = connection.execute("SELECT MIN(started_at) FROM version_publications WHERE status='STAGING'").fetchone()[0]
            durations = dict(connection.execute("SELECT index_name,COALESCE(SUM(duration_seconds),0) "
                                                "FROM index_stage_results GROUP BY index_name").fetchall())
        return {"tombstones": tombstones, "retired": retired, "rollbacks": rollbacks, "activations": activations,
                "staging_age": max(0.0, time.time() - oldest) if oldest else 0.0, "stage_durations": durations}

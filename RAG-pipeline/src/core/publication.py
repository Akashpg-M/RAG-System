from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from src.core.contracts import ChildChunk, ParentChunk, Vector


class PublicationError(RuntimeError):
    pass


class StaleFencingToken(PublicationError):
    pass


class PublicationValidationError(PublicationError):
    pass


@dataclass(frozen=True)
class ManifestEntry:
    chunk_id: str
    parent_id: str
    content_hash: str
    ordinal: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexStageResult:
    index_name: str
    outcome: str
    chunk_count: int
    checksum: str
    duration_seconds: float = 0.0
    error_code: Optional[str] = None


@dataclass(frozen=True)
class PublicationSnapshot:
    revision: int
    active_versions: Dict[str, str]
    tombstones: frozenset[str]
    degraded_versions: frozenset[tuple[str, str]] = frozenset()

    def allows(self, document_id: str, version_id: str, namespace: str = "default") -> bool:
        return document_id not in self.tombstones and self.active_versions.get(document_id) == version_id


@dataclass
class PreparedDocument:
    document_id: str
    version_id: Optional[str]
    parents: list[ParentChunk]
    children: list[ChildChunk]
    triples: list[Dict[str, Any]]
    embeddings: list[Vector] = field(default_factory=list)


def chunk_manifest(chunks: Iterable[ChildChunk]) -> tuple[list[ManifestEntry], str]:
    entries = [
        ManifestEntry(chunk.chunk_id, chunk.parent_id, chunk.content_hash, ordinal, dict(chunk.metadata))
        for ordinal, chunk in enumerate(chunks)
    ]
    return entries, manifest_checksum(entries)


def manifest_checksum(entries: Iterable[ManifestEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: value.ordinal):
        digest.update(f"{entry.ordinal}\0{entry.chunk_id}\0{entry.parent_id}\0{entry.content_hash}\n".encode("utf-8"))
    return digest.hexdigest()

from __future__ import annotations

import hashlib
import math
import re
import uuid
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.core.contracts import ChildChunk, DocumentVersion, IngestionTask, ParentChunk, Vector


class InMemoryCache:
    def __init__(self):
        self.values: Dict[str, Vector] = {}

    def get(self, key: str) -> Optional[Vector]:
        value = self.values.get(key)
        return list(value) if value is not None else None

    def set(self, key: str, value: Vector) -> None:
        self.values[key] = list(value)


class DeterministicEmbeddingProvider:
    def __init__(self, vector_dim: int = 32):
        self.vector_dim = vector_dim

    def get_embeddings_batched(self, texts: List[str], batch_size: int = 64) -> List[Vector]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.vector_dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.vector_dim
                vector[bucket] += 1.0
            magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / magnitude for value in vector])
        return vectors


class InMemoryDenseIndex:
    def __init__(self):
        self.points: Dict[str, tuple[ChildChunk, Vector]] = {}

    def upsert_chunks_bulk(self, chunks: List[ChildChunk], embeddings: List[Vector]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        self.points.update((chunk.chunk_id, (chunk, list(vector))) for chunk, vector in zip(chunks, embeddings))

    def search_similar(
        self, query_vector: Vector, limit: int = 3, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        hits = []
        for chunk, vector in self.points.values():
            if metadata_filter and any(chunk.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            score = sum(left * right for left, right in zip(query_vector, vector))
            hits.append({
                "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
                "parent_id": chunk.parent_id, "text": chunk.text, "content_hash": chunk.content_hash,
                "metadata": chunk.metadata.copy(), "score": score,
            })
        return sorted(hits, key=lambda item: item["score"], reverse=True)[:limit]

    def delete_document(self, document_id: str) -> None:
        self.points = {
            chunk_id: value for chunk_id, value in self.points.items() if value[0].document_id != document_id
        }

    def delete_version(self, document_id: str, version_id: str) -> None:
        self.points = {key: value for key, value in self.points.items()
                       if not (value[0].document_id == document_id and
                               value[0].metadata.get("version_id") == version_id)}

    def version_chunks(self, document_id: str, version_id: str) -> Dict[str, str]:
        return {chunk.chunk_id: chunk.content_hash for chunk, _ in self.points.values()
                if chunk.document_id == document_id and chunk.metadata.get("version_id") == version_id}

    def is_ready(self) -> bool:
        return True


class InMemorySparseIndex:
    def __init__(self):
        self.documents: Dict[str, ChildChunk] = {}

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def add_documents(self, chunks: List[ChildChunk]) -> None:
        self.documents.update((chunk.chunk_id, chunk) for chunk in chunks)

    def search(
        self, query: str, top_k: int = 10, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        results = []
        for chunk in self.documents.values():
            if metadata_filter:
                document_id = metadata_filter.get("document_id")
                if document_id and chunk.document_id != document_id:
                    continue
                if any(
                    chunk.metadata.get(key) != value
                    for key, value in metadata_filter.items() if key != "document_id"
                ):
                    continue
            document_tokens = self._tokens(chunk.text)
            score = float(sum(document_tokens.count(token) for token in query_tokens))
            if score:
                results.append(self._payload(chunk, sparse_score=score))
        return sorted(results, key=lambda item: item["sparse_score"], reverse=True)[:top_k]

    def get_by_chunk_or_parent(self, identifier: str) -> List[Dict[str, Any]]:
        return [
            self._payload(chunk) for chunk in self.documents.values()
            if chunk.chunk_id == identifier or chunk.parent_id == identifier
        ]

    def delete_document(self, document_id: str) -> List[str]:
        stale = [key for key, chunk in self.documents.items() if chunk.document_id == document_id]
        for key in stale:
            del self.documents[key]
        return stale

    def delete_version(self, document_id: str, version_id: str) -> List[str]:
        stale = [key for key, chunk in self.documents.items() if chunk.document_id == document_id and
                 chunk.metadata.get("version_id") == version_id]
        for key in stale:
            del self.documents[key]
        return stale

    def version_chunks(self, document_id: str, version_id: str) -> Dict[str, str]:
        return {chunk.chunk_id: chunk.content_hash for chunk in self.documents.values()
                if chunk.document_id == document_id and chunk.metadata.get("version_id") == version_id}

    def is_ready(self) -> bool:
        return True

    @staticmethod
    def _payload(chunk: ChildChunk, **scores: Any) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id, "parent_id": chunk.parent_id, "text": chunk.text,
            "metadata": chunk.metadata.copy(), **scores,
        }


class InMemoryGraphIndex:
    def __init__(self):
        self.triples: List[Dict[str, Any]] = []

    def add_triples_bulk(self, triples: List[Dict[str, Any]], chunk_id: Optional[str] = None) -> None:
        for triple in triples:
            prepared = dict(triple)
            prepared["chunk_id"] = prepared.get("chunk_id") or chunk_id
            if prepared.get("source") and prepared.get("relation") and prepared.get("target") and prepared["chunk_id"]:
                prepared["source"] = prepared["source"].strip().lower()
                prepared["relation"] = prepared["relation"].strip().lower()
                prepared["target"] = prepared["target"].strip().lower()
                if prepared not in self.triples:
                    self.triples.append(prepared)

    def traverse_graph_hops(self, seed_entities: List[str], max_hops: int = 1) -> List[Dict[str, Any]]:
        discovered = []
        visited = set()
        pending = deque((entity.strip().lower(), 0) for entity in seed_entities)
        while pending:
            entity, hop = pending.popleft()
            if entity in visited:
                continue
            visited.add(entity)
            if hop >= max_hops:
                continue
            for triple in self.triples:
                if triple["source"] != entity:
                    continue
                discovered.append({**triple, "hop_level": hop + 1})
                pending.append((triple["target"], hop + 1))
        return discovered

    def delete_document(self, document_id: str) -> None:
        self.triples = [item for item in self.triples if not item["chunk_id"].startswith(f"{document_id}#")]

    def delete_version(self, document_id: str, version_id: str) -> None:
        prefix = f"{document_id}#{version_id}#"
        self.triples = [item for item in self.triples if not item["chunk_id"].startswith(prefix)]

    def is_ready(self) -> bool:
        return True


class PlainTextChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 30):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: str, document_id: str) -> tuple[List[ParentChunk], List[ChildChunk]]:
        text = Path(file_path).read_text(encoding="utf-8")
        parent_id = f"{document_id}#p0"
        metadata = {"source": str(Path(file_path).resolve()), "title": Path(file_path).name}
        parent = ParentChunk(parent_id, document_id, text, metadata)
        tokens = text.split()
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for index, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start:start + self.chunk_size]
            if not window:
                break
            child_text = " ".join(window)
            digest = hashlib.sha256(child_text.encode("utf-8")).hexdigest()[:16]
            chunks.append(ChildChunk(
                f"{document_id}#c{index}", document_id, parent_id, child_text,
                len(window), digest, {"source": metadata["source"], "parent_id": parent_id},
            ))
            if start + self.chunk_size >= len(tokens):
                break
        return [parent], chunks


class NoOpGraphExtractor:
    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        return []


class IdentityQueryProcessor:
    def process_query(self, raw_query: str) -> Dict[str, str]:
        return {"original_query": raw_query, "rewritten_query": raw_query, "hyde_document": raw_query}


class TokenOverlapReranker:
    def predict(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        scores = []
        for query, text in pairs:
            query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
            text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            scores.append(float(len(query_tokens & text_tokens)))
        return scores


class DeterministicAnswerGenerator:
    def generate_stream(
        self, query: str, context_pool: List[Dict[str, Any]], score_threshold: float = -2.0
    ) -> Iterator[str]:
        if not context_pool:
            yield "No indexed context was found."
            return
        yield "\n\n".join(item["text"] for item in context_pool)


class InMemoryObjectStorage:
    def __init__(self):
        self.objects: Dict[str, bytes] = {}

    def exists(self, uri: str) -> bool:
        return uri in self.objects

    def put_bytes(self, uri: str, data: bytes, content_type: str) -> None:
        self.objects[uri] = bytes(data)

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def delete(self, uri: str) -> None:
        self.objects.pop(uri, None)


class InMemoryDocumentRepository:
    def __init__(self):
        self.versions: Dict[str, DocumentVersion] = {}
        self.lock = threading.Lock()

    def save(self, version: DocumentVersion) -> None:
        with self.lock:
            self.versions[version.document_id] = version

    def get_latest(self, document_id: str) -> Optional[DocumentVersion]:
        with self.lock:
            return self.versions.get(document_id)

    def delete(self, document_id: str) -> None:
        with self.lock:
            self.versions.pop(document_id, None)


class InMemoryTaskRepository:
    def __init__(self):
        self.tasks: Dict[str, IngestionTask] = {}
        self.lock = threading.Lock()

    def save(self, task: IngestionTask) -> None:
        with self.lock:
            self.tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[IngestionTask]:
        with self.lock:
            return self.tasks.get(task_id)

    def get_latest_for_document(self, document_id: str) -> Optional[IngestionTask]:
        with self.lock:
            matching = [task for task in self.tasks.values() if task.document_id == document_id]
            return max(matching, key=lambda task: task.updated_at) if matching else None


class InMemoryQueue:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def submit_task(self, file_path: str) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": "PENDING", "file_path": file_path, "error": None}
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        return task.copy() if task else None

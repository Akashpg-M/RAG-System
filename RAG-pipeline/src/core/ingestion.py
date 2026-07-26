from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, List, Optional

from src.core.contracts import Vector
from src.core.ports import (
    CacheProvider,
    DenseIndex,
    DocumentChunker,
    EmbeddingProvider,
    GraphExtractionProvider,
    GraphIndex,
    SparseIndex,
)


def stable_document_id(source_uri: str) -> str:
    normalized = os.path.normcase(str(Path(source_uri).resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class IngestionService:
    def __init__(
        self,
        sparse_index: SparseIndex,
        graph_index: GraphIndex,
        dense_index: DenseIndex,
        embedding_provider: EmbeddingProvider,
        cache: CacheProvider,
        graph_extractor: GraphExtractionProvider,
        document_id_factory: Callable[[str], str] = stable_document_id,
        embedding_batch_size: int = 64,
        graph_extraction_enabled: bool = True,
    ):
        self.sparse_store = sparse_index
        self.graph_store = graph_index
        self.vector_store = dense_index
        self.embedder = embedding_provider
        self.cache = cache
        self.extractor = graph_extractor
        self.document_id_factory = document_id_factory
        self.embedding_batch_size = embedding_batch_size
        self.graph_extraction_enabled = graph_extraction_enabled

    def document_id_for(self, source_uri: str) -> str:
        return self.document_id_factory(source_uri)

    def delete_document(self, source_uri: str) -> int:
        document_id = self.document_id_for(source_uri)
        return self.delete_document_by_id(document_id)

    def delete_document_by_id(self, document_id: str) -> int:
        chunk_ids = self.sparse_store.delete_document(document_id)
        self.graph_store.delete_document(document_id)
        self.vector_store.delete_document(document_id)
        return len(chunk_ids)

    def ingest_document(
        self,
        source_uri: str,
        chunker: DocumentChunker,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        source_path = Path(source_uri).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Document does not exist: {source_uri}")
        resolved_document_id = document_id or self.document_id_for(str(source_path))
        if progress_callback:
            progress_callback("PARSING")
        parent_chunks, child_chunks = chunker.process_file(str(source_path), resolved_document_id)
        if progress_callback:
            progress_callback("CHUNKING")
        for parent in parent_chunks:
            parent.metadata.setdefault("document_id", resolved_document_id)
            if version_id:
                parent.metadata.setdefault("version_id", version_id)
        for child in child_chunks:
            child.metadata.setdefault("document_id", resolved_document_id)
            if version_id:
                child.metadata.setdefault("version_id", version_id)

        triples = []
        if self.graph_extraction_enabled:
            for parent in parent_chunks:
                for triple in self.extractor.extract_triples(parent.text):
                    prepared = dict(triple)
                    prepared["chunk_id"] = parent.parent_id
                    triples.append(prepared)

        if progress_callback:
            progress_callback("EMBEDDING")
        embeddings: List[Vector | None] = [None] * len(child_chunks)
        uncached_chunks = []
        uncached_positions = []
        for position, chunk in enumerate(child_chunks):
            cached = self.cache.get(chunk.content_hash)
            if cached is None:
                uncached_chunks.append(chunk)
                uncached_positions.append(position)
            else:
                embeddings[position] = cached

        if uncached_chunks:
            texts = [chunk.text for chunk in uncached_chunks]
            try:
                computed = self.embedder.get_embeddings_batched(texts, batch_size=self.embedding_batch_size)
            except TypeError:
                # Compatibility for Stage 0 providers that implemented the
                # original single-argument method before the batch-size port.
                computed = self.embedder.get_embeddings_batched(texts)
            if len(computed) != len(uncached_chunks):
                raise RuntimeError("Embedding provider returned an unexpected vector count")
            for position, chunk, vector in zip(uncached_positions, uncached_chunks, computed):
                self.cache.set(chunk.content_hash, vector)
                embeddings[position] = vector

        final_embeddings = [embedding for embedding in embeddings if embedding is not None]
        if len(final_embeddings) != len(child_chunks):
            raise RuntimeError("Embedding preparation did not produce one vector per chunk")

        self.sparse_store.delete_document(resolved_document_id)
        self.graph_store.delete_document(resolved_document_id)
        self.vector_store.delete_document(resolved_document_id)
        if child_chunks:
            if progress_callback:
                progress_callback("INDEXING_DENSE")
            self.vector_store.upsert_chunks_bulk(child_chunks, final_embeddings)
            if progress_callback:
                progress_callback("INDEXING_SPARSE")
            self.sparse_store.add_documents(child_chunks)
        if triples:
            if progress_callback:
                progress_callback("INDEXING_GRAPH")
            self.graph_store.add_triples_bulk(triples)

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from src.core.ports import CacheProvider, DenseIndex, EmbeddingProvider, GraphIndex, SparseIndex


class DenseRetriever:
    def __init__(self, vector_store: DenseIndex, embedder: EmbeddingProvider, cache: CacheProvider):
        self.store = vector_store
        self.embedder = embedder
        self.cache = cache

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        query_vector = self.cache.get(query_hash)
        if query_vector is None:
            query_vector = self.embedder.get_embeddings_batched([query])[0]
            self.cache.set(query_hash, query_vector)
        hits = self.store.search_similar(query_vector, limit=top_k, metadata_filter=filters)
        return [{
            "chunk_id": hit["chunk_id"], "text": hit["text"], "parent_id": hit.get("parent_id", ""),
            "metadata": hit.get("metadata", {}), "retriever": "dense",
            "dense_score": hit.get("score", 1.0), "sparse_score": None, "graph_score": None,
            "rrf_score": 0.0, "rerank_score": None,
        } for hit in hits]


class SparseRetriever:
    def __init__(self, sparse_store: SparseIndex):
        self.store = sparse_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        hits = self.store.search(query, top_k=top_k)
        return [{
            "chunk_id": hit["chunk_id"], "text": hit["text"], "parent_id": hit.get("parent_id", ""),
            "metadata": hit.get("metadata", {}), "retriever": "sparse",
            "dense_score": None, "sparse_score": hit["sparse_score"], "graph_score": None,
            "rrf_score": 0.0, "rerank_score": None,
        } for hit in hits]


class GraphRetriever:
    def __init__(
        self,
        graph_store: GraphIndex,
        sparse_store: SparseIndex,
        tokenizer: Any,
        hop_decay: float = 0.5,
        max_hops: int = 2,
    ):
        self.graph_store = graph_store
        self.sparse_store = sparse_store
        self.tokenizer = tokenizer
        self.hop_decay = hop_decay
        self.max_hops = max_hops

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        seed_tokens = self.tokenizer.tokenize(query)
        if not seed_tokens:
            return []
        triples = self.graph_store.traverse_graph_hops(seed_tokens, max_hops=self.max_hops)
        if not triples:
            return []
        chunk_scores: Dict[str, float] = {}
        triples_by_chunk: Dict[str, List[str]] = {}
        for record in triples:
            chunk_id = record["chunk_id"]
            if not chunk_id:
                continue
            hop = record.get("hop_level", 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + self.hop_decay ** (hop - 1)
            path = f"({record['source']}) -[{record['relation']}]-> ({record['target']})"
            triples_by_chunk.setdefault(chunk_id, []).append(path)

        ranked_ids = sorted(chunk_scores, key=lambda chunk_id: chunk_scores[chunk_id], reverse=True)
        results = []
        for identifier in ranked_ids:
            for payload in self.sparse_store.get_by_chunk_or_parent(identifier):
                results.append({
                    "chunk_id": payload["chunk_id"], "text": payload["text"],
                    "parent_id": payload.get("parent_id", ""),
                    "metadata": {
                        **payload.get("metadata", {}),
                        "graph_context_relations": triples_by_chunk.get(identifier, []),
                        "graph_hop_score": chunk_scores[identifier],
                    },
                    "retriever": "graph", "dense_score": None, "sparse_score": None,
                    "graph_score": chunk_scores[identifier], "rrf_score": 0.0, "rerank_score": None,
                })
                if len(results) >= top_k:
                    return results
        return results

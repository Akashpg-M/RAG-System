import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Protocol

from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore
from src.sparse_store import SparseStore
from src.graph_store import KnowledgeGraphStore
from src.tokenizer import canonical_tokenizer

# THE RETRIEVER INTERFACE 
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...

# CONCRETE RETRIEVER IMPLEMENTATIONS 
class DenseRetriever:
    def __init__(self, vector_store: ProductionVectorStore, embedder: ProductionEmbedder, cache: Any):
        self.store = vector_store
        self.embedder = embedder
        self.cache = cache

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        import hashlib
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
        query_vector = self.cache.get(query_hash)
        
        if not query_vector:
            query_vector = self.embedder.get_embeddings_batched([query])[0]
            self.cache.set(query_hash, query_vector)
            
        hits = self.store.search_similar(query_vector, limit=top_k)
        return [{
            "chunk_id": h["chunk_id"], "text": h["text"], "parent_id": h.get("parent_id", ""),
            "metadata": h.get("metadata", {}), "retriever": "dense",
            "dense_score": h.get("score", 1.0), "sparse_score": None, "graph_score": None,
            "rrf_score": 0.0, "rerank_score": None
        } for h in hits]

class SparseRetriever:
    def __init__(self, sparse_store: SparseStore):
        self.store = sparse_store

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        hits = self.store.search(query, top_k=top_k)
        return [{
            "chunk_id": h["chunk_id"], "text": h["text"], "parent_id": h.get("parent_id", ""),
            "metadata": h.get("metadata", {}), "retriever": "sparse",
            "dense_score": None, "sparse_score": h["sparse_score"], "graph_score": None,
            "rrf_score": 0.0, "rerank_score": None
        } for h in hits]

class GraphRetriever:
    def __init__(self, graph_store: KnowledgeGraphStore, sparse_store: SparseStore, hop_decay: float = 0.5):
        self.graph_store = graph_store
        self.sparse_store = sparse_store
        self.hop_decay = hop_decay  # Penalty multiplier per structural link crossed

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        seed_tokens = canonical_tokenizer.tokenize(query)
        if not seed_tokens: return []
            
        # 1. Traversal up to 2 hops out to find deeper system connections
        triples_with_hops = self.graph_store.traverse_graph_hops(seed_tokens, max_hops=2)
        if not triples_with_hops: return []
            
        # 2. Compute decayed score based on structural distance from the seed query
        chunk_scores: Dict[str, float] = {}
        triples_by_chunk: Dict[str, List[str]] = {}
        
        for record in triples_with_hops:
            cid = record["chunk_id"]
            if not cid: continue
                
            hop_level = record.get("hop_level", 1)
            # Mathematical Decay function: Score = 1.0 * (decay_factor ^ (hop - 1))
            calculated_weight = 1.0 * (self.hop_decay ** (hop_level - 1))
            
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + calculated_weight
            
            path_string = f"({record['source']}) -[{record['relation']}]-> ({record['target']})"
            triples_by_chunk.setdefault(cid, []).append(path_string)

        # Sort candidate chunk identifiers by their total accumulated structural score
        ranked_chunk_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

        standardized_hits = []
        with sqlite3.connect(self.sparse_store.db_path) as conn:
            cursor = conn.cursor()
            for cid in ranked_chunk_ids[:top_k]:
                # If a parent ID was indexed, resolve down to matching child entries
                cursor.execute("SELECT payload FROM documents WHERE chunk_id = ? OR parent_id = ?", (cid, cid))
                rows = cursor.fetchall()
                
                for row in rows:
                    payload = json.loads(row[0])
                    standardized_hits.append({
                        "chunk_id": payload["chunk_id"],
                        "text": payload["text"],
                        "parent_id": payload.get("parent_id", ""),
                        "metadata": {
                            **payload.get("metadata", {}),
                            "graph_context_relations": triples_by_chunk.get(cid, []),
                            "graph_hop_score": chunk_scores[cid]
                        },
                        "retriever": "graph",
                        "dense_score": None,
                        "sparse_score": None,
                        "graph_score": chunk_scores[cid],
                        "rrf_score": 0.0,
                        "rerank_score": None
                    })
                    
        return standardized_hits[:top_k]
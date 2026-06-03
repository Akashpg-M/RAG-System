# src/models.py
import hashlib
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class ParentChunk:
    """
    Represents a broad structural context block of text. 
    Maintains large window integrity to give the LLM full semantic context.
    """
    parent_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChildChunk:
    """
    Represents a highly targeted sub-segment nested inside a ParentChunk.
    Optimized for high-resolution mathematical vector matching.
    """
    chunk_id: str
    parent_id: str  # Critical link pointing back to full parental context
    text: str
    token_count: int
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qdrant_id(self) -> int:
        """
        Generates a perfectly deterministic, stable 64-bit unsigned integer ID 
        for Qdrant database engines, completely avoiding unstable Python hash().
        """
        hex_hash = hashlib.sha256(self.chunk_id.encode('utf-8')).hexdigest()
        # Extract 15 hex characters to easily fit within signed 64-bit int boundaries maxing at 2^63 - 1
        return int(hex_hash[:15], 16)

@dataclass
class RawDocument:
    document_id: str
    filename: str
    raw_text: str
    
    @classmethod
    def from_text(cls, filename: str, raw_text: str):
        hash_input = f"{filename}{raw_text}".encode('utf-8')
        doc_id = hashlib.sha256(hash_input).hexdigest()[:16]
        return cls(document_id=doc_id, filename=filename, raw_text=raw_text)

class KnowledgeGraphIndex:
    """
    Implements an internal Entity-Relationship Knowledge Graph index for Graph RAG.
    Enables tracking multi-hop connections across separate textual ideas.
    """
    def __init__(self):
        # In-memory adjacency index: { entity_source: [(relationship, entity_target, chunk_id)] }
        self.graph: Dict[str, List[tuple]] = {}

    def add_triple(self, source: str, relation: str, target: str, chunk_id: str):
        """Adds an absolute directed relational edge between semantic concepts."""
        src, tgt = source.strip().lower(), target.strip().lower()
        if src not in self.graph:
            self.graph[src] = []
        self.graph[src].append((relation.strip().lower(), tgt, chunk_id))
        logger = sqlite3.logging.getLogger("GraphIndex") # Safe log access
        
    def traverse_hops(self, seed_entities: List[str], max_hops: int = 1) -> List[Dict[str, str]]:
        """
        Traverses the structural adjacency matrix to extract linked conceptual context
        for a set of queried keywords or seed concepts.
        """
        discovered_triples = []
        visited = set()
        queue = [(entity.lower(), 0) for entity in seed_entities]

        while queue:
            current_entity, current_hop = queue.pop(0)
            if current_entity in visited or current_hop > max_hops:
                continue
            visited.add(current_entity)

            if current_entity in self.graph:
                for relation, target, cid in self.graph[current_entity]:
                    discovered_triples.append({
                        "source": current_entity,
                        "relation": relation,
                        "target": target,
                        "chunk_id": cid
                    })
                    # Add unvisited neighbors to queue for multi-hop expansion
                    if target not in visited:
                        queue.append((target, current_hop + 1))
        return discovered_triples

class EmbeddingCache:
    """
    Durable Optimization Layer: Persistent local SQLite engine mapping content hashes 
    directly to vector coordinates to prevent re-computing vectors for unchanged text.
    """
    def __init__(self, db_path: str = "embedding_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    content_hash TEXT PRIMARY KEY,
                    embedding TEXT
                )
            """)
            conn.commit()

    def get(self, content_hash: str) -> Optional[List[float]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM cache WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, content_hash: str, embedding: List[float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (content_hash, embedding) VALUES (?, ?)",
                (content_hash, json.dumps(embedding))
            )
            conn.commit()
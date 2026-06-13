import hashlib
import sqlite3
import json
import logging
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
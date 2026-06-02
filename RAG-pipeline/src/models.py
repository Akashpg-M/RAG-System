# src/models.py
import hashlib
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class DocumentChunk:
    chunk_id: str         
    document_id: str      
    text: str
    token_count: int
    content_hash: str    
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qdrant_id(self) -> int:
        """
        Converts the stable string chunk_id into a highly reliable 64-bit unsigned 
        integer ID for Qdrant database engines, completely bypassing unstable Python hash().
        """
        hex_hash = hashlib.sha256(self.chunk_id.encode('utf-8')).hexdigest()
        # Take 15 hex characters to easily fit inside signed 64-bit int boundaries maxing at 2^63 - 1
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
    Optimization: Local SQLite cache engine mapping content hashes 
    directly to vector arrays to prevent re-computing vectors for duplicate text text blocks.
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
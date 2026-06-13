import sqlite3
import math
import json
import logging
from typing import List, Dict, Any
from src.models import ChildChunk
from src.tokenizer import canonical_tokenizer

logger = logging.getLogger("SQLiteSparseStore")

class SparseStore:
    def __init__(self, db_path: str = "sparse_index.db", b: float = 0.75, k1: float = 1.5):
        self.db_path = db_path
        self.b = b
        self.k1 = k1
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Added parent_id to support seamless downward mapping resolutions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    chunk_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    doc_len INTEGER,
                    payload TEXT,
                    content_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS postings (
                    token TEXT,
                    chunk_id TEXT,
                    term_freq INTEGER,
                    PRIMARY KEY (token, chunk_id),
                    FOREIGN KEY(chunk_id) REFERENCES documents(chunk_id)
                )
            """)
            # Key-Value store for operational parameters
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corpus_metadata (
                    key TEXT PRIMARY KEY,
                    value REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_parent ON documents(parent_id)")
            conn.commit()

    def delete_by_content_hash(self, content_hash: str) -> List[str]:
        """Removes all stale tracking postings to enable clean incremental updates."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id FROM documents WHERE content_hash = ?", (content_hash,))
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            if chunk_ids:
                placeholders = ",".join(["?"] * len(chunk_ids))
                cursor.execute(f"DELETE FROM postings WHERE chunk_id IN ({placeholders})", chunk_ids)
                cursor.execute(f"DELETE FROM documents WHERE content_hash = ?", (content_hash,))
                conn.commit()
            return chunk_ids

    def recalculate_corpus_stats(self):
        """Pre-computes and caches corpus metrics to completely eliminate real-time table scans."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(doc_len) FROM documents")
            size, avg_len = cursor.fetchone()
            size = size if size else 0
            avg_len = avg_len if avg_len else 0.0
            
            cursor.execute("INSERT OR REPLACE INTO corpus_metadata VALUES ('corpus_size', ?)", (float(size),))
            cursor.execute("INSERT OR REPLACE INTO corpus_metadata VALUES ('avg_doc_len', ?)", (float(avg_len),))
            conn.commit()

    def add_documents(self, chunks: List[ChildChunk]):
        if not chunks: return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for chunk in chunks:
                tokens = canonical_tokenizer.tokenize(chunk.text)
                if not tokens: continue
                
                payload = json.dumps({
                    "chunk_id": chunk.chunk_id, "parent_id": chunk.parent_id,
                    "text": chunk.text, "metadata": chunk.metadata
                })
                cursor.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.parent_id, len(tokens), payload, chunk.content_hash)
                )
                
                local_freqs = {}
                for token in tokens:
                    local_freqs[token] = local_freqs.get(token, 0) + 1
                    
                for token, tf in local_freqs.items():
                    cursor.execute("INSERT OR REPLACE INTO postings VALUES (?, ?, ?)", (token, chunk.chunk_id, tf))
            conn.commit()
        self.recalculate_corpus_stats()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = canonical_tokenizer.tokenize(query)
        if not query_tokens: return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fetch cached values in O(1) time
            cursor.execute("SELECT value FROM corpus_metadata WHERE key = 'corpus_size'")
            size_row = cursor.fetchone()
            cursor.execute("SELECT value FROM corpus_metadata WHERE key = 'avg_doc_len'")
            len_row = cursor.fetchone()
            
            corpus_size = size_row[0] if size_row else 0
            avg_doc_len = len_row[0] if len_row else 0.0
            if corpus_size == 0: return []

            placeholders = ",".join(["?"] * len(query_tokens))
            cursor.execute(f"""
                SELECT token, COUNT(distinct chunk_id) FROM postings 
                WHERE token IN ({placeholders}) GROUP BY token
            """, query_tokens)
            dfs = {token: count for token, count in cursor.fetchall()}

            cursor.execute(f"""
                SELECT p.token, p.chunk_id, p.term_freq, d.doc_len, d.payload
                FROM postings p
                JOIN documents d ON p.chunk_id = d.chunk_id
                WHERE p.token IN ({placeholders})
            """, query_tokens)
            raw_postings = cursor.fetchall()

        candidate_scores = {}
        for token, chunk_id, tf, doc_len, payload_str in raw_postings:
            doc_count = dfs.get(token, 0)
            if doc_count == 0: continue
            
            idf = math.log((corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            
            if chunk_id not in candidate_scores:
                candidate_scores[chunk_id] = {"score": 0.0, "payload": json.loads(payload_str)}
            candidate_scores[chunk_id]["score"] += score

        results = []
        for chunk_id, data in candidate_scores.items():
            if data["score"] > 0:
                payload = data["payload"]
                payload["sparse_score"] = data["score"]
                results.append(payload)

        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k]
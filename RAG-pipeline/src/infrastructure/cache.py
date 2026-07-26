import json
import sqlite3
from typing import List, Optional


class SQLiteEmbeddingCache:
    def __init__(self, db_path: str = "embedding_cache.db"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS cache (content_hash TEXT PRIMARY KEY, embedding TEXT)")
            connection.commit()

    def get(self, content_hash: str) -> Optional[List[float]]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT embedding FROM cache WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, content_hash: str, embedding: List[float]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO cache (content_hash, embedding) VALUES (?, ?)",
                (content_hash, json.dumps(embedding)),
            )
            connection.commit()


EmbeddingCache = SQLiteEmbeddingCache


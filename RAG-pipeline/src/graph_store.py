import sqlite3
import logging
from collections import deque
from typing import List, Dict, Any

logger = logging.getLogger("SQLiteGraphStore")

class KnowledgeGraphStore:
    def __init__(self, db_path: str = "graph_store.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    source TEXT,
                    relation TEXT,
                    target TEXT,
                    chunk_id TEXT,
                    PRIMARY KEY (source, relation, target, chunk_id)
                )
            """)
            # Explicitly added indexes to avoid full table scans during graph walks
            conn.execute("CREATE INDEX IF NOT EXISTS idx_triples_source ON triples(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_triples_target ON triples(target)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_triples_chunk ON triples(chunk_id)")
            conn.commit()

    def delete_by_chunk_id(self, chunk_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM triples WHERE chunk_id = ?", (chunk_id,))
            conn.commit()

    def add_triples_bulk(self, triples: List[Dict[str, str]], chunk_id: str):
        if not triples: return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for t in triples:
                source = t.get("source", "").strip().lower()
                relation = t.get("relation", "").strip().lower()
                target = t.get("target", "").strip().lower()
                if source and relation and target:
                    cursor.execute("INSERT OR IGNORE INTO triples VALUES (?, ?, ?, ?)", (source, relation, target, chunk_id))
            conn.commit()

    def traverse_graph_hops(self, seed_entities: List[str], max_hops: int = 1) -> List[Dict[str, Any]]:
        if not seed_entities: return []

        discovered_triples = []
        visited_nodes = set()
        
        # Upgraded to deque for performance optimizations on popleft steps
        queue = deque([(e.strip().lower(), 0) for e in seed_entities])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            while queue:
                current_entity, current_hop = queue.popleft() # Fast O(1) pointer shifts
                
                if current_entity in visited_nodes or current_hop > max_hops: continue
                visited_nodes.add(current_entity)

                cursor.execute("SELECT relation, target, chunk_id FROM triples WHERE source = ?", (current_entity,))
                edges = cursor.fetchall()

                for relation, target, chunk_id in edges:
                    discovered_triples.append({
                        "source": current_entity, "relation": relation, "target": target,
                        "chunk_id": chunk_id, "hop_level": current_hop + 1
                    })
                    if target not in visited_nodes:
                        queue.append((target, current_hop + 1))

        return discovered_triples
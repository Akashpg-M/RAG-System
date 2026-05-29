# src/retrieval.py
import logging
from typing import List, Dict, Any
from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore

logger = logging.getLogger("RetrievalEngine")

class RetrievalEngine:
    def __init__(self, vector_store: ProductionVectorStore, embedder: ProductionEmbedder):
        self.store = vector_store
        self.embedder = embedder

    def retrieve_context(self, query_text: str, top_k: int = 3, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes real semantic RAG retrieval: Query -> Embed -> Search Vector DB
        """
        logger.info(f"Executing retrieval workflow for text query: '{query_text}'")
        # 1. Transform query text to raw semantic vector
        query_vector = self.embedder.get_embeddings_batched([query_text])[0]
        
        # 2. Query against indexed points
        matches = self.store.search_similar(query_vector, limit=top_k, metadata_filter=filters)
        return matches
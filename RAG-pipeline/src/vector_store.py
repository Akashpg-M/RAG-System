# src/vector_store.py
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.models import DocumentChunk
from src.config import Config

logger = logging.getLogger("VectorStore")

class ProductionVectorStore:
    def __init__(self, collection_name: str, vector_dim: int):
        self.collection_name = collection_name
        # Durable, local file-system path persistence
        self.client = QdrantClient(path=Config.QDRANT_STORAGE_PATH)
        self._ensure_collection(vector_dim)

    def _ensure_collection(self, vector_dim: int):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(
                f"Creating collection '{self.collection_name}' with dimension scale {vector_dim}"
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                ),
            )

    def filter_existing_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Idempotency Guard: Compares document chunk IDs against existing DB targets
        to skip redundant embedding generation steps.
        """
        new_chunks = []
        for chunk in chunks:
            # Check if this precise chunk ID already exists
            res = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[hash(chunk.chunk_id) % 10**8],
                with_payload=True
            )
            if res and res[0].payload.get("content_hash") == chunk.content_hash:
                continue # Skip matched identity, no changes detected

            new_chunks.append(chunk) #contains new chunks that need to be embedded and upserted
        return new_chunks

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "content_hash": chunk.content_hash,
                **chunk.metadata
            }
            points.append(
                PointStruct(
                    id=hash(chunk.chunk_id) % 10**8,
                    vector=embedding,
                    payload=payload
                )
            )
        self.client.upsert(collection_name=self.collection_name, wait=True, points=points)
        logger.info(f"Successfully synchronized {len(points)} vector records into persistent index.")

    def search_similar(self, query_vector: List[float], limit: int = 3, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes query matching combined with optional structural metadata scoping.
        """
        qdrant_filter = None
        if metadata_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit
        )
        return [hit.payload for hit in results]
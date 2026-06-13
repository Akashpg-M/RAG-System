import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.config import Config
from src.models import ChildChunk

logger = logging.getLogger("VectorStore")

class ProductionVectorStore:
    def __init__(self, collection_name: str, vector_dim: int):
        self.collection_name = collection_name
        self.client = QdrantClient(path=Config.QDRANT_STORAGE_PATH)
        self._ensure_collection(vector_dim)

    def _ensure_collection(self, vector_dim: int):
        # if not self.client.collection_exists(self.collection_name):
        #     self.client.create_collection(
        #         collection_name=self.collection_name,
        #         vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        #     )

        collections_response = self.client.get_collections()
        existing_collections = [collection.name for collection in collections_response.collections]
        
        # Create it if it doesn't exist
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

    def filter_existing_chunks_batched(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """
        Optimized Batch Idempotency: Gathers all requested IDs and hits the database 
        in a single combined lookup call to maximize scalability.
        """
        if not chunks:
            return []

        # Map stable database integer representation back to objects
        id_map = {chunk.qdrant_id: chunk for chunk in chunks}
        target_ids = list(id_map.keys())
        
        # Single batched roundtrip call
        existing_records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=target_ids,
            with_payload=True
        )
        
        existing_payloads = {record.id: record.payload for record in existing_records if record.payload}
        
        new_chunks = []
        for qid, chunk in id_map.items():
            if qid in existing_payloads:
                # Double-check data integrity hashes match perfectly
                if existing_payloads[qid].get("content_hash") == chunk.content_hash:
                    continue # Extracted content is completely identical, safe to skip
            new_chunks.append(chunk)
            
        return new_chunks

    def upsert_chunks_bulk(self, chunks: List[ChildChunk], embeddings: List[List[float]], batch_size: int = 100):
        """
        Streams vector uploads across bounded chunks instead of choking database network lines.
        """
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                # "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "content_hash": chunk.content_hash,
                **chunk.metadata
            }
            points.append(PointStruct(id=chunk.qdrant_id, vector=embedding, payload=payload))
            
        # Segment arrays into physical sub-batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, wait=True, points=batch)
            
        logger.info(f"Bulk-uploaded {len(points)} vector nodes via batch sized windows.")

    def search_similar(self, query_vector: List[float], limit: int = 3, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        qdrant_filter = None
        if metadata_filter:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit
        )
        
        formatted_hits = []
        for hit in results:
            payload_data = hit.payload.copy() if hit.payload else {}
            payload_data["score"] = hit.score 
            formatted_hits.append(payload_data)

        return formatted_hits
    
    def delete_vector(self, chunk_id: str):
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[chunk_id])
        )
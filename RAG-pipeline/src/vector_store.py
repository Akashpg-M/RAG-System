# src/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.models import DocumentChunk
from typing import List
import os

class VectorStoreManager:
    def __init__(self, collection_name: str = "kb_chunks", vector_size: int = 1536):
        """
        Initializes a local, in-memory Qdrant database.
        For production, replace location=":memory:" with your Qdrant cloud URL/API key.
        """
        self.client = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Ensure the collection exists in the database
        self._ensure_collection()

    def _ensure_collection(self):

      collections = self.client.get_collections().collections
      collection_names = [collection.name for collection in collections]

      if self.collection_name not in collection_names:

          self.client.create_collection(
              collection_name=self.collection_name,
              vectors_config=VectorParams(
                  size=self.vector_size,
                  distance=Distance.COSINE
              ),
          )

          print(f"Created collection: {self.collection_name}")

      else:
          print(f"Collection already exists: {self.collection_name}")

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """
        Inserts or updates vector points inside the database using deterministic chunk IDs.
        """
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Qdrant requires IDs to be valid UUIDs or integers. 
            # For simplicity in local testing, we use a numerical hash or an incrementing integer,
            # but we preserve our deterministic chunk_id inside the payload metadata.
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                **chunk.metadata
            }
            
            # Using a simple integer hash of the chunk string ID for local unique indexing
            idx = int(hash(chunk.chunk_id) % 10**8)
            
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload=payload
                )
            )
            
        # Perform an upsert operation (idempotent)
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        print(f"Successfully indexed {len(points)} chunks into Vector DB.")
# src/main.py
import logging
import time
from src.models import RawDocument, EmbeddingCache
from src.chunker import SemanticChunker
from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore
from src.retrieval import RetrievalEngine

logger = logging.getLogger("PipelineApplication")

def main():
    logger.info("Initializing Highly Scalable Ingestion Pipeline Ecosystem.")
    
    # Initialize Core Modules
    chunker = SemanticChunker()
    embedder = ProductionEmbedder()
    db = ProductionVectorStore(collection_name="production_enterprise_kb", vector_dim=embedder.vector_dim)
    cache = EmbeddingCache()
    retrieval_system = RetrievalEngine(vector_store=db, embedder=embedder)
    
    # Raw multi-paragraph ingestion data setup
    sample_document_text = """
    ## Platform System Overview
    The engine infrastructure runs cleanly on Python frameworks. It relies heavily on vectorized tensor math matrices to evaluate incoming human prompts.
    
    ## Cloud Operations Infrastructure
    Deployment runtime automation targets Docker image configurations inside scalable cloud orchestration frameworks. Kubernetes setups drive system resilience, abstracting compute instances out of memory bounds seamlessly.
    """
    
    document = RawDocument.from_text(filename="cloud_architecture_spec.md", raw_text=sample_document_text)
    
    # 1. Semantic Slid-Window Context Splitting
    chunks = chunker.chunk_document(document)
    logger.info(f"Chunking complete. Yielded {len(chunks)} items with context boundaries.")
    
    # 2. Batch-Optimized Database Idempotency Filter Checks
    uncached_chunks_to_embed = []
    final_embeddings = []
    
    # Quick filter out items already stored directly inside Vector DB 
    needed_chunks = db.filter_existing_chunks_batched(chunks)
    
    # 3. Layer-2 Local Encoding Cache Check
    for chunk in needed_chunks:
        cached_vector = cache.get(chunk.content_hash)
        if cached_vector:
            final_embeddings.append(cached_vector)
        else:
            uncached_chunks_to_embed.append(chunk)

    # 4. Generate Missing Vectors and Update Cache
    if uncached_chunks_to_embed:
        texts_to_compute = [c.text for c in uncached_chunks_to_embed]
        new_vectors = embedder.get_embeddings_batched(texts_to_compute)
        
        # Populate SQLite Cache to avoid re-computing during future runs
        for chunk, vec in zip(uncached_chunks_to_embed, new_vectors):
            cache.set(chunk.content_hash, vec)
            final_embeddings.append(vec)
            
    # 5. Continuous Bulk Streaming Sync
    if needed_chunks:
        db.upsert_chunks_bulk(needed_chunks, final_embeddings)
    else:
        logger.info("Everything is synchronized perfectly. Ingestion pipelines skipped cleanly.")

    # 6. Real Live RAG Semantic Retrieval Verification
    print("\n" + "="*50)
    print("LIVE RETRIEVAL SYSTEM TEST")
    print("="*50)
    search_query = "How do we manage python infrastructure and kubernetes?"
    results = retrieval_system.retrieve_context(search_query, top_k=1)
    for res in results:
        print(f"Matched text: {res.get('text')}")
    print("="*50)

if __name__ == "__main__":
    main()
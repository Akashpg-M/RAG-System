# src/main.py
import logging
import time
from src.models import RawDocument
from src.chunker import SemanticChunker
from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore
from src.retrieval import RetrievalEngine

logger = logging.getLogger("PipelineApplication")

def main():
    logger.info("Starting Enterprise Ingestion and Retrieval App Framework.")
    start_time = time.time()
    
    # 1. Pipeline Component Orchestration setup
    chunker = SemanticChunker()
    embedder = ProductionEmbedder()
    db = ProductionVectorStore(collection_name="production_knowledge_base", vector_dim=embedder.vector_dim)
    retrieval_system = RetrievalEngine(vector_store=db, embedder=embedder)
    
    # 2. Ingest structured multi-paragraph data
    sample_document_text = """
## Platform System Overview
The engine infrastructure runs cleanly on Python frameworks. It relies heavily on vectorized tensor math matrices to evaluate incoming human prompts.

## Cloud Operations Infrastructure
Deployment runtime automation targets Docker image configurations inside scalable cloud orchestration frameworks. Kubernetes setups drive system resilience, abstracting compute instances out of memory bounds seamlessly.
"""
    
    document = RawDocument.from_text(filename="cloud_architecture_spec.md", raw_text=sample_document_text)
    
    # 3. Execution Processing Core
    chunks = chunker.chunk_document(document)
    logger.info(f"Document chunking complete. Total segments: {len(chunks)}")
    
    # 4. Idempotency Filtering Check
    chunks_to_process = db.filter_existing_chunks(chunks)
    logger.info(f"Idempotency filter analysis: {len(chunks_to_process)} / {len(chunks)} chunks require processing.")
    
    if chunks_to_process:
        texts_to_embed = [c.text for c in chunks_to_process]
        embeddings = embedder.get_embeddings_batched(texts_to_embed)
        db.upsert_chunks(chunks_to_process, embeddings)
    else:
        logger.info("All computed chunk identities exist and are up to date. Processing skipped.")

    # 5. Live Search Validation Run
    print("\n" + "="*50)
    print("LIVE RETRIEVAL SYSTEM SEARCH TEST")
    print("="*50)
    
    search_query = "Tell me about container automation and Kubernetes deployment"
    retrieved_contexts = retrieval_system.retrieve_context(search_query, top_k=1)
    
    for i, context in enumerate(retrieved_contexts):
        print(f"\nResult #{i+1} (Source File: {context.get('source')})")
        print(f"Content Match: {context.get('text')}")
    print("="*50)
    
    logger.info(f"Total pipeline cycle completed execution inside {time.time() - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()
# src/main.py
import logging
import time
from typing import List, Dict, Any
import tiktoken
from groq import Groq
from src.config import Config

from src.sparse_store import SparseStore
from src.graph_store import KnowledgeGraphStore
from src.vector_store import ProductionVectorStore
from src.embedder import ProductionEmbedder
from src.models import EmbeddingCache, ChildChunk
from src.chunker import SemanticDoclingChunker
from src.retrievers import DenseRetriever, SparseRetriever, GraphRetriever
from src.semantic_processor import SemanticQueryProcessor
from src.retrieval import AgenticRetrievalEngine, RetrieverManager
from src.fusion import ReciprocalRankFusion
from src.generation import ProductionResponseGenerator
from src.queue_worker import IngestionQueueManager
from src.telemetry import QueryTelemetryTracker
from src.graph.graph_extractor import GraphExtractor

from src.graph.ontology import DomainOntology
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EnterpriseRAG")

class ProductionIngestionPipeline:
    def __init__(self, sparse_store, graph_store, vector_store, embedder, cache, graph_extractor: GraphExtractor):
        self.sparse_store = sparse_store
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.cache = cache
        self.extractor = graph_extractor

    def ingest_document(self, file_path: str, chunker: SemanticDoclingChunker):
        logger.info(f"Background thread starting processing loop for file: {file_path}")
        
        # 1. Parse and Chunk
        parent_chunks, child_chunks = chunker.process_file(file_path, "doc_hash_123")
        
        # 2. Incremental Cleanup
        for chunk in child_chunks:
            purged = self.sparse_store.delete_by_content_hash(chunk.content_hash)
            for old_cid in purged:
                self.graph_store.delete_by_chunk_id(old_cid)
                if hasattr(self.vector_store, 'delete_vector'):
                    self.vector_store.delete_vector(old_cid)

        # 3. Add to Sparse Store
        self.sparse_store.add_documents(child_chunks)
        
        all_extracted_triples = []
        for p_chunk in parent_chunks:
            # Long-range structural relationship context window preserved cleanly
            triples = self.extractor.extract_triples(p_chunk.text)
            for t in triples:
                t["chunk_id"] = p_chunk.id  # Correlate back to target chunk identification
                all_extracted_triples.append(t)
                
        # Phase 4 Optimization: Execute a batch graph insertion using a single transactional block commit
        if all_extracted_triples:
            self.graph_store.add_triples_bulk(all_extracted_triples)
            
        # 4. Add to Vector Store
        needed_chunks = self.vector_store.filter_existing_chunks_batched(child_chunks)
        uncached = []
        final_embeddings = []
        
        for chunk in needed_chunks:
            cached_vec = self.cache.get(chunk.content_hash)
            if cached_vec:
                final_embeddings.append(cached_vec)
            else:
                uncached.append(chunk)

        if uncached:
            computed_vectors = self.embedder.get_embeddings_batched([c.text for c in uncached])
            for chunk, vector in zip(uncached, computed_vectors):
                self.cache.set(chunk.content_hash, vector)
                final_embeddings.append(vector)

        if needed_chunks:
            self.vector_store.upsert_chunks_bulk(needed_chunks, final_embeddings)
            
        logger.info(f"Successfully processed and indexed document in background thread.")

def main():
    logger.info("Initializing Telemetry-monitored RAG Node...")
    
    # 1. Storage & Primitives
    graph_db = KnowledgeGraphStore(db_path="graph_store.db")
    sparse_db = SparseStore(db_path="sparse_index.db")
    embedder = ProductionEmbedder()
    cache = EmbeddingCache()
    vector_db = ProductionVectorStore(collection_name="enterprise_kb", vector_dim=embedder.vector_dim)

    llm_client = Groq(api_key=Config.GROQ_API_KEY)

    # Define Domain Ontology (Phase 2 Requirement)
    ontology_path = os.path.join("config", "ontologies", "software.json")
    aliases_path = os.path.join("config", "ontologies", "aliases.json")

    try:
        domain_ontology = DomainOntology.load_pipeline_configs(ontology_path, aliases_path)
    except Exception as e:
        logger.critical(f"System boot aborted. Configuration error: {str(e)}")
        raise SystemExit(1)
    
    # Initialize the decoupled Graph Extractor
    graph_extractor = GraphExtractor(
        llm_client=llm_client, 
        model_name="llama3-70b-8192", 
        ontology=domain_ontology
    )

    # 2. Ingestion Setup
    # Fixed: Chunker now receives the underlying embedder/tokenizer instance, not the graph database
    # Initialize the token counter for the chunker
    token_encoder = tiktoken.get_encoding("cl100k_base")
    
    # Pass the token encoder (NOT the vector embedder) to the chunker
    chunker = SemanticDoclingChunker(encoder_instance=token_encoder)

    # Fixed: Injected the newly configured graph_extractor instance into the pipeline pipeline
    pipeline = ProductionIngestionPipeline(
        sparse_store=sparse_db, 
        graph_store=graph_db, 
        vector_store=vector_db, 
        embedder=embedder, 
        cache=cache,
        graph_extractor=graph_extractor
    )
    
    queue_manager = IngestionQueueManager(ingestion_pipeline=pipeline, chunker=chunker)

    # Submit an initial ingestion task smoothly
    try:
        task_id = queue_manager.submit_task("data/raw/system_design.md")
        logger.info(f"Check execution state via Endpoint UUID: {task_id}")
        logger.info("Waiting 15 seconds for Docling parsing and LLM Graph Extraction to complete...")
        time.sleep(15)
    except Exception as e:
        logger.warning(f"Skipping test ingestion: {str(e)}")

    # 3. Retrieval Engine Setup
    dense_ret = DenseRetriever(vector_store=vector_db, embedder=embedder, cache=cache)
    sparse_ret = SparseRetriever(sparse_store=sparse_db)
    graph_ret = GraphRetriever(graph_store=graph_db, sparse_store=sparse_db)
    
    manager = RetrieverManager(dense=dense_ret, sparse=sparse_ret, graph=graph_ret)
    engine = AgenticRetrievalEngine(processor=SemanticQueryProcessor(), manager=manager, fusion_strategy=ReciprocalRankFusion())
    generator = ProductionResponseGenerator()

    # 4. Execution & Telemetry
    telemetry = QueryTelemetryTracker()
    search_query = "What is the retry policy configuration for database connections?"
    
    t_start = telemetry.start_timer()
    fused_context = engine.retrieve_context(search_query, top_k=2)
    telemetry.stop_timer("total_retrieval_and_rerank", t_start)
    
    if fused_context:
        t_gen = telemetry.start_timer()
        print("\nSTREAMING RESPONSE:")
        for token in generator.generate_stream(search_query, fused_context):
            print(token, end="", flush=True)
        print("\n")
        telemetry.stop_timer("llm_generation", t_gen)
    else:
        logger.warning("No context found for the query.")

    telemetry.emit_telemetry_report()

if __name__ == "__main__":
    main()
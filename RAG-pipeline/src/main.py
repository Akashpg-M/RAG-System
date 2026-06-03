# src/main.py
import logging
import time
from src.models import RawDocument, EmbeddingCache, KnowledgeGraphIndex
from src.chunker import HierarchicalGraphChunker
from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore
from src.retrieval import AgenticRetrievalEngine

logger = logging.getLogger("PipelineApplication")

def compile_production_prompt(query: str, retrieved_nodes: list) -> str:
    """
    Production Prompt Framework: Assembles clear context instructions, XML formatting blocks, 
    Hierarchical (Parent) boundaries, and absolute anti-hallucination guardrails.
    """
    context_str = ""
    graph_str = ""
    
    for idx, node in enumerate(retrieved_nodes):
        # HIERARCHICAL RAG OPTIMIZATION: Pull the broad Parent Context instead of short Child snippets
        source_text = node.get("parent_context", node.get("text"))
        context_str += f"\n<document id=\"{node.get('chunk_id')}\" source=\"{node.get('source')}\">\n{source_text}\n</document>\n"
        
        if "graph_context_triples" in node:
            graph_str += json.dumps(node["graph_context_triples"], indent=2)

    prompt_blueprint = f"""
    You are an expert Enterprise Systems Architecture assistant. Answer the user query using ONLY the verified technical context blocks provided inside the XML tags below.

    <system_operational_rules>
    1. If the answer cannot be explicitly derived from the provided context, state clearly: "INFORMATION NOT AVAILABLE IN ENTERPRISE KNOWLEDGE BASE."
    2. Do NOT use outside knowledge or hallucinate assumptions.
    3. Maintain an authoritative, professional engineering tone.
    4. Reference the document IDs when drawing factual assertions.
    </system_operational_rules>

    <structural_knowledge_graph_triples>
    {graph_str if graph_str else "No explicit relational graph triples extracted for this partition."}
    </structural_knowledge_graph_triples>

    <verified_knowledge_context>
    {context_str}
    </verified_knowledge_context>

    USER QUERY: {query}
    FINAL EXPERT RESPONSE:
    """
    return prompt_blueprint.strip()

def main():
    logger.info("Starting Enterprise Ingestion & Agentic Retrieval Runtime Application.")
    
    # 1. Setup Architecture Dependencies
    graph_db = KnowledgeGraphIndex()
    chunker = HierarchicalGraphChunker(graph_index=graph_db)
    embedder = ProductionEmbedder()
    db = ProductionVectorStore(collection_name="production_enterprise_kb", vector_dim=embedder.vector_dim)
    cache = EmbeddingCache()
    agentic_search = AgenticRetrievalEngine(store=db, embedder=embedder, graph_index=graph_db)
    
    # 2. Document Parsing Pre-processing Phase
    with open("data/raw/system_design.md", "r") as f:
        raw_markdown = f.read()
        
    doc = RawDocument.from_text(filename="system_design.md", raw_text=raw_markdown)
    parent_chunks, child_chunks = chunker.process_document(doc)
    logger.info(f"Ingested text parsed into {len(parent_chunks)} Parent envelopes and {len(child_chunks)} nested Child vectors.")

    # 3. Batch-Optimized DB Idempotency Filter Checks
    needed_child_chunks = db.filter_existing_chunks_batched(child_chunks)
    logger.info(f"Idempotency Analysis: {len(needed_child_chunks)} / {len(child_chunks)} vectors require ingestion sync.")
    
    uncached_chunks = []
    final_embeddings = []
    
    # 4. Layer-2 Local Encoding Cache Processing Core
    for chunk in needed_child_chunks:
        cached_vector = cache.get(chunk.content_hash)
        if cached_vector:
            final_embeddings.append(cached_vector)
        else:
            uncached_chunks.append(chunk)

    # 5. Generate Missing Coordinates safely via Fault-Tolerant Batched Engine
    if uncached_chunks:
        logger.info(f"Generating vectors for {len(uncached_chunks)} uncached items...")
        computed_vectors = embedder.get_embeddings_batched([c.text for c in uncached_chunks])
        for chunk, vector in zip(uncached_chunks, computed_vectors):
            cache.set(chunk.content_hash, vector)
            final_embeddings.append(vector)

    # 6. Bulk Streaming Upsert Sync
    if needed_child_chunks:
        db.upsert_chunks_bulk(needed_child_chunks, final_embeddings)
        logger.info("Persistent index data state updated successfully.")
    else:
        logger.info("All components synchronized perfectly. Ingestion bypassed cleanly.")

    # 7. Live System Run & Prompt Compilation
    user_query = "How do we manage python infrastructure and kubernetes?"
    retrieved_contexts = agentic_search.retrieve_context(user_query, top_k=1)
    
    if retrieved_contexts:
        production_prompt = compile_production_prompt(user_query, retrieved_contexts)
        print("\n" + "="*70)
        print("COMPILED PRODUCTION ENTERPRISE LLM PROMPT")
        print("="*70)
        print(production_prompt)
        print("="*70)
    else:
        logger.error("System failed to isolate contextual document reference points.")

if __name__ == "__main__":
    main()
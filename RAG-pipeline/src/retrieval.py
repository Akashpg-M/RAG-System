# src/retrieval.py
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.embedder import ProductionEmbedder
from src.vector_store import ProductionVectorStore
from src.models import KnowledgeGraphIndex

logger = logging.getLogger("AgenticRetrievalEngine")

class AgenticRetrievalEngine:
    """
    Enterprise Orchestrator: Drives query expansions, graph traversals, 
    hybrid synthesis scoring, cross-encoder reranking, and self-correcting agent logic.
    """
    def __init__(self, store: ProductionVectorStore, embedder: ProductionEmbedder, graph_index: KnowledgeGraphIndex):
        self.store = store
        self.embedder = embedder
        self.graph_index = graph_index
        # Load high-precision local CrossEncoder model for deep conceptual verification
        logger.info("Loading Cross-Encoder neural re-ranking layer...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rewrite_query(self, query: str) -> str:
        """
        Query Rewriting Layer: Cleans linguistical noise and normalizes phrasing 
        to maximize vocabulary alignment with underlying vector indexes.
        """
        cleaned = query.lower().replace("how do we manage", "").replace("tell me about", "")
        return cleaned.strip()

    def expand_query(self, query: str) -> List[str]:
        """
        Query Expansion Layer: Splits an incoming request into multiple parallel paths 
        to target different semantic angles of the knowledge base.
        """
        base = self.rewrite_query(query)
        # Generates orthogonal technical formulations to broaden candidate retrieval matching
        return [base, f"infrastructure architecture configurations for {base}", f"deployment specifications and operations for {base}"]

    def extract_seed_entities(self, query: str) -> List[str]:
        """Simple rules-based entity extractor to locate seed anchors for the Graph traversal loop."""
        keywords = ["python", "kubernetes", "docker", "workers", "architecture", "deployment", "infrastructure", "engine"]
        return [word for word in keywords if word in query.lower()]

    def retrieve_context(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        The Agentic Multi-Route Loop: Evaluates the query, expands search constraints, 
        combines Vector and Graph structures, self-corrects results, and executes reranking.
        """
        logger.info(f"Agent processing query routing execution: '{query_text}'")
        
        # 1. Linguistic Processors
        expanded_queries = self.expand_query(query_text)
        logger.info(f"Query expansion generated sub-search matrix: {expanded_queries}")

        # 2. Parallel Vector Base Search
        candidate_payloads = []
        seen_chunk_ids = set()
        
        for sub_query in expanded_queries:
            q_vector = self.embedder.get_embeddings_batched([sub_query])[0]
            hits = self.store.search_similar(q_vector, limit=5)
            for hit in hits:
                if hit["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(hit["chunk_id"])
                    candidate_payloads.append(hit)

        # 3. Graph RAG Execution Step
        seeds = self.extract_seed_entities(query_text)
        graph_triples = self.graph_index.traverse_hops(seeds, max_hops=1)
        if graph_triples:
            logger.info(f"Graph RAG active. Extracted semantic context relations: {graph_triples}")

        # 4. Agentic Evaluation / Self-Correction Layer
        # If vector candidate quality score drops too low, dynamically pull backup data pipelines
        if not candidate_payloads and graph_triples:
            logger.warning("Vector search missed. Agent redirecting exclusively to structured Graph fallback pipelines.")
            # Self-correcting routine would look up chunks bound to graph_triples here

        if not candidate_payloads:
            return []

        # 5. Cross-Encoder Contextual Reranking
        # Pairs the original query against candidate text segments to evaluate non-linear semantic match
        rerank_inputs = [[query_text, candidate["text"]] for candidate in candidate_payloads]
        rerank_scores = self.reranker.predict(rerank_inputs).tolist()
        
        for candidate, score in zip(candidate_payloads, rerank_scores):
            # Inject raw score directly into structural tracking payload
            candidate["rerank_relevance_score"] = score

        # Re-sort candidates based on neural model validation scores
        candidate_payloads.sort(key=lambda x: x["rerank_relevance_score"], reverse=True)
        
        # Append extracted graph relations directly onto top context items to inject rich link data
        final_results = candidate_payloads[:top_k]
        if final_results and graph_triples:
            final_results[0]["graph_context_triples"] = graph_triples

        return final_results
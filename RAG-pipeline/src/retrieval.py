import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Protocol
from sentence_transformers import CrossEncoder

from src.semantic_processor import SemanticQueryProcessor
from src.retrievers import DenseRetriever, SparseRetriever, GraphRetriever
from src.fusion import FusionStrategy, ReciprocalRankFusion

logger = logging.getLogger("AgenticRetrievalEngine")

# FUSION STRATEGY PATTERN
class FusionStrategy(Protocol):
    def fuse(self, list_of_ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]: ...

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, list_of_ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        fused_map = {}
        for ranked_list in list_of_ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                cid = doc["chunk_id"]
                if cid not in fused_map:
                    fused_map[cid] = doc.copy()
                    fused_map[cid]["rrf_score"] = 0.0
                
                # Consolidate raw scores for debugging visibility
                for score_type in ["dense_score", "sparse_score", "graph_score"]:
                    if doc.get(score_type) is not None:
                        fused_map[cid][score_type] = doc[score_type]
                        
                fused_map[cid]["rrf_score"] += 1.0 / (self.k + rank)
                
        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_list
 
# PARALLEL RETRIEVER MANAGER
class RetrieverManager:
    """Manages thread-safe concurrent execution across distinct indices."""
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, graph: GraphRetriever):
        self.dense = dense
        self.sparse = sparse
        self.graph = graph

    def execute_routing(self, semantic_payload: Dict[str, Any], top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        results_matrix = []
        
        # Execute all retrieval streams in parallel across an optimized worker pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_sparse = executor.submit(self.sparse.retrieve, semantic_payload["original_query"], top_k, filters)
            future_dense_rewrite = executor.submit(self.dense.retrieve, semantic_payload["rewritten_query"], top_k, filters)
            future_dense_hyde = executor.submit(self.dense.retrieve, semantic_payload["hyde_document"], top_k, filters)
            future_graph = executor.submit(self.graph.retrieve, semantic_payload["original_query"], top_k, filters)
            
            # Re-collect the standardized payloads as threads finish execution
            results_matrix.append(future_sparse.result())
            results_matrix.append(future_dense_rewrite.result())
            results_matrix.append(future_dense_hyde.result())
            results_matrix.append(future_graph.result())
            
        return results_matrix

# CORE ORCHESTRATOR
class AgenticRetrievalEngine:
    def __init__(self, processor: SemanticQueryProcessor, manager: RetrieverManager, fusion_strategy: FusionStrategy):
        self.processor = processor
        self.manager = manager
        self.fusion = fusion_strategy
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve_context(self, query_text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # 1. Semantic Processing
        semantic_data = self.processor.process_query(query_text)
        
        # 2. Concurrent Retrieval execution
        retrieved_matrices = self.manager.execute_routing(semantic_data, top_k=10, filters=filters)
        
        # 3. Apply Pluggable Fusion Strategy
        fused_pool = self.fusion.fuse(retrieved_matrices)
        if not fused_pool: return []
            
        # 4. Capped Cross-Encoder Reranking
        # Limit the pool to 3x the top_k requested to save massive compute time
        rerank_cap = top_k * 3
        candidate_pool = fused_pool[:rerank_cap]
        
        rerank_inputs = [[query_text, candidate["text"]] for candidate in candidate_pool]
        rerank_scores = self.reranker.predict(rerank_inputs).tolist()
        
        for candidate, score in zip(candidate_pool, rerank_scores):
            candidate["rerank_score"] = score
            
        candidate_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_pool[:top_k]
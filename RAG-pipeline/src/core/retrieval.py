from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

from src.core.ports import FusionStrategy, QueryProcessor, RerankingProvider, Retriever

logger = logging.getLogger("AgenticRetrievalEngine")


class RetrieverManager:
    def __init__(self, dense: Retriever, sparse: Retriever, graph: Retriever, enable_hyde: bool = True):
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.enable_hyde = enable_hyde

    def execute_routing(
        self, semantic_payload: Dict[str, Any], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        jobs = [
            ("sparse", self.sparse, semantic_payload["original_query"]),
            ("dense_rewrite", self.dense, semantic_payload["rewritten_query"]),
        ]
        if self.enable_hyde:
            jobs.append(("dense_hyde", self.dense, semantic_payload["hyde_document"]))
        jobs.append(("graph", self.graph, semantic_payload["original_query"]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [(name, executor.submit(retriever.retrieve, query, top_k, filters)) for name, retriever, query in jobs]
            results = []
            for name, future in futures:
                try:
                    results.append(future.result())
                except Exception:
                    logger.exception("Retriever %s failed; continuing with remaining indexes", name)
                    results.append([])
        return results


class RetrievalService:
    def __init__(
        self,
        processor: QueryProcessor,
        manager: RetrieverManager,
        fusion_strategy: FusionStrategy,
        reranker: RerankingProvider,
        candidate_top_k: int = 10,
        rerank_pool_multiplier: int = 3,
    ):
        self.processor = processor
        self.manager = manager
        self.fusion = fusion_strategy
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.rerank_pool_multiplier = rerank_pool_multiplier

    def retrieve_context(
        self, query_text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        semantic_data = dict(self.processor.process_query(query_text))
        matrices = self.manager.execute_routing(semantic_data, top_k=self.candidate_top_k, filters=filters)
        fused_pool = self.fusion.fuse(matrices)
        if not fused_pool:
            return []
        candidate_pool = fused_pool[:top_k * self.rerank_pool_multiplier]
        pairs = [[query_text, candidate["text"]] for candidate in candidate_pool]
        try:
            predictions = self.reranker.predict(pairs)
            scores = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        except Exception:
            logger.exception("Cross-encoder reranking failed; returning RRF-ranked candidates")
            return candidate_pool[:top_k]
        for candidate, score in zip(candidate_pool, scores):
            candidate["rerank_score"] = float(score)
        candidate_pool.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidate_pool[:top_k]


from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import time
from typing import Any, Dict, List, Optional

from src.core.ports import FusionStrategy, QueryProcessor, RerankingProvider, Retriever
from src.observability import get_observability

logger = logging.getLogger("AgenticRetrievalEngine")


class RetrieverManager:
    def __init__(self, dense: Retriever, sparse: Retriever, graph: Retriever, enable_hyde: bool = True):
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.enable_hyde = enable_hyde

    def execute_routing(
        self,
        semantic_payload: Dict[str, Any],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
    ) -> List[List[Dict[str, Any]]]:
        jobs = []
        if mode in ("hybrid", "sparse"):
            jobs.append(("sparse", self.sparse, semantic_payload["original_query"]))
        if mode in ("hybrid", "dense"):
            jobs.append(("dense_rewrite", self.dense, semantic_payload["rewritten_query"]))
        if mode in ("hybrid", "dense") and self.enable_hyde:
            jobs.append(("dense_hyde", self.dense, semantic_payload["hyde_document"]))
        if mode in ("hybrid", "graph"):
            jobs.append(("graph", self.graph, semantic_payload["original_query"]))
        if not jobs:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        telemetry = get_observability()

        def retrieve(name: str, retriever: Retriever, query: str) -> List[Dict[str, Any]]:
            started = time.perf_counter()
            with telemetry.span(f"query.retrieve.{name}", {"rag.retriever": name}):
                result = retriever.retrieve(query, top_k, filters)
            elapsed = time.perf_counter() - started
            telemetry.metrics.labels(telemetry.metrics.retrieval_duration, retriever=name).observe(elapsed)
            telemetry.metrics.labels(telemetry.metrics.candidates_returned, retriever=name).observe(len(result))
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            # Each branch receives an independent copy of the caller's OTEL context. This
            # makes the retrievers overlapping siblings even though they run in threads.
            futures = [
                (name, executor.submit(contextvars.copy_context().run, retrieve, name, retriever, query))
                for name, retriever, query in jobs
            ]
            results = []
            for name, future in futures:
                try:
                    results.append(future.result())
                except Exception:
                    logger.exception("retriever_failed", extra={"component": name, "error_code": "retrieval"})
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
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        telemetry = get_observability()
        with telemetry.span("query.representations"):
            semantic_data = dict(self.processor.process_query(query_text))
        matrices = self.manager.execute_routing(
            semantic_data, top_k=max(self.candidate_top_k, top_k), filters=filters, mode=mode
        )
        started = time.perf_counter()
        with telemetry.span("query.fusion"):
            fused_pool = self.fusion.fuse(matrices)
        telemetry.metrics.labels(telemetry.metrics.fusion_duration).observe(time.perf_counter() - started)
        if not fused_pool:
            return []
        candidate_pool = fused_pool[:top_k * self.rerank_pool_multiplier]
        pairs = [[query_text, candidate["text"]] for candidate in candidate_pool]
        try:
            started = time.perf_counter()
            with telemetry.span("query.rerank", {"rag.candidate_count": len(candidate_pool)}):
                predictions = self.reranker.predict(pairs)
            telemetry.metrics.labels(telemetry.metrics.rerank_duration).observe(time.perf_counter() - started)
            scores = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        except Exception:
            logger.exception("reranking_failed", extra={"component": "reranker", "error_code": "retrieval"})
            return candidate_pool[:top_k]
        for candidate, score in zip(candidate_pool, scores):
            candidate["rerank_score"] = float(score)
        candidate_pool.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidate_pool[:top_k]

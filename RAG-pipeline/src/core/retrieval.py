from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from src.core.ports import FusionStrategy, QueryProcessor, RerankingProvider, Retriever
from src.core.performance import BoundedExecutor, Bulkhead, CapacityExhausted, Deadline, DeadlineExceeded
from src.observability import get_observability

logger = logging.getLogger("AgenticRetrievalEngine")


class RetrieverManager:
    def __init__(self, dense: Retriever, sparse: Retriever, graph: Retriever, enable_hyde: bool = True,
                 max_workers: int = 16, max_pending: int = 32, per_retriever_timeout: float = 2.5):
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.enable_hyde = enable_hyde
        self.per_retriever_timeout = per_retriever_timeout
        self.executor = BoundedExecutor(max_workers, max_pending, "rag-retriever")

    def execute_routing(
        self,
        semantic_payload: Dict[str, Any],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
        deadline: Optional[Deadline] = None,
        mandatory: Optional[Iterable[str]] = None,
    ) -> "RetrievalMatrices":
        jobs = []
        if mode in ("hybrid", "fast", "adaptive_graph", "adaptive_hyde", "sparse"):
            jobs.append(("sparse", self.sparse, semantic_payload["original_query"]))
        if mode in ("hybrid", "fast", "adaptive_graph", "adaptive_hyde", "dense"):
            jobs.append(("dense_rewrite", self.dense, semantic_payload["rewritten_query"]))
        if mode in ("hybrid", "adaptive_hyde", "dense") and self.enable_hyde:
            jobs.append(("dense_hyde", self.dense, semantic_payload["hyde_document"]))
        if mode in ("hybrid", "adaptive_graph", "graph"):
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

        required = set(mandatory or (
            {"dense_rewrite"}
            if mode in ("hybrid", "fast", "adaptive_graph", "adaptive_hyde") else {jobs[0][0]}
        ))
        futures = []
        degraded: list[str] = []
        for name, retriever, query in jobs:
            try:
                future = self.executor.submit(contextvars.copy_context().run, retrieve, name, retriever, query)
                futures.append((name, time.monotonic(), future))
            except CapacityExhausted:
                if name in required:
                    raise
                degraded.append(f"{name}_capacity")
        results = RetrievalMatrices()
        for name, submitted, future in futures:
            per_stage = max(0.0, self.per_retriever_timeout - (time.monotonic() - submitted))
            timeout = per_stage if deadline is None else min(per_stage, deadline.remaining)
            try:
                if timeout <= 0:
                    raise concurrent.futures.TimeoutError()
                results.append(future.result(timeout=timeout))
            except concurrent.futures.TimeoutError as error:
                future.cancel()
                if name in required:
                    raise DeadlineExceeded(f"mandatory retriever {name} timed out") from error
                degraded.append(f"{name}_timeout")
                results.append([])
            except Exception:
                if name in required:
                    raise
                logger.exception("retriever_failed", extra={"component": name, "error_code": "retrieval"})
                degraded.append(f"{name}_failure")
                results.append([])
        results.degraded_reasons = tuple(degraded)
        return results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


class RetrievalMatrices(list[List[Dict[str, Any]]]):
    degraded_reasons: tuple[str, ...] = ()


class RetrievedContext(list[Dict[str, Any]]):
    degraded_reasons: tuple[str, ...] = ()


class RetrievalService:
    def __init__(
        self,
        processor: QueryProcessor,
        manager: RetrieverManager,
        fusion_strategy: FusionStrategy,
        reranker: RerankingProvider,
        candidate_top_k: int = 10,
        rerank_pool_multiplier: int = 3,
        rewrite_concurrency: int = 2,
        reranker_concurrency: int = 1,
        reranker_candidate_cap: int = 30,
        reranker_batch_size: int = 8,
        reranker_skip_below: int = 2,
        stage_pending: int = 8,
    ):
        self.processor = processor
        self.manager = manager
        self.fusion = fusion_strategy
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.rerank_pool_multiplier = rerank_pool_multiplier
        self.rewrite_bulkhead = Bulkhead(rewrite_concurrency, "query-rewrite")
        self.reranker_bulkhead = Bulkhead(reranker_concurrency, "reranker")
        self.rewrite_executor = BoundedExecutor(
            rewrite_concurrency, max(rewrite_concurrency, stage_pending), "rag-rewrite"
        )
        self.reranker_executor = BoundedExecutor(
            reranker_concurrency, max(reranker_concurrency, stage_pending), "rag-reranker"
        )
        self.reranker_candidate_cap = reranker_candidate_cap
        self.reranker_batch_size = reranker_batch_size
        self.reranker_skip_below = reranker_skip_below

    def retrieve_context(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
        deadline: Optional[Deadline] = None,
    ) -> List[Dict[str, Any]]:
        telemetry = get_observability()
        if deadline:
            deadline.require()
        with telemetry.span("query.representations"):
            try:
                future = self.rewrite_executor.submit(
                    contextvars.copy_context().run, self.rewrite_bulkhead.run, self.processor.process_query, query_text
                )
                semantic_data = dict(future.result(timeout=deadline.remaining if deadline else None))
            except concurrent.futures.TimeoutError as error:
                future.cancel()
                raise DeadlineExceeded("query representation generation timed out") from error
        if deadline:
            deadline.require()
        matrices = self.manager.execute_routing(
            semantic_data, top_k=max(self.candidate_top_k, top_k), filters=filters, mode=mode, deadline=deadline
        )
        started = time.perf_counter()
        with telemetry.span("query.fusion"):
            fused_pool = self.fusion.fuse(matrices)
        telemetry.metrics.labels(telemetry.metrics.fusion_duration).observe(time.perf_counter() - started)
        if not fused_pool:
            empty = RetrievedContext()
            empty.degraded_reasons = matrices.degraded_reasons
            return empty
        candidate_pool = fused_pool[:min(self.reranker_candidate_cap, top_k * self.rerank_pool_multiplier)]
        if len(candidate_pool) < self.reranker_skip_below:
            skipped = RetrievedContext(candidate_pool[:top_k])
            skipped.degraded_reasons = matrices.degraded_reasons
            return skipped
        if deadline:
            deadline.require(0.001)
        pairs = [[query_text, candidate["text"]] for candidate in candidate_pool]
        try:
            started = time.perf_counter()
            with telemetry.span("query.rerank", {"rag.candidate_count": len(candidate_pool)}):
                predictions = []
                for offset in range(0, len(pairs), self.reranker_batch_size):
                    if deadline:
                        deadline.require(0.001)
                    batch = pairs[offset:offset + self.reranker_batch_size]
                    future = self.reranker_executor.submit(
                        contextvars.copy_context().run, self.reranker_bulkhead.run, self.reranker.predict, batch
                    )
                    try:
                        batch_scores = future.result(timeout=deadline.remaining if deadline else None)
                    except concurrent.futures.TimeoutError as error:
                        future.cancel()
                        raise DeadlineExceeded("reranking timed out") from error
                    predictions.extend(batch_scores.tolist() if hasattr(batch_scores, "tolist") else batch_scores)
            telemetry.metrics.labels(telemetry.metrics.rerank_duration).observe(time.perf_counter() - started)
            scores = list(predictions)
        except Exception:
            logger.exception("reranking_failed", extra={"component": "reranker", "error_code": "retrieval"})
            fallback = RetrievedContext(candidate_pool[:top_k])
            fallback.degraded_reasons = (*matrices.degraded_reasons, "reranker_failure")
            return fallback
        for candidate, score in zip(candidate_pool, scores):
            candidate["rerank_score"] = float(score)
        candidate_pool.sort(key=lambda item: item["rerank_score"], reverse=True)
        result = RetrievedContext(candidate_pool[:top_k])
        result.degraded_reasons = matrices.degraded_reasons
        return result

    def shutdown(self) -> None:
        self.manager.shutdown()
        self.rewrite_executor.shutdown(wait=False)
        self.reranker_executor.shutdown(wait=False)

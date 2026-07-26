"""Backward-compatible retrieval orchestration imports."""

from src.core.retrieval import RetrievalService, RetrieverManager


class AgenticRetrievalEngine(RetrievalService):
    def __init__(self, processor, manager, fusion_strategy, reranker=None):
        if reranker is None:
            from src.infrastructure.providers import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        super().__init__(processor, manager, fusion_strategy, reranker)


__all__ = ["AgenticRetrievalEngine", "RetrievalService", "RetrieverManager"]

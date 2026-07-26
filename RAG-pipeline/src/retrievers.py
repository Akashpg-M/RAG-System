"""Backward-compatible retriever imports."""

from src.core.ports import Retriever
from src.core.retrievers import DenseRetriever, GraphRetriever as CoreGraphRetriever, SparseRetriever
from src.tokenizer import canonical_tokenizer


class GraphRetriever(CoreGraphRetriever):
    def __init__(self, graph_store, sparse_store, hop_decay: float = 0.5):
        super().__init__(graph_store, sparse_store, canonical_tokenizer, hop_decay=hop_decay, max_hops=2)


__all__ = ["DenseRetriever", "GraphRetriever", "Retriever", "SparseRetriever"]

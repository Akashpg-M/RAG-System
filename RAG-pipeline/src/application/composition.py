from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.application.config import AppConfig, Profile, profile_config
from src.core.contracts import QueryResult, RetrievalCandidate, SourceCitation
from src.core.fusion import ReciprocalRankFusion
from src.core.ingestion import IngestionService
from src.core.ports import AnswerGenerator, DocumentChunker, QueueOperations
from src.core.retrieval import RetrievalService, RetrieverManager
from src.core.retrievers import DenseRetriever, GraphRetriever, SparseRetriever


@dataclass
class RagApplication:
    config: AppConfig
    ingestion: IngestionService
    retrieval: RetrievalService
    generator: AnswerGenerator
    chunker: DocumentChunker
    queue: Optional[QueueOperations] = None

    def ingest(self, source_uri: str) -> None:
        self.ingestion.ingest_document(source_uri, self.chunker)

    def delete(self, source_uri: str) -> int:
        return self.ingestion.delete_document(source_uri)

    def query(self, query: str, top_k: int = 5) -> QueryResult:
        context = self.retrieval.retrieve_context(query, top_k=top_k)
        answer = "".join(self.generator.generate_stream(query, context))
        candidates = [RetrievalCandidate(**candidate) for candidate in context]
        citations = [SourceCitation(
            chunk_id=candidate.chunk_id,
            source_uri=str(candidate.metadata.get("source", "unknown")),
            parent_id=candidate.parent_id,
            title=candidate.metadata.get("title"),
        ) for candidate in candidates]
        return QueryResult(query=query, answer=answer, candidates=candidates, citations=citations)


def build_in_memory_application(
    config: Optional[AppConfig] = None,
    *,
    embedding_provider: Any = None,
    query_processor: Any = None,
    reranker: Any = None,
    answer_generator: Any = None,
) -> RagApplication:
    from src.infrastructure.memory import (
        DeterministicAnswerGenerator,
        DeterministicEmbeddingProvider,
        IdentityQueryProcessor,
        InMemoryCache,
        InMemoryDenseIndex,
        InMemoryGraphIndex,
        InMemoryQueue,
        InMemorySparseIndex,
        NoOpGraphExtractor,
        PlainTextChunker,
        TokenOverlapReranker,
    )

    settings = config or profile_config(Profile.TEST)
    embedder = embedding_provider or DeterministicEmbeddingProvider()
    cache = InMemoryCache()
    dense_index = InMemoryDenseIndex()
    sparse_index = InMemorySparseIndex()
    graph_index = InMemoryGraphIndex()
    tokenizer = type("SimpleTokenizer", (), {"tokenize": staticmethod(InMemorySparseIndex._tokens)})()
    ingestion = IngestionService(
        sparse_index, graph_index, dense_index, embedder, cache, NoOpGraphExtractor(),
        embedding_batch_size=settings.pipeline.embedding_batch_size,
        graph_extraction_enabled=settings.pipeline.enable_graph_extraction,
    )
    dense = DenseRetriever(dense_index, embedder, cache)
    sparse = SparseRetriever(sparse_index)
    graph = GraphRetriever(graph_index, sparse_index, tokenizer, max_hops=settings.pipeline.graph_max_hops)
    manager = RetrieverManager(dense, sparse, graph, enable_hyde=settings.pipeline.enable_hyde)
    retrieval = RetrievalService(
        query_processor or IdentityQueryProcessor(), manager, ReciprocalRankFusion(),
        reranker or TokenOverlapReranker(), candidate_top_k=settings.pipeline.retrieval_top_k,
        rerank_pool_multiplier=settings.pipeline.rerank_pool_multiplier,
    )
    chunker = PlainTextChunker(settings.pipeline.chunk_size, settings.pipeline.chunk_overlap)
    return RagApplication(
        settings, ingestion, retrieval, answer_generator or DeterministicAnswerGenerator(), chunker, InMemoryQueue()
    )


def build_local_application(config: AppConfig, include_queue: bool = True) -> RagApplication:
    import tiktoken
    from groq import Groq

    from src.chunker import SemanticDoclingChunker
    from src.graph.graph_extractor import GraphExtractor
    from src.graph.ontology import DomainOntology
    from src.infrastructure.cache import SQLiteEmbeddingCache
    from src.infrastructure.providers import (
        CrossEncoderReranker,
        ProductionEmbedder,
        ProductionResponseGenerator,
        SemanticQueryProcessor,
    )
    from src.infrastructure.queue import IngestionQueueManager
    from src.infrastructure.storage import KnowledgeGraphStore, ProductionVectorStore, SparseStore
    from src.tokenizer import canonical_tokenizer

    graph_index = KnowledgeGraphStore(config.storage.graph_db_path)
    sparse_index = SparseStore(config.storage.sparse_db_path)
    embedder = ProductionEmbedder(config.models.embedding_model)
    cache = SQLiteEmbeddingCache(config.storage.cache_db_path)
    dense_index = ProductionVectorStore(
        config.storage.collection_name, embedder.vector_dim, storage_path=config.storage.qdrant_path
    )
    client = Groq(api_key=config.models.groq_api_key) if config.models.groq_api_key else None
    ontology_root = Path(__file__).resolve().parents[1] / "graph" / "ontologies"
    ontology = DomainOntology.load_pipeline_configs(
        str(ontology_root / "software.json"), str(ontology_root / "aliases.json")
    )
    extractor = GraphExtractor(client, config.models.groq_model, ontology)
    ingestion = IngestionService(
        sparse_index, graph_index, dense_index, embedder, cache, extractor,
        embedding_batch_size=config.pipeline.embedding_batch_size,
        graph_extraction_enabled=config.pipeline.enable_graph_extraction,
    )
    dense = DenseRetriever(dense_index, embedder, cache)
    sparse = SparseRetriever(sparse_index)
    graph = GraphRetriever(
        graph_index, sparse_index, canonical_tokenizer, max_hops=config.pipeline.graph_max_hops
    )
    manager = RetrieverManager(dense, sparse, graph, enable_hyde=config.pipeline.enable_hyde)
    processor = SemanticQueryProcessor(
        semantic_enabled=True, llm_client=client, model_name=config.models.groq_model,
        api_key=config.models.groq_api_key,
    )
    retrieval = RetrievalService(
        processor, manager, ReciprocalRankFusion(), CrossEncoderReranker(config.models.reranker_model),
        candidate_top_k=config.pipeline.retrieval_top_k,
        rerank_pool_multiplier=config.pipeline.rerank_pool_multiplier,
    )
    generator = ProductionResponseGenerator(
        llm_client=client, model_name=config.models.groq_model, api_key=config.models.groq_api_key
    )
    chunker = SemanticDoclingChunker(
        tiktoken.get_encoding("cl100k_base"), config.pipeline.chunk_size, config.pipeline.chunk_overlap
    )
    queue = IngestionQueueManager(ingestion, chunker) if include_queue else None
    return RagApplication(config, ingestion, retrieval, generator, chunker, queue)


def build_application(config: AppConfig, include_queue: bool = True) -> RagApplication:
    if config.providers.dense_index == "memory":
        return build_in_memory_application(config)
    return build_local_application(config, include_queue=include_queue)

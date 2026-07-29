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
    retrieval_cache: Any = None

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

    def shutdown(self) -> None:
        """Close long-lived executors and provider transports owned by this process."""
        if hasattr(self.retrieval, "shutdown"):
            self.retrieval.shutdown()
        for dependency in (
            self.ingestion.vector_store, self.ingestion.sparse_store, self.ingestion.graph_store,
            getattr(self.ingestion, "cache", None), self.retrieval_cache,
        ):
            close = getattr(dependency, "close", None)
            if close:
                close()


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
        embedding_model_version=settings.models.embedding_model,
        pipeline_version=settings.pipeline.pipeline_version,
        embedding_batch_max_tokens=settings.performance.embedding_batch_max_tokens,
        embedding_batch_max_bytes=settings.performance.embedding_batch_max_bytes,
        embedding_memory_budget_bytes=settings.performance.embedding_memory_budget_bytes,
        parsing_concurrency=settings.performance.parsing_concurrency,
        embedding_concurrency=settings.performance.embedding_concurrency,
        graph_concurrency=settings.performance.graph_concurrency,
        index_concurrency=settings.performance.index_concurrency,
    )
    dense = DenseRetriever(dense_index, embedder, cache, settings.models.embedding_model)
    sparse = SparseRetriever(sparse_index)
    graph = GraphRetriever(graph_index, sparse_index, tokenizer, max_hops=settings.pipeline.graph_max_hops)
    manager = RetrieverManager(
        dense, sparse, graph, enable_hyde=settings.pipeline.enable_hyde,
        max_workers=settings.performance.retriever_workers,
        max_pending=settings.performance.retriever_pending,
        per_retriever_timeout=settings.performance.per_retriever_timeout_seconds,
    )
    retrieval = RetrievalService(
        query_processor or IdentityQueryProcessor(), manager, ReciprocalRankFusion(),
        reranker or TokenOverlapReranker(), candidate_top_k=settings.pipeline.retrieval_top_k,
        rerank_pool_multiplier=settings.pipeline.rerank_pool_multiplier,
        rewrite_concurrency=settings.performance.rewrite_concurrency,
        reranker_concurrency=settings.performance.reranker_concurrency,
        reranker_candidate_cap=settings.performance.reranker_candidate_cap,
        reranker_batch_size=settings.performance.reranker_batch_size,
        reranker_skip_below=settings.performance.reranker_skip_below,
        stage_pending=settings.performance.query_max_concurrency,
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
    from src.application.cached_providers import CachingGraphExtractor, CachingQueryProcessor
    from src.infrastructure.cache import SQLiteEmbeddingCache
    from src.infrastructure.redis_cache import RedisVersionedCache
    from src.infrastructure.providers import (
        CrossEncoderReranker,
        ProductionEmbedder,
        ProductionResponseGenerator,
        SemanticQueryProcessor,
    )
    from src.infrastructure.queue import IngestionQueueManager
    from src.infrastructure.storage import KnowledgeGraphStore, ProductionVectorStore, SparseStore
    from src.tokenizer import canonical_tokenizer
    from src.application.performance_adapters import ObservableCircuitBreaker

    if config.providers.graph_index == "postgres":
        from src.infrastructure.postgres import PostgresGraphIndex
        graph_index = PostgresGraphIndex(config.storage.control_database_url, config.publication.namespace)
    else:
        graph_index = KnowledgeGraphStore(config.storage.graph_db_path)
    sparse_index = SparseStore(config.storage.sparse_db_path)
    embedder = ProductionEmbedder(config.models.embedding_model)
    redis_client = None
    if config.providers.cache == "redis":
        import redis
        redis_client = redis.Redis.from_url(
            config.queue.redis_url, max_connections=max(8, config.performance.query_max_concurrency * 2)
        )
        cache = RedisVersionedCache(
            redis_client, "embedding", config.performance.cache_ttl_seconds,
            config.performance.cache_max_value_bytes, config.performance.cache_lock_seconds,
        )
        dense_cache = RedisVersionedCache(
            redis_client, "query", config.performance.cache_ttl_seconds,
            config.performance.cache_max_value_bytes, config.performance.cache_lock_seconds,
        )
    else:
        cache = SQLiteEmbeddingCache(config.storage.cache_db_path)
        dense_cache = cache
    dense_index = ProductionVectorStore(
        config.storage.collection_name, embedder.vector_dim,
        storage_path=None if config.storage.qdrant_url else config.storage.qdrant_path,
        url=config.storage.qdrant_url or None,
        pool_size=config.performance.qdrant_pool_size,
        breaker_failure_threshold=config.performance.breaker_failure_threshold,
        breaker_recovery_seconds=config.performance.breaker_recovery_seconds,
        breaker_half_open_probes=config.performance.breaker_half_open_probes,
    )
    client = Groq(api_key=config.models.groq_api_key, timeout=config.pipeline.provider_timeout_seconds) \
        if config.models.groq_api_key else None
    groq_breaker = ObservableCircuitBreaker(
        "groq", config.performance.breaker_failure_threshold,
        config.performance.breaker_recovery_seconds, config.performance.breaker_half_open_probes,
    )
    ontology_root = Path(__file__).resolve().parents[1] / "graph" / "ontologies"
    ontology = DomainOntology.load_pipeline_configs(
        str(ontology_root / "software.json"), str(ontology_root / "aliases.json")
    )
    extractor: Any = GraphExtractor(client, config.models.groq_model, ontology, groq_breaker)
    if redis_client:
        extractor = CachingGraphExtractor(
            extractor,
            RedisVersionedCache(redis_client, "graph", config.performance.cache_ttl_seconds,
                                config.performance.cache_max_value_bytes, config.performance.cache_lock_seconds),
            config.models.groq_model, parser_chunker_version=(
                f"{config.pipeline.parser_version}:{config.pipeline.chunker_config_version}"
            ),
        )
    ingestion = IngestionService(
        sparse_index, graph_index, dense_index, embedder, cache, extractor,
        embedding_batch_size=config.pipeline.embedding_batch_size,
        graph_extraction_enabled=config.pipeline.enable_graph_extraction,
        embedding_model_version=config.models.embedding_model,
        pipeline_version=config.pipeline.pipeline_version,
        embedding_batch_max_tokens=config.performance.embedding_batch_max_tokens,
        embedding_batch_max_bytes=config.performance.embedding_batch_max_bytes,
        embedding_memory_budget_bytes=config.performance.embedding_memory_budget_bytes,
        parsing_concurrency=config.performance.parsing_concurrency,
        embedding_concurrency=config.performance.embedding_concurrency,
        graph_concurrency=config.performance.graph_concurrency,
        index_concurrency=config.performance.index_concurrency,
    )
    dense = DenseRetriever(dense_index, embedder, dense_cache, config.models.embedding_model)
    sparse = SparseRetriever(sparse_index)
    graph = GraphRetriever(
        graph_index, sparse_index, canonical_tokenizer, max_hops=config.pipeline.graph_max_hops
    )
    manager = RetrieverManager(
        dense, sparse, graph, enable_hyde=config.pipeline.enable_hyde,
        max_workers=config.performance.retriever_workers, max_pending=config.performance.retriever_pending,
        per_retriever_timeout=config.performance.per_retriever_timeout_seconds,
    )
    processor: Any = SemanticQueryProcessor(
        semantic_enabled=True, llm_client=client, model_name=config.models.groq_model,
        api_key=config.models.groq_api_key, circuit_breaker=groq_breaker,
    )
    if redis_client:
        processor = CachingQueryProcessor(
            processor,
            RedisVersionedCache(redis_client, "representation", config.performance.cache_ttl_seconds,
                                config.performance.cache_max_value_bytes, config.performance.cache_lock_seconds),
            "groq", config.models.groq_model,
        )
    retrieval = RetrievalService(
        processor, manager, ReciprocalRankFusion(), CrossEncoderReranker(config.models.reranker_model),
        candidate_top_k=config.pipeline.retrieval_top_k,
        rerank_pool_multiplier=config.pipeline.rerank_pool_multiplier,
        rewrite_concurrency=config.performance.rewrite_concurrency,
        reranker_concurrency=config.performance.reranker_concurrency,
        reranker_candidate_cap=config.performance.reranker_candidate_cap,
        reranker_batch_size=config.performance.reranker_batch_size,
        reranker_skip_below=config.performance.reranker_skip_below,
        stage_pending=config.performance.query_max_concurrency,
    )
    generator = ProductionResponseGenerator(
        llm_client=client, model_name=config.models.groq_model, api_key=config.models.groq_api_key,
        circuit_breaker=groq_breaker,
    )
    chunker = SemanticDoclingChunker(
        tiktoken.get_encoding("cl100k_base"), config.pipeline.chunk_size, config.pipeline.chunk_overlap
    )
    queue = IngestionQueueManager(ingestion, chunker) if include_queue else None
    retrieval_cache = RedisVersionedCache(
        redis_client, "retrieval", config.performance.cache_ttl_seconds,
        config.performance.cache_max_value_bytes, config.performance.cache_lock_seconds,
    ) if redis_client else None
    return RagApplication(config, ingestion, retrieval, generator, chunker, queue, retrieval_cache)


def build_application(config: AppConfig, include_queue: bool = True) -> RagApplication:
    if config.providers.dense_index == "memory":
        return build_in_memory_application(config)
    return build_local_application(config, include_queue=include_queue)

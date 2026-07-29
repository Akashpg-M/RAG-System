from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.api.errors import ApiError
from src.application.cached_providers import CachingGraphExtractor
from src.application.config import PerformanceSettings, Profile, profile_config
from src.application.query_runtime import QueryRuntime
from src.application.snapshot_cache import PublicationSnapshotCache
from src.core.cache_keys import (
    graph_extraction_key, normalize_query, query_embedding_key, representation_key, retrieval_key,
)
from src.core.ingestion import IngestionService
from src.core.publication import PreparedDocument
from src.core.performance import (
    BoundedExecutor, CapacityExhausted, CircuitBreaker, CircuitOpen, CircuitState, Deadline, DeadlineExceeded,
)
from src.core.publication import PublicationSnapshot
from src.core.retrieval import RetrievalMatrices, RetrievalService, RetrieverManager
from src.infrastructure.redis_cache import RedisVersionedCache
from src.infrastructure.memory import InMemoryCache, InMemoryDenseIndex, InMemoryGraphIndex, InMemorySparseIndex
from src.models import ChildChunk
from src.observability import Observability


class FakeRedis:
    def __init__(self):
        self.values: dict[str, bytes | str] = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.values.get(key)

    def set(self, key, value, ex=None, nx=False):
        with self.lock:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

    def delete(self, *keys):
        with self.lock:
            for key in keys:
                self.values.pop(key, None)

    def scan(self, cursor=0, match=None, count=100):
        prefix = (match or "").rstrip("*")
        return 0, [key for key in self.values if key.startswith(prefix)]


def test_deadline_uses_one_monotonic_budget():
    deadline = Deadline.after(0.03)
    first = deadline.require()
    time.sleep(0.02)
    assert deadline.remaining < first
    time.sleep(0.02)
    with pytest.raises(DeadlineExceeded):
        deadline.require()


def test_bounded_executor_rejects_without_growing_queue():
    release = threading.Event()
    executor = BoundedExecutor(1, 1, "bounded-test")
    running = executor.submit(release.wait, 1)
    with pytest.raises(CapacityExhausted):
        executor.submit(lambda: None)
    release.set()
    running.result(1)
    executor.shutdown()


def test_cancelled_http_waiter_does_not_release_underlying_query_capacity():
    async def scenario():
        telemetry = Observability("runtime-test", "test", "6", "stage-6", sample_ratio=0)
        runtime = QueryRuntime(1, 1, telemetry)
        release = threading.Event()
        started = threading.Event()

        def slow():
            started.set()
            release.wait(1)

        task = asyncio.create_task(runtime.execute(slow))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ApiError) as error:
            await runtime.execute(lambda: None)
        assert error.value.code == "query_capacity_exhausted"
        release.set()
        for _ in range(50):
            if runtime.bulkhead.active == 0:
                break
            await asyncio.sleep(0.01)
        assert runtime.bulkhead.active == 0
        runtime.shutdown()

    asyncio.run(scenario())


def test_circuit_breaker_closed_open_half_open_and_recovery():
    now = [0.0]
    breaker = CircuitBreaker("provider", failure_threshold=2, recovery_timeout=5, clock=lambda: now[0])

    def fail():
        raise TimeoutError("transient")

    with pytest.raises(TimeoutError):
        breaker.call(fail)
    with pytest.raises(TimeoutError):
        breaker.call(fail)
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpen):
        breaker.call(lambda: "blocked")
    now[0] = 6
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is CircuitState.CLOSED


def test_circuit_ignores_capacity_and_deadline_controls():
    breaker = CircuitBreaker("provider", failure_threshold=1)
    for error in (CapacityExhausted("full"), DeadlineExceeded("done")):
        with pytest.raises(type(error)):
            breaker.call(lambda error=error: (_ for _ in ()).throw(error))
        assert breaker.state is CircuitState.CLOSED


def test_cache_keys_are_hashed_versioned_and_scope_isolated():
    query = "  Secret\tquery  "
    assert normalize_query(query) == "Secret query"
    embedding = query_embedding_key(query, "embed-v1")
    assert "Secret" not in embedding
    assert embedding != query_embedding_key(query, "embed-v2")
    common = (query, 7, {"mode": "hybrid"}, {"department": "eng"}, "tenant-a")
    first = retrieval_key(*common, "scope-a", 5, "schema-v1")
    assert first != retrieval_key(*common, "scope-b", 5, "schema-v1")
    assert first != retrieval_key(query, 8, common[2], common[3], common[4], "scope-a", 5, "schema-v1")
    assert first != retrieval_key(*common, "scope-a", 10, "schema-v1")
    assert query not in first


def test_representation_and_graph_keys_include_correctness_versions():
    one = representation_key("q", "groq", "m1", "p1", {"temperature": 0}, "hybrid")
    two = representation_key("q", "groq", "m1", "p2", {"temperature": 0}, "hybrid")
    assert one != two
    graph = graph_extraction_key("content", "m", "p", "ontology", "schema", "parser")
    assert graph != graph_extraction_key("content", "m", "p", "ontology-v2", "schema", "parser")


def test_redis_cache_corruption_is_a_fail_open_miss():
    client = FakeRedis()
    cache = RedisVersionedCache(client, "query", ttl_seconds=10)
    client.values[cache._key("key")] = b"not-json"
    assert cache.get("key") is None
    assert cache.get_or_compute("key", lambda: [1, 2]) == [1, 2]
    assert cache.get("key") == [1, 2]


def test_redis_cache_single_flight_computes_once_per_process():
    cache = RedisVersionedCache(FakeRedis(), "query", ttl_seconds=10)
    calls = 0
    calls_lock = threading.Lock()

    def compute():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)
        return {"value": 1}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: cache.get_or_compute("shared", compute), range(8)))
    assert results == [{"value": 1}] * 8
    assert calls == 1


def test_cache_rejects_oversize_and_unknown_schema_entries():
    client = FakeRedis()
    cache = RedisVersionedCache(client, "retrieval", max_value_bytes=80)
    assert not cache.set("large", "x" * 100)
    client.values[cache._key("old")] = json.dumps({"schema": "old", "value": 1}).encode()
    assert cache.get("old") is None


def test_snapshot_cache_reuses_only_current_immutable_revision():
    class Publication:
        revision = 1
        loads = 0

        def current_revision(self):
            return self.revision

        def snapshot(self):
            self.loads += 1
            return PublicationSnapshot(self.revision, {"doc": f"v{self.revision}"}, frozenset(), frozenset())

    publication = Publication()
    cache = PublicationSnapshotCache(max_entries=2, ttl_seconds=30)
    assert cache.load(publication).active_versions["doc"] == "v1"
    assert cache.load(publication).active_versions["doc"] == "v1"
    assert publication.loads == 1
    publication.revision = 2
    assert cache.load(publication).active_versions["doc"] == "v2"
    assert publication.loads == 2


class Retriever:
    def __init__(self, name: str, delay: float = 0, failure: Exception | None = None):
        self.name, self.delay, self.failure = name, delay, failure

    def retrieve(self, query, top_k, filters=None):
        time.sleep(self.delay)
        if self.failure:
            raise self.failure
        return [{"chunk_id": self.name, "text": self.name}]


def test_optional_retriever_timeout_degrades_while_mandatory_results_survive():
    manager = RetrieverManager(
        Retriever("dense"), Retriever("sparse"), Retriever("graph", delay=0.08),
        enable_hyde=False, max_workers=3, max_pending=3, per_retriever_timeout=0.02,
    )
    payload = {"original_query": "q", "rewritten_query": "q", "hyde_document": "q"}
    result = manager.execute_routing(payload, mode="hybrid", deadline=Deadline.after(0.2))
    assert len(result) == 3
    assert result[0] and result[1] and result[2] == []
    assert "graph_timeout" in result.degraded_reasons
    manager.shutdown()


def test_mandatory_retriever_timeout_is_controlled_failure():
    manager = RetrieverManager(
        Retriever("dense"), Retriever("sparse", delay=0.08), Retriever("graph"),
        enable_hyde=False, max_workers=3, max_pending=3, per_retriever_timeout=0.02,
    )
    payload = {"original_query": "q", "rewritten_query": "q", "hyde_document": "q"}
    with pytest.raises(DeadlineExceeded):
        manager.execute_routing(payload, mode="fast", deadline=Deadline.after(0.2), mandatory={"sparse"})
    manager.shutdown()


def test_graph_cache_hit_returns_fresh_records_for_new_version_identity():
    class Extractor:
        calls = 0

        def extract_triples(self, text):
            self.calls += 1
            return [{"source": "a", "relation": "uses", "target": "b"}]

    extractor = Extractor()
    cached = CachingGraphExtractor(extractor, RedisVersionedCache(FakeRedis(), "graph"), "model")
    first = cached.extract_triples("same parent")
    first[0]["chunk_id"] = "doc#v1#p0"
    second = cached.extract_triples("same parent")
    second[0]["chunk_id"] = "doc#v2#p0"
    assert extractor.calls == 1
    assert first[0]["chunk_id"] != second[0]["chunk_id"]


def test_mixed_embedding_cache_hits_and_memory_bounded_batches_preserve_order():
    class Embedder:
        calls: list[list[str]] = []

        def get_embeddings_batched(self, texts, batch_size=64):
            self.calls.append(list(texts))
            return [[float(int(text.removeprefix("chunk-")))] for text in texts]

    class Extractor:
        def extract_triples(self, text):
            return []

    cache = InMemoryCache()
    embedder = Embedder()
    service = IngestionService(
        InMemorySparseIndex(), InMemoryGraphIndex(), InMemoryDenseIndex(), embedder, cache, Extractor(),
        embedding_batch_size=2, embedding_batch_max_tokens=10, embedding_batch_max_bytes=30,
        embedding_memory_budget_bytes=10_000,
    )
    chunks = [
        ChildChunk(f"doc#v1#c{i}", "doc", "doc#v1#p0", f"chunk-{i}", 1, f"hash-{i}", {"version_id": "v1"})
        for i in range(5)
    ]
    from src.core.cache_keys import content_embedding_key
    cache.set(content_embedding_key("hash-1", "configured", "configured"), [1.0])
    cache.set(content_embedding_key("hash-3", "configured", "configured"), [3.0])
    prepared = PreparedDocument("doc", "v1", [], chunks, [])
    service.prepare_embeddings(prepared)
    assert prepared.embeddings == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    assert embedder.calls == [["chunk-0", "chunk-2"], ["chunk-4"]]


def test_reranker_batches_keep_scores_attached_to_the_original_candidate():
    candidates = [
        {"chunk_id": f"c{i}", "text": str(i), "rrf_score": 1.0, "metadata": {}} for i in range(7)
    ]

    class Processor:
        def process_query(self, query):
            return {"original_query": query, "rewritten_query": query, "hyde_document": query}

    class Manager:
        def execute_routing(self, *args, **kwargs):
            matrices = RetrievalMatrices([candidates])
            matrices.degraded_reasons = ()
            return matrices

        def shutdown(self):
            return None

    class Fusion:
        def fuse(self, matrices):
            return list(matrices[0])

    class Reranker:
        batches: list[list[str]] = []

        def predict(self, pairs):
            self.batches.append([pair[1] for pair in pairs])
            return [float(pair[1]) for pair in pairs]

    reranker = Reranker()
    service = RetrievalService(
        Processor(), Manager(), Fusion(), reranker, candidate_top_k=7, rerank_pool_multiplier=2,
        reranker_candidate_cap=7, reranker_batch_size=3, reranker_skip_below=0,
    )
    result = service.retrieve_context("q", top_k=7)
    assert [item["chunk_id"] for item in result] == [f"c{i}" for i in range(6, -1, -1)]
    assert reranker.batches == [["0", "1", "2"], ["3", "4", "5"], ["6"]]
    service.shutdown()


def test_candidate_refill_is_bounded_and_fills_after_stale_candidates(tmp_path):
    from src.api.metrics import ApiMetrics
    from src.api.schemas import QueryRequest
    from src.api.services import QueryApplicationService

    calls: list[int] = []

    class Retrieval:
        def retrieve_context(self, query, top_k, **kwargs):
            calls.append(top_k)
            stale = [{"chunk_id": f"doc#old#c{i}", "text": "stale", "metadata": {
                "document_id": "doc", "version_id": "old", "namespace": "default",
            }} for i in range(top_k)]
            valid = [{"chunk_id": "doc#v1#c0", "text": "first", "metadata": {
                "document_id": "doc", "version_id": "v1", "namespace": "default",
            }}]
            if top_k >= 8:
                valid.append({"chunk_id": "doc#v1#c1", "text": "second", "metadata": {
                    "document_id": "doc", "version_id": "v1", "namespace": "default",
                }})
            return [*stale, *valid]

    class Generator:
        def generate_stream(self, query, context):
            yield "grounded"

    class App:
        retrieval = Retrieval()
        generator = Generator()
        retrieval_cache = None

    class Publication:
        def current_revision(self):
            return 1

        def snapshot(self):
            return PublicationSnapshot(1, {"doc": "v1"}, frozenset(), frozenset())

    class Documents:
        def get_latest(self, document_id):
            return None

    config = profile_config(Profile.TEST, tmp_path)
    config = config.model_copy(update={"performance": config.performance.model_copy(update={
        "refill_max_rounds": 1, "refill_candidate_cap": 8,
    })})
    service = QueryApplicationService(App(), config, Documents(), ApiMetrics(lambda: 0), Publication())
    response = service.execute(QueryRequest(query="q", top_k=2), "trace")
    assert calls == [4, 8]
    assert [source.chunk_id for source in response.sources] == ["doc#v1#c0", "doc#v1#c1"]
    service.close()


def test_performance_configuration_rejects_unsafe_limits():
    with pytest.raises(ValueError):
        PerformanceSettings(query_max_concurrency=4, query_executor_pending=2)
    with pytest.raises(ValueError):
        PerformanceSettings(query_timeout_seconds=1, per_retriever_timeout_seconds=2)
    with pytest.raises(ValueError):
        PerformanceSettings(adaptive_retrieval_mode="mystery")


def test_adaptive_routing_is_disabled_by_default_and_shadow_is_non_mutating(tmp_path):
    from src.api.metrics import ApiMetrics
    from src.api.services import QueryApplicationService

    class App:
        retrieval = object()
        generator = object()

    config = profile_config(Profile.TEST, tmp_path)
    service = QueryApplicationService(App(), config, object(), ApiMetrics(lambda: 0))
    assert service._execution_mode("how are A and B connected", "hybrid")[0] == "hybrid"
    shadow = config.model_copy(update={
        "performance": config.performance.model_copy(update={"adaptive_retrieval_mode": "shadow"})
    })
    shadow_service = QueryApplicationService(App(), shadow, object(), ApiMetrics(lambda: 0))
    route, decision, reason = shadow_service._execution_mode("how are A and B connected", "hybrid")
    assert (route, decision, reason) == ("hybrid", "graph", "entity_query")
    service.close()
    shadow_service.close()

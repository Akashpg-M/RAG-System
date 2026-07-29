# Stage 6 performance and stability

## Execution architecture

The API admits at most `QUERY_MAX_CONCURRENCY` blocking query pipelines. A persistent bounded executor moves Qdrant, SQLite, model inference, reranking, and generation off the ASGI event loop. Cancelling an HTTP waiter does not release its capacity token until the underlying blocking operation exits, which prevents cancelled requests from accumulating hidden work.

Retrieval submits sparse, rewritten-dense, HyDE-dense, and graph branches as sibling operations to a long-lived bounded executor. Dense is mandatory for hybrid/fast routing; sparse, graph, and HyDE failures are explicitly degradable unless an application call overrides the mandatory set. Each branch receives the lesser of its configured timeout and the monotonic query deadline remaining.

Separate bulkheads protect query admission, query rewriting/HyDE, retrieval workers, cross-encoder reranking, generation, parsing, ingestion embedding, graph extraction, and each index writer. Queues are bounded by the already admitted request/task count. Capacity exhaustion fails before allocating expensive work with a bounded `Retry-After`; API rate policy uses 429, temporary execution capacity uses 503.

## Deadline and cancellation model

`Deadline` stores one monotonic expiration. Snapshot loading, representation generation, each retrieval branch, candidate refill, fusion, reranking batches, and generation consume the same remaining budget. Optional generation or reranking may return verified sources with bounded degradation metadata. A mandatory retrieval timeout returns controlled `query_timeout`/503.

Python cannot forcibly stop arbitrary blocking native code in a running thread. Timeouts cancel futures that have not started; already-running calls retain their bulkhead/executor slot until completion. Provider client timeouts and circuit breakers bound that residual work.

## Clients and pools

- Redis caches and queues reuse long-lived connection pools.
- PostgreSQL repositories share a psycopg 3 `ConnectionPool` per process. Wait duration, checked-out connections, exhaustion, and failures use bounded metrics.
- Qdrant reuses one client/HTTP transport with bounded keep-alive and connection limits.
- Groq reuses one configured client across rewrite, graph extraction, and generation.
- Lifespan shutdown closes executors, providers, queue transports, readiness workers, and telemetry.

Pool limits should be smaller than the database/server limit after accounting for every API and worker container. Increasing a pool is not a substitute for measuring wait and exhaustion metrics.

## Circuit breakers and degradation

Provider-neutral breakers implement CLOSED, OPEN, and HALF_OPEN states, transient-failure thresholds, recovery timeouts, limited probes, and successful closure. Capacity rejection, query-deadline exhaustion, deliberate cancellation, validation errors, and empty results do not count as provider failures. Qdrant is mandatory for dense retrieval; Groq-backed rewriting, optional graph extraction, and generation follow their documented fallback/source-only behavior.

## Cache specification

Redis keys contain hashes, never raw queries. Values have a serialization schema, TTL, maximum byte size, corruption fallback, local and Redis-backed single-flight locks, bounded lock duration, and fail-open dependency behavior.

| Cache | Correctness dimensions |
| --- | --- |
| Query embedding | normalized-query hash, normalization version, embedding model/version |
| Rewrite/HyDE | query hash, provider/model, prompt version, generation parameters, retrieval mode |
| Retrieval | representation hash, corpus revision, retrieval config, filters, namespace, authorization scope, top-k, index schema |
| Graph extraction | parent content hash, extractor model, prompt, ontology, schema, parser/chunker versions |

NFKC and whitespace normalization are deliberately conservative. Final-answer caching remains disabled. Activation, rollback, and tombstone creation increment the corpus revision, making earlier retrieval keys unreachable immediately; old revisions can be removed asynchronously.

## Publication snapshots and refill

Queries first read the scalar PostgreSQL corpus revision. An immutable bounded LRU/TTL cache stores the complete publication manifest by revision. Cache failure falls back to PostgreSQL and can never synthesize a staging version.

Candidate retrieval begins with a bounded overfetch. Active-version, tombstone, namespace, authorization, and requested filters are applied. If valid candidates remain below `top_k`, the service doubles the requested count until it reaches `CANDIDATE_REFILL_ROUNDS`, `CANDIDATE_REFILL_CAP`, or the query deadline. Shortfall is explicit in logs/metrics and never weakens publication filtering.

## Ingestion batching and backpressure

Embedding batches are bounded simultaneously by item count, approximate token count, UTF-8 bytes, and configured memory budget. Preallocated result slots preserve chunk-to-vector ordering across cache hits and multiple batches. Parsing, embedding, graph extraction, dense indexing, sparse indexing, and graph indexing have distinct bulkheads.

Worker intake acquires a worker slot before receiving another message. Queue depth/capacity plus worker and downstream bulkheads therefore bound pressure. Messages remain unacknowledged until durable activation and are recovered through Stage 3 leases/retries.

Cleanup claims a conclusively retired version transactionally and marks it `CLEANING` before touching external indexes. Rollback accepts only validated active/retiring/retired versions, so it cannot race a physical removal already in progress; failed cleanup returns the version to `RETIRED`.

## Adaptive routing

`ADAPTIVE_RETRIEVAL_MODE=off` is the default. `shadow` records an inspectable decision but executes full hybrid. `adaptive` selects dense+sparse by default, adds graph for explicit relationship terms, and adds HyDE for deterministically classified complex queries. Decisions and reason codes are bounded telemetry fields. No quality improvement is claimed; a later evaluation stage must validate thresholds before production enablement.

## Reproduction

```powershell
# Core services and containerized application
docker compose --profile application --profile observability up -d --build

# Deterministic API, then benchmark
$env:RAG_PROFILE = "test"
$env:AUTH_ENABLED = "false"
python -m uvicorn src.api.app:create_api --factory --host 127.0.0.1 --port 8000
python benchmarks/run_benchmark.py --config benchmarks/configs/deterministic.json --label rerun

# Controlled experiments and report
python benchmarks/run_experiments.py
python benchmarks/generate_report.py

# Locust (benchmark-only high client limit is required to isolate execution capacity)
python -m locust -f benchmarks/locustfile.py DeterministicInfrastructureUser --headless `
  --users 8 --spawn-rate 2 --run-time 15s --host http://127.0.0.1:8000
```

Results are in `benchmarks/results/`; PromQL used for measurements is in `benchmarks/promql.txt`. The optional Groq workload exits without running if `GROQ_API_KEY` is absent.

## Acceptance method and limitations

Maximum stable concurrency requires P95 within the configured SLO, error rate below the limit, bounded queue/memory behavior, and no uncontrolled task accumulation. An optimization is retained only if its target improves, correctness remains green, tails/errors stay controlled, and memory stays bounded.

The current shared Docker runtime image is functional but large because the Linux dependency solve selects CUDA-capable PyTorch transitive packages. A CPU-only model image and a separate lightweight dispatcher/migration image are deferred image-distribution improvements. See the generated Stage 6 report for actual results and rejected experiments.

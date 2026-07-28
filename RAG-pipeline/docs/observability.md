# Stage 5 observability

## Architecture and local startup

Application processes emit Prometheus metrics independently and export OTLP traces through bounded, fail-open SDK queues. The OpenTelemetry Collector batches traces into Tempo. Services write one-line JSON to stdout; Grafana Alloy discovers Docker containers and forwards their logs to Loki. Prometheus scrapes the API, worker, dispatcher, Collector, cAdvisor, PostgreSQL exporter, Redis exporter, and Qdrant. Grafana is provisioned with all three data sources and the `RAG Stage 5` dashboard.

Start the core dependencies with `docker compose up -d`. Add the local observability stack with:

```console
docker compose --profile observability up -d
```

Grafana is at <http://localhost:3000> (`admin` / `rag-local-only`), Prometheus at <http://localhost:9090>, Tempo at <http://localhost:3200>, and Loki at <http://localhost:3100>. The credentials are deliberately local-only. Node Exporter is omitted because its Linux host view is misleading under Docker Desktop; cAdvisor supplies container CPU and memory instead.

## Service identity and sampling

The service names are `rag-api`, `rag-worker`, and `rag-outbox-dispatcher`. Resources also contain `service.version`, `deployment.environment`, `rag.pipeline.version`, and a process/container instance identity. `OTEL_TRACES_SAMPLER_RATIO` controls a parent-based ratio sampler. Tests use an in-memory deterministic exporter; local defaults to all traces. `OTEL_EXPORTER_OTLP_ENDPOINT` selects the Collector. A missing backend never fails an application operation: exporters use a 2,048-span queue, 256-span batches, two-second export timeout, and controlled SDK retry behavior.

## Trace hierarchy

Query traces begin at the ASGI request and include authentication, `query.publication_snapshot`, representations, concurrent sibling `query.retrieve.*` spans, fusion, publication filtering, reranking, prompt construction, provider generation, and response middleware. Thread branches receive separate copies of the current context so their timings overlap correctly.

Ingestion traces contain upload acceptance, the atomic task/outbox operation, producer publication, receive, the lease-protected consumer operation, storage download, parsing, chunking, graph extraction, manifest persistence, embedding, each staged index, atomic activation, acknowledgement, temporary cleanup, and cleanup jobs. The ingestion envelope persists W3C `traceparent`/`tracestate`; Redis preserves the envelope, while SQS additionally mirrors the carrier into message attributes. Retries and DLQ copies preserve the same carrier. The consumer creates a new span linked by the extracted parent even after the HTTP request has ended.

## JSON log schema and redaction

Every configured operational log has UTC `timestamp`, `severity`, `service`, `environment`, an allow-shaped `event` name, `trace_id`, `span_id`, and `request_id`. Optional bounded fields are `operation`, `component`, `lifecycle_stage`, `outcome`, `retry_attempt`, `duration_seconds`, and `error_code`. Trace identifiers are injected automatically from the active context.

The formatter centrally suppresses free-form log messages and redacts authorization, credential, key, secret, password, and token fields, bearer values, and filesystem paths. Call sites log sanitized error codes, not exception text. Query text, prompts, retrieved text, document content, model responses, raw filenames/object keys, and credentials are prohibited. Opaque record identifiers remain control-plane data and are not metric labels or resource attributes.

## Application metrics

All metrics use a private registry per service instance, so repeated test/app construction cannot register the same collector twice. Scrapes read cached queue gauges and never perform Redis or PostgreSQL calls. Labels are centrally bounded; route paths are normalized before collection. Cache ratio is calculated from hit/miss counters rather than maintained as mutable state.

| Metric | Type | Controlled labels | Meaning |
|---|---|---|---|
| `rag_query_duration_seconds` | histogram | service | End-to-end query latency |
| `rag_retrieval_duration_seconds` | histogram | service, retriever | Dense, sparse, HyDE, or graph latency |
| `rag_fusion_duration_seconds` | histogram | service | Fusion/RRF latency |
| `rag_publication_filter_duration_seconds` | histogram | service | Active-version filter latency |
| `rag_rerank_duration_seconds` | histogram | service | Cross-encoder latency |
| `rag_generation_duration_seconds` | histogram | service | Complete generation latency |
| `rag_time_to_first_token_seconds` | histogram | service | First real streamed provider token; absent for non-streaming providers |
| `rag_ingestion_duration_seconds` | histogram | service, stage | Ingestion stage or delivery duration |
| `rag_embedding_batch_duration_seconds` | histogram | service | Embedding batch latency |
| `rag_publication_snapshot_duration_seconds` | histogram | service | Snapshot acquisition latency |
| `rag_index_publication_duration_seconds` | histogram | service, index | Version-staged index write latency |
| `rag_cleanup_duration_seconds` | histogram | service | Physical cleanup latency |
| `rag_query_requests_total` | counter | service, outcome, strategy | Query volume and outcome |
| `rag_query_errors_total` | counter | service, error_type | Bounded query failures |
| `rag_ingestion_tasks_total` | counter | service, status | Accepted tasks and lifecycle transitions |
| `rag_ingestion_failures_total` | counter | service, stage, error_type | Classified ingestion failures |
| `rag_cache_requests_total` | counter | service, cache, result | Cache hits and misses |
| `rag_empty_context_total` | counter | service | Queries with no approved context |
| `rag_ingestion_retries_total` / `rag_dlq_messages_total` | counter | service, stage / service | Retry and DLQ volume |
| `rag_llm_tokens_total` | counter | service, direction | Only provider-reported or defined tokenizer counts |
| `rag_publication_attempts_total` / `rag_publication_degraded_total` | counter | service, outcome / service | Atomic activation outcomes and optional-index degradation |
| `rag_cleanup_jobs_total` | counter | service, outcome | Cleanup outcomes |
| `rag_reconciliation_discrepancies_total` | counter | service, type | Missing, orphaned, checksum, count, staging, retirement, or tombstone discrepancy |
| `rag_candidate_discarded_total` | counter | service, reason | Publication filter rejection |
| `rag_queue_depth`, `rag_queue_pending_messages`, `rag_queue_oldest_message_age_seconds` | gauge | service | Cached queue state |
| `rag_active_workers`, `rag_active_ingestions` | gauge | service | Current worker/process activity |
| `rag_tombstoned_documents`, `rag_retired_versions_awaiting_cleanup` | gauge | service | Durable deletion and retirement backlog |
| `rag_staging_version_age_seconds`, `rag_rollback_operations` | gauge | service | Oldest staging age and durable rollback revision count |
| `rag_retrieval_candidates`, `rag_publication_candidates`, `rag_candidate_refill_rounds` | histogram | bounded | Retrieval/filter distributions |
| `rag_embedding_batch_size`, `rag_publication_snapshot_documents` | histogram | service | Workload-size distributions |

Latency buckets are dense from 5 ms through 10 s and extend to 60 s. Ingestion buckets extend to five minutes. Candidate and batch buckets are explicit powers/useful operating thresholds rather than default buckets. Infrastructure metrics originate from cAdvisor and the dependency exporters, not the application registry.

TTFT is deliberately absent for deterministic and other non-streaming generators. Token totals remain zero/absent when the provider does not report usage; they are never inferred from character counts.

## Recording rules and PromQL

Provisioned rules calculate request rate, error ratio, query P50/P95/P99, retrieval P95 by retriever, cache hit ratio, ingestion throughput/failure rate, publication failure ratio, and candidate discard ratio. Classic histogram quantiles preserve `le`, for example:

```promql
histogram_quantile(0.95, sum by (le) (rate(rag_query_duration_seconds_bucket[5m])))
sum(rate(rag_cache_requests_total{result="hit"}[5m])) / clamp_min(sum(rate(rag_cache_requests_total[5m])), 0.000001)
```

## Grafana correlation and troubleshooting

The Loki data source derives a Tempo link from the JSON `trace_id`. Tempo's `tracesToLogsV2` query searches Loki for the same ID inside the span time window. Dashboard panels cover query, retrieval, ingestion/publication, consistency, and infrastructure. Use a one-hour default range and the bounded service variable.

If traces are absent, check Collector `/metrics`, Tempo `/ready`, the OTLP endpoint, and sampling ratio. If logs are absent, check Alloy's UI on port 12345 and Docker socket access. If application panels are empty, confirm each process exposes its own port (`8000/metrics`, `9465`, `9466`) and that `host.docker.internal` resolves inside Docker Desktop. Prometheus target failures do not alter API/worker readiness. Loki/Tempo outages may drop bounded telemetry during a long outage but never block processing.

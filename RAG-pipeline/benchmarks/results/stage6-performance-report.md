# Stage 6 performance report

Generated from versioned JSON/CSV artifacts in this directory. Measurements were taken on Windows 11 with 6 physical/12 logical CPUs, 7.89 GB host memory, and Docker Desktop limited to 2 CPUs and 4.11 GB. The worktree was intentionally dirty because Stage 6 was under implementation.

## End-to-end benchmark results

| Workload | Samples | Concurrency | P50 before (ms) | P50 after (ms) | P95 before (ms) | P95 after (ms) | P99 before (ms) | P99 after (ms) | QPS before | QPS after | Errors after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic infrastructure | 200 | 8 | 88.58 | 62.71 | 116.13 | 99.92 | 126.16 | 115.00 | 87.61 | 118.72 | 0.0% |
| Real local retrieval, warm revisioned cache | 40 | 2 | 6178.58 | 25.97 | 11149.64 | 55.45 | 12375.93 | 71.85 | 0.31 | 65.59 | 0.0% |

The deterministic accepted result changed P95 by -14.0% and throughput by +35.5%. The real local result measures the newly reachable warm retrieval-cache path; it is not a claim about cold model inference.

## Stability boundary

Concurrency 8 met the benchmark criterion with P95 99.92 ms and zero errors. At concurrency 16, the service returned controlled 503 responses at 6.5%; P95 was 174.77 ms and the bounded admission layer prevented uncontrolled queue growth. The measured maximum stable concurrency for this configuration is therefore 8, not 16.

## Controlled one-variable experiments

| Experiment | Before variant | After variant | P95 before (ms) | P95 after (ms) |
| --- | --- | --- | ---: | ---: |
| retrieval_execution | sequential | parallel | 20.03 | 7.56 |
| query_cache | cold | warm | 2.54 | 0.00 |
| embedding_delivery | individual | batched | 66.92 | 3.27 |
| publication_snapshot | database_10000 | cached_10000 | 8.29 | 0.59 |
| candidate_filtering | without_refill | bounded_refill | 0.00 | 0.00 |

These are deterministic provider-stub micro-experiments. They validate direction, ordering, and boundedness; they do not substitute for retrieval-quality evaluation or production provider load.

## Rejected experiments

The first optimized deterministic run used stage queues sized only to stage worker counts. It achieved P95 62.73 ms but rejected 37.5% of requests. That setting was rejected. Stage queues now remain bounded by the already admitted query count; the repeated final run had zero errors.

The first Locust comparison was also rejected because the fixed client rate-limit window had already been consumed by a preceding saturation run. The final Locust run restarted the server and used a documented benchmark-only client limit: 1,481 requests, 0 failures, 32 ms P50, 59 ms P95, and 98 ms P99.

## Container and observability evidence

The API, worker, and dispatcher ran as distinct containers with limits of 2 GiB, 2 GiB, and 384 MiB. During first initialization they used 749.3 MiB, 691.9 MiB, and 60.51 MiB respectively. Prometheus reported all three application targets and cAdvisor `up`.

Tempo trace `3bcc32c0358babefe69e35099d0ec828` contains API upload/outbox, dispatcher publication, and worker parsing, embedding, three index stages, activation, and acknowledgement. Post-fix query trace `ec8c0395dee4adca4441a2b2281f7ea2` contains generation as a child of the same query root; Loki returned a correlated JSON record with trace/span/request IDs and no raw query.

## Limitations

- Live Groq load was skipped because no credential was configured; no provider cost or rate budget was consumed.
- The local retrieval after-result is a warm versioned-cache measurement. Cold inference is represented by the Stage 5 baseline and controlled cache experiment.
- Docker's shared runtime image currently includes CUDA-capable PyTorch transitive packages. It is functional but larger and slower to build than a CPU-only/split dispatcher image should be.
- After a Docker Desktop restart and host-clock resynchronization, Prometheus still reported target metadata as `up` but rejected new samples as out-of-order against future-dated samples retained in its persistent local TSDB. Application processing was unaffected; the telemetry volume was deliberately not deleted to conceal the condition.
- Adaptive routing remains disabled by default. Shadow mode records deterministic decisions while executing full hybrid retrieval; no quality improvement is claimed.
- Candidate refill is bounded by deadline, round count, and candidate cap; it can still return an explicit shortfall when stale content dominates all fetched candidates.

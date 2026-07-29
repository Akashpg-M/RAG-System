from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def milliseconds(value: float) -> str:
    return f"{value * 1000:.2f}"


def delta(before: float, after: float, lower_is_better: bool = True) -> str:
    change = (after - before) / before * 100 if before else 0
    return f"{change:+.1f}%"


def main() -> None:
    baseline = load("stage5-deterministic-baseline.json")["result"]
    optimized = load("stage6-deterministic-optimized-final.json")["result"]
    local_before = load("stage5-local-retrieval-baseline.json")["result"]
    local_after = load("stage6-local-retrieval-optimized.json")["result"]
    saturated = load("stage6-deterministic-saturation-16.json")["result"]
    rejected = load("stage6-deterministic-optimized.json")["result"]
    experiments = load("stage6-controlled-experiments.json")["records"]
    selected = {(item["experiment"], item["variant"]): item for item in experiments}
    rows = []
    for experiment, left, right in (
        ("retrieval_execution", "sequential", "parallel"),
        ("query_cache", "cold", "warm"),
        ("embedding_delivery", "individual", "batched"),
        ("publication_snapshot", "database_10000", "cached_10000"),
        ("candidate_filtering", "without_refill", "bounded_refill"),
    ):
        first, second = selected[(experiment, left)], selected[(experiment, right)]
        rows.append(
            f"| {experiment} | {left} | {right} | {milliseconds(first['latency_seconds']['p95'])} | "
            f"{milliseconds(second['latency_seconds']['p95'])} |"
        )
    report = f"""# Stage 6 performance report

Generated from versioned JSON/CSV artifacts in this directory. Measurements were taken on Windows 11 with 6 physical/12 logical CPUs, 7.89 GB host memory, and Docker Desktop limited to 2 CPUs and 4.11 GB. The worktree was intentionally dirty because Stage 6 was under implementation.

## End-to-end benchmark results

| Workload | Samples | Concurrency | P50 before (ms) | P50 after (ms) | P95 before (ms) | P95 after (ms) | P99 before (ms) | P99 after (ms) | QPS before | QPS after | Errors after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic infrastructure | 200 | 8 | {milliseconds(baseline['latency_seconds']['p50'])} | {milliseconds(optimized['latency_seconds']['p50'])} | {milliseconds(baseline['latency_seconds']['p95'])} | {milliseconds(optimized['latency_seconds']['p95'])} | {milliseconds(baseline['latency_seconds']['p99'])} | {milliseconds(optimized['latency_seconds']['p99'])} | {baseline['throughput_qps']:.2f} | {optimized['throughput_qps']:.2f} | {optimized['error_rate']:.1%} |
| Real local retrieval, warm revisioned cache | 40 | 2 | {milliseconds(local_before['latency_seconds']['p50'])} | {milliseconds(local_after['latency_seconds']['p50'])} | {milliseconds(local_before['latency_seconds']['p95'])} | {milliseconds(local_after['latency_seconds']['p95'])} | {milliseconds(local_before['latency_seconds']['p99'])} | {milliseconds(local_after['latency_seconds']['p99'])} | {local_before['throughput_qps']:.2f} | {local_after['throughput_qps']:.2f} | {local_after['error_rate']:.1%} |

The deterministic accepted result changed P95 by {delta(baseline['latency_seconds']['p95'], optimized['latency_seconds']['p95'])} and throughput by {delta(baseline['throughput_qps'], optimized['throughput_qps'], False)}. The real local result measures the newly reachable warm retrieval-cache path; it is not a claim about cold model inference.

## Stability boundary

Concurrency 8 met the benchmark criterion with P95 {milliseconds(optimized['latency_seconds']['p95'])} ms and zero errors. At concurrency 16, the service returned controlled 503 responses at {saturated['rejection_rate']:.1%}; P95 was {milliseconds(saturated['latency_seconds']['p95'])} ms and the bounded admission layer prevented uncontrolled queue growth. The measured maximum stable concurrency for this configuration is therefore 8, not 16.

## Controlled one-variable experiments

| Experiment | Before variant | After variant | P95 before (ms) | P95 after (ms) |
| --- | --- | --- | ---: | ---: |
{chr(10).join(rows)}

These are deterministic provider-stub micro-experiments. They validate direction, ordering, and boundedness; they do not substitute for retrieval-quality evaluation or production provider load.

## Rejected experiments

The first optimized deterministic run used stage queues sized only to stage worker counts. It achieved P95 {milliseconds(rejected['latency_seconds']['p95'])} ms but rejected {rejected['rejection_rate']:.1%} of requests. That setting was rejected. Stage queues now remain bounded by the already admitted query count; the repeated final run had zero errors.

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
"""
    target = RESULTS / "stage6-performance-report.md"
    target.write_text(report, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

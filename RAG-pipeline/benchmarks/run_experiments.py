from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import psutil


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def measure(experiment: str, variant: str, operation: Callable[[], Any], samples: int = 30) -> dict[str, Any]:
    process = psutil.Process()
    cpu_start = process.cpu_times()
    rss_start = process.memory_info().rss
    latencies: list[float] = []
    errors = 0
    started = time.perf_counter()
    for _ in range(samples):
        item_started = time.perf_counter()
        try:
            operation()
        except Exception:
            errors += 1
        latencies.append(time.perf_counter() - item_started)
    duration = time.perf_counter() - started
    cpu_end = process.cpu_times()
    return {
        "experiment": experiment,
        "variant": variant,
        "sample_count": samples,
        "warmup_seconds": 0,
        "measurement_seconds": duration,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": statistics.fmean(latencies),
        },
        "throughput_per_second": samples / duration if duration else 0,
        "error_rate": errors / samples,
        "cpu_seconds": (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system),
        "rss_delta_bytes": process.memory_info().rss - rss_start,
        "cache_state": "not_applicable",
        "provider": "deterministic_stub",
    }


def sleep_branches(parallel: bool) -> None:
    durations = (0.003, 0.004, 0.005, 0.006)
    if parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(time.sleep, durations))
    else:
        for duration in durations:
            time.sleep(duration)


def embedding_batches(batch_size: int, items: int = 64) -> None:
    for offset in range(0, items, batch_size):
        batch = min(batch_size, items - offset)
        time.sleep(0.0005 + batch * 0.00002)


def rerank(cap: int, batch_size: int = 8) -> None:
    scores = []
    for offset in range(0, cap, batch_size):
        batch = list(range(offset, min(cap, offset + batch_size)))
        time.sleep(0.0004 + len(batch) * 0.00001)
        scores.extend(float(value) for value in batch)
    assert scores == [float(value) for value in range(cap)]


class SnapshotSource:
    def __init__(self, documents: int):
        self.documents = documents
        self.revision = 7

    def current_revision(self) -> int:
        time.sleep(0.00005)
        return self.revision

    def snapshot(self) -> dict[str, str]:
        time.sleep(self.documents * 0.0000002)
        return {f"d{index}": "v1" for index in range(self.documents)}


def snapshot_lookup(source: SnapshotSource, cache: dict[int, dict[str, str]] | None) -> None:
    revision = source.current_revision()
    if cache is None or revision not in cache:
        snapshot = source.snapshot()
        if cache is not None:
            cache[revision] = snapshot


def filter_candidates(refill: bool) -> None:
    top_k, requested, rounds = 5, 10, 0
    while True:
        valid = max(0, requested - 8)
        if valid >= top_k or not refill or rounds >= 2 or requested >= 40:
            return
        requested = min(40, requested * 2)
        rounds += 1


def cache_lookup(cache: dict[str, list[int]], warm: bool) -> None:
    key = "versioned-hash"
    if not warm:
        cache.pop(key, None)
    value = cache.get(key)
    if value is None:
        time.sleep(0.002)
        cache[key] = list(range(64))


def bounded_concurrency(workers: int, tasks: int = 32) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _: time.sleep(0.001), range(tasks)))


def main() -> None:
    records: list[dict[str, Any]] = []
    records += [
        measure("retrieval_execution", "sequential", lambda: sleep_branches(False)),
        measure("retrieval_execution", "parallel", lambda: sleep_branches(True)),
    ]
    cache: dict[str, list[int]] = {}
    records += [
        measure("query_cache", "cold", lambda: cache_lookup(cache, False)),
        measure("query_cache", "warm", lambda: cache_lookup(cache, True)),
    ]
    branch_counts = {"dense": 1, "dense_sparse": 2, "full_hybrid": 4}
    for name, count in branch_counts.items():
        records.append(measure("retrieval_strategy", name, lambda count=count: time.sleep(0.002 * count)))
    records += [
        measure("reranker_toggle", "disabled", lambda: None),
        measure("reranker_toggle", "enabled", lambda: rerank(30)),
        measure("embedding_delivery", "individual", lambda: embedding_batches(1)),
        measure("embedding_delivery", "batched", lambda: embedding_batches(32)),
    ]
    for batch_size in (8, 16, 32, 64):
        records.append(measure("embedding_batch_size", str(batch_size), lambda size=batch_size: embedding_batches(size)))
    for cap in (10, 20, 30, 50):
        records.append(measure("reranker_candidate_cap", str(cap), lambda cap=cap: rerank(cap)))
    for concurrency in (1, 4, 8):
        records.append(measure("query_concurrency", str(concurrency),
                               lambda workers=concurrency: bounded_concurrency(workers)))
    for concurrency in (1, 2, 4):
        records.append(measure("ingestion_worker_concurrency", str(concurrency),
                               lambda workers=concurrency: bounded_concurrency(workers, 16)))
    records += [
        measure("adaptive_routing", "full_hybrid", lambda: sleep_branches(True)),
        measure("adaptive_routing", "shadow", lambda: ("fast", "default_fast_path", sleep_branches(True))),
    ]
    for size in (10, 1_000, 10_000):
        source = SnapshotSource(size)
        records.append(measure("publication_snapshot", f"database_{size}", lambda source=source: snapshot_lookup(source, None)))
        snapshot_cache: dict[int, dict[str, str]] = {}
        snapshot_lookup(source, snapshot_cache)
        records.append(measure("publication_snapshot", f"cached_{size}",
                               lambda source=source, cache=snapshot_cache: snapshot_lookup(source, cache)))
    records += [
        measure("candidate_filtering", "without_refill", lambda: filter_candidates(False)),
        measure("candidate_filtering", "bounded_refill", lambda: filter_candidates(True)),
    ]
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": "Deterministic micro-experiments; one primary variable per pair; no external providers.",
        "records": records,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "stage6-controlled-experiments.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def git_metadata(root: Path) -> dict[str, Any]:
    try:
        options = ["git", "-c", f"safe.directory={root.parent.as_posix()}", "-C", str(root.parent)]
        revision = subprocess.check_output([*options, "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output([*options, "status", "--porcelain", "--", root.name],
                                             text=True).strip())
        return {"revision": revision, "dirty": dirty}
    except Exception:
        return {"revision": "unavailable", "dirty": True}


def docker_metadata() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=10)
        info = json.loads(raw)
        return {"cpus": info.get("NCPU"), "memory_bytes": info.get("MemTotal"),
                "operating_system": info.get("OperatingSystem")}
    except Exception:
        return {"available": False}


def environment(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": sys.version.split()[0], "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False), "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available, "docker": docker_metadata(), "git": git_metadata(root),
        "models": {"embedding": os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
                   "reranker": os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                   "generation": os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")},
        "dataset_version": config["dataset_version"], "cache_state": config["cache_state"],
        "concurrency": config["concurrency"], "warmup_requests": config["warmup_requests"],
        "measurement_requests": config["measurement_requests"],
    }


def request(client: httpx.Client, base_url: str, query: str, timeout: float) -> tuple[float, int]:
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url}/api/v1/query",
                               json={"query": query, "top_k": 5, "retrieval_mode": "hybrid"}, timeout=timeout)
        return time.perf_counter() - started, response.status_code
    except httpx.TimeoutException:
        return time.perf_counter() - started, 504
    except httpx.HTTPError:
        return time.perf_counter() - started, 599


def run(config: dict[str, Any], base_url: str) -> dict[str, Any]:
    queries, timeout = config["queries"], float(config["query_timeout_seconds"])
    with httpx.Client() as client:
        for index in range(config["warmup_requests"]):
            request(client, base_url, queries[index % len(queries)], timeout)
        started_at, started = datetime.now(timezone.utc), time.perf_counter()
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            futures = [executor.submit(request, client, base_url, queries[index % len(queries)], timeout)
                       for index in range(config["measurement_requests"])]
            samples = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - started
    latencies = [sample[0] for sample in samples]
    errors = sum(status >= 400 for _, status in samples)
    rejected = sum(status in (429, 503) for _, status in samples)
    return {"started_at": started_at.isoformat(), "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": elapsed, "sample_count": len(samples),
            "latency_seconds": {"p50": percentile(latencies, .50), "p95": percentile(latencies, .95),
                                "p99": percentile(latencies, .99),
                                "mean": statistics.fmean(latencies) if latencies else 0},
            "throughput_qps": len(samples) / elapsed if elapsed else 0,
            "error_rate": errors / len(samples) if samples else 0,
            "rejection_rate": rejected / len(samples) if samples else 0,
            "status_counts": {str(code): sum(status == code for _, status in samples)
                              for code in sorted({status for _, status in samples})}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("requires_live_groq") and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("live Groq benchmark skipped: GROQ_API_KEY is not configured")
    result = {"schema_version": "1.0", "label": args.label, "benchmark": config,
              "environment": environment(root, config), "result": run(config, args.base_url)}
    output = Path(args.output_dir) / f"{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["result"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()

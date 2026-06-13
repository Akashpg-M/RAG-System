import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("TelemetryTracker")

class QueryTelemetryTracker:
    """Captures and structures deep operational performance tracking across the RAG engine."""
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "latencies": {},
            "counts": {},
            "cache_hits": 0,
            "cache_misses": 0
        }

    def start_timer(self) -> float:
        return time.perf_counter()

    def stop_timer(self, stage_name: str, start_time: float):
        elapsed = (time.perf_counter() - start_time) * 1000.0  # Normalized to milliseconds
        self.metrics["latencies"][stage_name] = round(elapsed, 2)

    def record_count(self, key: str, value: int):
        self.metrics["counts"][key] = value

    def record_cache(self, hit: bool):
        if hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def emit_telemetry_report(self):
        """Prints a structured breakdown of the execution path performance metrics."""
        logger.info("=== RETRIEVAL LOOP COMPONENT METRICS ===")
        for stage, lat in self.metrics["latencies"].items():
            logger.info(f" - [Stage Latency] {stage.upper().ljust(15)}: {lat} ms")
        for metric_name, qty in self.metrics["counts"].items():
            logger.info(f" - [Data Volume]   {metric_name.upper().ljust(15)}: {qty}")
        
        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_requests > 0:
            hit_ratio = (self.metrics["cache_hits"] / total_cache_requests) * 100
            logger.info(f" - [Cache Accuracy] HIT RATIO      : {hit_ratio:.1f}%")
        logger.info("========================================")
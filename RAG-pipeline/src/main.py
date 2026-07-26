"""Local composition root and Stage 0 compatibility entry point."""

import logging

from src.application.composition import build_local_application
from src.application.config import Profile, load_config
from src.core.ingestion import IngestionService
from src.telemetry import QueryTelemetryTracker

ProductionIngestionPipeline = IngestionService
logger = logging.getLogger("LocalRAG")


def main() -> None:
    config = load_config(Profile.LOCAL)
    application = build_local_application(config)
    source = "data/raw/system_design.md"
    application.ingest(source)

    telemetry = QueryTelemetryTracker()
    query = "What deployment technology does the system use?"
    started = telemetry.start_timer()
    result = application.query(query, top_k=2)
    telemetry.stop_timer("total_query", started)
    print(result.answer)
    telemetry.record_count("retrieved_candidates", len(result.candidates))
    telemetry.emit_telemetry_report()


if __name__ == "__main__":
    main()


__all__ = ["ProductionIngestionPipeline", "main"]

# ADR 0001: Provider-independent RAG core

Status: accepted

Business orchestration is separated from infrastructure because ingestion consistency, retrieval routing, fusion, and failure behavior must be reusable by APIs, workers, and evaluation jobs. Core modules depend only on shared contracts and protocols. Provider modules implement those protocols and are selected at the composition boundary.

Compatibility facades preserve Stage 0 import paths during the transition. Cross-provider transactions remain best-effort because Qdrant and SQLite do not share a transaction manager; changing that behavior is outside this refactor.


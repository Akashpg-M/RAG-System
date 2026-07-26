# Modular RAG service architecture

## Dependency direction

```text
HTTP routes / CLI / future workers and evaluations
                         |
            API and application services
                         |
           core services, contracts, ports
                         ^
                         |
       infrastructure adapters and providers
```

`src/core` owns shared contracts, provider protocols, ingestion orchestration, retrieval routing, RRF, and retriever behavior. It imports no Qdrant, SQLite, Groq, SentenceTransformers, Docling, NLTK, or FastAPI modules.

`src/application` validates profiles and wires the reusable RAG application. `src/api` adds HTTP schemas, lifecycle/query services, thin routes, authentication, validation, readiness, exception mapping, and metrics. Routes reference application services through `request.app.state`; they do not construct or import concrete indexes or model providers.

`src/infrastructure` implements Qdrant, SQLite, local-file, background-thread, model-provider, and deterministic in-memory adapters. Historical top-level imports remain Stage 0 compatibility facades.

## HTTP control plane

The versioned API exposes JSON queries, multipart uploads, task status, document deletion, liveness, readiness, and Prometheus metrics. Local upload bytes are validated before the object-storage port persists them. The request returns `202` after the task and document version are saved and background work is queued. The worker calls the existing `IngestionService`; it does not reproduce ingestion in the API layer.

API-assigned document and version IDs are injected into chunk metadata and used across dense, sparse, and graph indexes. This makes document filters, citations, status records, and deletion reference the same identity. Sparse filtering is applied before truncating ranked results, while all retrievers also enforce a common candidate filter.

## Lifecycle

```text
UPLOADING -> QUEUED -> PARSING -> CHUNKING -> EMBEDDING
  -> INDEXING_DENSE -> INDEXING_SPARSE -> [INDEXING_GRAPH] -> READY

active/ready/failure -> DELETE_PENDING -> DELETED
active stage -> FAILED_RETRYABLE | FAILED_PERMANENT
```

Transitions are validated and persisted through `TaskRepository`. Status responses expose controlled error codes, never provider exception text. Per-document locks prevent a queued ingestion from returning a deleted document to `READY`.

## Security and operations

Protected `/api/v1` routes use `X-API-Key` and a single-process sliding-window rate limiter. Upload validation enforces size, extension, MIME/signature agreement, safe filenames, UTF-8 text limits, PDF page/text limits, and DOCX archive limits. Validation and exception handlers return traceable errors without stack traces or raw input.

`/health` is process-only. `/ready` runs bounded lightweight probes for configured mandatory indexes; optional graph/Groq capabilities do not block readiness when fallback behavior is enabled. `/metrics` uses normalized route templates and static status/method labels, never queries, filenames, IDs, or content.

## Profiles and future adapters

- `test`: deterministic in-memory RAG providers, local temporary object storage, and in-memory control repositories.
- `local`: Qdrant, SQLite, local files, thread queues, SentenceTransformers, cross-encoder, and optional Groq.
- `benchmark`: larger retrieval/batch settings with generation and graph extraction disabled.
- `aws_demo`: validated placeholders only; AWS infrastructure remains deferred.

Environment variables are read once at composition. A future API process, external worker, evaluator, or AWS adapter can implement existing ports without modifying route contracts or core orchestration.


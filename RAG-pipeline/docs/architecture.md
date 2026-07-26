# Modular RAG architecture

## Dependency direction

```text
entry points / future API / workers / evaluation
                  |
          application composition
                  |
       core services + contracts + ports
                  ^
                  |
 infrastructure adapters and model providers
```

`src/core` owns stable contracts, provider protocols, ingestion orchestration, retrieval routing, RRF, and retriever behavior. It imports no Qdrant, SQLite, Groq, SentenceTransformers, Docling, or NLTK modules.

`src/infrastructure` implements local adapters: Qdrant dense storage, SQLite sparse/graph/cache/repositories, the thread queue, SentenceTransformer embeddings, cross-encoder reranking, Groq query/generation, and deterministic in-memory substitutes.

`src/application` validates configuration and wires a complete application. `build_local_application` preserves the current local providers. `build_in_memory_application` provides a model-free, database-free offline application. Existing top-level modules remain compatibility facades for Stage 0 imports.

## Core contracts and ports

Contracts cover parent/child chunks, upload/deletion events, document versions, ingestion tasks and statuses, retrieval candidates, query results, and citations. Ports cover object storage, document/task repositories, queues, chunking, dense/sparse/graph indexes, embeddings, reranking, graph extraction, query processing, answer generation, caching, retrieval, and fusion.

The ingestion service prepares graph and embedding artifacts, preserves cache ordering, and replaces document-owned records across all indexes. Retrieval runs sparse, rewritten dense, optional HyDE dense, and graph routes concurrently, then applies RRF and an injected reranker.

## Profiles

- `test`: in-memory indexes/cache/queue, deterministic embeddings/reranking/generation, no HyDE or graph extraction.
- `local`: Qdrant, SQLite stores/cache, thread queue, SentenceTransformers, cross-encoder, and optional Groq.
- `benchmark`: larger retrieval/batch limits with graph extraction and answer generation disabled.
- `aws_demo`: validated placeholder selections and timeouts only. AWS adapters are intentionally not implemented in this stage.

Environment variables are read once by `load_config` at the application boundary. Core services receive configuration values and providers through constructors.

## Future consumers

FastAPI handlers will call `RagApplication.ingest`, `delete`, and `query` without importing infrastructure. A process worker can consume upload/deletion event contracts and invoke `IngestionService`. Evaluation pipelines can inject deterministic or benchmark providers. Future AWS adapters will implement the existing ports and be selected in composition without changing core orchestration.


# Modular Multi-Index RAG

This repository provides a reusable RAG core with explicit provider boundaries. The local application preserves the stabilized Docling → hierarchical chunking → Qdrant/SQLite/graph ingestion flow and sparse + rewritten-dense + HyDE-dense + graph → RRF → cross-encoder retrieval flow.

## Module boundaries

```text
src/core/             contracts, ports, ingestion, retrieval, retrievers, RRF
src/application/      validated profiles and dependency composition
src/infrastructure/   local and deterministic adapters
src/graph/            ontology and graph extraction provider
src/*.py              Stage 0 compatibility facades and local adapters
```

Dependencies point inward: entry points and future services use application composition; composition selects infrastructure implementations of core ports; core never imports Qdrant, SQLite, Groq, SentenceTransformers, Docling, or NLTK.

Detailed boundaries and extension guidance are in [docs/architecture.md](docs/architecture.md). The separation decision is recorded in [ADR 0001](docs/decisions/0001-provider-independent-core.md).

## Configuration profiles

`src.application.config` provides validated `test`, `local`, `benchmark`, and `aws_demo` profiles. The AWS profile reserves provider choices and operational settings; it does not implement AWS integrations.

Environment variables are read once by `load_config`:

```env
GROQ_API_KEY=
GROQ_MODEL_NAME=llama-3.3-70b-versatile
QDRANT_STORAGE_PATH=./qdrant_local_data
SPARSE_DB_PATH=./sparse_index.db
GRAPH_DB_PATH=./graph_store.db
EMBEDDING_CACHE_PATH=./embedding_cache.db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=256
CHUNK_OVERLAP=30
```

## Setup and validation

Use Python 3.10 or newer and create a fresh environment; copied virtual environments are not portable.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check src tests
python -m mypy
python -m compileall -q src tests
```

Tests do not call Groq or download models. They include a complete in-memory ingest/query/delete flow and a temporary local Qdrant/SQLite integration check.

## Running locally

```powershell
python -m src.main
```

The local composition root ingests `data/raw/system_design.md` and executes a sample query. Local SentenceTransformer, cross-encoder, and Docling artifacts may require network access on first use. Groq is optional for query fallback but required for graph extraction and answer generation.

For an entirely offline application:

```python
from src.application.composition import build_application
from src.application.config import Profile, profile_config

app = build_application(profile_config(Profile.TEST))
app.ingest("document.md")
result = app.query("What does the document say?")
```

Re-ingestion replaces all records owned by the stable path-derived document ID. `app.delete(path)` consistently deletes the corresponding dense, sparse, and graph records.


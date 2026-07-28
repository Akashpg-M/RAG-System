# Versioned Multi-Index RAG Service

This repository provides a provider-independent RAG core and a versioned FastAPI query/document-control service. The local profile preserves the stabilized Docling -> hierarchical chunking -> Qdrant/SQLite/graph ingestion flow and sparse + rewritten-dense + HyDE-dense + graph -> RRF -> cross-encoder retrieval flow.

## Structure

```text
src/core/             contracts, ports, ingestion, retrieval, retrievers, RRF
src/application/      validated profiles and RAG dependency composition
src/api/              HTTP schemas, services, routes, security, metrics, validation
src/infrastructure/   local, SQLite, Qdrant, queue, model, and memory adapters
src/graph/            ontology and graph extraction provider
src/*.py              Stage 0 compatibility facades and local adapters
```

See [architecture.md](docs/architecture.md), [ADR 0001](docs/decisions/0001-provider-independent-core.md), and [ADR 0002](docs/decisions/0002-versioned-http-control-plane.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Do not reuse a copied `venv`; always install with `python -m pip` from the active environment.

## Run the offline API

The test profile uses deterministic in-memory indexes/models and requires no Groq, Qdrant, SQLite, or model downloads:

```powershell
$env:RAG_PROFILE = "test"
$env:RAG_API_KEY = "test-api-key"
python -m src.api
```

OpenAPI and interactive documentation:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

Example health and readiness calls:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Example upload:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/upload `
  -H "X-API-Key: test-api-key" `
  -F "file=@data/raw/system_design.md;type=text/markdown" `
  -F "category=architecture"
```

Poll the returned `status_url` until the state is `READY`, then query:

```powershell
$body = @{
  query = "What technology is used for deployment?"
  retrieval_mode = "hybrid"
  top_k = 3
  stream = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/query `
  -Headers @{"X-API-Key"="test-api-key"} `
  -ContentType "application/json" `
  -Body $body
```

## Run the full local API

```powershell
$env:RAG_PROFILE = "local"
$env:RAG_API_KEY = "replace-with-a-local-secret"
$env:GROQ_API_KEY = "your-groq-key"  # optional for retrieval fallback; required for generated answers/graph extraction
python -m src.api
```

The first local start may download Docling and SentenceTransformer/cross-encoder artifacts. Runtime data is written to ignored local Qdrant, SQLite, cache, and upload paths.

### Durable ingestion worker

Stage 3 runs upload ingestion outside the API. Start Redis, the API, and one or more
workers in separate terminals:

```powershell
docker compose up -d redis
python -m src.api
python -m src.worker
```

The local backend uses Redis Streams and consumer groups. Upload acceptance persists a
task plus outbox intent before returning `202`; workers acknowledge only after `READY`,
recover abandoned pending entries, renew persistent leases, use scheduled delayed
retries, and route terminal failures to a separate DLQ stream. Set
`INGESTION_QUEUE_BACKEND=sqs`, `SQS_QUEUE_URL`, and `SQS_DLQ_URL` to select SQS Standard.
SQS infrastructure provisioning and live-AWS tests are intentionally deferred.

## API endpoints

```text
POST   /api/v1/query
POST   /api/v1/documents/upload
GET    /api/v1/documents/{task_id}/status
DELETE /api/v1/documents/{document_id}
GET    /health
GET    /ready
GET    /metrics
```

Protected endpoints require `X-API-Key`. JSON queries currently reject `stream=true` with a controlled `422` response. Uploads return `202` after storage and queue acceptance; parsing and indexing never run in the request handler.

## Validation

```powershell
python -m pytest
python -m ruff check src tests
python -m mypy
python -m compileall -q src tests
```

The suite uses deterministic providers and temporary storage. It does not call Groq or download models, while retaining the temporary local Qdrant/SQLite compatibility test.

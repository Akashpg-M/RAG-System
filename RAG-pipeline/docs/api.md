# HTTP API contract

All `/api/v1` endpoints require `X-API-Key` when authentication is enabled. Operational endpoints are public. Every response includes `X-Trace-ID`; controlled errors contain `error`, `message`, and `trace_id`.

## `POST /api/v1/query`

JSON fields: `query`, optional `filters`, `retrieval_mode` (`hybrid`, `dense`, `sparse`, `graph`), `top_k`, optional `conversation_id`, and `stream`. The server bounds query length and `top_k`. Filters support one `document_id` plus the configured metadata allowlist. `stream=true` returns controlled `422` in this version.

Success is `200` with answer, retrieval strategy, model/configuration versions, trace ID, refusal flags, and bounded sources containing document/version/chunk identity, page/section metadata, excerpts, and available retrieval scores. Empty matching context also returns `200` with `empty_context=true` and `refused=true`.

## `POST /api/v1/documents/upload`

Multipart fields: required `file`; optional `document_id`, `category`, `department`, and `language`. Local uploads return `202` with document/version/task IDs, `QUEUED`, status URL, and a nullable future upload descriptor. File processing happens after the request on the configured queue.

Validation failures return controlled `400`, `413`, `415`, or `422` responses. Storage or queue dependency failures return `503` without provider details.

## `GET /api/v1/documents/{task_id}/status`

Returns `200` with current status, ordered status history, IDs, and a controlled error code. Unknown tasks return `404`. Repeated reads are side-effect free.

## `DELETE /api/v1/documents/{document_id}`

Returns `200` after cross-index and object cleanup. Repeated deletion returns `200` with `already_deleted=true`. Unknown documents return `404`. Cleanup failure returns `503` while lifecycle state remains non-READY.

## Operations

- `GET /health`: `200` if the API process can answer; no dependency work.
- `GET /ready`: `200` only when mandatory lightweight probes pass, otherwise `503`.
- `GET /metrics`: Prometheus text exposition with normalized route/method/status labels, request latency, query/upload/task/empty-context counters, and queue depth.


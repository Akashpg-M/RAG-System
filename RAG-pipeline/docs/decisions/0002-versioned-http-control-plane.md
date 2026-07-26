# ADR 0002: Versioned HTTP query and document control plane

Status: accepted

FastAPI is an outer adapter. Routes validate HTTP data and delegate to query and document-control application services. The existing ingestion and retrieval services remain the only owners of indexing and retrieval behavior.

Uploads return after durable local storage, version/task persistence, and queue acceptance. Ingestion runs on an in-process FIFO adapter for local development. Lifecycle transitions are explicit because four-state queue status was insufficient for operations and safe deletion.

Document IDs are assigned by the control plane and injected into core ingestion. This resolves the prior mismatch between path-derived index IDs and externally addressable document IDs. Streaming, presigned uploads, distributed rate limits, and external workers are deferred until they can share durable contracts without partial production infrastructure.


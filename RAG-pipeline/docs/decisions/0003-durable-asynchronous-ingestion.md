# 0003: Durable asynchronous ingestion

## Status

Accepted for Stage 3.

## Decision

Upload acceptance stores the object and records document version, task, idempotency
claim, and outbox publication intent. SQLite performs task/idempotency/outbox writes in
one immediate transaction. A restart-safe dispatcher publishes by event ID and marks the
outbox row only after the transport accepts it. It also reconstructs missing publication
intents for queued tasks.

Redis Streams is the free local transport. Consumer groups provide explicit pending
state, acknowledgement occurs only after indexing and lifecycle persistence, and
`XAUTOCLAIM` recovers abandoned deliveries. A sorted set schedules delayed retries; a
separate stream is the DLQ. Capacity is admission controlled without trimming pending
entries.

SQS Standard is the AWS transport. It long-polls, deletes only on acknowledgement,
renews visibility during processing, uses receive counts as delivery attempts, and sends
terminal work to the configured DLQ before deleting the source message. Native S3
ObjectCreated envelopes are treated as at-least-once, unordered notifications.

Workers acquire a persistent lease for `document_id:version_id`. Every takeover advances
a fencing token. Renewal, final lifecycle writes, and release require current ownership,
preventing an expired worker from finalizing over its successor. Delivery remains at
least once; deterministic chunk IDs, version-aware replacement, and persisted
idempotency provide effectively-once indexing.

Provider clients remain in infrastructure and composition. The RAG core contains no
Redis, boto3, SQS, or storage-specific implementation logic.

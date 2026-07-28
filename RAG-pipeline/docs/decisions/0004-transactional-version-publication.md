# 0004: Transactional versioned multi-index publication

## Status

Accepted for Stage 4.

## Context

External indexes cannot participate in the same transaction. Deleting an active version
before writing its replacement exposed partial data, and lifecycle `READY` alone could
not prove that dense, sparse, and graph content agreed.

## Decision

PostgreSQL is the shared control plane and authoritative visibility boundary. Alembic
migrations create immutable document versions, normalized lifecycle history,
idempotency and outbox records, persistent leases, chunk manifests, index-stage results,
active-version references, corpus revisions, tombstones, cleanup jobs, and versioned
graph relationships.

Workers create version-qualified chunk IDs and persist the complete provenance manifest
before writing external indexes. Each mandatory index records a successful count and
checksum matching that manifest. Activation takes a document-scoped transaction lock,
validates the current lease and fencing token, validates mandatory stages, advances the
corpus revision, swaps the active-version reference, and schedules retirement of the
previous version. Repeated activation is idempotent.

Queries capture one publication snapshot before retrieval. Candidates are over-fetched,
then filtered against that snapshot, tombstones, document version, and namespace before
reranked context reaches generation. Index presence is therefore insufficient to make a
chunk visible.

Deletion first creates a tombstone and advances the corpus revision. Physical cleanup is
an idempotent job. Rollback may select a retained validated version but cannot select a
staging, failed, removed, or tombstoned version.

Qdrant remains the dense index, SQLite BM25 remains the sparse compatibility index, and
version-associated graph entities and relationships move to PostgreSQL for the local
distributed profile.

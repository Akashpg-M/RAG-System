"""Stage 4 transactional publication control plane."""

from alembic import op


revision = "0001_stage4"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE documents (
      document_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE document_versions (
      document_id TEXT NOT NULL REFERENCES documents(document_id), version_id TEXT NOT NULL,
      source_uri TEXT NOT NULL, source_version TEXT NOT NULL, content_hash TEXT NOT NULL,
      pipeline_version TEXT NOT NULL, parser_version TEXT NOT NULL, chunker_config_version TEXT NOT NULL,
      embedding_model_version TEXT NOT NULL, index_schema_version TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}',
      created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY(document_id,version_id)
    );
    CREATE TABLE ingestion_tasks (
      task_id TEXT PRIMARY KEY, source_uri TEXT NOT NULL, document_id TEXT REFERENCES documents(document_id),
      version_id TEXT, status TEXT NOT NULL, error_code TEXT, created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
      idempotency_key TEXT, fencing_token BIGINT NOT NULL DEFAULT 0
    );
    CREATE TABLE ingestion_task_history (
      sequence BIGSERIAL PRIMARY KEY, task_id TEXT NOT NULL REFERENCES ingestion_tasks(task_id),
      status TEXT NOT NULL, error_code TEXT, occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE idempotency_claims (
      idempotency_key TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT NOT NULL,
      task_id TEXT NOT NULL REFERENCES ingestion_tasks(task_id), created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE outbox_events (
      event_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES ingestion_tasks(task_id), payload JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(), published_at TIMESTAMPTZ,
      publish_attempts INTEGER NOT NULL DEFAULT 0, last_error_code TEXT
    );
    CREATE INDEX outbox_pending_idx ON outbox_events(created_at) WHERE published_at IS NULL;
    CREATE TABLE worker_leases (
      resource_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, ownership_token TEXT NOT NULL,
      fencing_token BIGINT NOT NULL, expires_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE version_publications (
      document_id TEXT NOT NULL, version_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'STAGING',
      expected_chunk_count INTEGER, manifest_checksum TEXT, degraded BOOLEAN NOT NULL DEFAULT false,
      completed_at TIMESTAMPTZ, activated_at TIMESTAMPTZ, retired_at TIMESTAMPTZ, failed_at TIMESTAMPTZ,
      activation_fence BIGINT, PRIMARY KEY(document_id,version_id),
      FOREIGN KEY(document_id,version_id) REFERENCES document_versions(document_id,version_id)
    );
    CREATE TABLE chunk_manifests (
      document_id TEXT NOT NULL, version_id TEXT NOT NULL, chunk_id TEXT NOT NULL, parent_id TEXT NOT NULL,
      content_hash TEXT NOT NULL, ordinal INTEGER NOT NULL, metadata JSONB NOT NULL,
      PRIMARY KEY(document_id,version_id,chunk_id),
      FOREIGN KEY(document_id,version_id) REFERENCES document_versions(document_id,version_id)
    );
    CREATE TABLE index_stage_results (
      document_id TEXT NOT NULL, version_id TEXT NOT NULL, index_name TEXT NOT NULL, outcome TEXT NOT NULL,
      chunk_count INTEGER NOT NULL, checksum TEXT NOT NULL, duration_seconds DOUBLE PRECISION NOT NULL,
      error_code TEXT, completed_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(document_id,version_id,index_name),
      FOREIGN KEY(document_id,version_id) REFERENCES document_versions(document_id,version_id)
    );
    CREATE TABLE corpus_publication (singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK(singleton), revision BIGINT NOT NULL);
    INSERT INTO corpus_publication(singleton,revision) VALUES (true,0);
    CREATE TABLE active_versions (
      document_id TEXT PRIMARY KEY REFERENCES documents(document_id), version_id TEXT NOT NULL,
      revision BIGINT NOT NULL, activated_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE deletion_tombstones (
      document_id TEXT PRIMARY KEY REFERENCES documents(document_id), revision BIGINT NOT NULL,
      deleted_at TIMESTAMPTZ NOT NULL, reason_code TEXT
    );
    CREATE TABLE cleanup_jobs (
      job_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id), version_id TEXT,
      job_type TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error_code TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE(document_id,version_id,job_type)
    );
    CREATE TABLE graph_entities (
      entity_id BIGSERIAL PRIMARY KEY, namespace TEXT NOT NULL, normalized_name TEXT NOT NULL,
      document_id TEXT NOT NULL, version_id TEXT NOT NULL, UNIQUE(namespace,normalized_name,document_id,version_id)
    );
    CREATE TABLE graph_relationships (
      relationship_id BIGSERIAL PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT NOT NULL,
      source_entity_id BIGINT NOT NULL REFERENCES graph_entities(entity_id),
      target_entity_id BIGINT NOT NULL REFERENCES graph_entities(entity_id), relation TEXT NOT NULL, chunk_id TEXT NOT NULL
    );
    """)


def downgrade() -> None:
    op.execute("""
    DROP TABLE graph_relationships,graph_entities,cleanup_jobs,deletion_tombstones,active_versions,
      corpus_publication,index_stage_results,chunk_manifests,version_publications,worker_leases,outbox_events,
      idempotency_claims,ingestion_task_history,ingestion_tasks,document_versions,documents CASCADE;
    """)

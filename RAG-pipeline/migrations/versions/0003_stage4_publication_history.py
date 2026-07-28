"""Persist corpus publication revision history."""

from alembic import op


revision = "0003_stage4"
down_revision = "0002_stage4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE publication_revisions (
      revision BIGINT PRIMARY KEY, action TEXT NOT NULL, document_id TEXT,
      version_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE publication_revisions")

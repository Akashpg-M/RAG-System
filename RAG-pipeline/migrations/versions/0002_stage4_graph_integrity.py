"""Associate graph rows with immutable document versions."""

from alembic import op


revision = "0002_stage4"
down_revision = "0001_stage4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE graph_entities ADD CONSTRAINT graph_entities_version_fk
      FOREIGN KEY(document_id,version_id) REFERENCES document_versions(document_id,version_id) ON DELETE CASCADE;
    ALTER TABLE graph_relationships ADD CONSTRAINT graph_relationships_version_fk
      FOREIGN KEY(document_id,version_id) REFERENCES document_versions(document_id,version_id) ON DELETE CASCADE;
    CREATE UNIQUE INDEX graph_relationship_identity_idx ON graph_relationships(
      document_id,version_id,source_entity_id,target_entity_id,relation,chunk_id
    );
    """)


def downgrade() -> None:
    op.execute("""
    DROP INDEX graph_relationship_identity_idx;
    ALTER TABLE graph_relationships DROP CONSTRAINT graph_relationships_version_fk;
    ALTER TABLE graph_entities DROP CONSTRAINT graph_entities_version_fk;
    """)

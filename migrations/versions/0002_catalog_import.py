"""Durable immutable-source checkpoints for bounded canonical imports.

Revision ID: 0002_catalog_import
Revises: 0001_foundation
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = "0002_catalog_import"
down_revision = "0001_foundation"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("document_revisions", "canonical_text", nullable=True, schema="rag")
    op.add_column("document_revisions", sa.Column("canonical_utf8", sa.LargeBinary()), schema="rag")
    op.drop_constraint("ck_document_revisions_content_hash", "document_revisions", schema="rag", type_="check")
    op.create_check_constraint("ck_document_revisions_one_body", "document_revisions",
        "(canonical_text IS NULL) <> (canonical_utf8 IS NULL)", schema="rag")
    op.create_check_constraint("ck_document_revisions_binary_bound", "document_revisions",
        "canonical_utf8 IS NULL OR octet_length(canonical_utf8) <= 1048576", schema="rag")
    op.create_check_constraint("ck_document_revisions_content_hash", "document_revisions",
        "revision_id = encode(sha256(COALESCE(convert_to(canonical_text, 'UTF8'), canonical_utf8)), 'hex')", schema="rag")
    # PostgreSQL text rejects NUL. Count strict UTF-8 characters on a temporary
    # byte-token representation with NUL replaced solely for validation. The
    # original bytea and its content hash are never changed.
    op.execute(r"""
CREATE FUNCTION rag.canonical_utf8_length(data bytea) RETURNS bigint
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE SET search_path=pg_catalog AS $$
BEGIN
    IF octet_length(data)>1048576 THEN
        RAISE EXCEPTION 'Canonical binary document exceeds 1MiB' USING ERRCODE='23514';
    END IF;
    RETURN char_length(convert_from(decode(replace(replace(
        regexp_replace(encode(data,'hex'), '([0-9a-f]{2})', '\1 ', 'g'),
        '00 ', '01 '), ' ', ''),'hex'),'UTF8'));
END; $$
""")
    op.execute("REVOKE ALL ON FUNCTION rag.canonical_utf8_length(bytea) FROM PUBLIC")
    op.execute("""
CREATE OR REPLACE FUNCTION rag.guard_catalog_insert() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog, rag AS $$
DECLARE snapshot_state text; expected_count bigint; expected_length bigint; actual_length bigint;
BEGIN
    SELECT state, document_count INTO snapshot_state, expected_count
      FROM rag.dataset_snapshots WHERE snapshot_id=NEW.snapshot_id FOR SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Snapshot is unavailable' USING ERRCODE='23503';
    END IF;
    IF snapshot_state<>'importing' THEN
        RAISE EXCEPTION 'Catalog insertion requires an importing snapshot' USING ERRCODE='55000';
    END IF;
    IF TG_TABLE_NAME='documents' THEN
        IF NEW.position>=expected_count THEN
            RAISE EXCEPTION 'Document position exceeds declared snapshot size' USING ERRCODE='23514';
        END IF;
    ELSE
        SELECT text_length INTO expected_length FROM rag.documents WHERE snapshot_id=NEW.snapshot_id
          AND doc_id=NEW.doc_id AND revision_id=NEW.revision_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Canonical catalog revision is unavailable' USING ERRCODE='23503';
        END IF;
        IF (NEW.canonical_text IS NULL) = (NEW.canonical_utf8 IS NULL) THEN
            RAISE EXCEPTION 'Exactly one canonical body is required' USING ERRCODE='23514';
        END IF;
        IF NEW.canonical_utf8 IS NOT NULL THEN
            actual_length:=rag.canonical_utf8_length(NEW.canonical_utf8);
        ELSE
            actual_length:=char_length(NEW.canonical_text);
        END IF;
        IF actual_length<>expected_length THEN
            RAISE EXCEPTION 'Canonical text length differs from its catalog entry' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END; $$
""")
    op.create_table("catalog_imports",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("source_set_sha256", sa.Text(), nullable=False),
        sa.Column("artifact_hashes", pg.JSONB(), nullable=False),
        sa.Column("expected_document_count", sa.BigInteger(), nullable=False),
        sa.Column("expected_passage_count", sa.BigInteger(), nullable=False),
        sa.Column("expected_shards", sa.BigInteger(), nullable=False),
        sa.Column("document_cursor", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("span_shard", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("span_cursor", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("passage_cursor", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("unique_chunk_count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("phase", sa.Text(), nullable=False, server_default=sa.text("'documents'")),
        sa.Column("max_document_bytes", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_batch_bytes_observed", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_catalog_imports"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"], name="fk_catalog_imports_snapshot"),
        sa.CheckConstraint("source_set_sha256 ~ '^[0-9a-f]{64}$'", name="ck_catalog_imports_source_set_sha256_sha256"),
        sa.CheckConstraint("jsonb_typeof(artifact_hashes) = 'object'", name="ck_catalog_imports_artifact_object"),
        sa.CheckConstraint("phase IN ('documents', 'spans', 'complete')", name="ck_catalog_imports_phase"),
        sa.CheckConstraint("expected_shards > 0", name="ck_catalog_imports_shards_positive"),
        sa.CheckConstraint("document_cursor <= expected_document_count AND passage_cursor <= expected_passage_count "
                           "AND unique_chunk_count <= passage_cursor AND span_shard <= expected_shards",
                           name="ck_catalog_imports_cursor_bounds"),
        *(sa.CheckConstraint(f"{column} >= 0", name=f"ck_catalog_imports_{column}_nonnegative") for column in (
            "expected_document_count", "expected_passage_count", "document_cursor", "span_shard", "span_cursor",
            "passage_cursor", "unique_chunk_count", "max_document_bytes", "max_batch_bytes_observed")), schema="rag")
    op.execute("REVOKE ALL ON rag.catalog_imports FROM PUBLIC, rag_runtime")
    op.execute("""
CREATE FUNCTION rag.guard_import_checkpoint() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, rag AS $$
DECLARE expected bigint;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Import checkpoints cannot be deleted' USING ERRCODE='55000';
    END IF;
    IF TG_OP = 'UPDATE' THEN
        IF OLD.phase = 'complete' AND NEW IS DISTINCT FROM OLD THEN
            RAISE EXCEPTION 'Complete import checkpoints are immutable' USING ERRCODE='55000';
        END IF;
        IF ROW(NEW.snapshot_id,NEW.source_set_sha256,NEW.artifact_hashes,NEW.expected_document_count,
               NEW.expected_passage_count,NEW.expected_shards) IS DISTINCT FROM
           ROW(OLD.snapshot_id,OLD.source_set_sha256,OLD.artifact_hashes,OLD.expected_document_count,
               OLD.expected_passage_count,OLD.expected_shards) THEN
            RAISE EXCEPTION 'Import source identity is immutable' USING ERRCODE='55000';
        END IF;
        IF NEW.document_cursor < OLD.document_cursor OR NEW.passage_cursor < OLD.passage_cursor
           OR NEW.unique_chunk_count < OLD.unique_chunk_count
           OR ROW(NEW.span_shard,NEW.span_cursor) < ROW(OLD.span_shard,OLD.span_cursor)
           OR array_position(ARRAY['documents','spans','complete'],NEW.phase)
              < array_position(ARRAY['documents','spans','complete'],OLD.phase) THEN
            RAISE EXCEPTION 'Import checkpoints cannot move backward' USING ERRCODE='55000';
        END IF;
    END IF;
    SELECT document_count INTO expected FROM rag.dataset_snapshots
      WHERE snapshot_id=NEW.snapshot_id FOR SHARE;
    IF expected IS DISTINCT FROM NEW.expected_document_count THEN
        RAISE EXCEPTION 'Import checkpoint differs from snapshot size' USING ERRCODE='23514';
    END IF;
    IF NEW.phase='documents' AND (NEW.span_shard<>0 OR NEW.span_cursor<>0 OR NEW.passage_cursor<>0) THEN
        RAISE EXCEPTION 'Document import cannot advance passage cursors' USING ERRCODE='23514';
    END IF;
    IF NEW.phase IN ('spans','complete') AND NEW.document_cursor<>expected THEN
        RAISE EXCEPTION 'Passage import requires the full catalogue' USING ERRCODE='23514';
    END IF;
    IF NEW.phase='complete' AND (NEW.passage_cursor<>NEW.expected_passage_count
       OR NEW.span_shard<>NEW.expected_shards OR NEW.span_cursor<>0) THEN
        RAISE EXCEPTION 'Complete import requires every passage row' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$
""")
    op.execute("CREATE TRIGGER trg_catalog_import_checkpoint BEFORE INSERT OR UPDATE OR DELETE "
               "ON rag.catalog_imports FOR EACH ROW EXECUTE FUNCTION rag.guard_import_checkpoint()")
    op.execute("""
CREATE FUNCTION rag.guard_import_readiness() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, rag AS $$
DECLARE checkpoint rag.catalog_imports%ROWTYPE; actual_chunks bigint;
BEGIN
    IF NEW.state='ready' THEN
        SELECT * INTO checkpoint FROM rag.catalog_imports WHERE snapshot_id=NEW.snapshot_id;
        IF FOUND THEN
            SELECT count(*) INTO actual_chunks FROM rag.chunks WHERE snapshot_id=NEW.snapshot_id;
            IF checkpoint.phase<>'complete' OR checkpoint.document_cursor<>NEW.document_count
               OR checkpoint.passage_cursor<>checkpoint.expected_passage_count
               OR actual_chunks<>checkpoint.unique_chunk_count THEN
                RAISE EXCEPTION 'Snapshot import is incomplete' USING ERRCODE='23514';
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END; $$
""")
    op.execute("CREATE TRIGGER trg_snapshot_import_readiness BEFORE INSERT OR UPDATE ON rag.dataset_snapshots "
               "FOR EACH ROW EXECUTE FUNCTION rag.guard_import_readiness()")


def downgrade():
    raise RuntimeError("Canonical imports require explicit forward recovery")

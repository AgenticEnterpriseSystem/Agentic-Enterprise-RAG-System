"""Immutable verified Jira records and independent feed/field coverage.

Revision ID: 0004_jira_feed
Revises: 0003_document_policy
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = "0004_jira_feed"
down_revision = "0003_document_policy"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("ticket_feeds",
        sa.Column("feed_id", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("coverage", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default=sa.text("'importing'")),
        sa.Column("repository", sa.Text(), nullable=False),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("corpus_sha256", sa.Text(), nullable=False),
        sa.Column("source_set_sha256", sa.Text(), nullable=False),
        sa.Column("expected_records", sa.BigInteger(), nullable=False),
        sa.Column("imported_records", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("field_coverage", pg.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_manifest", pg.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("feed_id", name="pk_ticket_feeds"),
        sa.UniqueConstraint("snapshot_id", "source_type", name="uq_ticket_feeds_snapshot_source"),
        sa.UniqueConstraint("feed_id", "snapshot_id", name="uq_ticket_feeds_binding"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"], name="fk_ticket_feeds_snapshot"),
        *(sa.CheckConstraint(f"{name} ~ '^[0-9a-f]{{64}}$'", name=f"ck_ticket_feeds_{name}_sha256")
          for name in ("feed_id", "corpus_sha256", "source_set_sha256")),
        sa.CheckConstraint("source_type = 'jira'", name="ck_ticket_feeds_source"),
        sa.CheckConstraint("coverage IN ('verified_records', 'synthetic_fixture')", name="ck_ticket_feeds_coverage"),
        sa.CheckConstraint("state IN ('importing', 'ready')", name="ck_ticket_feeds_state"),
        sa.CheckConstraint("git_commit ~ '^[0-9a-f]{40}$'", name="ck_ticket_feeds_commit"),
        sa.CheckConstraint("jsonb_typeof(field_coverage) = 'object' AND jsonb_typeof(source_manifest) = 'object'", name="ck_ticket_feeds_objects"),
        sa.CheckConstraint("imported_records <= expected_records", name="ck_ticket_feeds_counts"),
        *(sa.CheckConstraint(f"{name} >= 0", name=f"ck_ticket_feeds_{name}_nonnegative") for name in ("expected_records", "imported_records")), schema="rag")
    op.add_column("tickets", sa.Column("feed_id", sa.Text()), schema="rag")
    op.add_column("tickets", sa.Column("native_key", sa.Text()), schema="rag")
    op.create_foreign_key("fk_tickets_feed", "tickets", "ticket_feeds", ["feed_id", "snapshot_id"], ["feed_id", "snapshot_id"], source_schema="rag", referent_schema="rag")
    op.create_check_constraint("ck_tickets_verified_feed", "tickets", "coverage <> 'verified_records' OR feed_id IS NOT NULL", schema="rag")
    op.create_check_constraint("ck_tickets_canonical_record_id", "tickets", "feed_id IS NULL OR ticket_id = doc_id", schema="rag")
    op.create_index("ix_tickets_native_key", "tickets", ["snapshot_id", "native_key", "doc_id"], schema="rag")
    op.create_index("uq_tickets_feed_document", "tickets", ["feed_id", "doc_id"], unique=True, schema="rag", postgresql_where=sa.text("feed_id IS NOT NULL"))
    op.execute("REVOKE ALL ON rag.ticket_feeds FROM PUBLIC")
    # Runtime needs an opaque ready-feed header, never the global path/UUID
    # inventory or field-coverage totals. Authorized per-record provenance lives
    # in tickets and remains protected by the existing document RLS.
    op.execute("GRANT SELECT (feed_id,snapshot_id,source_type,coverage,state,repository,git_commit,corpus_sha256,source_set_sha256,expected_records,imported_records) ON rag.ticket_feeds TO rag_runtime")
    op.execute("""
CREATE FUNCTION rag.guard_ticket_feed() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
DECLARE expected bigint; actual bigint; field text; verified bigint; missing bigint; invalid bigint;
BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'Ticket feed deletion is unsupported' USING ERRCODE='55000'; END IF;
 IF TG_OP='UPDATE' THEN
  IF OLD.state='ready' AND NEW IS DISTINCT FROM OLD THEN
   RAISE EXCEPTION 'Ready ticket feeds are immutable' USING ERRCODE='55000';
  END IF;
  IF ROW(NEW.feed_id,NEW.snapshot_id,NEW.source_type,NEW.coverage,NEW.repository,NEW.git_commit,NEW.corpus_sha256,
     NEW.source_set_sha256,NEW.expected_records,NEW.field_coverage,NEW.source_manifest,NEW.created_at)
   IS DISTINCT FROM ROW(OLD.feed_id,OLD.snapshot_id,OLD.source_type,OLD.coverage,OLD.repository,OLD.git_commit,OLD.corpus_sha256,
     OLD.source_set_sha256,OLD.expected_records,OLD.field_coverage,OLD.source_manifest,OLD.created_at)
   OR NEW.imported_records<OLD.imported_records THEN
   RAISE EXCEPTION 'Ticket feed identity is immutable' USING ERRCODE='55000';
  END IF;
 END IF;
 IF NOT EXISTS(SELECT 1 FROM rag.dataset_snapshots s WHERE s.snapshot_id=NEW.snapshot_id
   AND s.state='ready' AND s.corpus_sha256=NEW.corpus_sha256) THEN
  RAISE EXCEPTION 'Ticket feed requires its ready canonical snapshot' USING ERRCODE='23514';
 END IF;
 IF NEW.state='ready' THEN
  SELECT count(*) INTO expected FROM rag.documents WHERE snapshot_id=NEW.snapshot_id AND source_type='jira';
  SELECT count(*) INTO actual FROM rag.tickets WHERE feed_id=NEW.feed_id;
  IF expected<>NEW.expected_records OR actual<>expected OR NEW.imported_records<>expected THEN
   RAISE EXCEPTION 'Verified Jira feed is incomplete' USING ERRCODE='23514';
  END IF;
  FOREACH field IN ARRAY ARRAY['native_key','status','priority','project_id'] LOOP
   verified:=(NEW.field_coverage->field->>'verified')::bigint;
   missing:=(NEW.field_coverage->field->>'missing')::bigint;
   invalid:=(NEW.field_coverage->field->>'invalid')::bigint;
   IF verified IS NULL OR missing IS NULL OR invalid IS NULL OR least(verified,missing,invalid)<0
    OR verified+missing+invalid<>expected THEN
    RAISE EXCEPTION 'Verified field coverage is incomplete' USING ERRCODE='23514';
   END IF;
   EXECUTE format('SELECT count(%I) FROM rag.tickets WHERE feed_id=$1',field) INTO actual USING NEW.feed_id;
   IF actual<>verified THEN RAISE EXCEPTION 'Verified field coverage differs from records' USING ERRCODE='23514'; END IF;
  END LOOP;
 END IF;
 RETURN NEW;
END; $$
""")
    op.execute("CREATE TRIGGER trg_ticket_feed_guard BEFORE INSERT OR UPDATE OR DELETE ON rag.ticket_feeds FOR EACH ROW EXECUTE FUNCTION rag.guard_ticket_feed()")
    op.execute("""
CREATE FUNCTION rag.guard_verified_ticket() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
DECLARE feed_state text; feed_coverage text;
BEGIN
 IF TG_OP<>'INSERT' THEN
  IF OLD.feed_id IS NOT NULL THEN RAISE EXCEPTION 'Verified records are immutable' USING ERRCODE='55000'; END IF;
  IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 END IF;
 IF NEW.feed_id IS NOT NULL THEN
  SELECT state,coverage INTO feed_state,feed_coverage FROM rag.ticket_feeds WHERE feed_id=NEW.feed_id AND snapshot_id=NEW.snapshot_id FOR SHARE;
  IF feed_state IS DISTINCT FROM 'importing' OR NEW.coverage IS DISTINCT FROM feed_coverage OR NEW.ticket_id<>NEW.doc_id
   OR NOT EXISTS(SELECT 1 FROM rag.documents d WHERE d.snapshot_id=NEW.snapshot_id AND d.doc_id=NEW.doc_id AND d.source_type='jira') THEN
   RAISE EXCEPTION 'Verified record requires its importing canonical Jira feed' USING ERRCODE='23514';
  END IF;
 END IF;
 RETURN NEW;
END; $$
""")
    op.execute("CREATE TRIGGER trg_verified_ticket_guard BEFORE INSERT OR UPDATE OR DELETE ON rag.tickets FOR EACH ROW EXECUTE FUNCTION rag.guard_verified_ticket()")
    for table in ("ticket_feeds", "tickets"):
        op.execute(f"CREATE TRIGGER trg_{table}_no_truncate BEFORE TRUNCATE ON rag.{table} FOR EACH STATEMENT EXECUTE FUNCTION rag.deny_policy_truncate()")


def downgrade():
    raise RuntimeError("Verified ticket feeds require explicit forward recovery")

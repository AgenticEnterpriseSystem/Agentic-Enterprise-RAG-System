"""Reviewed grants, immutable compact scopes and invoker RLS.

Revision ID:0003_document_policy
Revises:0002_catalog_import
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = "0003_document_policy"
down_revision = "0002_catalog_import"
branch_labels = None
depends_on = None


def _sha(table, name):
    return sa.CheckConstraint(f"{name} ~ '^[0-9a-f]{{64}}$'", name=f"ck_{table}_{name}_sha256")


def upgrade():
    op.add_column("users", sa.Column("account_kind", sa.Text(), nullable=False, server_default=sa.text("'human'")), schema="rag")
    op.create_check_constraint("ck_users_account_kind", "users", "account_kind IN ('human', 'benchmark')", schema="rag")
    op.create_table("policy_changes",
        sa.Column("change_id", sa.Text(), nullable=False), sa.Column("manifest_id", sa.Text(), nullable=False),
        sa.Column("manifest_sha256", sa.Text(), nullable=False), sa.Column("action", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False), sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("reviewed_by", sa.Text(), nullable=False), sa.Column("manifest", pg.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("change_id", name="pk_policy_changes"),
        sa.UniqueConstraint("manifest_id", "action", name="uq_policy_changes_manifest_action"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"], name="fk_policy_changes_snapshot"),
        _sha("policy_changes", "change_id"), _sha("policy_changes", "manifest_sha256"),
        sa.CheckConstraint("action IN ('apply', 'revoke', 'benchmark')", name="ck_policy_changes_action"),
        sa.CheckConstraint("origin IN ('operator', 'synthetic')", name="ck_policy_changes_origin"),
        sa.CheckConstraint("jsonb_typeof(manifest) = 'object'", name="ck_policy_changes_manifest_object"),
        sa.CheckConstraint("manifest_id ~ '[^[:space:]]'", name="ck_policy_changes_manifest_id_nonblank"),
        sa.CheckConstraint("reviewed_by ~ '[^[:space:]]'", name="ck_policy_changes_reviewed_by_nonblank"), schema="rag")
    for table, constraint in (("document_grants", "fk_grants_policy_change"), ("group_memberships", "fk_memberships_policy_change")):
        op.add_column(table, sa.Column("policy_change_id", sa.Text()), schema="rag")
        op.create_foreign_key(constraint, table, "policy_changes", ["policy_change_id"], ["change_id"], source_schema="rag", referent_schema="rag")
    op.create_table("group_requirements",
        sa.Column("group_id", sa.Text(), nullable=False), sa.Column("required_group_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("group_id", "required_group_id", name="pk_group_requirements"),
        sa.ForeignKeyConstraint(["group_id"], ["rag.groups.group_id"], name="fk_requirements_group"),
        sa.ForeignKeyConstraint(["required_group_id"], ["rag.groups.group_id"], name="fk_requirements_required_group"),
        sa.CheckConstraint("group_id <> required_group_id", name="ck_requirements_not_self"), schema="rag")
    op.create_table("policy_scopes",
        sa.Column("scope_id", sa.Text(), nullable=False), sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False), sa.Column("session_version", sa.BigInteger(), nullable=False),
        sa.Column("policy_revision", sa.BigInteger(), nullable=False), sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("corpus_sha256", sa.Text(), nullable=False), sa.Column("catalog_sha256", sa.Text(), nullable=False),
        sa.Column("document_count", sa.BigInteger(), nullable=False), sa.Column("allowed_positions", sa.LargeBinary(), nullable=False),
        sa.Column("bitmap_sha256", sa.Text(), nullable=False), sa.Column("restriction_sha256", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("scope_id", name="pk_policy_scopes"),
        sa.ForeignKeyConstraint(["user_id"], ["rag.users.user_id"], name="fk_policy_scopes_user"),
        sa.ForeignKeyConstraint(["session_id"], ["rag.sessions.session_id"], name="fk_policy_scopes_session"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"], name="fk_policy_scopes_snapshot"),
        sa.CheckConstraint("scope_id ~ '^scope:[0-9a-f]{64}$'", name="ck_policy_scopes_scope_id"),
        sa.CheckConstraint("octet_length(allowed_positions) = (document_count + 7) / 8", name="ck_policy_scopes_bitmap_size"),
        sa.CheckConstraint("bitmap_sha256 = encode(sha256(allowed_positions), 'hex')", name="ck_policy_scopes_bitmap_hash"),
        sa.CheckConstraint("CASE WHEN document_count % 8 = 0 THEN true ELSE "
                           "get_byte(allowed_positions, (document_count / 8)::integer) >> (document_count % 8)::integer = 0 END",
                           name="ck_policy_scopes_padding"),
        *(_sha("policy_scopes", name) for name in ("corpus_sha256", "catalog_sha256", "bitmap_sha256", "restriction_sha256")),
        *(sa.CheckConstraint(f"{name} >= 0", name=f"ck_policy_scopes_{name}_nonnegative")
          for name in ("session_version", "policy_revision", "document_count")), schema="rag")
    op.execute("REVOKE ALL ON rag.policy_changes, rag.policy_scopes, rag.group_requirements FROM PUBLIC")
    op.execute("GRANT SELECT ON rag.group_requirements, rag.policy_scopes TO rag_runtime")
    op.execute("GRANT INSERT ON rag.policy_scopes TO rag_runtime")
    op.execute("""
CREATE FUNCTION rag.guard_account_kind() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN
 IF NEW.account_kind IS DISTINCT FROM OLD.account_kind THEN
  RAISE EXCEPTION 'Account kind is immutable' USING ERRCODE='55000';
 END IF;
 RETURN NEW;
END; $$
""")
    op.execute("CREATE TRIGGER trg_users_account_kind BEFORE UPDATE ON rag.users FOR EACH ROW EXECUTE FUNCTION rag.guard_account_kind()")
    op.execute("""
CREATE FUNCTION rag.guard_flat_group_requirements() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN
 PERFORM pg_advisory_xact_lock(59800004);
 IF NEW.group_id=NEW.required_group_id OR
    EXISTS(SELECT1 FROM rag.group_requirements WHERE group_id=NEW.required_group_id) OR
    EXISTS(SELECT1 FROM rag.group_requirements WHERE required_group_id=NEW.group_id) THEN
  RAISE EXCEPTION 'Compound groups require flat distinct base groups' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END; $$
""".replace("SELECT1", "SELECT 1"))
    op.execute("CREATE TRIGGER trg_group_requirements_flat BEFORE INSERT OR UPDATE ON rag.group_requirements FOR EACH ROW EXECUTE FUNCTION rag.guard_flat_group_requirements()")
    op.execute("""
CREATE FUNCTION rag.bump_policy_epoch() RETURNS void LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN
 UPDATE rag.policy_state SET revision=revision+1 WHERE singleton_id=1;
 IF NOT FOUND THEN RAISE EXCEPTION 'Policy revision is unavailable' USING ERRCODE='55000'; END IF;
END; $$
""")
    op.execute("""
CREATE FUNCTION rag.policy_insert_epoch() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN IF EXISTS(SELECT1 FROM new_rows) THEN PERFORM rag.bump_policy_epoch(); END IF; RETURN NULL; END; $$
""".replace("SELECT1", "SELECT 1"))
    op.execute("""
CREATE FUNCTION rag.policy_delete_epoch() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN IF EXISTS(SELECT1 FROM old_rows) THEN PERFORM rag.bump_policy_epoch(); END IF; RETURN NULL; END; $$
""".replace("SELECT1", "SELECT 1"))
    op.execute("""
CREATE FUNCTION rag.policy_update_epoch() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN
 IF EXISTS(SELECT * FROM old_rows EXCEPT SELECT * FROM new_rows)
    OR EXISTS(SELECT * FROM new_rows EXCEPT SELECT * FROM old_rows) THEN PERFORM rag.bump_policy_epoch(); END IF;
 RETURN NULL;
END; $$
""")
    op.execute("""
CREATE FUNCTION rag.deny_policy_truncate() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN RAISE EXCEPTION 'Use reviewed policy row mutations' USING ERRCODE='55000'; END; $$
""")
    for table in ("document_grants", "group_memberships", "group_requirements"):
        op.execute(f"CREATE TRIGGER trg_{table}_insert_epoch AFTER INSERT ON rag.{table} REFERENCING NEW TABLE AS new_rows FOR EACH STATEMENT EXECUTE FUNCTION rag.policy_insert_epoch()")
        op.execute(f"CREATE TRIGGER trg_{table}_delete_epoch AFTER DELETE ON rag.{table} REFERENCING OLD TABLE AS old_rows FOR EACH STATEMENT EXECUTE FUNCTION rag.policy_delete_epoch()")
        op.execute(f"CREATE TRIGGER trg_{table}_update_epoch AFTER UPDATE ON rag.{table} REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows FOR EACH STATEMENT EXECUTE FUNCTION rag.policy_update_epoch()")
        op.execute(f"CREATE TRIGGER trg_{table}_no_truncate BEFORE TRUNCATE ON rag.{table} FOR EACH STATEMENT EXECUTE FUNCTION rag.deny_policy_truncate()")
    for table in ("policy_changes", "policy_scopes"):
        op.execute(f"CREATE TRIGGER trg_{table}_immutable BEFORE UPDATE OR DELETE ON rag.{table} FOR EACH ROW EXECUTE FUNCTION rag.reject_immutable_mutation()")
        op.execute(f"CREATE TRIGGER trg_{table}_no_truncate BEFORE TRUNCATE ON rag.{table} FOR EACH STATEMENT EXECUTE FUNCTION rag.deny_policy_truncate()")
    op.execute("""
CREATE FUNCTION rag.guard_scope_registration() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,rag AS $$
BEGIN
 IF NEW.user_id IS DISTINCT FROM nullif(current_setting('rag.user_id',true),'')
  OR NEW.session_id IS DISTINCT FROM nullif(current_setting('rag.session_id',true),'')
  OR NEW.session_version::text IS DISTINCT FROM nullif(current_setting('rag.session_version',true),'')
  OR NEW.policy_revision::text IS DISTINCT FROM nullif(current_setting('rag.policy_revision',true),'')
  OR NEW.snapshot_id IS DISTINCT FROM nullif(current_setting('rag.snapshot_id',true),'')
  OR NEW.scope_id IS DISTINCT FROM nullif(current_setting('rag.scope_id',true),'')
  OR NOT EXISTS(SELECT1 FROM rag.sessions s JOIN rag.users u ON u.user_id=s.user_id
   JOIN rag.policy_state p ON p.singleton_id=1 JOIN rag.dataset_snapshots d ON d.snapshot_id=NEW.snapshot_id
   WHERE s.session_id=NEW.session_id AND s.user_id=NEW.user_id AND s.revoked_at IS NULL
    AND s.created_at<=statement_timestamp() AND s.expires_at>statement_timestamp() AND u.is_active
    AND s.session_version=NEW.session_version AND u.session_version=NEW.session_version
    AND p.revision=NEW.policy_revision AND d.state='ready' AND d.corpus_sha256=NEW.corpus_sha256
    AND d.catalog_sha256=NEW.catalog_sha256 AND d.document_count=NEW.document_count) THEN
  RAISE EXCEPTION 'Scope registration requires a current bound session' USING ERRCODE='42501';
 END IF;
 RETURN NEW;
END; $$
""".replace("SELECT1", "SELECT 1"))
    op.execute("CREATE TRIGGER trg_policy_scopes_registration BEFORE INSERT ON rag.policy_scopes FOR EACH ROW EXECUTE FUNCTION rag.guard_scope_registration()")
    op.execute("""
CREATE FUNCTION rag.valid_policy_context(target_snapshot text) RETURNS boolean
LANGUAGE sql STABLE SECURITY INVOKER SET search_path=pg_catalog,rag AS $$
 SELECT EXISTS(SELECT1 FROM rag.sessions s JOIN rag.users u ON u.user_id=s.user_id
 JOIN rag.policy_state p ON p.singleton_id=1
 JOIN rag.dataset_snapshots d ON d.snapshot_id=target_snapshot
 JOIN rag.policy_scopes c ON c.scope_id=nullif(current_setting('rag.scope_id',true),'')
 WHERE s.session_id=nullif(current_setting('rag.session_id',true),'')
 AND s.user_id=nullif(current_setting('rag.user_id',true),'') AND u.is_active AND s.revoked_at IS NULL
 AND s.created_at<=statement_timestamp() AND s.expires_at>statement_timestamp()
 AND s.session_version=u.session_version AND s.session_version::text=nullif(current_setting('rag.session_version',true),'')
 AND p.revision::text=nullif(current_setting('rag.policy_revision',true),'')
 AND d.state='ready' AND d.snapshot_id=nullif(current_setting('rag.snapshot_id',true),'')
 AND c.user_id=s.user_id AND c.session_id=s.session_id AND c.session_version=s.session_version
 AND c.policy_revision=p.revision AND c.snapshot_id=d.snapshot_id
 AND c.corpus_sha256=d.corpus_sha256 AND c.catalog_sha256=d.catalog_sha256 AND c.document_count=d.document_count)
$$
""".replace("SELECT1", "SELECT 1"))
    op.execute("""
CREATE FUNCTION rag.document_permitted(target_snapshot text,target_doc text) RETURNS boolean
LANGUAGE sql STABLE SECURITY INVOKER SET search_path=pg_catalog,rag AS $$
 SELECT rag.valid_policy_context(target_snapshot) AND EXISTS(
 SELECT1 FROM rag.documents d JOIN rag.policy_scopes c ON c.scope_id=nullif(current_setting('rag.scope_id',true),'')
 WHERE d.snapshot_id=target_snapshot AND d.doc_id=target_doc AND c.snapshot_id=d.snapshot_id
 AND CASE WHEN d.position>=0 AND d.position<c.document_count THEN get_bit(c.allowed_positions,d.position::integer)=1 ELSE false END
 AND EXISTS(SELECT1 FROM rag.document_grants g WHERE g.snapshot_id=d.snapshot_id AND g.doc_id=d.doc_id
  AND (g.user_id=nullif(current_setting('rag.user_id',true),'') OR (g.group_id IS NOT NULL
   AND EXISTS(SELECT1 FROM rag.group_memberships m WHERE m.user_id=nullif(current_setting('rag.user_id',true),'') AND m.group_id=g.group_id)
   AND NOT EXISTS(SELECT1 FROM rag.group_requirements r WHERE r.group_id=g.group_id
    AND NOT EXISTS(SELECT1 FROM rag.group_memberships m WHERE m.user_id=nullif(current_setting('rag.user_id',true),'') AND m.group_id=r.required_group_id)))))
 )
$$
""".replace("SELECT1", "SELECT 1"))
    op.execute("""
CREATE FUNCTION rag.owned_run_context(target_snapshot text,target_owner text,target_scope text,target_policy bigint,target_session bigint)
RETURNS boolean LANGUAGE sql STABLE SECURITY INVOKER SET search_path=pg_catalog,rag AS $$
 SELECT rag.valid_policy_context(target_snapshot)
 AND target_owner=nullif(current_setting('rag.user_id',true),'')
 AND target_scope=nullif(current_setting('rag.scope_id',true),'')
 AND target_policy::text=nullif(current_setting('rag.policy_revision',true),'')
 AND target_session::text=nullif(current_setting('rag.session_version',true),'')
$$
""")
    for name, signature in (("valid_policy_context", "text"), ("document_permitted", "text,text"),
                            ("owned_run_context", "text,text,text,bigint,bigint")):
        op.execute(f"REVOKE ALL ON FUNCTION rag.{name}({signature}) FROM PUBLIC")
        op.execute(f"GRANT EXECUTE ON FUNCTION rag.{name}({signature}) TO rag_runtime")
    for table in ("document_revisions", "chunks", "tickets"):
        op.execute(f"CREATE POLICY {table}_scoped_read ON rag.{table} FOR SELECT TO rag_runtime USING (rag.document_permitted(snapshot_id,doc_id))")
    run_check = "rag.owned_run_context(snapshot_id,owner_user_id,scope_id,policy_revision,session_version)"
    op.execute(f"CREATE POLICY query_runs_owned ON rag.query_runs FOR ALL TO rag_runtime USING ({run_check}) WITH CHECK ({run_check})")
    step_check = "EXISTS(SELECT 1 FROM rag.query_runs r WHERE r.run_id=run_steps.run_id)"
    op.execute(f"CREATE POLICY run_steps_owned ON rag.run_steps FOR ALL TO rag_runtime USING ({step_check}) WITH CHECK ({step_check})")
    aggregate_check = run_check + " AND EXISTS(SELECT 1 FROM rag.query_runs r WHERE r.run_id=aggregate_results.run_id)"
    op.execute(f"CREATE POLICY aggregate_results_owned ON rag.aggregate_results FOR ALL TO rag_runtime USING ({aggregate_check}) WITH CHECK ({aggregate_check})")


def downgrade():
    raise RuntimeError("Document permissions require explicit forward recovery")

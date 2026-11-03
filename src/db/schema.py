"""Relational foundation owned by the platform lane.

Canonical catalog rows and text are append-only during import and closed when
their snapshot becomes ready. Chunks are derived spans outside the catalog
fingerprint: new bounded spans may be appended against an existing immutable
revision after readiness, while UPDATE/DELETE remain forbidden. The migration
requires a complete bounded catalog and every canonical body before readiness.
Policy/session counters cannot decrease, and a run's identity/provenance cannot
change when recording its mutable outcomes. The migration installs these rules
as database triggers; metadata alone is not enforcement. AUTH/ACL still decide
when to increment versions and how to authorize each operation.

Metadata catalogs contain no document title or body. Protected tables begin with
RLS enabled and no policies; the separate ACL ticket supplies runtime policies.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

SCHEMA = "rag"
PROTECTED_TABLES = (
    "document_revisions", "chunks", "tickets", "query_runs", "run_steps",
    "aggregate_results",
)


def _nonblank(table: str, *columns: str) -> list[sa.CheckConstraint]:
    return [sa.CheckConstraint(f"{column} ~ '[^[:space:]]'",
                              name=f"ck_{table}_{column}_nonblank") for column in columns]


def _sha(table: str, column: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(f"{column} ~ '^[0-9a-f]{{64}}$'",
                              name=f"ck_{table}_{column}_sha256")


def _nonnegative(table: str, *columns: str) -> list[sa.CheckConstraint]:
    return [sa.CheckConstraint(f"{column} >= 0", name=f"ck_{table}_{column}_nonnegative")
            for column in columns]


def _build_metadata() -> sa.MetaData:
    catalog = sa.MetaData(schema=SCHEMA)
    sa.Table("users", catalog,
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("login", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("session_version", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("role", sa.Text()),
        sa.Column("department", sa.Text()),
        sa.Column("project_ids", pg.ARRAY(sa.Text()), nullable=False,
                  server_default=sa.text("'{}'::text[]")),
        sa.PrimaryKeyConstraint("user_id", name="pk_users"),
        sa.UniqueConstraint("login", name="uq_users_login"),
        sa.CheckConstraint("login = lower(btrim(login)) AND login !~ '^[[:space:]]|[[:space:]]$'",
                           name="ck_users_login_normalized"),
        sa.CheckConstraint("array_position(project_ids, NULL) IS NULL",
                           name="ck_users_projects_nonnull"),
        *_nonblank("users", "user_id", "login", "password_hash", "display_name"),
        *_nonnegative("users", "session_version"),
    )
    sessions = sa.Table("sessions", catalog,
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("csrf_hash", sa.Text(), nullable=False),
        sa.Column("session_version", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("session_id", name="pk_sessions"),
        sa.ForeignKeyConstraint(["user_id"], ["rag.users.user_id"], name="fk_sessions_user"),
        sa.UniqueConstraint("token_hash", name="uq_sessions_token_hash"),
        sa.CheckConstraint("expires_at > created_at", name="ck_sessions_expiry_after_creation"),
        sa.CheckConstraint("revoked_at IS NULL OR revoked_at >= created_at",
                           name="ck_sessions_revocation_after_creation"),
        _sha("sessions", "token_hash"), _sha("sessions", "csrf_hash"),
        *_nonblank("sessions", "session_id"), *_nonnegative("sessions", "session_version"),
    )
    sa.Index("ix_sessions_user", sessions.c.user_id)
    sa.Index("ix_sessions_expiry", sessions.c.expires_at)

    sa.Table("groups", catalog,
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("group_id", name="pk_groups"),
        *_nonblank("groups", "group_id", "label"),
    )
    memberships = sa.Table("group_memberships", catalog,
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False, server_default=sa.text("'operator'")),
        sa.PrimaryKeyConstraint("user_id", "group_id", name="pk_group_memberships"),
        sa.ForeignKeyConstraint(["user_id"], ["rag.users.user_id"], name="fk_memberships_user"),
        sa.ForeignKeyConstraint(["group_id"], ["rag.groups.group_id"], name="fk_memberships_group"),
        sa.CheckConstraint("origin IN ('operator', 'synthetic')", name="ck_memberships_origin"),
    )
    sa.Index("ix_memberships_group_user", memberships.c.group_id, memberships.c.user_id)

    sa.Table("dataset_snapshots", catalog,
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("corpus_sha256", sa.Text(), nullable=False),
        sa.Column("catalog_sha256", sa.Text()),
        sa.Column("document_count", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default=sa.text("'importing'")),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_dataset_snapshots"),
        _sha("dataset_snapshots", "corpus_sha256"), _sha("dataset_snapshots", "catalog_sha256"),
        sa.CheckConstraint("state IN ('importing', 'ready', 'failed')", name="ck_snapshots_state"),
        sa.CheckConstraint("state <> 'ready' OR catalog_sha256 IS NOT NULL",
                           name="ck_snapshots_ready_catalog_hash"),
        *_nonblank("dataset_snapshots", "snapshot_id"),
        *_nonnegative("dataset_snapshots", "document_count"),
    )
    sa.Table("catalog_imports", catalog,
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
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_catalog_imports"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"],
                                name="fk_catalog_imports_snapshot"),
        _sha("catalog_imports", "source_set_sha256"),
        sa.CheckConstraint("jsonb_typeof(artifact_hashes) = 'object'", name="ck_catalog_imports_artifact_object"),
        sa.CheckConstraint("phase IN ('documents', 'spans', 'complete')", name="ck_catalog_imports_phase"),
        sa.CheckConstraint("expected_shards > 0", name="ck_catalog_imports_shards_positive"),
        sa.CheckConstraint("document_cursor <= expected_document_count AND passage_cursor <= expected_passage_count "
                           "AND unique_chunk_count <= passage_cursor AND span_shard <= expected_shards",
                           name="ck_catalog_imports_cursor_bounds"),
        *_nonnegative("catalog_imports", "expected_document_count", "expected_passage_count",
                      "document_cursor", "span_shard", "span_cursor", "passage_cursor",
                      "unique_chunk_count", "max_document_bytes", "max_batch_bytes_observed"),
    )
    documents = sa.Table("documents", catalog,
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("position", sa.BigInteger(), nullable=False),
        sa.Column("revision_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("text_length", sa.BigInteger(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True)),
        sa.Column("metadata_origin", sa.Text(), nullable=False, server_default=sa.text("'upstream'")),
        sa.PrimaryKeyConstraint("snapshot_id", "doc_id", name="pk_documents"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"],
                                name="fk_documents_snapshot"),
        sa.UniqueConstraint("snapshot_id", "position", name="uq_documents_position"),
        sa.UniqueConstraint("snapshot_id", "doc_id", "revision_id", name="uq_documents_revision"),
        sa.CheckConstraint("metadata_origin IN ('upstream', 'extracted', 'synthetic')",
                           name="ck_documents_metadata_origin"),
        sa.CheckConstraint("char_length(source_type) <= 64", name="ck_documents_source_length"),
        _sha("documents", "revision_id"),
        *_nonblank("documents", "doc_id", "source_type"),
        *_nonnegative("documents", "position", "text_length"),
    )
    sa.Index("ix_documents_source_timestamp", documents.c.snapshot_id,
             documents.c.source_type, documents.c.timestamp)

    sa.Table("document_revisions", catalog,
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("revision_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("canonical_text", sa.Text()),
        sa.Column("canonical_utf8", sa.LargeBinary()),
        sa.PrimaryKeyConstraint("snapshot_id", "doc_id", "revision_id", name="pk_document_revisions"),
        sa.ForeignKeyConstraint(["snapshot_id", "doc_id", "revision_id"],
            ["rag.documents.snapshot_id", "rag.documents.doc_id", "rag.documents.revision_id"],
            name="fk_document_revisions_document"),
        sa.CheckConstraint("(canonical_text IS NULL) <> (canonical_utf8 IS NULL)",
                           name="ck_document_revisions_one_body"),
        sa.CheckConstraint("canonical_utf8 IS NULL OR octet_length(canonical_utf8) <= 1048576",
                           name="ck_document_revisions_binary_bound"),
        sa.CheckConstraint("revision_id = encode(sha256(COALESCE(convert_to(canonical_text, 'UTF8'), canonical_utf8)), 'hex')",
                           name="ck_document_revisions_content_hash"),
    )
    chunks = sa.Table("chunks", catalog,
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("revision_id", sa.Text(), nullable=False),
        sa.Column("start_char", sa.BigInteger(), nullable=False),
        sa.Column("end_char", sa.BigInteger(), nullable=False),
        sa.Column("chunker_id", sa.Text(), nullable=False),
        sa.Column("provenance", pg.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("snapshot_id", "chunk_id", name="pk_chunks"),
        sa.ForeignKeyConstraint(["snapshot_id", "doc_id", "revision_id"],
            ["rag.document_revisions.snapshot_id", "rag.document_revisions.doc_id",
             "rag.document_revisions.revision_id"], name="fk_chunks_revision"),
        sa.CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_chunks_span"),
        sa.CheckConstraint("jsonb_typeof(provenance) = 'object'", name="ck_chunks_provenance_object"),
        *_nonblank("chunks", "chunk_id", "chunker_id"),
    )
    sa.Index("ix_chunks_revision", chunks.c.snapshot_id, chunks.c.doc_id, chunks.c.revision_id)

    grants = sa.Table("document_grants", catalog,
        sa.Column("grant_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text()),
        sa.Column("group_id", sa.Text()),
        sa.Column("origin", sa.Text(), nullable=False, server_default=sa.text("'operator'")),
        sa.PrimaryKeyConstraint("grant_id", name="pk_document_grants"),
        sa.ForeignKeyConstraint(["snapshot_id", "doc_id"],
            ["rag.documents.snapshot_id", "rag.documents.doc_id"], name="fk_grants_document"),
        sa.ForeignKeyConstraint(["user_id"], ["rag.users.user_id"], name="fk_grants_user"),
        sa.ForeignKeyConstraint(["group_id"], ["rag.groups.group_id"], name="fk_grants_group"),
        sa.CheckConstraint("(user_id IS NULL) <> (group_id IS NULL)", name="ck_grants_one_subject"),
        sa.CheckConstraint("origin IN ('operator', 'synthetic')", name="ck_grants_origin"),
    )
    sa.Index("uq_document_grants_user", grants.c.user_id, grants.c.snapshot_id, grants.c.doc_id,
             unique=True, postgresql_where=grants.c.user_id.is_not(None))
    sa.Index("uq_document_grants_group", grants.c.group_id, grants.c.snapshot_id, grants.c.doc_id,
             unique=True, postgresql_where=grants.c.group_id.is_not(None))
    sa.Index("ix_document_grants_document", grants.c.snapshot_id, grants.c.doc_id)

    sa.Table("policy_state", catalog,
        sa.Column("singleton_id", sa.SmallInteger(), nullable=False, server_default=sa.text("1")),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("singleton_id", name="pk_policy_state"),
        sa.CheckConstraint("singleton_id = 1", name="ck_policy_state_singleton"),
        *_nonnegative("policy_state", "revision"),
    )
    tickets = sa.Table("tickets", catalog,
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("ticket_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("revision_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text()),
        sa.Column("priority", sa.Text()),
        sa.Column("project_id", sa.Text()),
        sa.Column("field_provenance", pg.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("coverage", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "ticket_id", name="pk_tickets"),
        sa.ForeignKeyConstraint(["snapshot_id", "doc_id", "revision_id"],
            ["rag.document_revisions.snapshot_id", "rag.document_revisions.doc_id",
             "rag.document_revisions.revision_id"], name="fk_tickets_revision"),
        sa.CheckConstraint("jsonb_typeof(field_provenance) = 'object'",
                           name="ck_tickets_provenance_object"),
        sa.CheckConstraint("coverage IN ('verified_records', 'synthetic_fixture')",
                           name="ck_tickets_coverage"),
        *_nonblank("tickets", "ticket_id"),
    )
    sa.Index("ix_tickets_revision", tickets.c.snapshot_id, tickets.c.doc_id, tickets.c.revision_id)
    sa.Index("ix_tickets_status_project", tickets.c.snapshot_id, tickets.c.status, tickets.c.project_id)

    runs = sa.Table("query_runs", catalog,
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("scope_id", sa.Text(), nullable=False),
        sa.Column("policy_revision", sa.BigInteger(), nullable=False),
        sa.Column("session_version", sa.BigInteger(), nullable=False),
        sa.Column("config_sha256", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("api_calls", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("estimated_cost_usd", pg.DOUBLE_PRECISION()),
        sa.Column("latency_ms", pg.DOUBLE_PRECISION(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("run_id", name="pk_query_runs"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["rag.users.user_id"], name="fk_runs_owner"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["rag.dataset_snapshots.snapshot_id"],
                                name="fk_runs_snapshot"),
        sa.UniqueConstraint("run_id", "owner_user_id", "snapshot_id", "scope_id", "policy_revision",
                            "session_version", name="uq_query_runs_scope_binding"),
        sa.CheckConstraint("mode IN ('baseline', 'agentic')", name="ck_query_runs_mode"),
        sa.CheckConstraint("status IN ('pending', 'answered', 'insufficient_evidence', 'error', 'clarification')",
                           name="ck_query_runs_status"),
        sa.CheckConstraint("char_length(question) <= 8000", name="ck_query_runs_question_length"),
        sa.CheckConstraint("finished_at IS NULL OR finished_at >= created_at",
                           name="ck_query_runs_finished_after_created"),
        sa.CheckConstraint("estimated_cost_usd IS NULL OR (estimated_cost_usd >= 0 AND estimated_cost_usd < 'Infinity'::double precision)",
                           name="ck_query_runs_cost_finite_nonnegative"),
        sa.CheckConstraint("latency_ms >= 0 AND latency_ms < 'Infinity'::double precision",
                           name="ck_query_runs_latency_finite_nonnegative"),
        _sha("query_runs", "config_sha256"),
        *_nonblank("query_runs", "run_id", "scope_id", "question"),
        *_nonnegative("query_runs", "policy_revision", "session_version", "input_tokens",
                      "output_tokens", "api_calls"),
    )
    sa.Index("ix_query_runs_owner_created", runs.c.owner_user_id, runs.c.created_at)
    sa.Index("ix_query_runs_snapshot", runs.c.snapshot_id)

    sa.Table("run_steps", catalog,
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("step_number", sa.BigInteger(), nullable=False),
        sa.Column("payload", pg.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "step_number", name="pk_run_steps"),
        sa.ForeignKeyConstraint(["run_id"], ["rag.query_runs.run_id"], name="fk_run_steps_run"),
        sa.CheckConstraint("step_number >= 1", name="ck_run_steps_positive_number"),
        sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_run_steps_payload_object"),
    )
    aggregates = sa.Table("aggregate_results", catalog,
        sa.Column("artifact_id", sa.Text(), nullable=False),
        sa.Column("citation_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("scope_id", sa.Text(), nullable=False),
        sa.Column("policy_revision", sa.BigInteger(), nullable=False),
        sa.Column("session_version", sa.BigInteger(), nullable=False),
        sa.Column("template_id", sa.Text(), nullable=False),
        sa.Column("template_version", sa.Text(), nullable=False),
        sa.Column("filters", pg.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("operation", sa.Text(), nullable=False, server_default=sa.text("'count'")),
        sa.Column("count", sa.BigInteger(), nullable=False),
        sa.Column("lineage_sha256", sa.Text(), nullable=False),
        sa.Column("coverage", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("artifact_id", name="pk_aggregate_results"),
        sa.UniqueConstraint("citation_id", name="uq_aggregate_results_citation"),
        sa.ForeignKeyConstraint(["run_id", "owner_user_id", "snapshot_id", "scope_id",
                                 "policy_revision", "session_version"],
            ["rag.query_runs.run_id", "rag.query_runs.owner_user_id", "rag.query_runs.snapshot_id",
             "rag.query_runs.scope_id", "rag.query_runs.policy_revision", "rag.query_runs.session_version"],
            name="fk_aggregate_results_run_binding"),
        sa.CheckConstraint("jsonb_typeof(filters) = 'array'", name="ck_aggregate_results_filters_array"),
        sa.CheckConstraint("operation = 'count'", name="ck_aggregate_results_count_operation"),
        sa.CheckConstraint("coverage IN ('verified_records', 'synthetic_fixture')",
                           name="ck_aggregate_results_coverage"),
        _sha("aggregate_results", "lineage_sha256"),
        *_nonblank("aggregate_results", "artifact_id", "citation_id", "scope_id", "template_id",
                   "template_version"),
        *_nonnegative("aggregate_results", "policy_revision", "session_version", "count"),
    )
    sa.Index("ix_aggregate_results_run", aggregates.c.run_id)
    sa.Index("ix_aggregate_results_owner_created", aggregates.c.owner_user_id, aggregates.c.created_at)
    return catalog


metadata = _build_metadata()
users = metadata.tables["rag.users"]
sessions = metadata.tables["rag.sessions"]
groups = metadata.tables["rag.groups"]
group_memberships = metadata.tables["rag.group_memberships"]
dataset_snapshots = metadata.tables["rag.dataset_snapshots"]
catalog_imports = metadata.tables["rag.catalog_imports"]
documents = metadata.tables["rag.documents"]
document_revisions = metadata.tables["rag.document_revisions"]
chunks = metadata.tables["rag.chunks"]
document_grants = metadata.tables["rag.document_grants"]
policy_state = metadata.tables["rag.policy_state"]
tickets = metadata.tables["rag.tickets"]
query_runs = metadata.tables["rag.query_runs"]
run_steps = metadata.tables["rag.run_steps"]
aggregate_results = metadata.tables["rag.aggregate_results"]

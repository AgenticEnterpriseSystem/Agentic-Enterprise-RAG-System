"""Foundation migrations and database invariants on dedicated disposable PG."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from uuid import uuid4

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Database checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Database checks require the psycopg driver")
pytest.importorskip("alembic", reason="Migration checks require Alembic")

from tests.db_support import (
    db_engine, disposable_database, engine_for, fresh_db_engine, pg_urls, require_test_url,
    seed_catalog, url_string,
)


EXPECTED_TABLES = {
    "users", "sessions", "groups", "group_memberships", "dataset_snapshots", "documents",
    "document_revisions", "chunks", "document_grants", "policy_state", "tickets",
    "query_runs", "run_steps", "aggregate_results", "catalog_imports",
    "group_requirements", "policy_scopes", "policy_changes",
}


@contextmanager
def rejected(connection, *, sqlstate=None):
    with pytest.raises(sa.exc.DBAPIError) as error:
        with connection.begin_nested():
            yield
    if sqlstate:
        assert error.value.orig.sqlstate == sqlstate


@pytest.mark.parametrize("url", [
    "postgresql://username:private-marker@localhost/postgres",
    "postgresql://username:private-marker@localhost/production",
    "sqlite:///rag_test_not_postgresql.db", "not-a-url",
])
def test_database_helper_rejects_non_test_targets_without_disclosing_urls(url):
    with pytest.raises(ValueError) as error:
        require_test_url(url)
    assert "private-marker" not in str(error.value)


def test_database_helper_normalizes_only_an_explicit_test_target():
    url = require_test_url("postgresql://u:p@localhost/rag_test_foundation")
    assert url.drivername == "postgresql+psycopg" and url.database == "rag_test_foundation"


def test_migration_history_has_one_expected_head():
    from src.db.migrate import migration_heads
    assert migration_heads() == ("0003_document_policy",)


def test_fresh_database_upgrade_and_repeat_upgrade_preserve_schema_and_seed(pg_urls):
    from src.db.migrate import current_revision, upgrade_database
    with disposable_database(pg_urls) as fresh:
        assert current_revision(url_string(fresh.migration)) is None
        upgrade_database(url_string(fresh.migration))
        assert current_revision(url_string(fresh.migration)) == "0003_document_policy"
        engine = engine_for(fresh.migration)
        try:
            with engine.begin() as connection:
                from src.db import schema
                data = seed_catalog(connection)
            upgrade_database(url_string(fresh.migration))
            assert current_revision(url_string(fresh.migration)) == "0003_document_policy"
            with engine.connect() as connection:
                assert set(sa.inspect(connection).get_table_names(schema="rag")) == EXPECTED_TABLES | {"alembic_version"}
                assert connection.execute(sa.select(schema.document_revisions.c.canonical_text).where(
                    schema.document_revisions.c.snapshot_id == data["snapshot_id"])).scalar_one() == data["text"]
        finally:
            engine.dispose()


def test_fresh_runtime_can_read_migration_revision_but_cannot_change_version_table(pg_urls):
    from src.db.migrate import current_revision, upgrade_database
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        assert current_revision(url_string(fresh.runtime)) == "0003_document_policy"
        runtime = engine_for(fresh.runtime)
        try:
            with runtime.connect() as connection, connection.begin():
                with rejected(connection, sqlstate="42501"):
                    connection.execute(sa.text(
                        "UPDATE rag.alembic_version SET version_num='0001_foundation'"))
                # A false predicate still requires DELETE permission and cannot
                # remove version state even if an incorrect grant slips through.
                with rejected(connection, sqlstate="42501"):
                    connection.execute(sa.text("DELETE FROM rag.alembic_version WHERE false"))
            assert current_revision(url_string(fresh.runtime)) == "0003_document_policy"
        finally:
            runtime.dispose()


def test_migrated_tables_match_core_metadata_and_protected_tables_keep_rls(db_engine):
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext
    from src.db import schema

    assert {table.name for table in schema.metadata.tables.values()} == EXPECTED_TABLES
    with db_engine.connect() as connection:
        def include_object(obj, name, kind, reflected, compare_to):
            return name != "alembic_version" and (kind != "table" or obj.schema == "rag")
        migration = MigrationContext.configure(connection, opts={"include_schemas": True,
            "include_object": include_object, "compare_type": True, "compare_server_default": True})
        assert compare_metadata(migration, schema.metadata) == []
        protected = connection.execute(sa.text("""
            SELECT c.relname, c.relrowsecurity
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='rag' AND c.relkind='r'
        """)).all()
        enabled = {name for name, active in protected if active}
        assert set(schema.PROTECTED_TABLES) <= enabled
        # Explicit ACL policies now exist. Functional nonowner/default-denial
        # cases exercise them; metadata parity must not demand no policies.
        policies = connection.execute(sa.text("""
            SELECT tablename, qual, with_check FROM pg_policies WHERE schemaname='rag'
        """)).all()
        assert not any(qual == "true" or check == "true"
                       for table, qual, check in policies if table in schema.PROTECTED_TABLES)


def test_unique_normalized_logins_membership_fk_and_catalog_position(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection)
        with rejected(connection, sqlstate="23505"):
            connection.execute(schema.users.insert().values(user_id="other-" + uuid4().hex,
                login=data["login"], password_hash="synthetic", display_name="Other"))
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.users.insert().values(user_id="other-" + uuid4().hex,
                login=" MixedCase ", password_hash="synthetic", display_name="Other"))
        with rejected(connection, sqlstate="23503"):
            connection.execute(schema.group_memberships.insert().values(
                user_id=data["user_id"], group_id="missing-" + uuid4().hex))
        with rejected(connection, sqlstate="23505"):
            connection.execute(schema.documents.insert().values(snapshot_id=data["snapshot_id"],
                doc_id="other-" + uuid4().hex, position=0, revision_id=data["revision_id"],
                source_type="jira", text_length=len(data["text"])))


def test_grants_require_one_subject_and_unique_principal_document_pairs(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection)
        group_id = "group-" + uuid4().hex
        connection.execute(schema.groups.insert().values(group_id=group_id, label="Synthetic group"))
        base = {"snapshot_id": data["snapshot_id"], "doc_id": data["doc_id"]}
        for subjects in ({}, {"user_id": data["user_id"], "group_id": group_id}):
            with rejected(connection, sqlstate="23514"):
                connection.execute(schema.document_grants.insert().values(**(base | subjects)))
        for subject in ({"user_id": data["user_id"]}, {"group_id": group_id}):
            connection.execute(schema.document_grants.insert().values(**(base | subject)))
            with rejected(connection, sqlstate="23505"):
                connection.execute(schema.document_grants.insert().values(**(base | subject)))
        connection.execute(schema.group_memberships.insert().values(user_id=data["user_id"], group_id=group_id))
        with rejected(connection, sqlstate="23505"):
            connection.execute(schema.group_memberships.insert().values(user_id=data["user_id"], group_id=group_id))


def test_session_tokens_are_unique_hashed_values_and_expiry_is_validated(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection)
        session = {"session_id": "session-" + uuid4().hex, "user_id": data["user_id"],
                   "token_hash": hashlib.sha256(uuid4().bytes).hexdigest(), "csrf_hash": "d" * 64,
                   "expires_at": datetime(2030, 1, 1, tzinfo=timezone.utc)}
        connection.execute(schema.sessions.insert().values(**session))
        with rejected(connection, sqlstate="23505"):
            connection.execute(schema.sessions.insert().values(**(session | {"session_id": "session-" + uuid4().hex})))
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.sessions.insert().values(**(session | {
                "session_id": "session-" + uuid4().hex, "token_hash": "invalid-plaintext-token"})))
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.sessions.insert().values(**(session | {
                "session_id": "session-" + uuid4().hex,
                "token_hash": hashlib.sha256(uuid4().bytes).hexdigest(),
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "expires_at": datetime(2025, 1, 1, tzinfo=timezone.utc)})))


def test_zero_aggregate_is_bound_to_its_owning_run_snapshot_and_scope_versions(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection)
        run = {"run_id": "run-" + uuid4().hex, "owner_user_id": data["user_id"],
               "snapshot_id": data["snapshot_id"], "scope_id": "scope-fixture",
               "policy_revision": 7, "session_version": 2, "config_sha256": "a" * 64,
               "mode": "agentic", "question": "Count accessible open tickets."}
        connection.execute(schema.query_runs.insert().values(**run))
        artifact = {key: run[key] for key in ("run_id", "owner_user_id", "snapshot_id", "scope_id",
                                            "policy_revision", "session_version")}
        artifact |= {"artifact_id": "artifact-" + uuid4().hex, "citation_id": "aggregate-" + uuid4().hex,
                     "template_id": "verified-ticket-count", "template_version": "1", "count": 0,
                     "lineage_sha256": "c" * 64, "coverage": "synthetic_fixture"}
        connection.execute(schema.aggregate_results.insert().values(**artifact))
        for field, changed in (("owner_user_id", "another-user"), ("snapshot_id", "another-snapshot"),
                               ("scope_id", "another-scope"), ("policy_revision", 8), ("session_version", 3)):
            forged = artifact | {"artifact_id": "artifact-" + uuid4().hex,
                                  "citation_id": "aggregate-" + uuid4().hex, field: changed}
            with rejected(connection, sqlstate="23503"):
                connection.execute(schema.aggregate_results.insert().values(**forged))
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.aggregate_results.insert().values(**(artifact | {
                "artifact_id": "artifact-" + uuid4().hex, "citation_id": "aggregate-" + uuid4().hex, "count": -1})))


@pytest.mark.parametrize("change", [
    {"catalog_sha256": None}, {"corpus_sha256": "invalid"},
    {"catalog_sha256": "A" * 64}, {"document_count": -1},
])
def test_ready_snapshot_requires_valid_hashes_and_nonnegative_count(db_engine, change):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        values = dict(snapshot_id="snapshot-" + uuid4().hex, state="ready", document_count=0,
                      corpus_sha256="a" * 64, catalog_sha256="b" * 64) | change
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.dataset_snapshots.insert().values(**values))


def test_revision_body_and_ready_catalog_identity_are_immutable(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection, ready=True)
        revision = schema.document_revisions.c.snapshot_id == data["snapshot_id"]
        snapshot = schema.dataset_snapshots.c.snapshot_id == data["snapshot_id"]
        with rejected(connection):
            connection.execute(schema.document_revisions.update().where(revision).values(
                canonical_text="x" * len(data["text"])))
        with rejected(connection):
            connection.execute(schema.document_revisions.delete().where(revision))
        with rejected(connection):
            connection.execute(schema.dataset_snapshots.update().where(snapshot).values(corpus_sha256="c" * 64))
        with rejected(connection):
            connection.execute(schema.documents.update().where(
                schema.documents.c.snapshot_id == data["snapshot_id"]).values(source_type="slack"))
        # Exact no-op writes do not count as a new version or force import retries.
        connection.execute(schema.dataset_snapshots.update().where(snapshot).values(state="ready"))
        assert connection.execute(sa.select(schema.document_revisions.c.canonical_text).where(revision)).scalar_one() == data["text"]


@pytest.mark.parametrize("mismatch", ["hash", "character_length"])
def test_canonical_revision_hash_and_character_length_are_checked_by_database(db_engine, mismatch):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        snapshot_id, doc_id = "snapshot-" + uuid4().hex, "doc-" + uuid4().hex
        text = "Café 🧭"
        revision_id = "c" * 64 if mismatch == "hash" else hashlib.sha256(text.encode()).hexdigest()
        text_length = len(text) if mismatch == "hash" else len(text) + 1
        connection.execute(schema.dataset_snapshots.insert().values(
            snapshot_id=snapshot_id, corpus_sha256="a" * 64, document_count=1))
        connection.execute(schema.documents.insert().values(snapshot_id=snapshot_id,
            doc_id=doc_id, position=0, revision_id=revision_id, source_type="jira", text_length=text_length))
        with rejected(connection):
            connection.execute(schema.document_revisions.insert().values(snapshot_id=snapshot_id,
                doc_id=doc_id, revision_id=revision_id, title="", canonical_text=text))


def test_derived_chunks_can_append_after_ready_but_spans_are_bounded_and_immutable(db_engine):
    from src.db import schema
    with db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection, ready=True)
        chunk = {"snapshot_id": data["snapshot_id"], "chunk_id": "chunk-" + uuid4().hex,
                 "doc_id": data["doc_id"], "revision_id": data["revision_id"],
                 "start_char": len(data["text"]) - 3, "end_char": len(data["text"]),
                 "chunker_id": "another-bounded-window"}
        connection.execute(schema.chunks.insert().values(**chunk))
        with rejected(connection, sqlstate="23514"):
            connection.execute(schema.chunks.insert().values(**(chunk | {
                "chunk_id": "chunk-" + uuid4().hex, "end_char": len(data["text"]) + 1})))
        target = schema.chunks.c.chunk_id == chunk["chunk_id"]
        with rejected(connection):
            connection.execute(schema.chunks.update().where(target).values(chunker_id="changed"))
        with rejected(connection):
            connection.execute(schema.chunks.delete().where(target))
        assert connection.execute(sa.select(schema.dataset_snapshots.c.catalog_sha256).where(
            schema.dataset_snapshots.c.snapshot_id == data["snapshot_id"])).scalar_one() == "b" * 64


def test_ready_snapshot_requires_complete_catalog_and_each_revision_body(fresh_db_engine):
    from src.db import schema
    with fresh_db_engine.connect() as connection, connection.begin():
        with rejected(connection):
            connection.execute(schema.dataset_snapshots.insert().values(
                snapshot_id="premature-" + uuid4().hex, document_count=1, state="ready",
                corpus_sha256="a" * 64, catalog_sha256="b" * 64))
        data = seed_catalog(connection)
        target = schema.dataset_snapshots.c.snapshot_id == data["snapshot_id"]
        connection.execute(schema.dataset_snapshots.update().where(target).values(document_count=2))
        with rejected(connection):
            connection.execute(schema.dataset_snapshots.update().where(target).values(state="ready", catalog_sha256="b" * 64))
        second_text = "Second canonical record"
        second_doc = "second-" + uuid4().hex
        second_revision = hashlib.sha256(second_text.encode()).hexdigest()
        connection.execute(schema.documents.insert().values(snapshot_id=data["snapshot_id"],
            doc_id=second_doc, position=1, revision_id=second_revision, source_type="jira", text_length=len(second_text)))
        with rejected(connection):
            connection.execute(schema.dataset_snapshots.update().where(target).values(state="ready", catalog_sha256="b" * 64))
        connection.execute(schema.document_revisions.insert().values(snapshot_id=data["snapshot_id"],
            doc_id=second_doc, revision_id=second_revision, title="", canonical_text=second_text))
        connection.execute(schema.dataset_snapshots.update().where(target).values(state="ready", catalog_sha256="b" * 64))
        assert connection.execute(sa.select(schema.dataset_snapshots.c.state).where(target)).scalar_one() == "ready"


def test_ready_rechecks_positions_after_importing_snapshot_count_changes(fresh_db_engine):
    from src.db import schema
    with fresh_db_engine.connect() as connection, connection.begin():
        snapshot, doc = "snapshot-" + uuid4().hex, "doc-" + uuid4().hex
        text = "Position one"
        revision = hashlib.sha256(text.encode()).hexdigest()
        connection.execute(schema.dataset_snapshots.insert().values(
            snapshot_id=snapshot, corpus_sha256="a" * 64, document_count=2))
        connection.execute(schema.documents.insert().values(snapshot_id=snapshot, doc_id=doc,
            position=1, revision_id=revision, source_type="jira", text_length=len(text)))
        connection.execute(schema.document_revisions.insert().values(snapshot_id=snapshot,
            doc_id=doc, revision_id=revision, title="", canonical_text=text))
        target = schema.dataset_snapshots.c.snapshot_id == snapshot
        connection.execute(schema.dataset_snapshots.update().where(target).values(document_count=1))
        # Row/body counts now match, but position 1 is outside the new 0..0 range.
        with rejected(connection):
            connection.execute(schema.dataset_snapshots.update().where(target).values(state="ready", catalog_sha256="b" * 64))


def test_policy_and_session_versions_never_decrease_or_lose_the_singleton(fresh_db_engine):
    from src.db import schema
    with fresh_db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection)
        singleton = schema.policy_state.c.singleton_id == 1
        connection.execute(schema.policy_state.update().where(singleton).values(revision=2))
        connection.execute(schema.policy_state.update().where(singleton).values(revision=2))
        with rejected(connection):
            connection.execute(schema.policy_state.update().where(singleton).values(revision=1))
        with rejected(connection):
            connection.execute(schema.policy_state.delete().where(singleton))
        assert connection.execute(sa.select(schema.policy_state.c.revision).where(singleton)).scalar_one() == 2
        user = schema.users.c.user_id == data["user_id"]
        connection.execute(schema.users.update().where(user).values(session_version=2))
        connection.execute(schema.users.update().where(user).values(session_version=2))
        with rejected(connection):
            connection.execute(schema.users.update().where(user).values(session_version=1))
        assert connection.execute(sa.select(schema.users.c.session_version).where(user)).scalar_one() == 2


def test_run_provenance_is_frozen_before_any_aggregate_but_outcome_can_progress(fresh_db_engine):
    from src.db import schema
    with fresh_db_engine.connect() as connection, connection.begin():
        data = seed_catalog(connection, ready=True)
        other = seed_catalog(connection, ready=True)
        run = {"run_id": "run-" + uuid4().hex, "owner_user_id": data["user_id"],
               "snapshot_id": data["snapshot_id"], "scope_id": "original-scope", "policy_revision": 7,
               "session_version": 2, "config_sha256": "a" * 64, "mode": "baseline", "question": "Original question"}
        connection.execute(schema.query_runs.insert().values(**run))
        target = schema.query_runs.c.run_id == run["run_id"]
        original = connection.execute(sa.select(schema.query_runs).where(target)).mappings().one()
        changes = {"run_id": "other-run-" + uuid4().hex, "owner_user_id": other["user_id"],
                   "snapshot_id": other["snapshot_id"], "scope_id": "changed-scope", "policy_revision": 8,
                   "session_version": 3, "config_sha256": "b" * 64, "mode": "agentic", "question": "Changed question",
                   "created_at": original["created_at"] + timedelta(seconds=1)}
        for field, replacement in changes.items():
            with rejected(connection):
                connection.execute(schema.query_runs.update().where(target).values(**{field: replacement}))
        connection.execute(schema.query_runs.update().where(target).values(status="answered",
            finished_at=original["created_at"] + timedelta(seconds=10), input_tokens=10, output_tokens=5,
            api_calls=1, estimated_cost_usd=.01, latency_ms=25))
        stored = connection.execute(sa.select(schema.query_runs).where(target)).mappings().one()
        assert all(stored[field] == original[field] for field in changes)
        assert stored["status"] == "answered" and stored["input_tokens"] == 10


def test_frozen_initial_revision_installs_no_allow_policy_and_denies_runtime(pg_urls):
    from alembic import command
    from src.db.migrate import migration_config
    with disposable_database(pg_urls) as fresh:
        owner = engine_for(fresh.migration)
        try:
            with owner.begin() as connection:
                connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS rag")
                config = migration_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "0001_foundation")
                canonical = "Frozen\ntext"
                revision = hashlib.sha256(canonical.encode()).hexdigest()
                connection.execute(sa.text("INSERT INTO rag.dataset_snapshots "
                    "(snapshot_id,corpus_sha256,document_count) VALUES ('frozen',:corpus,1)"),
                    {"corpus": "a" * 64})
                connection.execute(sa.text("INSERT INTO rag.documents "
                    "(snapshot_id,doc_id,position,revision_id,source_type,text_length) "
                    "VALUES ('frozen','frozen-doc',0,:revision,'jira',:length)"),
                    {"revision": revision, "length": len(canonical)})
                connection.execute(sa.text("INSERT INTO rag.document_revisions "
                    "(snapshot_id,doc_id,revision_id,title,canonical_text) "
                    "VALUES ('frozen','frozen-doc',:revision,'Frozen',:body)"),
                    {"revision": revision, "body": canonical})
                assert connection.execute(sa.text(
                    "SELECT count(*) FROM rag.document_revisions")).scalar_one() == 1
                assert connection.execute(sa.text(
                    "SELECT count(*) FROM pg_policies WHERE schemaname='rag'")).scalar_one() == 0
            runtime = engine_for(fresh.runtime)
            try:
                with runtime.connect() as connection:
                    assert connection.execute(sa.text(
                        "SELECT count(*) FROM rag.document_revisions")).scalar_one() == 0
            finally:
                runtime.dispose()
        finally:
            owner.dispose()

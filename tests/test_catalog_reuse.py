"""Real PostgreSQL metadata reuse without cached authority or extra pools.

Only fresh disposable databases from the existing ACL fixture are used. These
checks exercise the new explicit proof path; the original ACL/body suites retain
the exhaustive policy and excerpt matrix.
"""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from threading import Barrier, Event

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Catalog reuse checks require DB dependencies")
pytest.importorskip("psycopg", reason="Catalog reuse checks require the psycopg driver")
pytest.importorskip("alembic", reason="Catalog reuse checks require Alembic")

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, PolicyUnavailable, RequestContext,
    SessionExpired, SnapshotMismatch, StaleContext,
)
from src.access.policy import PolicyResolver
from src.db import schema
from src.db.runtime import Database, DatabaseSettings
from src.contracts import SnapshotRef
from src.storage.fingerprints import ValidatedCatalog, catalog_fingerprint
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import (
    DOC_IDS, ENTRIES, TEXTS, acl_db, allowed_rows, candidate, identity,
    raw_token, seed_explicit_grants,
)


@contextmanager
def queries(database):
    statements = []
    def record(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)
    sa.event.listen(database.engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        sa.event.remove(database.engine, "before_cursor_execute", record)


def catalog_queries(statements):
    return tuple(statement for statement in statements
        if statement.lstrip().lower().startswith("select ")
        and "from rag.documents" in " ".join(statement.lower().split()))


def reused(fixture, **restrictions):
    store = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    resolver = PolicyResolver(fixture.database, fixture.snapshot, catalog=proof, **restrictions)
    return store, proof, resolver


def test_one_constructor_scan_is_shared_by_getters_and_multiple_resolvers_without_body_columns(acl_db):
    fixture = acl_db
    with queries(fixture.database) as statements:
        store = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True)
        initial = catalog_queries(statements)
        assert initial
        proof = store.validated_catalog()
        first = PolicyResolver(fixture.database, fixture.snapshot, catalog=proof)
        second = PolicyResolver(fixture.database, fixture.snapshot, catalog=store.validated_catalog(), doc_ids=("general",))
        assert store.validated_catalog() is proof
        assert catalog_queries(statements) == initial
    assert proof.entries == ENTRIES
    assert first._catalog is not second._catalog
    assert first._catalog.doc_ids is second._catalog.doc_ids is proof.doc_ids
    assert first._catalog.sources is second._catalog.sources is proof.sources
    assert first._catalog.positions is second._catalog.positions is proof.positions
    normalized = "\n".join(statements).lower()
    assert "canonical_text" not in normalized and "canonical_utf8" not in normalized
    assert "document_revisions" not in normalized
    assert "rag.documents.title" not in normalized and "rag.documents.content" not in normalized
    assert sum("from rag.dataset_snapshots" in statement.lower() for statement in statements) >= 5
    assert fixture.database.engine.pool.checkedout() == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}, object()],
    ids=["none", "zero", "one", "true-string", "false-string", "list", "dict", "object"])
def test_retention_flag_is_a_strict_boolean_before_database_io(monkeypatch, value):
    database = object.__new__(Database)
    snapshot = SnapshotRef(snapshot_id="flag-fixture", corpus_sha256="a" * 64,
        catalog_sha256=catalog_fingerprint(ENTRIES, expected_count=len(ENTRIES)), document_count=len(ENTRIES))
    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid retention flag reached database work")
    monkeypatch.setattr(Database, "transaction", unexpected)
    with pytest.raises((ValueError, TypeError)):
        PostgresDocumentStore(database, snapshot, retain_catalog=value)


def test_default_streams_verification_without_retention_then_lazily_builds_once(acl_db, monkeypatch):
    fixture, builds = acl_db, []
    original = ValidatedCatalog.build
    def tracked(cls, snapshot, entries, *, issuer):
        builds.append(True)
        return original(snapshot, entries, issuer=issuer)
    monkeypatch.setattr(ValidatedCatalog, "build", classmethod(tracked))
    with queries(fixture.database) as statements:
        store = PostgresDocumentStore(fixture.database, fixture.snapshot)
        initial = len(catalog_queries(statements))
        assert initial > 0 and builds == []
        assert getattr(store, "_validated_catalog", None) is None
        proof = store.validated_catalog()
        after_lazy_build = len(catalog_queries(statements))
        assert after_lazy_build > initial and builds == [True]
        assert store.validated_catalog() is proof
        resolver = PolicyResolver(fixture.database, fixture.snapshot, catalog=store.validated_catalog())
        assert resolver._catalog.positions is proof.positions
        assert len(catalog_queries(statements)) == after_lazy_build and builds == [True]
    assert proof.entries == ENTRIES and fixture.database.engine.pool.checkedout() == 0


def test_default_streaming_verification_still_rejects_corrupt_metadata(acl_db, monkeypatch):
    fixture = acl_db
    original = PostgresDocumentStore.iter_catalog
    def corrupt(self, *, batch_size=1000):
        for batch in original(self, batch_size=batch_size):
            yield tuple(entry.model_copy(update={"source_type": "changed"}) if entry.position == 0 else entry
                        for entry in batch)
    monkeypatch.setattr(PostgresDocumentStore, "iter_catalog", corrupt)
    with pytest.raises(SnapshotMismatch):
        PostgresDocumentStore(fixture.database, fixture.snapshot)
    assert fixture.database.engine.pool.checkedout() == 0


def test_simultaneous_lazy_getters_publish_only_one_built_capability(acl_db, monkeypatch):
    fixture = acl_db
    store = PostgresDocumentStore(fixture.database, fixture.snapshot)
    original, builds = ValidatedCatalog.build, []
    start, entered, release = Barrier(4, timeout=5), Event(), Event()
    def blocked_build(cls, snapshot, entries, *, issuer):
        builds.append(True)
        entered.set()
        assert release.wait(timeout=5), "Test did not release the paused metadata build"
        return original(snapshot, entries, issuer=issuer)
    def getter():
        start.wait()
        return store.validated_catalog()
    monkeypatch.setattr(ValidatedCatalog, "build", classmethod(blocked_build))
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(getter) for _ in range(4)]
        try:
            assert entered.wait(timeout=5)
            # Keep the build paused while competing real getters contend. A
            # second unsynchronized builder would be counted independently.
            done, _pending = wait(futures, timeout=.1)
            assert not done
        finally:
            release.set()
        proofs = tuple(future.result(timeout=10) for future in futures)
    assert builds == [True]
    assert all(proof is proofs[0] for proof in proofs)
    assert fixture.database.engine.pool.checkedout() == 0


def test_old_keyset_iteration_keeps_batches_and_releases_the_only_connection_before_yield(acl_db):
    fixture = acl_db
    store = PostgresDocumentStore(fixture.database, fixture.snapshot)
    with queries(fixture.database) as statements:
        iterator = store.iter_catalog(batch_size=2)
        try:
            first = next(iterator)
            assert first == ENTRIES[:2]
            assert fixture.database.engine.pool.checkedout() == 0
            # This actual live-ready operation would time out on pool_size1 if
            # the generator retained its read transaction across the yield.
            proof = store.validated_catalog()
            resolver = PolicyResolver(fixture.database, fixture.snapshot, catalog=proof)
            assert resolver._catalog.positions is proof.positions
            assert fixture.database.engine.pool.checkedout() == 0
            remainder = tuple(iterator)
        finally:
            iterator.close()
    assert tuple(entry for batch in (first, *remainder) for entry in batch) == ENTRIES
    assert all(len(batch) <= 2 for batch in (first, *remainder))
    assert len(catalog_queries(statements)) > 1
    assert all("order by rag.documents.position" in query.lower() and "limit" in query.lower()
        for query in catalog_queries(statements))
    assert fixture.database.engine.pool.checkedout() == 0


def test_constructor_rechecks_live_ready_after_complete_build_before_publishing_proof(acl_db, monkeypatch):
    fixture = acl_db
    original, completed = ValidatedCatalog.build, []

    def lose_readiness_after_build(cls, snapshot, entries, *, issuer):
        proof = original(snapshot, entries, issuer=issuer)
        completed.append(True)
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql("REVOKE SELECT ON rag.dataset_snapshots FROM rag_runtime")
        return proof

    monkeypatch.setattr(ValidatedCatalog, "build", classmethod(lose_readiness_after_build))
    unpublished = object.__new__(PostgresDocumentStore)
    try:
        with pytest.raises(sa.exc.SQLAlchemyError):
            unpublished.__init__(fixture.database, fixture.snapshot, retain_catalog=True)
        assert completed == [True]
        assert getattr(unpublished, "_validated_catalog", None) is None
    finally:
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql("GRANT SELECT ON rag.dataset_snapshots TO rag_runtime")
    assert fixture.database.engine.pool.checkedout() == 0


def test_failed_lazy_final_ready_check_keeps_no_proof_and_next_call_rebuilds(acl_db, monkeypatch):
    fixture = acl_db
    store = PostgresDocumentStore(fixture.database, fixture.snapshot)
    original, completed = ValidatedCatalog.build, []
    def fail_first_final_check(cls, snapshot, entries, *, issuer):
        proof = original(snapshot, entries, issuer=issuer)
        completed.append(True)
        if len(completed) == 1:
            with fixture.owner.begin() as connection:
                connection.exec_driver_sql("REVOKE SELECT ON rag.dataset_snapshots FROM rag_runtime")
        return proof
    monkeypatch.setattr(ValidatedCatalog, "build", classmethod(fail_first_final_check))
    try:
        with pytest.raises(sa.exc.SQLAlchemyError):
            store.validated_catalog()
        assert store._validated_catalog is None and completed == [True]
    finally:
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql("GRANT SELECT ON rag.dataset_snapshots TO rag_runtime")
    proof = store.validated_catalog()
    assert completed == [True, True]
    assert store.validated_catalog() is proof and completed == [True, True]
    assert fixture.database.engine.pool.checkedout() == 0


def test_cached_getter_rechecks_ready_after_proof_require_before_release(acl_db, monkeypatch):
    fixture = acl_db
    store = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    original = ValidatedCatalog.require
    def lose_readiness_after_require(self, snapshot, *, issuer=None):
        result = original(self, snapshot, issuer=issuer)
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql("REVOKE SELECT ON rag.dataset_snapshots FROM rag_runtime")
        return result
    monkeypatch.setattr(ValidatedCatalog, "require", lose_readiness_after_require)
    try:
        with pytest.raises(sa.exc.SQLAlchemyError):
            store.validated_catalog()
        assert store._validated_catalog is proof
    finally:
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql("GRANT SELECT ON rag.dataset_snapshots TO rag_runtime")
    assert fixture.database.engine.pool.checkedout() == 0


def test_same_dsn_does_not_allow_another_database_object_to_claim_the_issuer(acl_db):
    fixture = acl_db
    store = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    other = Database(DatabaseSettings(dsn=url_string(fixture.urls.runtime), pool_size=1))
    try:
        with queries(other) as statements:
            with pytest.raises(EvidenceValidationError):
                PolicyResolver(other, fixture.snapshot, catalog=proof)
        assert catalog_queries(statements) == ()
        assert other.engine.pool.checkedout() == 0
    finally:
        other.close()


@pytest.mark.parametrize("changes", [
    {"snapshot_id": "other-ready-id"}, {"corpus_sha256": "b" * 64},
    {"catalog_sha256": "c" * 64}, {"document_count": len(ENTRIES) + 1},
])
def test_reuse_checks_the_complete_snapshot_before_any_catalog_scan(acl_db, changes):
    fixture = acl_db
    proof = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True).validated_catalog()
    with queries(fixture.database) as statements:
        with pytest.raises(SnapshotMismatch):
            PolicyResolver(fixture.database, fixture.snapshot.model_copy(update=changes), catalog=proof)
    assert catalog_queries(statements) == ()
    assert fixture.database.engine.pool.checkedout() == 0


def test_resolver_does_not_accept_a_duck_typed_or_unissued_catalog(acl_db):
    fixture = acl_db
    proof = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True).validated_catalog()
    called = []
    def self_attested(*_args, **_kwargs):
        called.append(True)
        return proof
    lookalike = SimpleNamespace(snapshot=proof.snapshot, entries=proof.entries,
        doc_ids=proof.doc_ids, sources=proof.sources, positions=proof.positions, require=self_attested)
    unissued = object.__new__(ValidatedCatalog)
    with queries(fixture.database) as statements:
        for supplied in (lookalike, unissued):
            with pytest.raises(EvidenceValidationError):
                PolicyResolver(fixture.database, fixture.snapshot, catalog=supplied)
    assert called == [] and catalog_queries(statements) == ()


@pytest.mark.parametrize("relation", ["dataset_snapshots", "policy_scopes"])
def test_ready_getter_and_reused_resolver_do_not_bypass_real_authoritative_sql_failure(acl_db, relation):
    fixture = acl_db
    seed_explicit_grants(fixture)
    store, proof, resolver = reused(fixture)
    principal = identity(fixture, "intern")
    context = resolver.resolve(principal)
    assert set(allowed_rows(fixture, context)) == {"general", "atlas"}
    with fixture.owner.begin() as connection:
        connection.exec_driver_sql(f"REVOKE SELECT ON rag.{relation} FROM rag_runtime")
    try:
        if relation == "dataset_snapshots":
            with pytest.raises(sa.exc.SQLAlchemyError):
                store.validated_catalog()
            with pytest.raises(PolicyUnavailable):
                PolicyResolver(fixture.database, fixture.snapshot, catalog=proof)
        else:
            # An intact metadata proof may still be obtained. It confers no
            # permission when the independent scope registry is unavailable.
            assert store.validated_catalog() is proof
        with pytest.raises(PolicyUnavailable):
            resolver.resolve(principal)
        with pytest.raises(PolicyUnavailable):
            resolver.assert_current(context)
    finally:
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql(f"GRANT SELECT ON rag.{relation} TO rag_runtime")
    assert fixture.database.engine.pool.checkedout() == 0


def test_metadata_capability_does_not_authorize_bodies_and_keeps_exact_permitted_reads(acl_db):
    fixture = acl_db
    seed_explicit_grants(fixture)
    store, proof, resolver = reused(fixture, doc_ids=("general", "finance-exec"))
    assert proof.positions["finance-exec"] == 2
    with fixture.database.transaction() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)).scalar_one() == 0
    context = resolver.resolve(identity(fixture, "intern"))
    assert allowed_rows(fixture, context) == ("general",)
    with queries(fixture.database) as statements:
        with pytest.raises(AccessDenied):
            store.read_passage(context, candidate(fixture, "finance-exec"), guard=resolver)
        assert not any("from rag.document_revisions" in statement.lower() for statement in statements)
    passage = store.read_passage(context, candidate(fixture, "general"), guard=resolver)
    assert passage.chunk.text == TEXTS[0][:8]
    assert passage.snapshot_id == fixture.snapshot.snapshot_id and passage.revision_id == ENTRIES[0].revision_id
    expansion = store.read_more(context, (passage,), guard=resolver, max_chars=24)
    assert expansion
    for evidence in expansion:
        assert evidence.chunk.text == TEXTS[0][evidence.chunk.start_char:evidence.chunk.end_char]
    assert fixture.database.engine.pool.checkedout() == 0


@pytest.mark.parametrize("mutation", ["grant", "compound_membership"])
def test_reused_metadata_cannot_keep_a_revoked_grant_or_compound_permission_alive(acl_db, mutation):
    fixture = acl_db
    seed_explicit_grants(fixture)
    store, proof, resolver = reused(fixture)
    name = "intern" if mutation == "grant" else "executive"
    principal = identity(fixture, name)
    original = resolver.resolve(principal)
    revoked_document = "general" if mutation == "grant" else "finance-exec"
    assert original.scope.permits(revoked_document)
    with fixture.owner.begin() as connection:
        if mutation == "grant":
            connection.execute(schema.document_grants.delete().where(
                schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id,
                schema.document_grants.c.doc_id == revoked_document))
        else:
            connection.execute(schema.group_memberships.delete().where(
                schema.group_memberships.c.user_id == principal.user_id,
                schema.group_memberships.c.group_id == "finance"))
    assert store.validated_catalog() is proof
    with pytest.raises(StaleContext):
        resolver.assert_current(original)
    assert allowed_rows(fixture, original) == ()
    current = resolver.resolve(principal)
    assert current.binding.policy_revision > original.binding.policy_revision
    assert not current.scope.permits(revoked_document)
    with pytest.raises(AccessDenied):
        store.read_passage(current, candidate(fixture, revoked_document), guard=resolver)


@pytest.mark.parametrize("mutation", ["logout", "expiry"])
def test_cached_catalog_and_scope_do_not_bypass_live_session_invalidation(acl_db, mutation):
    fixture = acl_db
    seed_explicit_grants(fixture)
    store, proof, resolver = reused(fixture)
    principal = identity(fixture, "intern")
    context = resolver.resolve(principal)
    if mutation == "logout":
        fixture.auth.logout(raw_token(fixture, "intern"), fixture.sessions["intern"].response.csrf_token)
    else:
        now = datetime.now(timezone.utc)
        with fixture.owner.begin() as connection:
            connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
                .values(created_at=now - timedelta(hours=1), expires_at=now - timedelta(seconds=1)))
    assert store.validated_catalog() is proof
    with pytest.raises(SessionExpired):
        resolver.resolve(principal)
    with pytest.raises(SessionExpired):
        store.read_passage(context, candidate(fixture), guard=resolver)
    assert allowed_rows(fixture, context) == ()


def test_reuse_keeps_resolver_local_scope_identity_and_rejects_forged_position_maps(acl_db):
    fixture = acl_db
    seed_explicit_grants(fixture)
    store, proof, first = reused(fixture)
    second = PolicyResolver(fixture.database, fixture.snapshot, catalog=proof)
    original = first.resolve(identity(fixture, "intern"))
    assert first._catalog.positions is second._catalog.positions is proof.positions
    assert first._catalog is not second._catalog
    with pytest.raises(EvidenceValidationError):
        second.assert_current(original)
    positions = dict(proof.positions)
    positions["general"], positions["finance-exec"] = positions["finance-exec"], positions["general"]
    forged_catalog = replace(first._catalog, positions=MappingProxyType(positions))
    forged = RequestContext(original.identity, replace(original.scope, _catalog=forged_catalog))
    assert forged.scope.permits("finance-exec")
    with pytest.raises(EvidenceValidationError):
        first.assert_current(forged)
    assert set(allowed_rows(fixture, forged)) == {"general", "atlas"}
    assert store.validated_catalog() is proof

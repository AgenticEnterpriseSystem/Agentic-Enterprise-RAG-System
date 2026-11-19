"""Real PostgreSQL paging/idle-transaction regressions, using disposable tiny ACL DBs."""
from contextlib import closing, contextmanager
import sqlite3
import time
from types import SimpleNamespace

import pytest
sa = pytest.importorskip("sqlalchemy", reason="Policy page checks require optional DB dependencies")

import src.access.policy as policy_module
from src.access.contracts import EvidenceValidationError, PolicyUnavailable, SessionExpired, StaleContext
from src.access.policy import PolicyResolver
from src.access.repository import GrantSpec, GroupSpec, MembershipSpec
from src.db import schema
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import (
    DOC_IDS, acl_db, allowed_rows, fixture_manifest, identity, repository, seed_explicit_grants,
)


@contextmanager
def transaction_depth(database):
    state = {"depth": 0}
    def begin(_connection): state["depth"] += 1
    def end(_connection): state["depth"] -= 1
    sa.event.listen(database.engine, "begin", begin)
    sa.event.listen(database.engine, "commit", end)
    sa.event.listen(database.engine, "rollback", end)
    try:
        yield state
    finally:
        sa.event.remove(database.engine, "begin", begin)
        sa.event.remove(database.engine, "commit", end)
        sa.event.remove(database.engine, "rollback", end)


def test_bitmap_work_runs_after_transaction_and_pool_checkout_are_closed(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    resolver = PolicyResolver(fixture.database, fixture.snapshot, cache_size=0)
    principal = identity(fixture, "intern")
    original = resolver._apply_grant_page
    applications = []
    with transaction_depth(fixture.database) as state:
        def apply(bits, documents):
            assert state["depth"] == 0
            assert fixture.database.engine.pool.checkedout() == 0
            applications.append(tuple(documents))
            original(bits, documents)
        monkeypatch.setattr(resolver, "_apply_grant_page", apply)
        context = resolver.resolve(principal)
        assert state["depth"] == 0
    assert applications and set(allowed_rows(fixture, context)) == {"atlas", "general"}


@pytest.mark.parametrize("page_size", [1, 2, 3])
def test_duplicate_grants_across_tiny_keyset_boundaries_preserve_exact_bitmap(acl_db, monkeypatch, page_size):
    fixture = acl_db
    base = fixture_manifest(fixture)
    user = fixture.users["direct_user"].user_id
    # Same document can be direct-granted and granted through multiple groups.
    # LIMIT may cut an equal-doc_id group; skipping its remaining duplicates is safe.
    manifest = base.model_copy(update={
        "groups": base.groups + (GroupSpec(group_id="extra-all", label="Synthetic all"),),
        "memberships": base.memberships + (MembershipSpec(user_id=user, group_id="extra-all"),),
        "grants": base.grants + (GrantSpec(doc_ids=DOC_IDS, group_id="extra-all"),),
    })
    repository(fixture).apply(manifest)
    monkeypatch.setattr(policy_module, "_GRANT_PAGE_SIZE", page_size)
    resolver = PolicyResolver(fixture.database, fixture.snapshot, cache_size=0)
    context = resolver.resolve(identity(fixture, "direct_user"))
    assert context.scope.bitmap == bytes([255, 255, 1])
    assert context.scope.allowed_count == len(DOC_IDS)
    assert set(allowed_rows(fixture, context)) == set(DOC_IDS)
    restricted = PolicyResolver(fixture.database, fixture.snapshot, doc_ids=("general", "boundary-eight"), cache_size=0)
    narrow = restricted.resolve(identity(fixture, "direct_user"))
    assert narrow.scope.allowed_count == 2
    assert set(allowed_rows(fixture, narrow)) == {"general", "boundary-eight"}


@pytest.mark.parametrize("mutation", ["grant", "membership", "session"])
def test_revocation_between_pages_fails_closed_without_registering_partial_scope(acl_db, monkeypatch, mutation):
    fixture = acl_db
    seed_explicit_grants(fixture)
    monkeypatch.setattr(policy_module, "_GRANT_PAGE_SIZE", 1)
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    principal = identity(fixture, "intern")
    original, calls = resolver._apply_grant_page, []
    def apply(bits, documents):
        original(bits, documents)
        calls.append(tuple(documents))
        if len(calls) != 1:
            return
        with fixture.owner.begin() as connection:
            if mutation == "grant":
                connection.execute(schema.document_grants.delete().where(schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id,
                    schema.document_grants.c.doc_id == "general"))
            elif mutation == "membership":
                connection.execute(schema.group_memberships.delete().where(schema.group_memberships.c.user_id == principal.user_id))
            else:
                connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id).values(revoked_at=sa.func.now()))
    monkeypatch.setattr(resolver, "_apply_grant_page", apply)
    with pytest.raises(SessionExpired if mutation == "session" else StaleContext):
        resolver.resolve(principal)
    assert len(calls) == 1 and not resolver._scope_cache
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.policy_scopes)).scalar_one() == 0


def test_revocation_after_last_page_is_rechecked_at_registration(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    principal = identity(fixture, "intern")
    original = resolver._apply_grant_page
    def apply(bits, documents):
        original(bits, documents)
        with fixture.owner.begin() as connection:
            connection.execute(schema.group_memberships.delete().where(schema.group_memberships.c.user_id == principal.user_id))
    monkeypatch.setattr(resolver, "_apply_grant_page", apply)
    with pytest.raises(StaleContext):
        resolver.resolve(principal)
    assert not resolver._scope_cache


def test_slow_bitmap_work_exceeding_low_idle_timeout_keeps_connection_healthy(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    principal = identity(fixture, "intern")
    database = Database(DatabaseSettings(dsn=url_string(fixture.urls.runtime), pool_size=1, idle_transaction_timeout_ms=100))
    pids = []
    sa.event.listen(database.engine, "connect", lambda connection, _record: pids.append(connection.info.backend_pid))
    try:
        resolver = PolicyResolver(database, fixture.snapshot, cache_size=0)
        monkeypatch.setattr(policy_module, "_GRANT_PAGE_SIZE", 1)
        original = resolver._apply_grant_page
        def apply(bits, documents):
            assert database.engine.pool.checkedout() == 0
            time.sleep(.30)  # Three times the configured DB idle-in-transaction timeout.
            original(bits, documents)
        monkeypatch.setattr(resolver, "_apply_grant_page", apply)
        context = resolver.resolve(principal)
        with database.transaction(context) as connection:
            assert connection.exec_driver_sql("SHOW idle_in_transaction_session_timeout").scalar_one() == "100ms"
            assert connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one() == pids[0]
        assert len(pids) == 1, "No timeout disconnect/reconnect may mask an open transaction"
        assert context.scope.allowed_count == 2
    finally:
        database.close()


def observation_fixture(monkeypatch):
    resolver = object.__new__(PolicyResolver)
    resolver._operator_bitmap = bytes([1])
    resolver._catalog = SimpleNamespace(positions={"synthetic": 0})
    resolver.snapshot = SimpleNamespace(document_count=1)
    events = []
    monkeypatch.setattr(policy_module, "emit", lambda stage, event, **fields: events.append({"stage": stage, "event": event, **fields}))
    return resolver, events


def test_many_grant_pages_emit_constant_size_separate_cumulative_observations(monkeypatch):
    resolver, events = observation_fixture(monkeypatch)
    monkeypatch.setattr(resolver, "_grant_pages", lambda *_: iter([("synthetic",)] * 200))
    assert resolver._resolve_bitmap(None, 0) == bytes([1])
    assert len(events) == 4  # Never exhaust the request event budget with page logs.
    for stage in ("policy.grants", "policy.bitmap"):
        terminal = next(e for e in events if e["stage"] == stage and e["event"] == "end")
        assert terminal["status"] == "ok" and terminal["duration_is_cumulative"] is True
        assert terminal["grant_page_count"] == terminal["grant_row_count"] == 200
        assert terminal["duration_ms"] >= 0


@pytest.mark.parametrize("failed_stage", ["policy.grants", "policy.bitmap"])
def test_aggregate_observation_preserves_exact_failed_stage_and_original_error(monkeypatch, failed_stage):
    resolver, events = observation_fixture(monkeypatch)
    error = PolicyUnavailable("Synthetic unavailable") if failed_stage == "policy.grants" else EvidenceValidationError("Synthetic bad bitmap input")
    def pages(*_):
        if failed_stage == "policy.grants":
            raise error
        yield ("synthetic",)
    monkeypatch.setattr(resolver, "_grant_pages", pages)
    if failed_stage == "policy.bitmap":
        def apply(*_): raise error
        monkeypatch.setattr(resolver, "_apply_grant_page", apply)
    with pytest.raises(type(error)) as caught:
        resolver._resolve_bitmap(None, 0)
    assert caught.value is error
    failed = [e for e in events if e["event"] == "error"]
    assert len(events) == 4 and len(failed) == 1
    assert failed[0]["stage"] == failed_stage and failed[0]["error"] is error
    assert all(e["status"] == "cancelled" for e in events if e["event"] == "end")


# Pre-optimization query, retained as an independent SQL equivalence oracle.
# These in-memory tests execute both SQL statements, rather than reimplementing
# their selection rules in a mock. PostgreSQL plan/timing checks remain separate.
_UNBOUNDED_GRANT_DOCUMENTS_SQL = """
WITH effective_groups AS MATERIALIZED (
 SELECT membership.group_id
 FROM rag.group_memberships membership
 WHERE membership.user_id=:user_id
 AND NOT EXISTS (
  SELECT 1 FROM rag.group_requirements requirement
  WHERE requirement.group_id=membership.group_id
  AND NOT EXISTS (
   SELECT 1 FROM rag.group_memberships required_membership
   WHERE required_membership.user_id=:user_id
   AND required_membership.group_id=requirement.required_group_id
  )
 )
), authorized_documents AS (
 SELECT grant_row.doc_id FROM rag.document_grants grant_row
 WHERE grant_row.user_id=:user_id AND grant_row.snapshot_id=:snapshot
 UNION ALL
 SELECT grant_row.doc_id FROM effective_groups effective
 JOIN rag.document_grants grant_row ON grant_row.group_id=effective.group_id
 WHERE grant_row.group_id IS NOT NULL AND grant_row.snapshot_id=:snapshot
)
SELECT authorized.doc_id FROM authorized_documents authorized
WHERE (CAST(:after AS text) IS NULL OR authorized.doc_id > CAST(:after AS text))
ORDER BY authorized.doc_id
LIMIT :page_size
"""


@contextmanager
def grant_sql_fixture(complete_requirements):
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("ATTACH DATABASE ':memory:' AS rag")
        connection.executescript("""
            CREATE TABLE rag.document_grants(snapshot_id TEXT, doc_id TEXT, user_id TEXT, group_id TEXT);
            CREATE TABLE rag.group_memberships(user_id TEXT, group_id TEXT);
            CREATE TABLE rag.group_requirements(group_id TEXT, required_group_id TEXT);
        """)
        groups = ["public", "team-a", "team-b", "restricted", "required-a"]
        if complete_requirements:
            groups.append("required-b")
        connection.executemany("INSERT INTO rag.group_memberships VALUES (?,?)", [("user", g) for g in groups])
        connection.executemany("INSERT INTO rag.group_requirements VALUES (?,?)",
            [("restricted", "required-a"), ("restricted", "required-b")])
        direct = {f"doc-{n:04d}" for n in range(0, 256, 2)}
        group_sets = {
            "public": {f"doc-{n:04d}" for n in range(64)},
            "team-a": {f"doc-{n:04d}" for n in range(0, 256, 3)},
            "team-b": {f"doc-{n:04d}" for n in range(100, 256)},
            "restricted": {"restricted-only", "doc-0000"},
            "nonmember": {"aaa-nonmember"},
        }
        grants = [("snapshot", doc, "user", None) for doc in direct]
        grants += [("snapshot", doc, None, group) for group, docs in group_sets.items() for doc in docs]
        grants += [("other-snapshot", "aaa-wrong-snapshot", "user", None),
                   ("other-snapshot", "aaa-wrong-group-snapshot", None, "public"),
                   ("snapshot", "aaa-other-user", "other-user", None)]
        connection.executemany("INSERT INTO rag.document_grants VALUES (?,?,?,?)", grants)
        expected = direct | group_sets["public"] | group_sets["team-a"] | group_sets["team-b"]
        if complete_requirements:
            expected |= group_sets["restricted"]
        yield connection, expected


@pytest.mark.parametrize("page_size", [1, 2, 7, 31])
@pytest.mark.parametrize("after", [None, "doc-0027", "doc-0100", "doc-0255", "zz"])
@pytest.mark.parametrize("complete_requirements", [False, True])
def test_branch_bounded_page_matches_old_sql_with_overlap_and_keyset_boundaries(
        page_size, after, complete_requirements):
    with grant_sql_fixture(complete_requirements) as (connection, _expected):
        parameters = {"snapshot": "snapshot", "user_id": "user", "after": after, "page_size": page_size}
        actual = connection.execute(policy_module._GRANT_DOCUMENTS_SQL, parameters).fetchall()
        old = connection.execute(_UNBOUNDED_GRANT_DOCUMENTS_SQL, parameters).fetchall()
        assert actual == old
        assert len(actual) <= page_size
        assert all(after is None or row[0] > after for row in actual)


@pytest.mark.parametrize("page_size", [1, 2, 7, 31])
@pytest.mark.parametrize("complete_requirements", [False, True])
def test_branch_bounded_full_walk_keeps_exact_grants_across_duplicate_cuts(page_size, complete_requirements):
    with grant_sql_fixture(complete_requirements) as (connection, expected):
        parameters = {"snapshot": "snapshot", "user_id": "user", "after": None, "page_size": page_size}
        seen = set()
        for _ in range(len(expected) + 1):
            rows = connection.execute(policy_module._GRANT_DOCUMENTS_SQL, parameters).fetchall()
            assert rows == connection.execute(_UNBOUNDED_GRANT_DOCUMENTS_SQL, parameters).fetchall()
            seen.update(row[0] for row in rows)
            if len(rows) < page_size:
                break
            assert parameters["after"] is None or rows[-1][0] > parameters["after"]
            parameters["after"] = rows[-1][0]
        else:
            pytest.fail("Grant cursor failed to make bounded progress")
        assert seen == expected

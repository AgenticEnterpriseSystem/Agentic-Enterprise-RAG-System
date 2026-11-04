"""Bounded immutable decision reuse with real, uncached PostgreSQL authority."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Barrier
from uuid import uuid4

import pytest
import sqlalchemy as sa

from src.access.contracts import PolicyUnavailable, SessionExpired, StaleContext
from src.access.policy import BitmapScope, PolicyResolver
from src.auth.accounts import disable_account, reset_password
from src.db import schema
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import (
    PASSWORD, acl_db, allowed_rows, identity, raw_token, seed_documents, seed_explicit_grants,
)


@contextmanager
def statements(database):
    queries = []

    def watch(_connection, _cursor, statement, _parameters, _context, _many):
        queries.append(statement)

    sa.event.listen(database.engine, "before_cursor_execute", watch)
    try:
        yield queries
    finally:
        sa.event.remove(database.engine, "before_cursor_execute", watch)


def scans(queries):
    return sum("WITH effective_groups AS MATERIALIZED" in query for query in queries)


def resolver_for(fixture, **options):
    seed_explicit_grants(fixture)
    return PolicyResolver(fixture.database, fixture.snapshot, **options)


@pytest.mark.parametrize("capacity", [-1, 33, True, False, 1.5, "32", None])
def test_cache_capacity_rejects_invalid_values_before_catalog_io(acl_db, monkeypatch, capacity):
    def unexpected_catalog(_self):
        pytest.fail("Invalid capacity reached database/catalog work")

    monkeypatch.setattr(PolicyResolver, "_load_catalog", unexpected_catalog)
    with pytest.raises(ValueError):
        PolicyResolver(acl_db.database, acl_db.snapshot, cache_size=capacity)


def test_warm_resolve_reuses_only_bitmap_and_keeps_current_identity_and_live_sql(acl_db):
    fixture = acl_db
    resolver = resolver_for(fixture)
    original_identity = identity(fixture, "intern")
    with statements(fixture.database) as first_sql:
        first = resolver.resolve(original_identity)
    assert scans(first_sql) == 1
    fixture.auth.clock = lambda: datetime.now(timezone.utc) + timedelta(seconds=5)
    current_identity = fixture.auth.authenticate(raw_token(fixture, "intern"), touch=True).identity
    assert current_identity.expires_at > original_identity.expires_at
    with statements(fixture.database) as warm_sql:
        warm = resolver.resolve(current_identity)
    assert scans(warm_sql) == 0
    assert warm is not first and warm.scope is first.scope
    assert warm.identity == current_identity and warm.identity.expires_at > first.identity.expires_at
    assert warm.binding == first.binding
    assert sum("FROM rag.sessions JOIN rag.users" in query for query in warm_sql) >= 3
    assert sum("FROM rag.policy_state" in query for query in warm_sql) >= 3
    assert any("FROM rag.policy_scopes" in query for query in warm_sql)
    assert set(allowed_rows(fixture, warm)) == {"general", "atlas"}
    assert all(isinstance(value, BitmapScope) for value in resolver._scope_cache.values())


def test_disabled_cache_is_exact_baseline_and_keeps_no_decisions(acl_db):
    fixture = acl_db
    resolver = resolver_for(fixture, cache_size=0)
    principal = identity(fixture, "intern")
    with statements(fixture.database) as queries:
        first, second = resolver.resolve(principal), resolver.resolve(principal)
    assert scans(queries) == 2
    assert first.scope is not second.scope and first.binding == second.binding
    assert first.scope.bitmap == second.scope.bitmap and not resolver._scope_cache


def test_lru_eviction_is_bounded_and_recomputes_the_evicted_principal(acl_db):
    fixture = acl_db
    resolver = resolver_for(fixture, cache_size=2)
    principals = {name: identity(fixture, name) for name in ("intern", "executive", "labels_only")}
    with statements(fixture.database) as queries:
        intern = resolver.resolve(principals["intern"])
        executive = resolver.resolve(principals["executive"])
        assert resolver.resolve(principals["intern"]).scope is intern.scope
        empty = resolver.resolve(principals["labels_only"])
        assert not empty.scope.allowed_count
        assert len(resolver._scope_cache) == 2 and scans(queries) == 3
        assert resolver.resolve(principals["intern"]).scope is intern.scope
        again = resolver.resolve(principals["executive"])
    assert scans(queries) == 4 and again.scope is not executive.scope
    assert again.binding == executive.binding and len(resolver._scope_cache) == 2
    assert set(allowed_rows(fixture, again)) == {"general", "finance-exec"}


@pytest.mark.parametrize("mutation", ["document_grants", "group_memberships", "group_requirements"])
def test_live_policy_epoch_invalidates_cached_grant_membership_and_compound_decisions(acl_db, mutation):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "executive")
    old = resolver.resolve(principal)
    assert old.scope.permits("finance-exec")
    with fixture.owner.begin() as connection:
        if mutation == "document_grants":
            connection.execute(schema.document_grants.delete().where(
                schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id,
                schema.document_grants.c.doc_id == "finance-exec"))
        elif mutation == "group_memberships":
            connection.execute(schema.group_memberships.delete().where(
                schema.group_memberships.c.user_id == principal.user_id,
                schema.group_memberships.c.group_id == "finance"))
        else:
            connection.execute(schema.group_requirements.update().where(
                schema.group_requirements.c.group_id == "finance_exec",
                schema.group_requirements.c.required_group_id == "finance").values(required_group_id="atlas-team"))
    with pytest.raises(StaleContext):
        resolver.assert_current(old)
    with statements(fixture.database) as queries:
        fresh = resolver.resolve(principal)
    assert scans(queries) == 1 and fresh.scope is not old.scope
    assert fresh.binding.policy_revision > old.binding.policy_revision
    assert not fresh.scope.permits("finance-exec")
    assert allowed_rows(fixture, fresh) == ("general",)


@pytest.mark.parametrize("mutation", ["logout", "reset", "disable", "expiry"])
def test_cached_decision_never_survives_live_session_or_account_invalidation(acl_db, mutation):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "intern")
    resolver.resolve(principal)
    if mutation == "logout":
        fixture.auth.logout(raw_token(fixture, "intern"), fixture.sessions["intern"].response.csrf_token)
    elif mutation == "reset":
        reset_password(fixture.logins["intern"], "Changed cache fixture password",
            url=url_string(fixture.urls.migration), hasher=fixture.hasher)
    elif mutation == "disable":
        disable_account(fixture.logins["intern"], url=url_string(fixture.urls.migration))
    else:
        now = datetime.now(timezone.utc)
        with fixture.owner.begin() as connection:
            connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
                .values(created_at=now - timedelta(hours=1), expires_at=now - timedelta(seconds=1)))
    with statements(fixture.database) as queries:
        with pytest.raises(SessionExpired):
            resolver.resolve(principal)
    assert scans(queries) == 0


@pytest.mark.parametrize("relation", ["policy_state", "policy_scopes"])
def test_cached_scope_fails_closed_when_authoritative_policy_or_registry_sql_is_unavailable(acl_db, relation):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "intern")
    resolver.resolve(principal)
    with fixture.owner.begin() as connection:
        connection.exec_driver_sql(f"REVOKE SELECT ON rag.{relation} FROM rag_runtime")
    try:
        with pytest.raises(PolicyUnavailable) as error:
            resolver.resolve(principal)
        assert "password" not in str(error.value).lower()
    finally:
        with fixture.owner.begin() as connection:
            connection.exec_driver_sql(f"GRANT SELECT ON rag.{relation} TO rag_runtime")


@pytest.mark.parametrize("mutation", ["policy", "logout", "registry_unavailable"])
def test_authority_change_immediately_after_genuine_cache_hit_cannot_escape_final_guard(acl_db, monkeypatch, mutation):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "intern")
    first = resolver.resolve(principal)
    lookup = resolver._cached_scope

    def changed_after_hit(key):
        scope = lookup(key)
        assert scope is first.scope
        if mutation == "logout":
            # Use the owner connection to avoid nesting authentication in the
            # fixture's single runtime connection while resolve holds it.
            with fixture.owner.begin() as connection:
                connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
                    .values(revoked_at=datetime.now(timezone.utc)))
        elif mutation == "policy":
            with fixture.owner.begin() as connection:
                connection.execute(schema.document_grants.delete().where(
                    schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id,
                    schema.document_grants.c.doc_id == "general"))
        else:
            with fixture.owner.begin() as connection:
                connection.exec_driver_sql("REVOKE SELECT ON rag.policy_scopes FROM rag_runtime")
        return scope

    monkeypatch.setattr(resolver, "_cached_scope", changed_after_hit)
    expected = SessionExpired if mutation == "logout" else StaleContext if mutation == "policy" else PolicyUnavailable
    try:
        with pytest.raises(expected):
            resolver.resolve(principal)
    finally:
        if mutation == "registry_unavailable":
            with fixture.owner.begin() as connection:
                connection.exec_driver_sql("GRANT SELECT ON rag.policy_scopes TO rag_runtime")


def test_cache_hit_expiry_after_registry_io_is_rechecked_before_release(acl_db):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "intern")
    resolver.resolve(principal)
    now = [datetime.now(timezone.utc)]
    resolver.clock = lambda: now[0]

    def expire(_connection, _cursor, statement, _parameters, _context, _many):
        if "FROM rag.policy_scopes" in statement:
            now[0] = principal.expires_at + timedelta(seconds=1)

    sa.event.listen(fixture.database.engine, "after_cursor_execute", expire)
    try:
        with pytest.raises(SessionExpired):
            resolver.resolve(principal)
    finally:
        sa.event.remove(fixture.database.engine, "after_cursor_execute", expire)


def test_failed_final_guard_never_populates_a_new_cache_entry(acl_db, monkeypatch):
    fixture = acl_db
    resolver = resolver_for(fixture)
    principal = identity(fixture, "intern")
    guard = resolver.assert_current

    def rejected(_context):
        raise PolicyUnavailable("Synthetic final authority check unavailable")

    monkeypatch.setattr(resolver, "assert_current", rejected)
    with statements(fixture.database) as queries:
        with pytest.raises(PolicyUnavailable):
            resolver.resolve(principal)
        assert not resolver._scope_cache
        monkeypatch.setattr(resolver, "assert_current", guard)
        fresh = resolver.resolve(principal)
    assert scans(queries) == 2 and fresh.scope.allowed_count == 2


def test_user_session_operator_and_snapshot_boundaries_never_share_cached_authority(acl_db):
    fixture = acl_db
    resolver = resolver_for(fixture)
    intern_identity = identity(fixture, "intern")
    first = resolver.resolve(intern_identity)
    empty = resolver.resolve(identity(fixture, "labels_only"))
    assert empty.scope.allowed_count == 0
    other_session = fixture.auth.login(fixture.logins["intern"], PASSWORD, client_ip="127.0.0.1")
    current_identity = fixture.auth.authenticate(other_session.token, touch=False).identity
    current = resolver.resolve(current_identity)
    assert current.identity.session_id != first.identity.session_id
    assert current.scope is not first.scope and current.binding.scope_id != first.binding.scope_id
    assert set(allowed_rows(fixture, current)) == {"general", "atlas"}
    restricted = PolicyResolver(fixture.database, fixture.snapshot, doc_ids=("general",))
    narrowed = restricted.resolve(current_identity)
    assert narrowed.scope.bitmap != current.scope.bitmap
    assert narrowed.binding.scope_id != current.binding.scope_id
    assert allowed_rows(fixture, narrowed) == ("general",)
    snapshot = fixture.snapshot.model_copy(update={"snapshot_id": "cache-other-" + uuid4().hex})
    with fixture.owner.begin() as connection:
        seed_documents(connection, snapshot)
        connection.execute(schema.document_grants.insert().values(snapshot_id=snapshot.snapshot_id,
            doc_id="direct", user_id=current_identity.user_id, origin="synthetic"))
    other = PolicyResolver(fixture.database, snapshot).resolve(current_identity)
    assert other.binding.snapshot == snapshot and other.scope.permits("direct")
    assert not other.scope.permits("general") and other.scope is not current.scope
    assert allowed_rows(fixture, other) == ("direct",)


def test_concurrent_misses_for_distinct_principals_never_mix_or_widen_bitmaps(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    database = Database(DatabaseSettings(dsn=url_string(fixture.urls.runtime), pool_size=4))
    try:
        resolver = PolicyResolver(database, fixture.snapshot, cache_size=2)
        principals = [identity(fixture, name) for name in ("intern", "executive", "intern", "executive")]
        lookup = resolver._cached_scope
        ready = Barrier(4, timeout=10)

        def simultaneous_miss(key):
            scope = lookup(key)
            assert scope is None
            ready.wait()
            return scope

        monkeypatch.setattr(resolver, "_cached_scope", simultaneous_miss)
        with ThreadPoolExecutor(max_workers=4) as workers:
            contexts = list(workers.map(resolver.resolve, principals))
        monkeypatch.setattr(resolver, "_cached_scope", lookup)
        assert len(resolver._scope_cache) == 2
        for index, context in enumerate(contexts):
            expected = {"general", "atlas"} if index % 2 == 0 else {"general", "finance-exec"}
            assert set(allowed_rows(fixture, context)) == expected
            again = resolver.resolve(context.identity)
            assert again.scope.bitmap == context.scope.bitmap and set(allowed_rows(fixture, again)) == expected
    finally:
        database.close()

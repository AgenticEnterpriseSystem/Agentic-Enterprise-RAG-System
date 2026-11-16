"""Real PostgreSQL buffered-page lifetime/idle regressions; synthetic ACL DBs only.

Injected pauses establish the transaction-lifetime failure mechanism; they do not
claim to reproduce the exact garbage-collection pause of the historical request.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest
sa = pytest.importorskip("sqlalchemy", reason="Buffered policy checks require optional DB dependencies")

import src.access.policy as policy_module
from src.access.contracts import PolicyUnavailable, SessionExpired, StaleContext
from src.access.policy import PolicyResolver
from src.db import schema
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import acl_db, identity, seed_explicit_grants
from tests.test_policy_grant_pages import transaction_depth


def test_buffered_page_survives_commit_discard_all_and_concurrent_same_connection_checkout(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    principal = identity(fixture, "intern")
    resolver = PolicyResolver(fixture.database, fixture.snapshot, cache_size=0)
    original, pages = resolver._materialize_grant_page, []
    with transaction_depth(fixture.database) as state:
        def materialize(result):
            assert state["depth"] == 0 and fixture.database.engine.pool.checkedout() == 0
            assert not getattr(result.cursor, "name", None), "Named/server cursors cannot escape their transaction"
            buffered_pid = result.cursor.connection.info.backend_pid
            entered = threading.Event()
            def concurrent_checkout():
                with fixture.database.transaction() as connection:
                    pid = connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
                    entered.set()
                    connection.exec_driver_sql("SELECT pg_sleep(.03)")
                    return pid
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(concurrent_checkout)
                assert entered.wait(2)
                # The original transaction has committed and returned to pool;
                # DISCARD ALL/role checkout/another transaction already occurred.
                documents = original(result)
                assert future.result(timeout=2) == buffered_pid
            assert result.closed and state["depth"] == 0
            pages.append(documents)
            return documents
        monkeypatch.setattr(resolver, "_materialize_grant_page", materialize)
        context = resolver.resolve(principal)
    assert pages == [("atlas", "general")]
    assert context.scope.allowed_count == 2


def test_materialization_pause_outside_transaction_exceeds_idle_timeout_without_reconnect(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    principal = identity(fixture, "intern")
    database = Database(DatabaseSettings(dsn=url_string(fixture.urls.runtime), pool_size=1, idle_transaction_timeout_ms=100))
    pids = []
    sa.event.listen(database.engine, "connect", lambda connection, _record: pids.append(connection.info.backend_pid))
    try:
        resolver = PolicyResolver(database, fixture.snapshot, cache_size=0)
        original = resolver._materialize_grant_page
        def slow(result):
            assert database.engine.pool.checkedout() == 0
            time.sleep(.30)
            return original(result)
        monkeypatch.setattr(resolver, "_materialize_grant_page", slow)
        context = resolver.resolve(principal)
        with database.transaction(context) as connection:
            assert connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one() == pids[0]
            assert connection.exec_driver_sql("SHOW idle_in_transaction_session_timeout").scalar_one() == "100ms"
        assert len(pids) == 1 and context.scope.allowed_count == 2
    finally:
        database.close()


def test_control_same_pause_inside_transaction_hits_real_idle_timeout_with_safe_sqlstate(acl_db):
    fixture = acl_db
    seed_explicit_grants(fixture)
    principal = identity(fixture, "intern")
    database = Database(DatabaseSettings(dsn=url_string(fixture.urls.runtime), pool_size=1, idle_transaction_timeout_ms=100))
    try:
        resolver = PolicyResolver(database, fixture.snapshot, cache_size=0)
        with pytest.raises(PolicyUnavailable) as caught:
            with resolver._transaction() as connection:
                epoch = resolver._identity(connection, principal)
                result = connection.execute(sa.text(policy_module._GRANT_DOCUMENTS_SQL), {
                    "snapshot": fixture.snapshot.snapshot_id, "user_id": principal.user_id, "after": None, "page_size": 8192})
                time.sleep(.30)  # Old conversion/later-check sequence held this TX open.
                tuple(result.scalars().all())
                resolver._identity(connection, principal, epoch=epoch)
        assert caught.value.database_sqlstate == "25P03"
        assert str(caught.value) == "Authoritative policy is unavailable"
    finally:
        database.close()


@pytest.mark.parametrize("mutation", ["grant", "session", "empty_page_session"])
def test_authority_changed_during_materialization_fails_before_any_bitmap_application(acl_db, monkeypatch, mutation):
    fixture = acl_db
    seed_explicit_grants(fixture)
    principal = identity(fixture, "labels_only" if mutation == "empty_page_session" else "intern")
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    original, checked = resolver._materialize_grant_page, []
    def materialize(result):
        documents = original(result)
        checked.append(documents)
        assert not documents if mutation == "empty_page_session" else documents
        with fixture.owner.begin() as connection:
            if mutation == "grant":
                connection.execute(schema.document_grants.delete().where(
                    schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id,
                    schema.document_grants.c.doc_id == "general"))
            else:
                connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
                    .values(revoked_at=sa.func.now()))
        return documents
    def forbidden_apply(*_): pytest.fail("Stale buffered data reached bitmap application")
    monkeypatch.setattr(resolver, "_materialize_grant_page", materialize)
    monkeypatch.setattr(resolver, "_apply_grant_page", forbidden_apply)
    with pytest.raises(StaleContext if mutation == "grant" else SessionExpired):
        resolver.resolve(principal)
    assert len(checked) == 1 and not resolver._scope_cache
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.policy_scopes)).scalar_one() == 0


def test_materialization_driver_failure_keeps_fixed_phase_state_and_closes_original_cursor(acl_db, monkeypatch):
    fixture = acl_db
    seed_explicit_grants(fixture)
    resolver = PolicyResolver(fixture.database, fixture.snapshot, cache_size=0)
    original, results, events = resolver._materialize_grant_page, [], []
    monkeypatch.setattr(policy_module, "emit", lambda stage, event, **fields: events.append({"stage": stage, "event": event, **fields}))
    def fail(result):
        results.append(result)
        class FailingResult:
            def __enter__(self): return self
            def __exit__(self, *_): result.close()
            def scalars(self): return self
            def all(self):
                raise sa.exc.OperationalError("PRIVATE SQL", {"secret": "PRIVATE VALUE"}, SimpleNamespace(sqlstate="25P03"))
        return original(FailingResult())
    monkeypatch.setattr(resolver, "_materialize_grant_page", fail)
    with pytest.raises(PolicyUnavailable) as caught:
        resolver.resolve(identity(fixture, "intern"))
    assert caught.value.database_phase == "materialize" and caught.value.database_sqlstate == "25P03"
    assert results[0].closed and not resolver._scope_cache
    failure = next(e for e in events if e["stage"] == "policy.grants" and e["event"] == "error")
    assert failure["database_phase"] == "materialize" and failure["database_sqlstate"] == "25P03"
    assert all("PRIVATE" not in str(v) for e in events for v in e.values())

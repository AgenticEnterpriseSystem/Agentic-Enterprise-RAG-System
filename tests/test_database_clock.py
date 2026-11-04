"""Deterministic application-host/DB-clock split; root executes actual PG.

The injected module-local datetime subclass models a Mac host clock120s ahead
or behind its Docker DB. psycopg, the shared datetime type and the new DB-clock
helper remain untouched. This proves a supported architecture boundary; it does
not attribute earlier intermittent failures to skew without evidence.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import sqlalchemy as sa

from src.access.contracts import PolicyUnavailable, SessionExpired
from src.access.policy import PolicyResolver
import src.access.policy as policy_module
from src.access.repository import verify_benchmark
import src.access.repository as repository_module
from src.auth.sessions import AuthError, AuthService
import src.auth.sessions as session_module
from src.db import schema
from tests.db_support import pg_urls
from tests.test_access_policy import (
    PASSWORD, acl_db, provision_benchmark, seed_explicit_grants,
)


REAL_DATETIME = datetime
UTC = timezone.utc


@pytest.fixture
def clock_data(acl_db):
    seed_explicit_grants(acl_db)
    assert acl_db.database.settings.pool_size == 1
    return acl_db


def skew_host(monkeypatch, seconds):
    class SkewedDateTime(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            actual = REAL_DATETIME.now(tz)
            return cls.fromtimestamp(actual.timestamp() + seconds, tz)

    for module in (session_module, policy_module, repository_module):
        monkeypatch.setattr(module, 'datetime', SkewedDateTime)
    return SkewedDateTime


def db_now(data):
    with data.database.transaction() as connection:
        return connection.execute(sa.select(sa.func.clock_timestamp())).scalar_one()


def service(data, *, clock=None):
    settings = data.auth.settings.model_copy(update={'idle_seconds': 60, 'absolute_seconds': 900})
    return AuthService(data.database, settings, clock=clock, hasher=data.hasher)


def login(data, auth, **kwargs):
    return auth.login(data.logins['intern'], PASSWORD, client_ip='127.0.0.1', **kwargs)


def stored(data, session_id):
    with data.owner.connect() as connection:
        return connection.execute(sa.select(schema.sessions).where(
            schema.sessions.c.session_id == session_id)).mappings().one()


def expired(call):
    with pytest.raises(AuthError) as failure:
        call()
    assert failure.value.code == 'session_expired' and failure.value.status == 401


@pytest.mark.parametrize('offset', [120, -120], ids=['host_ahead_120s', 'host_behind_120s'])
def test_default_session_registration_ignores_application_host_skew(clock_data, monkeypatch, offset):
    """Copy/run this node against old src: genuine42501, never a new-module import."""
    data = clock_data
    skew_host(monkeypatch, offset)
    auth = service(data)
    before = db_now(data)
    grant = login(data, auth)
    principal = auth.authenticate(grant.token, touch=False).identity
    states = []

    def sql_error(context):
        state = getattr(context.original_exception, 'sqlstate', None)
        if state:
            states.append(state)

    sa.event.listen(data.database.engine, 'handle_error', sql_error)
    try:
        try:
            context = PolicyResolver(data.database, data.snapshot).resolve(principal)
        except PolicyUnavailable:
            # Old auth+policy agree with each other's skewed host timestamp,
            # then the unmodified PostgreSQL scope trigger denies that session.
            if '42501' in states:
                pytest.fail('Default host-skewed session reached unchanged scope trigger42501')
            raise
    finally:
        sa.event.remove(data.database.engine, 'handle_error', sql_error)
    after = db_now(data)
    row = stored(data, principal.session_id)
    assert before <= row['created_at'] <= after
    assert row['expires_at'] - row['created_at'] == timedelta(seconds=60)
    assert principal.expires_at == row['expires_at'] and states == []
    assert {doc for doc in ('general', 'atlas', 'finance-exec') if context.scope.permits(doc)} == {'general', 'atlas'}


@pytest.mark.parametrize('offset', [120, -120], ids=['host_ahead_120s', 'host_behind_120s'])
def test_default_touch_and_logout_use_fresh_database_time_with_one_pool_slot(clock_data, monkeypatch, offset):
    data = clock_data
    skew_host(monkeypatch, offset)
    auth = service(data)
    grant = login(data, auth)
    principal = auth.authenticate(grant.token, touch=False).identity
    now = db_now(data)
    with data.owner.begin() as connection:
        connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
            .values(created_at=now-timedelta(seconds=300), expires_at=now+timedelta(seconds=10)))
    before = db_now(data)
    current = auth.authenticate(grant.token, touch=True)
    after = db_now(data)
    row = stored(data, principal.session_id)
    assert before <= row['expires_at'] - timedelta(seconds=60) <= after
    assert current.identity.expires_at == row['expires_at']
    before = db_now(data)
    auth.logout(grant.token, current.response.csrf_token)
    after = db_now(data)
    assert before <= stored(data, principal.session_id)['revoked_at'] <= after
    expired(lambda: auth.authenticate(grant.token, touch=True))


def test_host_behind_cannot_touch_or_resolve_a_session_already_expired_in_database(clock_data, monkeypatch):
    data = clock_data
    skew_host(monkeypatch, -120)
    auth = service(data)
    grant = login(data, auth)
    principal = auth.authenticate(grant.token, touch=False).identity
    now = db_now(data)
    with data.owner.begin() as connection:
        connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
            .values(created_at=now-timedelta(seconds=300), expires_at=now-timedelta(seconds=1)))
    expired(lambda: auth.authenticate(grant.token, touch=True))
    with pytest.raises(SessionExpired):
        PolicyResolver(data.database, data.snapshot).resolve(principal)
    assert stored(data, principal.session_id)['expires_at'] < now


@pytest.mark.parametrize('offset', [120, -120])
def test_default_benchmark_writer_and_scope_registration_share_database_time(clock_data, monkeypatch, tmp_path, offset):
    data = clock_data
    path = tmp_path / ('benchmark-' + uuid4().hex + '.json')
    provision_benchmark(data, path)
    skew_host(monkeypatch, offset)
    before = db_now(data)
    principal = verify_benchmark(path, data.database, hasher=data.hasher, ttl_seconds=60)
    context = PolicyResolver(data.database, data.snapshot).resolve(principal)
    after = db_now(data)
    row = stored(data, principal.session_id)
    assert before <= row['created_at'] <= after
    assert row['expires_at'] - row['created_at'] == timedelta(seconds=60)
    assert context.identity == principal and context.scope.allowed_count == data.snapshot.document_count


@pytest.mark.parametrize('injection', ['constructor', 'later_assignment'])
def test_explicit_auth_clock_remains_authoritative_for_creation_touch_and_logout(clock_data, injection):
    data = clock_data
    clock = SimpleNamespace(now=REAL_DATETIME(2030, 1, 1, tzinfo=UTC))
    callback = lambda: clock.now
    auth = service(data, clock=callback) if injection == 'constructor' else service(data)
    if injection == 'later_assignment':
        auth.clock = callback
    grant = login(data, auth)
    principal = auth.authenticate(grant.token, touch=False).identity
    assert stored(data, principal.session_id)['created_at'] == clock.now
    clock.now += timedelta(seconds=5)
    current = auth.authenticate(grant.token, touch=True)
    assert current.identity.expires_at == clock.now + timedelta(seconds=60)
    auth.logout(grant.token, current.response.csrf_token)
    assert stored(data, principal.session_id)['revoked_at'] == clock.now


@pytest.mark.parametrize('injection', ['constructor', 'later_assignment'])
def test_explicit_policy_clock_override_is_not_replaced_by_database_time(clock_data, injection):
    data = clock_data
    auth = service(data)
    grant = login(data, auth)
    principal = auth.authenticate(grant.token, touch=False).identity
    callback = lambda: db_reference + timedelta(seconds=120)
    db_reference = db_now(data)
    resolver = PolicyResolver(data.database, data.snapshot, clock=callback) if injection == 'constructor' else PolicyResolver(data.database, data.snapshot)
    if injection == 'later_assignment':
        resolver.clock = callback
    with pytest.raises(SessionExpired):
        resolver.resolve(principal)


def test_explicit_benchmark_clock_is_still_honored(clock_data, tmp_path):
    data = clock_data
    path = tmp_path / 'explicit-benchmark.json'
    provision_benchmark(data, path)
    target = REAL_DATETIME(2030, 1, 1, tzinfo=UTC)
    principal = verify_benchmark(path, data.database, hasher=data.hasher, clock=lambda: target, ttl_seconds=60)
    row = stored(data, principal.session_id)
    assert row['created_at'] == target and principal.expires_at == target + timedelta(seconds=60)


def test_rotating_legacy_host_ahead_cookie_revokes_it_without_rolling_back_new_login(clock_data):
    data = clock_data
    target = db_now(data) + timedelta(seconds=120)
    legacy = service(data, clock=lambda: target)
    previous = login(data, legacy)
    old = legacy.authenticate(previous.token, touch=False).identity
    before = db_now(data)
    auth = service(data)
    replacement = login(data, auth, previous_token=previous.token)
    current = auth.authenticate(replacement.token, touch=False).identity
    after = db_now(data)
    assert current.session_id != old.session_id
    assert before <= stored(data, current.session_id)['created_at'] <= after
    row = stored(data, old.session_id)
    assert row['revoked_at'] is not None and row['revoked_at'] >= row['created_at'] == target
    future = service(data, clock=lambda: target + timedelta(seconds=1))
    expired(lambda: future.authenticate(previous.token, touch=True))


def test_session_time_helper_uses_supplied_connection_and_preserves_explicit_callable():
    # Keep this import LOCAL so the old-source reproducer can collect/run its
    # default-host-skew node without the new utility module existing there.
    from src.db.clock import session_time
    now = REAL_DATETIME(2026, 11, 4, tzinfo=UTC)
    calls = []

    class Connection:
        def execute(self, statement):
            calls.append(str(statement))
            return SimpleNamespace(scalar_one=lambda: now)

    connection = Connection()
    assert session_time(connection) == now
    assert len(calls) == 1 and 'clock_timestamp' in calls[0]
    target = now + timedelta(days=5)
    assert session_time(connection, clock=lambda: target) == target
    assert len(calls) == 1, 'An explicitly injected clock must not acquire/query another connection'


@pytest.mark.parametrize('value', [None, True, '2026-11-04T00:00:00Z', REAL_DATETIME(2026, 11, 4)])
def test_session_time_helper_rejects_invalid_injected_clock_values(value):
    from src.db.clock import session_time
    with pytest.raises((ValueError, TypeError)):
        session_time(None, clock=lambda: value)

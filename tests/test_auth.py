"""Local-auth behavior on isolated real PostgreSQL and small HTTP fixtures.

Most lifecycle cases inject an intentionally cheap test hasher. The separate
production-hasher test exercises real Argon2id; fake timings/memory are never
presented as the production password profile. No corpus, model or paid service
is used by these checks.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

sa = pytest.importorskip("sqlalchemy", reason="Auth checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Auth checks require the psycopg driver")
pytest.importorskip("alembic", reason="Auth checks require Alembic")
pytest.importorskip("argon2", reason="Auth checks require the supported Argon2 implementation")

from src.api.main import create_app
from src.api.auth import AuthWorker, BOOTSTRAP_COOKIE_NAME
from src.api.runtime import RuntimeConfigurationError, WebSettings
from src.auth.accounts import PasswordHasher, create_account, disable_account, reset_password
import src.auth.accounts as account_module
from src.auth.contracts import CSRF_HEADER_NAME, SESSION_COOKIE_NAME
from src.auth.sessions import AuthError, AuthService
from src.auth.settings import AuthSettings
from src.db import schema
from src.db.migrate import upgrade_database
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import disposable_database, engine_for, pg_urls, url_string


ORIGIN = "http://localhost:8000"
PASSWORD = "fixture-only password 🧭"
NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class CheapHasher:
    """Deterministic test double, not a deployable password hash."""
    def __init__(self):
        self.verifications = []

    def hash(self, password):
        return "fixture-only$" + hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify(self, encoded_hash, password):
        self.verifications.append(encoded_hash)
        return encoded_hash == self.hash(password)


def auth_settings(**changes):
    values = dict(enabled=True, idle_seconds=10, absolute_seconds=30,
        bootstrap_seconds=10, max_waiting=2, worker_timeout_seconds=2,
        throttle_window_seconds=10, per_login_attempts=100, per_ip_attempts=200,
        global_attempts=300, throttle_capacity=1000, allowed_origins=(ORIGIN,),
        secure_cookies=False, allow_insecure_loopback=True)
    values.update(changes)
    return AuthSettings(**values)


@pytest.fixture(scope="module")
def auth_db(pg_urls):
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        owner = engine_for(fresh.migration)
        try:
            yield SimpleNamespace(urls=fresh, owner=owner)
        finally:
            owner.dispose()


@pytest.fixture
def auth(auth_db):
    hasher, clock = CheapHasher(), Clock()
    login = "reader_" + uuid4().hex
    user = create_account(login, PASSWORD, display_name="Fixture reader",
        role="intern", department="research", project_ids=("fixture-project",),
        url=url_string(auth_db.urls.migration), hasher=hasher)
    database = Database(DatabaseSettings(dsn=url_string(auth_db.urls.runtime), pool_size=2))
    settings = auth_settings()
    service = AuthService(database, settings, clock=clock, hasher=hasher,
                          csrf_secret=b"fixture-only-csrf-secret-32bytes-long")
    try:
        yield SimpleNamespace(database=database, db=auth_db, settings=settings,
            service=service, hasher=hasher, clock=clock, login=login, user=user)
    finally:
        database.close()


def service_for(auth, *, hasher=None, **settings_changes):
    settings = auth_settings(**settings_changes)
    return AuthService(auth.database, settings, clock=auth.clock,
        hasher=hasher or auth.hasher, csrf_secret=b"fixture-only-csrf-secret-32bytes-long")


def login(auth, *, service=None, password=PASSWORD, name=None, previous_token=None):
    return (service or auth.service).login(name or auth.login, password,
        client_ip="127.0.0.1", previous_token=previous_token)


def token(grant):
    assert isinstance(grant.token, SecretStr)
    return grant.token.get_secret_value()


def error_code(call, code, status=None):
    with pytest.raises(AuthError) as error:
        call()
    assert error.value.code == code
    if status is not None:
        assert error.value.status == status
    return error.value


def session_rows(auth):
    with auth.db.owner.connect() as connection:
        return connection.execute(sa.select(schema.sessions).where(
            schema.sessions.c.user_id == auth.user.user_id).order_by(schema.sessions.c.created_at,
                schema.sessions.c.session_id)).mappings().all()


def http_app(auth, *, settings=None, runtime_factory=None):
    kwargs = dict(auth_settings=settings or auth.settings, auth_service=auth.service)
    if runtime_factory is not None:
        kwargs["runtime_factory"] = runtime_factory
    return create_app(WebSettings(mode="fixture", demo_access=True), **kwargs)


def bootstrap(client):
    response = client.get("/api/auth/csrf")
    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "")
    value = response.json()["csrf_token"]
    assert value and client.cookies.get(BOOTSTRAP_COOKIE_NAME)
    return value


def http_login(client, auth, *, name=None, password=PASSWORD):
    csrf = bootstrap(client)
    response = client.post("/api/auth/login", json={"login": name or auth.login, "password": password},
        headers={"Origin": ORIGIN, CSRF_HEADER_NAME: csrf})
    assert "no-store" in response.headers.get("cache-control", "")
    return response


def test_real_argon2_profile_hashes_and_verifies_without_secret_repr(monkeypatch):
    from argon2 import extract_parameters
    from argon2 import PasswordHasher as Argon2Hasher
    from argon2.profiles import RFC_9106_LOW_MEMORY
    hasher = PasswordHasher()
    encoded = hasher.hash(PASSWORD)
    parameters = extract_parameters(encoded)
    assert parameters == RFC_9106_LOW_MEMORY
    assert hasher.verify(encoded, PASSWORD) is True
    assert hasher.verify(encoded, "a different fixture password") is False
    assert PASSWORD not in repr(hasher)
    monkeypatch.setattr(Argon2Hasher, "verify", lambda *_args, **_kwargs: pytest.fail(
        "Unapproved stored hash parameters reached the expensive backend"))
    for malformed in (encoded.replace("m=65536", "m=2147483647"),
                      encoded.replace("t=3", "t=99999"), encoded.replace("p=4", "p=1024"),
                      encoded.replace("v=19", "v=16"), "invalid-hash", "x" * 513):
        assert hasher.verify(malformed, PASSWORD) is False


@pytest.mark.parametrize("arguments", [("create", "fixture-reader", "--display-name", "Fixture reader"),
                                        ("reset-password", "fixture-reader")])
def test_account_password_cli_refuses_non_tty_before_reading_secret(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(account_module.getpass, "getpass", lambda *_args, **_kwargs: pytest.fail(
        "Non-interactive input attempted to read a password"))
    with pytest.raises(SystemExit) as error:
        account_module.main(list(arguments))
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert PASSWORD not in captured.out + captured.err


def test_account_normalization_and_password_reset_revoke_existing_versions(auth):
    grant = login(auth, name=auth.login.upper())
    first = auth.service.authenticate(token(grant))
    assert first.identity.user_id == auth.user.user_id and first.identity.session_version == 0
    replacement = "new fixture password 🧭"
    reset_password(auth.login.upper(), replacement, url=url_string(auth.db.urls.migration), hasher=auth.hasher)
    error_code(lambda: auth.service.authenticate(token(grant)), "session_expired", 401)
    error_code(lambda: login(auth), "invalid_credentials", 401)
    next_grant = login(auth, password=replacement)
    assert auth.service.authenticate(token(next_grant)).identity.session_version == 1


def test_provisioning_rejects_duplicate_and_weak_passwords_without_output(auth, capsys):
    for password in ("short", "x" * 1025):
        with pytest.raises(ValueError):
            create_account("invalid_" + uuid4().hex, password, display_name="Invalid fixture",
                url=url_string(auth.db.urls.migration), hasher=auth.hasher)
    with pytest.raises(ValueError):
        create_account(auth.login.upper(), PASSWORD, display_name="Duplicate fixture",
            url=url_string(auth.db.urls.migration), hasher=auth.hasher)
    captured = capsys.readouterr()
    assert PASSWORD not in captured.out + captured.err


def test_two_sessions_have_distinct_opaque_tokens_and_only_digests_are_persisted(auth):
    left, right = login(auth), login(auth)
    assert token(left) != token(right)
    assert left.response.csrf_token != right.response.csrf_token
    assert auth.service.authenticate(token(left)).identity.session_id != auth.service.authenticate(token(right)).identity.session_id
    rows = session_rows(auth)
    assert len(rows) == 2
    assert {row["token_hash"] for row in rows} == {hashlib.sha256(token(grant).encode()).hexdigest() for grant in (left, right)}
    assert {row["csrf_hash"] for row in rows} == {hashlib.sha256(grant.response.csrf_token.encode()).hexdigest() for grant in (left, right)}
    exported = json.dumps([dict(row) for row in rows], default=str)
    assert all(secret not in exported + repr(left) + repr(right) for secret in (token(left), token(right), PASSWORD))


def test_logout_replay_fails_while_other_session_is_independent(auth):
    left, right = login(auth), login(auth)
    auth.service.logout(token(left), left.response.csrf_token)
    error_code(lambda: auth.service.authenticate(token(left)), "session_expired", 401)
    assert auth.service.authenticate(token(right)).identity.user_id == auth.user.user_id
    assert sum(row["revoked_at"] is not None for row in session_rows(auth)) == 1


def test_login_rotation_revokes_previous_session_without_reusing_identity(auth):
    old = login(auth)
    new = login(auth, previous_token=token(old))
    assert token(new) != token(old)
    error_code(lambda: auth.service.authenticate(token(old)), "session_expired", 401)
    assert auth.service.authenticate(token(new)).identity.user_id == auth.user.user_id


def test_logout_requires_bound_csrf_and_never_revokes_another_session(auth):
    left, right = login(auth), login(auth)
    error_code(lambda: auth.service.logout(token(left), right.response.csrf_token), "csrf_failed", 403)
    assert auth.service.authenticate(token(left)).identity.user_id == auth.user.user_id
    assert auth.service.authenticate(token(right)).identity.user_id == auth.user.user_id


@pytest.mark.parametrize("value,code", [(None, "authentication_required"), ("", "authentication_required"),
    ("not-a-real-session-token", "session_expired"), ("x" * 10000, "session_expired")])
def test_missing_malformed_unknown_tokens_fail_with_content_free_errors(auth, value, code):
    error = error_code(lambda: auth.service.authenticate(value), code, 401)
    assert PASSWORD not in str(error) + repr(error)
    assert not session_rows(auth)


def test_idle_expiry_is_inclusive_and_me_style_reads_do_not_keep_session_alive(auth):
    grant = login(auth)
    auth.clock.advance(9)
    assert auth.service.authenticate(token(grant), touch=False).identity.user_id == auth.user.user_id
    auth.clock.advance(1)
    error_code(lambda: auth.service.authenticate(token(grant)), "session_expired", 401)


def test_activity_slides_idle_deadline_but_never_extends_absolute_lifetime(auth):
    grant = login(auth)
    created = session_rows(auth)[0]["created_at"]
    for seconds in (9, 9, 9):
        auth.clock.advance(seconds)
        assert auth.service.authenticate(token(grant), touch=True).identity.user_id == auth.user.user_id
    rows = session_rows(auth)
    assert rows[0]["created_at"] == created
    assert rows[0]["expires_at"] == created + timedelta(seconds=30)
    auth.clock.advance(3)
    error_code(lambda: auth.service.authenticate(token(grant)), "session_expired", 401)


@pytest.mark.parametrize("operation", ["authenticate", "logout"])
def test_expiry_while_waiting_for_session_row_cannot_renew_or_revoke_it(auth, monkeypatch, operation):
    grant = login(auth)
    before = session_rows(auth)
    original_session = auth.service._session

    def delayed_session(connection, raw):
        row = original_session(connection, raw)
        # Model time spent waiting for another transaction's row lock. The
        # expiry check must use time after that wait, never a pre-lock timestamp.
        auth.clock.advance(10)
        return row

    monkeypatch.setattr(auth.service, "_session", delayed_session)
    call = (lambda: auth.service.authenticate(token(grant))) if operation == "authenticate" else (
        lambda: auth.service.logout(token(grant), grant.response.csrf_token))
    error_code(call, "session_expired", 401)
    assert session_rows(auth) == before


def test_concurrent_logout_serializes_before_waiting_touch_and_token_stays_revoked(auth, monkeypatch):
    grant = login(auth)
    locked, waiting, release = threading.Event(), threading.Event(), threading.Event()
    original_session = auth.service._session
    counter, lock = 0, threading.Lock()

    def controlled_session(connection, raw):
        nonlocal counter
        with lock:
            counter += 1
            ordinal = counter
        if ordinal != 1:
            waiting.set()
        row = original_session(connection, raw)
        if ordinal == 1:
            locked.set()
            assert release.wait(3)
        return row

    monkeypatch.setattr(auth.service, "_session", controlled_session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        logout = pool.submit(auth.service.logout, token(grant), grant.response.csrf_token)
        assert locked.wait(2)
        touch = pool.submit(auth.service.authenticate, token(grant))
        try:
            assert waiting.wait(2)
            with pytest.raises(TimeoutError):
                touch.result(timeout=.05)
        finally:
            release.set()
        logout.result(timeout=2)
        error_code(lambda: touch.result(timeout=2), "session_expired", 401)
    assert session_rows(auth)[0]["revoked_at"] is not None


def test_disabled_account_revokes_all_sessions_and_cannot_log_in(auth):
    grants = (login(auth), login(auth))
    disable_account(auth.login, url=url_string(auth.db.urls.migration))
    for grant in grants:
        error_code(lambda: auth.service.authenticate(token(grant)), "session_expired", 401)
    error_code(lambda: login(auth), "invalid_credentials", 401)
    with auth.db.owner.connect() as connection:
        row = connection.execute(sa.select(schema.users).where(schema.users.c.user_id == auth.user.user_id)).mappings().one()
    assert not row["is_active"] and row["session_version"] == 1


@pytest.mark.parametrize("change", ["disable", "reset"])
def test_admin_change_during_password_verification_cannot_create_stale_session(auth, change):
    started, release = threading.Event(), threading.Event()

    class BlockingHasher(CheapHasher):
        def verify(self, encoded_hash, password):
            started.set()
            assert release.wait(3)
            return super().verify(encoded_hash, password)

    service = service_for(auth, hasher=BlockingHasher())
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempt = pool.submit(login, auth, service=service)
        assert started.wait(2)
        try:
            if change == "disable":
                mutation = pool.submit(disable_account, auth.login, url=url_string(auth.db.urls.migration))
            else:
                mutation = pool.submit(reset_password, auth.login, "replacement fixture password",
                    url=url_string(auth.db.urls.migration), hasher=auth.hasher)
            # The owner write must not wait for a password hash holding a row lock.
            mutation.result(timeout=1)
        finally:
            release.set()
        error_code(lambda: attempt.result(timeout=2), "invalid_credentials", 401)
    assert not session_rows(auth)


def test_bootstrap_is_signed_bound_and_expires_at_exact_boundary(auth):
    grant = auth.service.issue_bootstrap()
    value = token(grant)
    assert grant.response.csrf_token == value and value not in repr(grant)
    auth.service.validate_bootstrap(value, value)
    error_code(lambda: auth.service.validate_bootstrap(value, value + "tampered"), "csrf_failed", 403)
    error_code(lambda: auth.service.validate_bootstrap(value, "é"), "csrf_failed", 403)
    error_code(lambda: auth.service.validate_bootstrap(value + "tampered", value + "tampered"), "csrf_failed", 403)
    auth.clock.advance(10)
    error_code(lambda: auth.service.validate_bootstrap(value, value), "csrf_failed", 403)


def test_throttled_login_cannot_bypass_limit_with_correct_password_then_recovers(auth):
    service = service_for(auth, per_login_attempts=2)
    for _ in range(2):
        error_code(lambda: login(auth, service=service, password="wrong fixture password"), "invalid_credentials", 401)
    error_code(lambda: login(auth, service=service), "rate_limited", 429)
    assert not session_rows(auth)
    auth.clock.advance(10)
    assert service.authenticate(token(login(auth, service=service))).identity.user_id == auth.user.user_id


@pytest.mark.parametrize("kind", ["ip", "global"])
def test_ip_and_global_attempt_budgets_bound_unknown_account_spray(auth, kind):
    changes = {"per_ip_attempts": 2} if kind == "ip" else {"global_attempts": 2}
    service = service_for(auth, **changes)
    for index in range(2):
        error_code(lambda: service.login("unknown_" + str(index), PASSWORD,
            client_ip="127.0.0." + str(1 if kind == "ip" else index + 1)), "invalid_credentials", 401)
    error_code(lambda: login(auth, service=service), "rate_limited", 429)


def test_attempt_maps_remain_bounded_and_global_budget_survives_lru_evictions(auth):
    service = service_for(auth, throttle_capacity=2, global_attempts=5)
    for index in range(5):
        error_code(lambda: service.login("unknown_" + str(index), PASSWORD,
            client_ip="127.0.0." + str(index + 1)), "invalid_credentials", 401)
    assert len(service.limiter.logins) == len(service.limiter.ips) == 2
    error_code(lambda: login(auth, service=service), "rate_limited", 429)
    assert not session_rows(auth)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_auth_worker_keeps_its_slot_until_cancelled_or_timed_out_work_really_finishes(auth, interruption):
    started, release = threading.Event(), threading.Event()

    def blocked():
        started.set()
        assert release.wait(3)
        return "finished"

    async def scenario():
        worker = AuthWorker(auth.service, auth_settings(max_waiting=0, worker_timeout_seconds=.05))
        first = asyncio.create_task(worker.call(blocked))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            with pytest.raises(AuthError) as busy:
                await worker.call(lambda: "unadmitted")
            assert busy.value.code == "busy" and busy.value.status == 429
            if interruption == "cancel":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            else:
                with pytest.raises(AuthError) as timed_out:
                    await first
                assert timed_out.value.code == "busy"
            assert worker.pending == 1
            with pytest.raises(AuthError) as still_busy:
                await worker.call(lambda: "still-unadmitted")
            assert still_busy.value.code == "busy"
            release.set()
            deadline = time.monotonic() + 1
            while worker.pending and time.monotonic() < deadline:
                await asyncio.sleep(.005)
            assert worker.pending == 0
            assert await worker.call(lambda: "admitted") == "admitted"
        finally:
            release.set()
            await worker.aclose()

    asyncio.run(scenario())


def test_two_http_clients_have_independent_sessions_csrf_logout_and_no_token_json(auth):
    # Separate application lifespans avoid starting the same ASGI app twice;
    # both clients use the same real account/session store and service.
    with TestClient(http_app(auth), base_url=ORIGIN) as left, TestClient(http_app(auth), base_url=ORIGIN) as right:
        first, second = http_login(left, auth), http_login(right, auth)
        assert first.status_code == second.status_code == 200
        left_token, right_token = left.cookies.get(SESSION_COOKIE_NAME), right.cookies.get(SESSION_COOKIE_NAME)
        assert left_token and right_token and left_token != right_token
        assert left_token not in first.text and right_token not in second.text and PASSWORD not in first.text + second.text
        assert left.get("/api/auth/me").json()["user"]["user_id"] == auth.user.user_id
        assert right.get("/api/auth/me").status_code == 200
        denied = left.post("/api/auth/logout", headers={"Origin": ORIGIN,
            CSRF_HEADER_NAME: second.json()["csrf_token"]})
        assert denied.status_code == 403 and left.get("/api/auth/me").status_code == 200
        done = left.post("/api/auth/logout", headers={"Origin": ORIGIN,
            CSRF_HEADER_NAME: first.json()["csrf_token"]})
        assert done.status_code in (200, 204)
        assert left.get("/api/auth/me").status_code == 401
        assert right.get("/api/auth/me").status_code == 200
        left.cookies.set(SESSION_COOKIE_NAME, left_token)
        assert left.get("/api/auth/me").status_code == 401


def test_http_cookie_flags_and_login_rotate_old_session(auth):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        first = http_login(client, auth)
        cookie = next(value for value in first.headers.get_list("set-cookie") if value.startswith(SESSION_COOKIE_NAME + "="))
        assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
        assert "path=/" in cookie.lower() and "domain=" not in cookie.lower()
        old = client.cookies.get(SESSION_COOKIE_NAME)
        second = http_login(client, auth)
        assert second.status_code == 200 and client.cookies.get(SESSION_COOKIE_NAME) != old
        error_code(lambda: auth.service.authenticate(old), "session_expired", 401)


def test_https_profile_issues_secure_httponly_host_only_cookie(auth):
    origin = "https://auth.example"
    settings = auth_settings(allowed_origins=(origin,), secure_cookies=True, allow_insecure_loopback=False)
    service = AuthService(auth.database, settings, clock=auth.clock, hasher=auth.hasher,
                          csrf_secret=b"fixture-only-csrf-secret-32bytes-long")
    app = create_app(WebSettings(mode="fixture", demo_access=True), auth_settings=settings, auth_service=service)
    with TestClient(app, base_url=origin) as client:
        csrf = bootstrap(client)
        response = client.post("/api/auth/login", json={"login": auth.login, "password": PASSWORD},
            headers={"Origin": origin, CSRF_HEADER_NAME: csrf})
    assert response.status_code == 200
    cookie = next(value for value in response.headers.get_list("set-cookie") if value.startswith(SESSION_COOKIE_NAME + "="))
    assert all(flag in cookie.lower() for flag in ("; secure", "httponly", "samesite=strict", "path=/"))
    assert "domain=" not in cookie.lower()


@pytest.mark.parametrize("origin", [None, "https://evil.example", "null", ORIGIN + ".evil.example"])
def test_unsafe_auth_requires_exact_allowed_origin(auth, origin):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        csrf = bootstrap(client)
        headers = {CSRF_HEADER_NAME: csrf}
        if origin is not None:
            headers["Origin"] = origin
        response = client.post("/api/auth/login", json={"login": auth.login, "password": PASSWORD}, headers=headers)
        assert response.status_code == 403
        assert not session_rows(auth)


def test_login_requires_csrf_cookie_header_and_challenge_expiry(auth):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        response = client.post("/api/auth/login", json={"login": auth.login, "password": PASSWORD}, headers={"Origin": ORIGIN})
        assert response.status_code == 403
        csrf = bootstrap(client)
        response = client.post("/api/auth/login", json={"login": auth.login, "password": PASSWORD},
            headers={"Origin": ORIGIN, CSRF_HEADER_NAME: csrf + "tampered"})
        assert response.status_code == 403
        auth.clock.advance(10)
        response = client.post("/api/auth/login", json={"login": auth.login, "password": PASSWORD},
            headers={"Origin": ORIGIN, CSRF_HEADER_NAME: csrf})
        assert response.status_code == 403 and not session_rows(auth)


def test_wrong_unknown_and_disabled_http_logins_have_identical_generic_response(auth):
    bodies = []
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        for name, password in ((auth.login, "wrong fixture password"), ("unknown_" + uuid4().hex, PASSWORD)):
            response = http_login(client, auth, name=name, password=password)
            assert response.status_code == 401
            bodies.append(response.json())
        disable_account(auth.login, url=url_string(auth.db.urls.migration))
        response = http_login(client, auth)
        assert response.status_code == 401
        bodies.append(response.json())
    assert bodies[0] == bodies[1] == bodies[2]
    assert bodies[0]["detail"]["code"] == "invalid_credentials"
    assert auth.login not in json.dumps(bodies) and PASSWORD not in json.dumps(bodies)


@pytest.mark.parametrize("body", [{"login": "bad login", "password": PASSWORD},
    {"login": "reader", "password": "x" * 1025}, {"login": "reader", "password": PASSWORD, "role": "ceo"},
    {"login": "reader", "password": {"secret": PASSWORD}}, {"login": "reader"}])
def test_malformed_login_body_never_echoes_password_or_raw_input(auth, body, caplog):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        csrf = bootstrap(client)
        response = client.post("/api/auth/login", json=body,
            headers={"Origin": ORIGIN, CSRF_HEADER_NAME: csrf})
    assert response.status_code in (400, 401, 413, 422)
    if response.status_code == 401:
        assert response.json()["detail"]["code"] == "invalid_credentials"
    assert PASSWORD not in response.text + caplog.text and "x" * 1025 not in response.text
    assert not session_rows(auth)


@pytest.mark.parametrize("claimed_length", [None, "1"])
def test_login_http_body_limit_precedes_expensive_password_verification(auth, claimed_length):
    before = len(auth.hasher.verifications)
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        csrf = bootstrap(client)
        headers = {"Origin": ORIGIN, CSRF_HEADER_NAME: csrf, "Content-Type": "application/json"}
        if claimed_length is not None:
            headers["Content-Length"] = claimed_length
        response = client.post("/api/auth/login", content=json.dumps({"login": auth.login, "password": "x" * 100000}),
            headers=headers)
    assert response.status_code == 413
    assert len(auth.hasher.verifications) == before and not session_rows(auth)


def test_me_polling_does_not_refresh_idle_expiry_and_authenticated_queries_deny_until_acl(auth):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        response = http_login(client, auth)
        assert response.status_code == 200
        query = client.post("/api/query", json={"question": "Show CEO data"},
            headers={"Origin": ORIGIN, CSRF_HEADER_NAME: response.json()["csrf_token"]})
        assert query.status_code == 403 and query.json()["detail"]["code"] == "scope_denied"
        auth.clock.advance(9)
        assert client.get("/api/auth/me").status_code == 200
        auth.clock.advance(1)
        assert client.get("/api/auth/me").status_code == 401


def test_login_remains_responsive_while_custom_rag_initialization_is_blocked(auth):
    started, release = threading.Event(), threading.Event()

    def blocked_runtime(_settings):
        started.set()
        assert release.wait(3)
        return SimpleNamespace(info=lambda: {}, query=lambda _body: pytest.fail("Local auth called benchmark RAG"))

    with TestClient(http_app(auth, runtime_factory=blocked_runtime), base_url=ORIGIN) as client:
        try:
            assert started.wait(2)
            assert client.get("/api/health").status_code == 200
            with ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(http_login, client, auth).result(timeout=1)
            assert response.status_code == 200
            assert client.get("/api/auth/me").status_code == 200
            assert client.get("/api/ready").status_code == 503
        finally:
            release.set()


def test_injected_auth_service_does_not_enable_a_disabled_profile(auth):
    with pytest.raises(RuntimeConfigurationError) as error:
        create_app(WebSettings(mode="fixture", demo_access=True),
            auth_settings=auth_settings(enabled=False), auth_service=auth.service)
    assert PASSWORD not in str(error.value) and "authentication" in str(error.value).lower()


@pytest.mark.parametrize("explicit_auth_origins", [False, True])
def test_auth_origin_environment_precedence_and_fallback_control_routes_and_cors(auth, monkeypatch, explicit_auth_origins):
    fallback, override = "https://fallback.example", "https://auth.example"
    if explicit_auth_origins:
        monkeypatch.setenv("RAG_AUTH_ALLOWED_ORIGINS", override)
    else:
        monkeypatch.delenv("RAG_AUTH_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("RAG_AUTH_SECURE_COOKIES", "true")
    monkeypatch.setenv("RAG_AUTH_ALLOW_INSECURE_LOOPBACK", "false")
    selected = override if explicit_auth_origins else fallback
    app = create_app(WebSettings(mode="fixture", demo_access=True, auth_mode="local", allowed_origins=(fallback,)),
                     auth_service=auth.service)
    with TestClient(app, base_url=selected) as client:
        assert client.get("/api/auth/csrf").status_code == 200
        preflight = client.options("/api/auth/login", headers={"Origin": selected,
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-CSRF-Token, Content-Type"})
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == selected
        assert preflight.headers["access-control-allow-credentials"] == "true"
        if explicit_auth_origins:
            assert client.get("/api/auth/csrf", headers={"Origin": fallback}).status_code == 403
            denied = client.options("/api/auth/login", headers={"Origin": fallback,
                "Access-Control-Request-Method": "POST"})
            assert denied.status_code == 400 and "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize("enabled", ["true", "1"])
def test_environment_enabled_flag_selects_protected_profile_instead_of_benchmark(auth, monkeypatch, enabled):
    monkeypatch.setenv("RAG_AUTH_ENABLED", enabled)
    monkeypatch.setenv("RAG_AUTH_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setenv("RAG_AUTH_SECURE_COOKIES", "false")
    monkeypatch.setenv("RAG_AUTH_ALLOW_INSECURE_LOOPBACK", "true")
    app = create_app(WebSettings(mode="fixture", demo_access=True), auth_service=auth.service)
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/api/auth/csrf").status_code == 200
        assert client.get("/api/ready").status_code == 503
        assert client.post("/api/query", json={"question": "private data"},
                           headers={"Origin": ORIGIN}).status_code == 401
        assert app.state.runtime is None


def test_invalid_environment_auth_flag_rejects_startup_without_benchmark_fallback(auth, monkeypatch):
    monkeypatch.setenv("RAG_AUTH_ENABLED", "not-a-boolean")
    with pytest.raises(RuntimeConfigurationError):
        create_app(WebSettings(mode="fixture", demo_access=True), auth_service=auth.service)


def test_local_mode_conflicting_disabled_environment_flag_rejects_startup(auth, monkeypatch):
    monkeypatch.setenv("RAG_AUTH_ENABLED", "false")
    with pytest.raises(RuntimeConfigurationError):
        create_app(WebSettings(mode="fixture", demo_access=True, auth_mode="local"), auth_service=auth.service)


def test_non_ascii_query_csrf_header_fails_generically_instead_of_raising(auth):
    with TestClient(http_app(auth), base_url=ORIGIN) as client:
        assert http_login(client, auth).status_code == 200
        response = client.post("/api/query", json={"question": "private data"},
            headers=[(b"Origin", ORIGIN.encode("ascii")), (CSRF_HEADER_NAME.encode("ascii"), b"\xe9")])
    assert response.status_code == 403 and response.json()["detail"]["code"] == "csrf_failed"
    assert PASSWORD not in response.text


def test_secure_cookie_profile_requires_https_or_explicit_loopback_exception():
    with pytest.raises(ValueError):
        auth_settings(secure_cookies=False, allow_insecure_loopback=False)
    with pytest.raises(ValueError):
        auth_settings(secure_cookies=False, allow_insecure_loopback=True,
                      allowed_origins=("http://public.example",))

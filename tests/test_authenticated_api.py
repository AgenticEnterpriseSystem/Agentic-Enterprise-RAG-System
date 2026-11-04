"""Authenticated admission and physical-worker lifetime on tiny real PostgreSQL."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
import time
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

sa = pytest.importorskip('sqlalchemy', reason='Authenticated API checks require DB dependencies')
pytest.importorskip('psycopg', reason='Authenticated API checks require psycopg')
pytest.importorskip('alembic', reason='Authenticated API checks require Alembic')

import src.api.main as main_module
from src.api.main import create_app
from src.api.runtime import WebSettings
from src.auth.contracts import CSRF_HEADER_NAME, SESSION_COOKIE_NAME
from src.access.contracts import (
    AccessDenied, EvidenceValidationError, PolicyUnavailable, SessionExpired, SnapshotMismatch, StaleContext,
)
from src.contracts import Answer, QueryResponse, Trace, Usage
from tests.test_access_policy import PASSWORD, acl_db, raw_token
from tests.db_support import pg_urls
from tests.test_authenticated_runtime import QUESTION, app_db, make_runtime

ORIGIN = 'http://localhost:8000'


def wait_until(predicate, *, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    pytest.fail('Timed out waiting for the controlled test transition')


def wait_ready(client):
    result = None
    def ready():
        nonlocal result
        result = client.get('/api/ready')
        return result.status_code == 200
    wait_until(ready)
    return result.json()


def sign_in(client, fixture, principal='intern'):
    csrf = client.get('/api/auth/csrf')
    assert csrf.status_code == 200
    response = client.post('/api/auth/login', json={'login': fixture.logins[principal], 'password': PASSWORD},
        headers={'Origin': ORIGIN, CSRF_HEADER_NAME: csrf.json()['csrf_token']})
    assert response.status_code == 200
    return {'Origin': ORIGIN, CSRF_HEADER_NAME: response.json()['csrf_token']}


def app_for(fixture, bundle, *, factory=None):
    return create_app(bundle.settings, factory or (lambda _settings: bundle.runtime),
        auth_settings=fixture.auth.settings, auth_service=fixture.auth)


def empty_response(snapshot):
    return QueryResponse(snapshot=snapshot, answer=Answer(status='insufficient_evidence', text='Controlled worker finished.'),
        trace=Trace(run_id=str(uuid4()), steps=(), total_latency_ms=0, usage=Usage(estimated_cost_usd=0),
                    stop_reason='no_new_evidence'))


def block_query(bundle, monkeypatch, *, entered, release, after_release=None):
    def blocked(body, identity):
        entered.set()
        assert release.wait(10), 'Test did not release the admitted RAG worker'
        if after_release:
            after_release()
        return empty_response(bundle.resolver.snapshot)
    monkeypatch.setattr(bundle.runtime, 'query', blocked)


def test_two_browser_sessions_use_one_runtime_and_never_share_other_users_sources(app_db):
    bundle = make_runtime(app_db)
    app = app_for(app_db, bundle)
    with TestClient(app, base_url=ORIGIN) as intern:
        wait_ready(intern)
        # Share the application's existing portal/transport, not a second
        # lifespan or executor. Each HTTP client owns its own cookie jar.
        with httpx.Client(transport=intern._transport, base_url=ORIGIN) as executive:
            first, second = sign_in(intern, app_db), sign_in(executive, app_db, 'executive')
            for client, headers, expected in ((intern, first, {'general', 'atlas'}),
                    (executive, second, {'general', 'finance-exec'}), (intern, first, {'general', 'atlas'})):
                response = client.post('/api/query', json={'question': QUESTION}, headers=headers)
                assert response.status_code == 200 and 'no-store' in response.headers.get('cache-control', '')
                value = QueryResponse.model_validate(response.json())
                assert value.snapshot == app_db.snapshot
                assert {source.chunk.doc_id for source in value.sources} == expected
            assert intern.cookies.get(SESSION_COOKIE_NAME) != executive.cookies.get(SESSION_COOKIE_NAME)


def test_public_readiness_exposes_no_inventory_permissions_models_or_spending(app_db):
    bundle = make_runtime(app_db)
    with TestClient(app_for(app_db, bundle), base_url=ORIGIN) as client:
        ready = wait_ready(client)
        assert set(ready) <= {'status', 'busy', 'mode', 'retrieval', 'generation_enabled'}
        assert client.get('/api/health').json() == {'status': 'alive', 'ready': True}
        assert 'no-store' in client.get('/api/ready').headers['cache-control']
        payload = json.dumps(ready)
        assert all(name not in payload for name in ('general', 'finance-exec', 'source_types', 'budget', 'documents'))
        assert client.post('/api/query', json={'question': QUESTION}).status_code == 401


@pytest.mark.parametrize('selection', ['explicit-settings', 'environment-flag'])
def test_computed_auth_selection_reaches_default_builder_with_the_shared_database(app_db, monkeypatch, selection):
    bundle = make_runtime(app_db, provider=None, settings=WebSettings(mode='retrieval', auth_mode='benchmark'))
    calls = []
    def default_builder(settings, *, database=None, authenticated=False):
        calls.append((settings, database, authenticated))
        return bundle.runtime
    monkeypatch.setattr(main_module, 'build_runtime', default_builder)
    kwargs = {'auth_service': app_db.auth}
    if selection == 'explicit-settings':
        kwargs['auth_settings'] = app_db.auth.settings
    else:
        for key, value in {'RAG_AUTH_ENABLED':'true', 'RAG_AUTH_ALLOWED_ORIGINS':ORIGIN,
                'RAG_AUTH_SECURE_COOKIES':'false','RAG_AUTH_ALLOW_INSECURE_LOOPBACK':'true'}.items():
            monkeypatch.setenv(key, value)
    with TestClient(create_app(bundle.settings, **kwargs), base_url=ORIGIN) as client:
        wait_ready(client)
        assert len(calls) == 1 and calls[0][1] is app_db.database and calls[0][2] is True


def test_invalid_injected_benchmark_runtime_never_becomes_ready_or_handles_query(app_db, monkeypatch):
    bundle = make_runtime(app_db, provider=None)
    calls, touches = [], []
    original = app_db.auth.authenticate
    def authenticate(token, *, touch=True):
        touches.append(touch)
        return original(token, touch=touch)
    monkeypatch.setattr(app_db.auth, 'authenticate', authenticate)
    class BenchmarkPretender:
        def info(self):
            pytest.fail('Local readiness exposed injected benchmark metadata')
        def query(self, body):
            calls.append(body)
            pytest.fail('Authenticated query fell back to a benchmark identity')
    app = app_for(app_db, bundle, factory=lambda _settings: BenchmarkPretender())
    with TestClient(app, base_url=ORIGIN) as client:
        wait_until(lambda: app.state.error is not None)
        headers = sign_in(client, app_db)
        ready = client.get('/api/ready')
        assert ready.status_code == 503 and set(ready.json()) == {'status', 'message'}
        denied = client.post('/api/query', json={'question': QUESTION}, headers=headers)
        assert denied.status_code == 503 and denied.json()['detail']['code'] == 'not_ready'
        assert calls == [] and True not in touches


def test_no_touch_for_not_ready_and_auth_health_remain_responsive_during_loading(app_db, monkeypatch):
    bundle = make_runtime(app_db, provider=None)
    entered, release, touches = Event(), Event(), []
    original = app_db.auth.authenticate
    def authenticate(token, *, touch=True):
        touches.append(touch)
        return original(token, touch=touch)
    def factory(_settings):
        entered.set()
        assert release.wait(10)
        return bundle.runtime
    monkeypatch.setattr(app_db.auth, 'authenticate', authenticate)
    with TestClient(app_for(app_db, bundle, factory=factory), base_url=ORIGIN) as client:
        try:
            assert entered.wait(3)
            headers = sign_in(client, app_db)
            assert client.get('/api/health').json() == {'status': 'alive', 'ready': False}
            assert client.get('/api/auth/me').status_code == 200
            response = client.post('/api/query', json={'question': QUESTION}, headers=headers)
            assert response.status_code == 503 and response.json()['detail']['code'] == 'not_ready'
            assert True not in touches
        finally:
            release.set()


def test_timeout_keeps_raw_worker_busy_and_only_admitted_query_touches_session(app_db, monkeypatch):
    settings = WebSettings(mode='retrieval', auth_mode='local', max_seconds=.1)
    bundle = make_runtime(app_db, provider=None, settings=settings)
    entered, release, touches = Event(), Event(), []
    block_query(bundle, monkeypatch, entered=entered, release=release)
    original = app_db.auth.authenticate
    def authenticate(token, *, touch=True):
        touches.append(touch)
        return original(token, touch=touch)
    monkeypatch.setattr(app_db.auth, 'authenticate', authenticate)
    app = app_for(app_db, bundle)
    with TestClient(app, base_url=ORIGIN) as client:
        wait_ready(client)
        headers = sign_in(client, app_db)
        invalid = client.post('/api/query', json={'question': QUESTION}, headers={'Origin':ORIGIN})
        assert invalid.status_code == 403 and True not in touches
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, '/api/query', json={'question': QUESTION}, headers=headers)
            try:
                assert entered.wait(3)
                raw = app.state.active_work
                assert raw.running() and app.state.busy
                busy = client.post('/api/query', json={'question': QUESTION}, headers=headers)
                assert busy.status_code == 429 and busy.json()['detail']['code'] == 'busy'
                assert busy.headers['retry-after'] == '2'
                assert client.get('/api/health').status_code == client.get('/api/auth/me').status_code == 200
                assert first.result(timeout=3).status_code == 504
                assert app.state.active_work is raw and not raw.done() and app.state.busy
                assert client.post('/api/query', json={'question': QUESTION}, headers=headers).status_code == 429
                assert touches.count(True) == 1
            finally:
                release.set()
            raw.result(timeout=5)
        wait_until(lambda: not app.state.busy)
        assert client.post('/api/query', json={'question': QUESTION}, headers=headers).status_code == 200
        assert touches.count(True) == 2


@pytest.mark.parametrize('error,status,code', [
    (SessionExpired,401,'session_expired'), (PolicyUnavailable,503,'policy_unavailable'),
    (StaleContext,409,'stale_context'), (AccessDenied,403,'scope_denied'),
    (SnapshotMismatch,503,'not_ready'), (EvidenceValidationError,502,'query_failed'),
    (ValueError,502,'query_failed'),
])
def test_worker_access_errors_are_typed_sanitized_and_never_return_partial_sources(app_db, monkeypatch, caplog, error, status, code):
    bundle = make_runtime(app_db, provider=None)
    def fail(_body, _identity):
        raise error('PRIVATE_PROVIDER_BODY_AND_TOKEN_MARKER')
    monkeypatch.setattr(bundle.runtime, 'query', fail)
    with TestClient(app_for(app_db, bundle), base_url=ORIGIN) as client:
        wait_ready(client)
        response = client.post('/api/query', json={'question': QUESTION}, headers=sign_in(client, app_db))
        assert response.status_code == status and response.json()['detail']['code'] == code
        assert 'sources' not in response.json() and 'PRIVATE_' not in response.text
        assert 'no-store' in response.headers['cache-control'] and 'PRIVATE_' not in caplog.text


async def async_login(client, fixture):
    csrf = await client.get('/api/auth/csrf')
    assert csrf.status_code == 200
    response = await client.post('/api/auth/login', json={'login':fixture.logins['intern'],'password':PASSWORD},
        headers={'Origin':ORIGIN, CSRF_HEADER_NAME:csrf.json()['csrf_token']})
    assert response.status_code == 200
    return {'Origin':ORIGIN, CSRF_HEADER_NAME:response.json()['csrf_token']}


async def async_ready(client):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if (await client.get('/api/ready')).status_code == 200:
            return
        await asyncio.sleep(.005)
    pytest.fail('Authenticated runtime did not become ready')


def test_cancelled_http_waiter_does_not_release_the_physical_worker(app_db, monkeypatch):
    bundle = make_runtime(app_db, provider=None)
    entered, release = Event(), Event()
    block_query(bundle, monkeypatch, entered=entered, release=release)
    app = app_for(app_db, bundle)
    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as client:
                await async_ready(client)
                headers = await async_login(client, app_db)
                pending = asyncio.create_task(client.post('/api/query',json={'question':QUESTION},headers=headers))
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    raw = app.state.active_work
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert raw.running() and app.state.busy and app.state.active_work is raw
                    assert (await client.post('/api/query',json={'question':QUESTION},headers=headers)).status_code == 429
                    assert (await client.get('/api/auth/me')).status_code == 200
                finally:
                    release.set()
                await asyncio.to_thread(raw.result,5)
                deadline=time.monotonic()+3
                while app.state.busy and time.monotonic()<deadline:
                    await asyncio.sleep(.005)
                assert not app.state.busy
    asyncio.run(scenario())


@pytest.mark.parametrize('blocked_kind', ['rag', 'auth'])
def test_owned_shared_pool_closes_only_after_real_work_even_after_asgi_loop_exits(app_db, monkeypatch, blocked_kind):
    import src.auth.sessions as sessions_module
    import src.db.runtime as database_module
    bundle = make_runtime(app_db, provider=None)
    entered, release, closed, used_after_shutdown = Event(), Event(), Event(), Event()
    original_close = app_db.database.close
    def close():
        original_close()
        closed.set()
    monkeypatch.setattr(app_db.database,'close',close)
    def after_release():
        assert not closed.is_set(), 'Shared pool closed while its admitted worker was still active'
        with app_db.database.transaction() as connection:
            assert connection.execute(sa.text('SELECT 1')).scalar_one() == 1
        used_after_shutdown.set()
    if blocked_kind=='rag':
        block_query(bundle,monkeypatch,entered=entered,release=release,after_release=after_release)
    else:
        original_authenticate=app_db.auth.authenticate
        def blocked_authenticate(token,*,touch=True):
            entered.set()
            assert release.wait(10)
            after_release()
            return original_authenticate(token,touch=touch)
        monkeypatch.setattr(app_db.auth,'authenticate',blocked_authenticate)
    # Exercise create_app's OWNED pool path using the fixture's real Database
    # and AuthService, while avoiding a second pool or production password hash.
    monkeypatch.setattr(database_module.DatabaseSettings,'from_env',classmethod(lambda cls:app_db.database.settings))
    monkeypatch.setattr(database_module,'Database',lambda _settings:app_db.database)
    monkeypatch.setattr(sessions_module,'AuthService',lambda database,settings:app_db.auth)
    app=create_app(bundle.settings,lambda _settings:bundle.runtime,auth_settings=app_db.auth.settings)
    async def start_and_shutdown():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url=ORIGIN) as client:
                await async_ready(client)
                client.cookies.set(SESSION_COOKIE_NAME,raw_token(app_db,'intern'))
                if blocked_kind=='rag':
                    csrf=app_db.sessions['intern'].response.csrf_token
                    task=asyncio.create_task(client.post('/api/query',json={'question':QUESTION},
                        headers={'Origin':ORIGIN,CSRF_HEADER_NAME:csrf}))
                else:
                    task=asyncio.create_task(client.get('/api/auth/me'))
                assert await asyncio.to_thread(entered.wait,3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                raw=app.state.active_work
                assert not closed.is_set()
        assert app.state.closing and not closed.is_set()
        return raw
    try:
        raw=asyncio.run(start_and_shutdown())
        assert not closed.is_set()
    finally:
        release.set()
    if blocked_kind=='rag':
        raw.result(timeout=5)
    assert used_after_shutdown.wait(5) and closed.wait(5)

"""Live, content-free query progress: recorder registry and the polling endpoint."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.api.runtime import AuthenticatedRuntime, WebSettings
from src.contracts import Answer, QueryResponse, Trace, Usage
from src.observability import Recorder, bind_progress, observation_context, span
from tests.test_api import settings, wait_ready

PROGRESS_REQUEST_KEYS = {'status', 'http_status_code', 'worker_status', 'started_at', 'completed_at',
                         'duration_ms', 'rounds', 'error_code', 'error_type', 'usage'}
PROGRESS_STAGE_KEYS = {'sequence', 'stage', 'event', 'status', 'round_number', 'duration_ms',
                       'reason_code', 'stop_reason', 'sufficient', 'error_code', 'error_type',
                       'returned_count', 'unique_document_count', 'selected_document_count'}


def recorder(**kwargs):
    return Recorder(log=False, **kwargs)


def test_bind_then_read_returns_only_allow_listed_fields():
    rec, token = recorder(), str(uuid4())
    observation = rec.start_request()
    observation.run_id = str(uuid4())
    with observation_context(observation):
        bind_progress(token, 'user-1')
        with span('route', round_number=1):
            pass
        observation.emit('critic', 'end', status='ok', sufficient=True, exclusion_reasons={'other': 1},
                         input_tokens=3, candidate_count=9, provider_stop_reason='end_turn')
    result = rec.progress(token, 'user-1')
    assert set(result) == {'request', 'stages'}
    assert set(result['request']) == PROGRESS_REQUEST_KEYS
    assert result['stages'] and all(set(stage) <= PROGRESS_STAGE_KEYS for stage in result['stages'])
    assert [stage['sequence'] for stage in result['stages']] == sorted(stage['sequence'] for stage in result['stages'])
    assert 'request_id' not in result['request'] and 'run_id' not in result['request']
    # Uppercase form of the same UUID resolves identically.
    assert rec.progress(token.upper(), 'user-1')['stages'] == result['stages']


@pytest.mark.parametrize('owner,asked,found', [
    ('user-1', 'user-1', True), ('user-1', 'user-2', False), ('user-1', None, False),
    (None, None, True), (None, 'user-1', False),
])
def test_owner_must_match(owner, asked, found):
    rec, token = recorder(), str(uuid4())
    rec.bind_progress(rec.start_request(), token, owner)
    assert (rec.progress(token, asked) is not None) is found


@pytest.mark.parametrize('token', ['', 'not-a-uuid', None, 7, 'x' * 5000, '{}'])
def test_bad_tokens_are_ignored_without_raising(token):
    rec = recorder()
    observation = rec.start_request()
    rec.bind_progress(observation, token, 'u')
    assert rec.progress(token, 'u') is None
    assert rec.stats()['retained_requests'] == 1 and rec._progress == {}


def test_token_already_bound_to_another_request_is_not_hijacked():
    rec, token = recorder(), str(uuid4())
    first, second = rec.start_request(), rec.start_request()
    rec.bind_progress(first, token, 'owner')
    rec.bind_progress(second, token, 'attacker')
    first.emit('route', 'start', status='in_progress')
    assert rec.progress(token, 'attacker') is None
    assert rec.progress(token, 'owner')['stages'][-1]['stage'] == 'route'
    # A request keeps its first token, so one request cannot hold many tokens.
    other = str(uuid4())
    rec.bind_progress(first, other, 'owner')
    assert rec.progress(other, 'owner') is None


def test_unbound_without_current_observation_is_a_noop():
    bind_progress(str(uuid4()), 'u')


def test_registry_is_pruned_with_request_history():
    rec = recorder(max_requests=2)
    tokens = []
    for _ in range(4):
        token = str(uuid4())
        rec.bind_progress(rec.start_request(), token, None)
        tokens.append(token)
    assert [rec.progress(token, None) is not None for token in tokens] == [False, False, True, True]
    assert len(rec._progress) == 2


def test_registry_expires_with_retention():
    now = [0.0]
    rec = recorder(retention_seconds=10, clock=lambda: now[0])
    token = str(uuid4())
    rec.bind_progress(rec.start_request(), token, None)
    assert rec.progress(token, None) is not None
    now[0] = 11
    assert rec.progress(token, None) is None and rec._progress == {}


# Non-authenticated application -------------------------------------------------

def scripted(app, entered, release, *, fail=False):
    runtime = app.state.runtime
    original = runtime.query

    def query(body):
        with span('route', round_number=1):
            pass
        entered.set()
        assert release.wait(5)
        if fail:
            with span('critic', round_number=1):
                raise ValueError('boom')
        return original(body)
    runtime.query = query


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    pytest.fail('Timed out')


def test_owner_sees_in_progress_then_completed_stages_while_busy():
    app = create_app(settings())
    token, entered, release = str(uuid4()), threading.Event(), threading.Event()
    with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
        wait_ready(client)
        scripted(app, entered, release)
        future = pool.submit(client.post, '/api/query', json={'question': 'retention'},
                             headers={'X-Query-Progress-ID': token})
        assert entered.wait(3)
        assert app.state.busy is True
        running = client.get('/api/query/progress/' + token)
        assert running.status_code == 200 and 'no-store' in running.headers['cache-control']
        body = running.json()
        assert body['request']['status'] == 'in_progress' and body['request']['completed_at'] is None
        assert [(s['stage'], s['event']) for s in body['stages'] if s['stage'] == 'route'] == [
            ('route', 'start'), ('route', 'end')]
        assert all(set(s) <= PROGRESS_STAGE_KEYS for s in body['stages'])
        release.set()
        assert future.result(5).status_code == 200
        wait_for(lambda: not app.state.busy)
        done = client.get('/api/query/progress/' + token).json()
        assert done['request']['status'] == 'ok' and done['request']['worker_status'] == 'ok'
        assert done['request']['completed_at'] is not None
        assert 'question' not in str(done).lower() and 'retention' not in str(done)


def test_failed_run_is_reflected_with_error_status_and_code():
    app = create_app(settings())
    token, entered, release = str(uuid4()), threading.Event(), threading.Event()
    release.set()
    with TestClient(app) as client:
        wait_ready(client)
        scripted(app, entered, release, fail=True)
        response = client.post('/api/query', json={'question': 'retention'},
                               headers={'X-Query-Progress-ID': token})
        assert response.status_code == 502
        wait_for(lambda: not app.state.busy)
        body = client.get('/api/query/progress/' + token).json()
        failed = [s for s in body['stages'] if s['stage'] == 'critic' and s['event'] == 'error']
        assert failed and failed[0]['status'] == 'error' and failed[0]['error_type'] == 'ValueError'
        assert body['request']['status'] == 'error' and body['request']['http_status_code'] == 502
        assert body['request']['error_code'] == 'query_failed'


def test_header_absent_or_invalid_binds_nothing_and_query_works():
    app = create_app(settings())
    with TestClient(app) as client:
        wait_ready(client)
        for headers in ({}, {'X-Query-Progress-ID': 'nope'}):
            assert client.post('/api/query', json={'question': 'retention'}, headers=headers).status_code == 200
            wait_for(lambda: not app.state.busy)
        assert app.state.observability._progress == {}


def test_busy_rejection_is_visible_and_unknown_tokens_are_404():
    app = create_app(settings())
    token, entered, release = str(uuid4()), threading.Event(), threading.Event()
    with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
        wait_ready(client)
        scripted(app, entered, release)
        future = pool.submit(client.post, '/api/query', json={'question': 'retention'})
        assert entered.wait(3)
        rejected = client.post('/api/query', json={'question': 'retention'},
                               headers={'X-Query-Progress-ID': token})
        assert rejected.status_code == 429
        body = client.get('/api/query/progress/' + token).json()
        assert body['request']['http_status_code'] == 429 and body['request']['error_code'] == 'busy'
        missing = [client.get('/api/query/progress/' + value) for value in (str(uuid4()), 'not-a-uuid')]
        assert [r.status_code for r in missing] == [404, 404]
        assert missing[0].json() == {'detail': {'code': 'not_found', 'message': missing[0].json()['detail']['message']}}
        assert missing[0].json() == missing[1].json()
        assert client.get('/api/query/progress/' + token,
                          headers={'Origin': 'https://untrusted.invalid'}).status_code == 403
        release.set()
        future.result(5)


# Authenticated application, with a fake session service (no database) ---------

class FakeAuth:
    def __init__(self):
        self.database = object()
        self.sessions = {'cookie-a': 'user-a', 'cookie-b': 'user-b'}
        self.touches = []

    def authenticate(self, token, *, touch=True):
        from src.auth.sessions import AuthError
        user = self.sessions.get(token)
        if user is None:
            raise AuthError('session_expired', 401, 'This session is unavailable. Sign in again.')
        self.touches.append(touch)
        return SimpleNamespace(identity=SimpleNamespace(user_id=user),
                               response=SimpleNamespace(csrf_token='csrf-' + user))


class FakeAuthenticatedRuntime(AuthenticatedRuntime):
    def __init__(self, database, entered, release):
        self.database, self.entered, self.release = database, entered, release

    def assert_ready(self):
        pass

    def info(self):
        return {'mode': 'fixture', 'retrieval': 'fixture', 'generation_enabled': True}

    def query(self, body, identity):
        with span('route', round_number=1):
            pass
        self.entered.set()
        assert self.release.wait(5)
        return QueryResponse(answer=Answer(status='insufficient_evidence', text='Done.'),
            trace=Trace(run_id=str(uuid4()), steps=(), total_latency_ms=0,
                        usage=Usage(estimated_cost_usd=0), stop_reason='no_new_evidence'))


ORIGIN = 'http://localhost:8000'


@pytest.fixture
def authenticated():
    from src.auth.settings import AuthSettings
    auth = FakeAuth()
    entered, release = threading.Event(), threading.Event()
    config = WebSettings(mode='fixture', auth_mode='local', max_seconds=10)
    auth_settings = AuthSettings(enabled=True, allowed_origins=(ORIGIN,), secure_cookies=False,
                                 allow_insecure_loopback=True)
    app = create_app(config, lambda _s: FakeAuthenticatedRuntime(auth.database, entered, release),
                     auth_settings=auth_settings, auth_service=auth)
    with TestClient(app, base_url=ORIGIN) as client:
        wait_for(lambda: app.state.runtime is not None)
        yield SimpleNamespace(app=app, client=client, auth=auth, entered=entered, release=release)
    release.set()


def test_authenticated_owner_sees_progress_other_user_and_anonymous_do_not(authenticated):
    client, token = authenticated.client, str(uuid4())
    url = '/api/query/progress/' + token
    headers = {'Origin': ORIGIN, 'X-CSRF-Token': 'csrf-user-a', 'X-Query-Progress-ID': token}
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.post, '/api/query', json={'question': 'retention'},
                             headers={**headers, 'Cookie': 'rag_session=cookie-a'})
        assert authenticated.entered.wait(3)
        assert authenticated.app.state.busy is True
        owner = client.get(url, headers={'Cookie': 'rag_session=cookie-a'})
        assert owner.status_code == 200 and 'no-store' in owner.headers['cache-control']
        assert any(s['stage'] == 'route' for s in owner.json()['stages'])
        other = client.get(url, headers={'Cookie': 'rag_session=cookie-b'})
        unknown = client.get('/api/query/progress/' + str(uuid4()), headers={'Cookie': 'rag_session=cookie-b'})
        assert other.status_code == unknown.status_code == 404
        assert other.json() == unknown.json() and 'no-store' in other.headers['cache-control']
        anonymous = client.get(url)
        assert anonymous.status_code == 401 and 'no-store' in anonymous.headers['cache-control']
        assert client.get(url, headers={'Cookie': 'rag_session=expired'}).status_code == 401
        assert client.get(url, headers={'Cookie': 'rag_session=cookie-a',
                                                 'Origin': 'https://untrusted.invalid'}).status_code == 403
        authenticated.release.set()
        assert future.result(5).status_code == 200
    # Only the admitted query may extend a session; progress reads never do.
    assert authenticated.auth.touches.count(True) == 1


def test_authenticated_binding_requires_valid_csrf(authenticated):
    token = str(uuid4())
    response = authenticated.client.post('/api/query', json={'question': 'retention'}, headers={'Origin': ORIGIN, 'X-CSRF-Token': 'wrong', 'X-Query-Progress-ID': token,
                 'Cookie': 'rag_session=cookie-a'})
    assert response.status_code == 403
    assert authenticated.app.state.observability._progress == {}

"""Request correlation before admission; private admin history with no model I/O."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.api.runtime import WebSettings
from src.observability import emit
from tests.test_api import settings, wait_ready


def test_ingress_ids_cover_success_validation_and_denial_without_echoing_inputs():
    app = create_app(settings())
    with TestClient(app) as client:
        wait_ready(client)
        for payload, headers, status in (
            ({'question': 'retention'}, {}, 200),
            ({'question': 'DO-NOT-LOG-THIS', 'user': 'SECRET-ID'}, {}, 422),
            ({'question': 'SECRET-QUESTION'}, {'Origin': 'https://untrusted.invalid'}, 403),
        ):
            response = client.post('/api/query', json=payload, headers=headers)
            assert response.status_code == status
            request_id = response.headers['x-request-id']
            assert str(UUID(request_id)) == request_id
            detail = app.state.observability.get(request_id)
            assert detail['request']['http_status_code'] == status
            assert detail['request']['http_finished']
            assert all(event['request_id'] == request_id for event in detail['stages'])
            text = json.dumps(detail)
            assert all(secret not in text for secret in ('DO-NOT-LOG-THIS', 'SECRET-ID', 'SECRET-QUESTION', 'untrusted.invalid'))
        assert client.get('/api/observability/access').status_code == 401
        assert client.get('/api/observability/requests').status_code == 401
        assert client.get('/api/observability/requests/' + request_id).status_code == 401
        # Diagnostics/health polling never recursively fills query history.
        assert app.state.observability.list()['total'] == 3


def test_thread_context_and_http_timeout_are_separate_until_real_worker_finishes(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    app = create_app(settings(max_seconds=.08))
    with TestClient(app) as client:
        wait_ready(client)
        runtime = app.state.runtime
        original = runtime.query
        def controlled(body):
            emit('policy.resolve', 'start')
            entered.set()
            assert release.wait(3)
            emit('policy.resolve', 'end', status='ok')
            return original(body)
        monkeypatch.setattr(runtime, 'query', controlled)
        try:
            response = client.post('/api/query', json={'question': 'retention'})
            assert entered.is_set() and response.status_code == 504
            request_id = response.headers['x-request-id']
            pending = app.state.observability.get(request_id)
            assert pending['request']['worker_status'] == 'running'
            assert pending['request']['completed_at'] is None
            assert pending['request']['http_status_code'] == 504
            assert client.post('/api/query', json={'question': 'retention'}).status_code == 429
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while app.state.busy and time.monotonic() < deadline:
            time.sleep(.005)
        detail = app.state.observability.get(request_id)
        assert detail['request']['worker_status'] == 'ok'
        assert detail['request']['status'] == 'timeout'
        assert detail['request']['completed_at'] is not None
        # Worker time starts after admission, while HTTP time includes admission.
        assert detail['request']['duration_ms'] >= detail['request']['http_duration_ms']
        endings = [stage['stage'] for stage in detail['stages']
                   if stage['event'] == 'end' and stage['stage'] in ('http', 'worker')]
        assert endings == ['http', 'worker']
        assert [stage['event'] for stage in detail['stages'] if stage['stage'] == 'policy.resolve'] == ['start', 'end']


def test_failed_recorder_does_not_break_queries(monkeypatch):
    app = create_app(settings())
    def broken():
        raise RuntimeError('sensitive recorder exception')
    monkeypatch.setattr(app.state.observability, 'start_request', broken)
    with TestClient(app) as client:
        wait_ready(client)
        assert client.post('/api/query', json={'question': 'retention'}).status_code == 200


def test_evidence_environment_and_admin_defaults_are_explicit(monkeypatch):
    assert WebSettings().evidence_doc_limit == 8
    assert WebSettings().observability_admin_user_ids == ()
    monkeypatch.setenv('RAG_EVIDENCE_DOC_LIMIT', '5')
    monkeypatch.setenv('RAG_OBSERVABILITY_ADMIN_USER_IDS', 'user-one, user-two')
    value = WebSettings.from_env()
    assert value.evidence_doc_limit == 5
    assert value.observability_admin_user_ids == ('user-one', 'user-two')
    monkeypatch.setenv('RAG_EVIDENCE_DOC_LIMIT', '10')
    assert WebSettings.from_env().evidence_doc_limit == 10
    monkeypatch.setenv('RAG_EVIDENCE_DOC_LIMIT', '9')
    with pytest.raises(ValueError):
        WebSettings.from_env()

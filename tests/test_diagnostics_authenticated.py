"""Admin is an explicit stable-ID capability, independent of document grants."""
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg')

from src.api.main import create_app
from src.api.runtime import WebSettings
from src.auth.accounts import disable_account
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import acl_db
from tests.test_authenticated_runtime import app_db, make_runtime, QUESTION
from tests.test_authenticated_api import ORIGIN, sign_in, wait_ready


def test_admin_reads_require_current_session_and_explicit_id_not_role_or_document_scope(app_db):
    fixture = app_db
    # Labels-only account has zero document grants, yet explicit admin capability.
    admin_id = fixture.users['labels_only'].user_id
    config = WebSettings(mode='fixture', auth_mode='local',
        observability_admin_user_ids=(admin_id,), max_seconds=20)
    bundle = make_runtime(fixture, settings=config)
    app = create_app(config, lambda _settings: bundle.runtime,
        auth_settings=fixture.auth.settings, auth_service=fixture.auth)
    with TestClient(app, base_url=ORIGIN) as client:
        wait_ready(client)
        denied = client.post('/api/query', json={'question': QUESTION})
        key = denied.headers['x-request-id']
        for path in ('access', 'requests', 'requests/' + key):
            response = client.get('/api/observability/' + path)
            assert response.status_code == 401 and 'no-store' in response.headers['cache-control']
        sign_in(client, fixture, 'executive')  # CEO label never grants admin.
        for path in ('access', 'requests', 'requests/' + key):
            assert client.get('/api/observability/' + path).status_code == 403
        client.cookies.clear()
        sign_in(client, fixture, 'labels_only')
        assert client.get('/api/observability/access').json() == {'allowed': True}
        listing = client.get('/api/observability/requests')
        assert listing.status_code == 200 and 'no-store' in listing.headers['cache-control']
        assert listing.json()['history_scope'] == 'current_process'
        detail = client.get('/api/observability/requests/' + key)
        assert detail.status_code == 200
        assert detail.json()['request']['http_status_code'] == 401
        assert client.get('/api/observability/requests/' + str(uuid4())).status_code == 404
        assert client.get('/api/observability/requests/not-a-uuid').status_code == 404
        assert client.get('/api/observability/requests?limit=51').status_code == 422
        assert client.get('/api/observability/requests', headers={'Origin': 'https://untrusted.invalid'}).status_code == 403
        # Revocation takes effect on the next read; no cached admin identity.
        disable_account(fixture.logins['labels_only'], url=url_string(fixture.urls.migration))
        assert client.get('/api/observability/access').status_code == 401
        assert client.get('/api/observability/requests/' + key).status_code == 401


def test_empty_allowlist_denies_all_and_auth_worker_preserves_query_context(app_db):
    bundle = make_runtime(app_db)
    app = create_app(bundle.settings, lambda _settings: bundle.runtime,
        auth_settings=app_db.auth.settings, auth_service=app_db.auth)
    with TestClient(app, base_url=ORIGIN) as client:
        wait_ready(client)
        headers = sign_in(client, app_db)
        assert client.get('/api/observability/access').status_code == 403
        response = client.post('/api/query', json={'question': QUESTION}, headers=headers)
        assert response.status_code == 200
        detail = app.state.observability.get(response.headers['x-request-id'])
        assert detail['request']['run_id'] == response.json()['trace']['run_id']
        assert detail['request']['worker_status'] == 'ok'
        assert any(event['stage'] == 'policy.resolve' for event in detail['stages'])
        assert all(event['request_id'] == response.headers['x-request-id'] for event in detail['stages'])
        encoded = json.dumps(detail)
        assert QUESTION not in encoded and 'Synthetic document' not in encoded

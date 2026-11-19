"""Deployment readiness is live metadata, never a model/body reload.

Pure cases use a real Database type with a synthetic held-connection fixture;
no service, credentials, models or network are accessed. Root owns PG/Docker
acceptance separately, including the genuine Compose smoke fixture.
"""

from contextlib import contextmanager
import asyncio
import json
from pathlib import Path
import stat
import subprocess
import sys
import time
from threading import Event, current_thread
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest
import sqlalchemy as sa

from src.access.contracts import PolicyUnavailable
from src.api import deployment_check as checks
from src.api.main import create_app
from src.api.runtime import AuthenticatedRuntime, WebSettings, build_runtime
from src.auth.settings import AuthSettings
from src.contracts import SnapshotRef
from src.db.runtime import Database


SNAPSHOT = SnapshotRef(snapshot_id='readiness-fixture', corpus_sha256='a' * 64,
    catalog_sha256='b' * 64, document_count=3)
READY = {'state': 'ready', 'corpus_sha256': SNAPSHOT.corpus_sha256,
    'catalog_sha256': SNAPSHOT.catalog_sha256, 'document_count': 3}
FEED = {'state': 'ready', 'corpus_sha256': SNAPSHOT.corpus_sha256,
    'expected_records': 2, 'imported_records': 2, 'coverage': 'synthetic_fixture'}


class Result:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def all(self):
        return self.value

    def mappings(self):
        return self

    def one_or_none(self):
        return self.value

    def scalar_one(self):
        return self.value


def database(*, head=('0004_jira_feed',), snapshot=READY, jira_count=2, feed=FEED,
             known_count=False, failure=None):
    actual = object.__new__(Database)
    actual.queries, actual.transactions = [], 0

    class Connection:
        def __init__(self):
            self.responses = iter([list(head), snapshot] + ([] if known_count else [jira_count])
                + ([feed] if jira_count else []))

        def execute(self, statement):
            actual.queries.append(str(statement))
            if failure is not None:
                raise failure
            return Result(next(self.responses))

    @contextmanager
    def transaction(context=None):
        assert context is None, 'Readiness must never acquire document authority'
        actual.transactions += 1
        yield Connection()

    actual.transaction = transaction
    return actual


@pytest.mark.parametrize('head', [(), ('0003_document_policy',), ('0004_jira_feed', 'another_head')])
def test_wrong_or_missing_schema_fails_before_catalogue_queries(head):
    actual = database(head=head)
    with pytest.raises(PolicyUnavailable):
        checks.assert_storage_ready(actual, SNAPSHOT)
    assert actual.transactions == 1 and len(actual.queries) == 1


@pytest.mark.parametrize('changes', [{'state': 'importing'}, {'corpus_sha256': 'c' * 64},
    {'catalog_sha256': 'c' * 64}, {'document_count': 4}])
def test_ready_snapshot_requires_every_fingerprint_field(changes):
    with pytest.raises(PolicyUnavailable):
        checks.assert_storage_ready(database(snapshot={**READY, **changes}), SNAPSHOT)


@pytest.mark.parametrize('feed', [None, {**FEED, 'state': 'importing'},
    {**FEED, 'imported_records': 1}, {**FEED, 'expected_records': 1},
    {**FEED, 'corpus_sha256': 'c' * 64}, {**FEED, 'coverage': 'unknown'}])
def test_catalogue_jira_requires_complete_matching_ready_feed(feed):
    with pytest.raises(PolicyUnavailable):
        checks.assert_storage_ready(database(feed=feed), SNAPSHOT)


@pytest.mark.parametrize('coverage', ['synthetic_fixture', 'verified_records'])
def test_ready_metadata_uses_one_transaction_and_no_protected_body_query(coverage):
    actual = database(feed={**FEED, 'coverage': coverage})
    assert checks.assert_storage_ready(actual, SNAPSHOT) is None
    assert actual.transactions == 1 and len(actual.queries) == 4
    text = '\n'.join(actual.queries).lower()
    assert 'canonical_text' not in text and 'document_revisions' not in text
    assert 'chunks' not in text and 'rag.tickets ' not in text


def test_existing_runtime_catalogue_count_avoids_live_catalogue_scan():
    actual = database(known_count=True)
    runtime = object.__new__(AuthenticatedRuntime)
    runtime.database, runtime.snapshot = actual, SNAPSHOT
    runtime.settings = WebSettings(auth_mode='local')
    runtime.jira_tool = SimpleNamespace(_jira_doc_ids=('one', 'two'))
    assert runtime.assert_ready() is None
    assert len(actual.queries) == 3
    assert not any('rag.documents' in query for query in actual.queries)


def test_no_jira_catalogue_needs_no_feed_but_false_receipt_cannot_skip_jira():
    receipt = checks.DeploymentReceipt(snapshot=SNAPSHOT, jira_required=False)
    actual = database(jira_count=0)
    checks.assert_storage_ready(actual, SNAPSHOT, receipt=receipt)
    assert len(actual.queries) == 3
    with pytest.raises(PolicyUnavailable):
        checks.assert_storage_ready(database(), SNAPSHOT, receipt=receipt)


def test_database_failure_is_sanitized_and_never_returns_ready():
    actual = database(failure=sa.exc.SQLAlchemyError('synthetic private DSN marker'))
    with pytest.raises(PolicyUnavailable) as failed:
        checks.assert_storage_ready(actual, SNAPSHOT)
    assert 'private DSN' not in str(failed.value)


@pytest.mark.parametrize('count', [True, -1, 4, 2.0])
def test_invalid_prevalidated_count_is_rejected_before_connection(count):
    actual = database()
    with pytest.raises(PolicyUnavailable):
        checks.assert_storage_ready(actual, SNAPSHOT, jira_documents=count)
    assert actual.transactions == 0


def test_receipt_publication_is_live_verified_readable_exclusive_and_idempotent(tmp_path):
    actual = database()
    path = tmp_path / 'deployment.json'
    first = checks.publish_receipt(actual, SNAPSHOT, path)
    original = path.read_bytes()
    assert first.jira_required and first.snapshot == SNAPSHOT
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert checks.publish_receipt(actual, SNAPSHOT, path) == first
    assert path.read_bytes() == original and actual.transactions == 2
    changed = {**first.model_dump(mode='json'), 'jira_required': False}
    path.write_text(json.dumps(changed))
    with pytest.raises(PolicyUnavailable):
        checks.publish_receipt(actual, SNAPSHOT, path)
    assert json.loads(path.read_text()) == changed
    assert list(tmp_path.iterdir()) == [path]


def test_no_receipt_is_published_after_failed_live_validation(tmp_path):
    path = tmp_path / 'deployment.json'
    with pytest.raises(PolicyUnavailable):
        checks.publish_receipt(database(feed=None), SNAPSHOT, path)
    assert not path.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', ['{', '{"format":"x","format":"y"}',
    '{"snapshot":NaN}', 'x' * 65537])
def test_partial_ambiguous_or_oversized_receipts_fail_closed(tmp_path, value):
    path = tmp_path / 'deployment.json'
    path.write_text(value)
    with pytest.raises(PolicyUnavailable):
        checks.load_receipt(path, SNAPSHOT)


@pytest.mark.parametrize('failure', ['schema', 'feed'])
def test_build_runtime_rejects_incompatible_metadata_before_loading_models(tmp_path, monkeypatch, failure):
    from src.storage import postgres_store
    from src.api import runtime as runtime_module
    path = tmp_path / 'snapshot.json'
    path.write_text(SNAPSHOT.model_dump_json())
    actual = database(head=('old_schema',)) if failure == 'schema' else database(feed=None)
    monkeypatch.setattr(postgres_store, 'PostgresDocumentStore', lambda *a, **kw: pytest.fail('Store loaded early'))
    monkeypatch.setattr(runtime_module, 'load_store_retriever', lambda *a, **kw: pytest.fail('Model loaded early'))
    with pytest.raises(PolicyUnavailable):
        build_runtime(WebSettings(auth_mode='local', snapshot_file=path), database=actual, authenticated=True)


class ObservedRuntime(AuthenticatedRuntime):
    def __init__(self, actual_database, *, failing=False):
        self.database, self.failing = actual_database, failing
        self.checked, self.threads = Event(), []

    def assert_ready(self):
        self.threads.append(current_thread().name)
        self.checked.set()
        if self.failing:
            raise PolicyUnavailable('synthetic private dependency marker')

    def info(self):
        return {'mode': 'retrieval', 'retrieval': 'bm25', 'generation_enabled': False,
                'documents': 511958, 'source_types': ('hidden_inventory',)}


def application(runtime, auth=None):
    auth = auth or AuthSettings(enabled=True)
    return create_app(WebSettings(auth_mode='local'), lambda _settings: runtime,
        auth_settings=auth, auth_service=SimpleNamespace(database=runtime.database))


def test_live_ready_recovers_from_dependency_failure_on_existing_bounded_auth_worker():
    runtime = ObservedRuntime(object())
    with TestClient(application(runtime)) as client:
        assert runtime.checked.wait(2)
        deadline = time.monotonic() + 2
        while (ready := client.get('/api/ready')).status_code != 200 and time.monotonic() < deadline:
            time.sleep(.005)
        assert ready.status_code == 200 and ready.headers['cache-control'] == 'no-store'
        assert set(ready.json()) == {'status', 'busy', 'mode', 'retrieval', 'generation_enabled'}
        runtime.failing = True
        failed = client.get('/api/ready')
        assert failed.status_code == 503 and failed.json()['status'] == 'not_ready'
        assert failed.headers['cache-control'] == 'no-store'
        assert 'hidden_inventory' not in failed.text and 'private dependency' not in failed.text
        assert client.get('/api/health').status_code == 200
        runtime.failing = False
        assert client.get('/api/ready').status_code == 200
        assert runtime.threads[0].startswith('rag_')
        assert runtime.threads[1:] and all(name.startswith('rag-auth') for name in runtime.threads[1:])


def test_startup_final_validation_cannot_fail_on_busy_auth_admission():
    """Hold real bounded auth admission exactly as the factory finishes.

    Old code permanently discarded the runtime on AuthWorker busy429. Startup
    now completes on the original RAG worker, while live /ready still fails
    closed until the physically occupied authentication worker is released.
    """
    factory_entered, factory_release = Event(), Event()
    auth_entered, auth_release = Event(), Event()
    runtime = ObservedRuntime(object())

    def factory(_settings):
        factory_entered.set()
        assert factory_release.wait(5), 'Test did not release runtime factory'
        return runtime

    def occupied_auth():
        auth_entered.set()
        assert auth_release.wait(5), 'Test did not release authentication worker'

    settings = AuthSettings(enabled=True, max_waiting=0, worker_timeout_seconds=5)
    app = create_app(WebSettings(auth_mode='local'), factory,
        auth_settings=settings, auth_service=SimpleNamespace(database=runtime.database))

    async def scenario():
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(factory_entered.wait, 2)
            pending = asyncio.create_task(app.state.auth_worker.call(occupied_auth))
            try:
                assert await asyncio.to_thread(auth_entered.wait, 2)
                assert app.state.auth_worker.pending == 1
                factory_release.set()
                await asyncio.wait_for(asyncio.shield(app.state.initialization), 2)
                assert app.state.runtime is runtime and app.state.error is None
                assert runtime.threads == ['rag_0']
                assert app.state.auth_worker.pending == 1
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url='https://localhost:8000') as client:
                    assert (await client.get('/api/ready')).status_code == 503
                    assert (await client.get('/api/health')).status_code == 200
                    auth_release.set()
                    await pending
                    assert (await client.get('/api/ready')).status_code == 200
                    assert runtime.threads[-1].startswith('rag-auth')
            finally:
                factory_release.set()
                auth_release.set()
                await pending

    asyncio.run(scenario())


def test_timed_out_readiness_keeps_physical_auth_worker_slot_until_completion():
    class BlockedRuntime(ObservedRuntime):
        blocked = False

        def assert_ready(self):
            super().assert_ready()
            if self.blocked:
                entered.set()
                assert release.wait(5), 'Test did not release bounded readiness worker'

    entered, release = Event(), Event()
    runtime = BlockedRuntime(object())
    app = application(runtime, AuthSettings(enabled=True, max_waiting=0, worker_timeout_seconds=.05))
    with TestClient(app) as client:
        try:
            assert runtime.checked.wait(2)
            deadline = time.monotonic() + 2
            while client.get('/api/ready').status_code != 200 and time.monotonic() < deadline:
                time.sleep(.005)
            runtime.blocked = True
            assert client.get('/api/ready').status_code == 503
            assert entered.is_set() and app.state.auth_worker.pending == 1
            # A timed-out HTTP waiter cannot free admission or add an unbounded
            # executor queue; the physical worker still owns the only slot.
            assert client.get('/api/ready').status_code == 503
            assert app.state.auth_worker.pending == 1
            assert client.get('/api/health').status_code == 200
        finally:
            runtime.blocked = False
            release.set()
        deadline = time.monotonic() + 2
        while (ready := client.get('/api/ready')).status_code != 200 and time.monotonic() < deadline:
            time.sleep(.005)
        assert ready.status_code == 200 and app.state.auth_worker.pending == 0


def test_failed_startup_never_publishes_runtime_but_liveness_stays_200():
    runtime = ObservedRuntime(object(), failing=True)
    with TestClient(application(runtime)) as client:
        assert runtime.checked.wait(2)
        result = client.get('/api/ready')
        assert result.status_code == 503 and result.headers['cache-control'] == 'no-store'
        assert 'private dependency' not in result.text
        assert client.get('/api/health').status_code == 200


def test_prestart_help_never_imports_database_or_native_model_modules():
    program = '''
import importlib.abc, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sqlalchemy','psycopg','alembic','numpy','torch','pyarrow','sentence_transformers'}:
            raise AssertionError('Unexpected heavy/dependency import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
sys.argv=['deployment_check','--help']
runpy.run_module('src.api.deployment_check',run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--publish-receipt' in result.stdout and '--receipt' in result.stdout


def test_cli_failure_never_prints_database_or_configuration_details(tmp_path, monkeypatch, capsys):
    from src.db.runtime import DatabaseSettings
    path = tmp_path / 'snapshot.json'
    path.write_text(SNAPSHOT.model_dump_json())

    def unavailable(*a, **kw):
        raise RuntimeError('synthetic private credential marker')

    monkeypatch.setattr(DatabaseSettings, 'from_env', unavailable)
    assert checks.main(['--snapshot', str(path), '--receipt', str(tmp_path / 'absent.json')]) == 1
    captured = capsys.readouterr()
    assert captured.out == 'Deployment is not ready.\n' and captured.err == ''

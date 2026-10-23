"""Application behavior without network, credentials, or model downloads."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.runtime import (FixtureProvider, FixtureRetriever, Runtime, WebSettings,
                             build_runtime, load_retriever)
from src.contracts import QueryRequest, QueryResponse


def settings(**kwargs):
    return WebSettings(mode='fixture', demo_access=True, **kwargs)


def test_empty_optional_environment_values_use_defaults(monkeypatch):
    # Compose emits empty optional fields when no local override was supplied.
    for key in WebSettings.model_fields:
        monkeypatch.delenv('RAG_' + key.upper(), raising=False)
    for key in ('FP16', 'ALLOWED_DOC_IDS_FILE', 'ALLOWED_SOURCES', 'MODEL'):
        monkeypatch.setenv('RAG_' + key, '')
    monkeypatch.setenv('RAG_MODE', 'fixture')
    config = WebSettings.from_env()
    assert config.fp16 is None and config.allowed_doc_ids_file is None
    assert config.allowed_sources == () and config.mode == 'fixture'
    monkeypatch.setenv('RAG_FP16', 'not-a-boolean')
    with pytest.raises(ValueError):
        WebSettings.from_env()


def wait_ready(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get('/api/ready')
        if response.status_code == 200:
            return response.json()
        time.sleep(.005)
    pytest.fail('Application did not become ready')


def test_lifecycle_loads_once_and_workflows_return_contracts():
    loaded = []
    def factory(config):
        loaded.append(config)
        return build_runtime(config)
    with TestClient(create_app(settings(), factory)) as client:
        info = wait_ready(client)
        assert info['mode'] == 'fixture' and info['generation_enabled']
        assert client.get('/api/health').json() == {'status': 'alive', 'ready': True}
        for mode in ('baseline', 'agentic'):
            response = client.post('/api/query', json={'question': 'How long are artifacts retained?', 'mode': mode})
            assert response.status_code == 200
            body = QueryResponse.model_validate(response.json())
            assert body.answer.status == 'answered'
            assert body.answer.citation_ids == (body.sources[0].citation_id,)
            assert '90 days' in body.sources[0].chunk.text
            assert bool([s for s in body.trace.steps if s.critic]) == (mode == 'agentic')
        assert len(loaded) == 1


@pytest.mark.parametrize('body', [
    {'question': ''}, {'question': '   '}, {'question': 'x'*8001},
    {'question': 'policy', 'mode': 'invalid'}, {'question': 'policy', 'user': {'user_id': 'admin'}},
    {'question': 'policy', 'allowed_doc_ids': ['fixture-policy']},
    {'question': 'policy', 'after': '2026-01-01'},
    {'question': 'policy', 'after': '2026-02-01T00:00:00Z', 'before': '2026-01-01T00:00:00Z'},
])
def test_request_validation_rejects_identity_and_invalid_inputs(body):
    with TestClient(create_app(settings())) as client:
        wait_ready(client)
        assert client.post('/api/query', json=body).status_code == 422


def test_partial_missing_evidence_and_provider_failure():
    with TestClient(create_app(settings())) as client:
        wait_ready(client)
        partial = client.post('/api/query', json={'question': 'Who pays the storage fee?', 'mode': 'agentic'})
        body = QueryResponse.model_validate(partial.json())
        assert partial.status_code == 200 and body.answer.status == 'insufficient_evidence'
        assert body.trace.stop_reason == 'no_new_evidence'
        assert any(s.critic and 'Storage fee payer' in s.critic.missing_facts for s in body.trace.steps)
        assert len([s for s in body.trace.steps if s.tool == 'text_search']) == 2
        empty = client.post('/api/query', json={'question': 'unavailable information', 'mode': 'agentic'})
        assert empty.json()['answer']['status'] == 'insufficient_evidence'
        assert not empty.json()['sources']
        failed = client.post('/api/query', json={'question': 'demo failure', 'mode': 'agentic'})
        assert failed.status_code == 502
        assert failed.json()['answer']['status'] == 'error'
        assert failed.json()['trace']['stop_reason'] == 'error'


def test_source_date_origin_and_empty_document_scope(tmp_path):
    with TestClient(create_app(settings())) as client:
        wait_ready(client)
        assert client.post('/api/query', json={'question': 'policy', 'source_types': ['gmail']}).status_code == 403
        assert client.post('/api/query', json={'question': 'policy'}, headers={'Origin': 'https://untrusted.example'}).status_code == 403
        filtered = client.post('/api/query', json={'question': 'policy', 'after': '2020-01-01T00:00:00Z'})
        assert filtered.status_code == 200 and not filtered.json()['sources']
    scope = tmp_path / 'scope.json'
    scope.write_text('[]')
    with TestClient(create_app(settings(allowed_doc_ids_file=scope))) as client:
        wait_ready(client)
        response = client.post('/api/query', json={'question': 'policy'})
        assert response.status_code == 403
        assert response.json()['detail']['code'] == 'scope_denied'


@pytest.mark.parametrize('violation', ['document', 'source', 'date'])
def test_final_boundary_rejects_misbehaving_retriever(violation):
    class BadRetriever(FixtureRetriever):
        def search(self, request):
            # Simulate an adapter ignoring every supplied filter.
            evidence = FixtureRetriever().search(request.model_copy(update={'source_types': (), 'after': None}))
            if violation == 'document':
                return (evidence[0].model_copy(update={'chunk': evidence[0].chunk.model_copy(update={'doc_id': 'SECRET'})}),)
            if violation == 'source':
                return (evidence[0].model_copy(update={'chunk': evidence[0].chunk.model_copy(update={'source_type': 'gmail'})}),)
            return evidence
    config = WebSettings(mode='retrieval', demo_access=True)
    with TestClient(create_app(config, lambda s: Runtime(s, BadRetriever()))) as client:
        wait_ready(client)
        response = client.post('/api/query', json={'question': 'policy', **({'after': '2020-01-01T00:00:00Z'} if violation == 'date' else {})})
        assert response.status_code == 502
        assert '90 days' not in response.text and 'SECRET' not in response.text


def test_readiness_during_loading_and_failed_startup():
    began, release = threading.Event(), threading.Event()
    def factory(config):
        began.set()
        assert release.wait(3)
        raise FileNotFoundError('private path and secret value')
    with TestClient(create_app(settings(), factory)) as client:
        try:
            assert began.wait(2)
            assert client.get('/api/health').status_code == 200
            assert client.get('/api/ready').status_code == 503
            assert client.post('/api/query', json={'question': 'policy'}).status_code == 503
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while client.app.state.error is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert 'private path' not in client.get('/api/ready').text
        assert 'missing' in client.get('/api/ready').json()['message']


def test_timeout_keeps_worker_reserved_and_health_responsive():
    began, release = threading.Event(), threading.Event()
    class Slow(FixtureRetriever):
        def search(self, request):
            began.set()
            assert release.wait(3)
            return super().search(request)
    config = WebSettings(mode='retrieval', demo_access=True, max_seconds=.08)
    with TestClient(create_app(config, lambda s: Runtime(s, Slow()))) as client:
        wait_ready(client)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(client.post, '/api/query', json={'question': 'policy'})
                assert began.wait(2)
                assert client.get('/api/health').status_code == 200
                assert client.post('/api/query', json={'question': 'policy'}).status_code == 429
                assert first.result(timeout=2).status_code == 504
                assert client.post('/api/query', json={'question': 'policy'}).status_code == 429
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while client.get('/api/ready').json()['busy'] and time.monotonic() < deadline:
            time.sleep(.005)
        assert client.post('/api/query', json={'question': 'policy'}).status_code == 200


def test_explicit_demo_and_live_opt_in_and_process_budget(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(ValueError, match='RAG_DEMO_ACCESS'):
        build_runtime(WebSettings(mode='fixture'))
    with pytest.raises(ValueError, match='Live mode requires'):
        build_runtime(WebSettings(mode='live', demo_access=True))
    provider = FixtureProvider()
    provider.reserved_usd = .02
    config = WebSettings(mode='live', demo_access=True, budget_usd=.02)
    with TestClient(create_app(config, lambda s: Runtime(s, FixtureRetriever(), provider))) as client:
        assert wait_ready(client)['budget_remaining_usd'] == 0
        response = client.post('/api/query', json={'question': 'policy'})
        assert response.status_code == 429 and response.json()['detail']['code'] == 'spending_limit'


def test_retrieval_only_never_constructs_provider_and_static_shell(tmp_path, monkeypatch):
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'index.html').write_text('<main>Built UI</main>')
    (tmp_path / 'assets/app.js').write_text('/* built */')
    import src.api.runtime as runtime_module
    monkeypatch.setattr(runtime_module, 'AnthropicProvider', lambda *a: pytest.fail('Paid provider constructed'))
    monkeypatch.setattr(runtime_module, 'load_retriever', lambda s: FixtureRetriever())
    config = WebSettings(mode='retrieval', demo_access=True, frontend_dist=tmp_path)
    with TestClient(create_app(config)) as client:
        assert wait_ready(client)['generation_enabled'] is False
        response = client.post('/api/query', json={'question': 'policy', 'mode': 'agentic'})
        assert response.status_code == 200
        assert response.json()['trace']['usage']['api_calls'] == 0
        assert response.json()['sources']
        assert client.get('/').text == '<main>Built UI</main>'
        assert client.get('/assets/app.js').status_code == 200


def test_shared_spending_reservations_reject_next_generation_without_paid_calls():
    from src.generation.generator import AnthropicConfig, AnthropicProvider
    class Messages:
        created = 0
        def count_tokens(self, **kwargs):
            return SimpleNamespace(input_tokens=100)
        def create(self, **kwargs):
            self.created += 1
            payload = json.loads(kwargs['messages'][0]['content'])
            answer = dict(status='answered', text='Synthetic test answer.',
                          citation_ids=[payload['evidence'][0]['citation_id']])
            return SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(answer))],
                usage=SimpleNamespace(input_tokens=100, output_tokens=10), stop_reason='end_turn')
    messages = Messages()
    config = WebSettings(mode='live', demo_access=True, budget_usd=.0003)
    provider = AnthropicProvider(AnthropicConfig(model='mock-only', allow_paid=True,
        budget_usd=.0003, input_usd_per_million=1, output_usd_per_million=1,
        max_output_tokens=100), client=SimpleNamespace(messages=messages))
    with TestClient(create_app(config, lambda s: Runtime(s, FixtureRetriever(), provider))) as client:
        wait_ready(client)
        assert client.post('/api/query', json={'question': 'policy'}).status_code == 200
        second = client.post('/api/query', json={'question': 'policy'})
        assert second.status_code == 429
        assert second.json()['trace']['stop_reason'] == 'budget'
        assert messages.created == 1 and provider.reserved_usd == pytest.approx(.0002)
        assert client.get('/api/ready').json()['budget_remaining_usd'] == pytest.approx(.0001)


def test_agent_rejects_scope_violation_before_critic_or_generator():
    class Malicious(FixtureRetriever):
        def search(self, request):
            hit = super().search(request)[0]
            return (hit.model_copy(update={'chunk': hit.chunk.model_copy(update={'doc_id': 'private'})}),)
    class NeverCall(FixtureProvider):
        def complete(self, *args):
            pytest.fail('Unauthorized evidence reached a model')
    with TestClient(create_app(settings(), lambda s: Runtime(s, Malicious(), NeverCall()))) as client:
        wait_ready(client)
        failed = client.post('/api/query', json={'question': 'policy', 'mode': 'agentic'})
        assert failed.status_code == 502
        assert failed.json()['sources'] == []
        assert '90 days' not in failed.text


def test_hybrid_manifest_failures_and_shared_corpus(tmp_path, monkeypatch):
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import BGEModel, CorpusDense, INDEX_VERSION
    from src.config import BGE_QUERY_PREFIX as QUERY_PREFIX
    from src.retrieval.hybrid import HybridRetriever
    corpus = object()
    bm25 = SimpleNamespace(doc_ids=['fixture-policy'], corpus=corpus, manifest={'corpus_sha256': 'abc'})
    monkeypatch.setattr(CorpusBM25, 'load', lambda *a, **k: bm25)
    model = {'name': 'BAAI/bge-base-en-v1.5', 'revision': 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', 'max_seq_length': 512, 'fp16': True, 'query_prefix': QUERY_PREFIX, 'normalize': True}
    manifest = {'version': INDEX_VERSION, 'complete': True, 'corpus_sha256': 'abc', 'shards': 1, 'model': model}
    def save():
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    save()
    config = WebSettings(mode='retrieval', demo_access=True, retrieval='hybrid', dense_index=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_retriever(config)
    (tmp_path / 'doc_ids.json').write_text('["fixture-policy"]')
    with pytest.raises(ValueError, match='precision'):
        load_retriever(config.model_copy(update={'fp16': False}))
    with pytest.raises(ValueError, match='shard'):
        load_retriever(config)
    for suffix in ('.npy', '.parquet'):
        (tmp_path / ('shard_00000'+suffix)).touch()
    manifest['corpus_sha256'] = 'different'
    save()
    with pytest.raises(ValueError, match='same corpus'):
        load_retriever(config)
    manifest['corpus_sha256'] = 'abc'
    save()
    captured = {}
    def fake_init(self, device, **kwargs):
        captured.update(kwargs)
        self.identity = model
    monkeypatch.setattr(BGEModel, '__init__', fake_init)
    def load(*args, **kwargs):
        assert kwargs['corpus'] is corpus
        assert kwargs['model'].identity == model
        return SimpleNamespace(doc_ids=bm25.doc_ids, manifest=manifest)
    monkeypatch.setattr(CorpusDense, 'load', load)
    hybrid = load_retriever(config)
    assert isinstance(hybrid, HybridRetriever)
    assert captured['model'] == 'bge-base' and captured['fp16']
    assert hybrid.config.bm25_weight == .7
    assert hybrid.bm25.corpus is corpus

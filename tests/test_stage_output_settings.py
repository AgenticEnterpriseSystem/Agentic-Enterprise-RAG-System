"""Explicit stage limits and truthful exports, with no services or SDK clients."""
import json
import socket
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.api import runtime as R
from src.contracts import QueryRequest, SnapshotRef
from src.evaluation.experiment_contracts import GenerationBudgets, ModelProvenance
from src.generation.generator import AnthropicConfig, AnthropicProvider
from src.retrieval.store import StoreBM25Retriever


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('Stage settings tests prohibit network and SDK clients')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def test_stage_defaults_preserve_existing_run_and_cost_limits():
    settings = R.WebSettings()
    assert settings.model_max_output_tokens == 1024
    assert settings.critic_max_output_tokens == 2048
    assert settings.max_tokens == 16000 and settings.max_seconds == 120
    assert settings.question_budget_usd == .1 and settings.budget_usd == 0


@pytest.mark.parametrize('field', ['model_max_output_tokens', 'critic_max_output_tokens'])
@pytest.mark.parametrize('value', [True, False, 127, 8193, 1024.0, '2048'])
def test_stage_settings_are_strict_bounded_server_integers(field, value):
    with pytest.raises(ValueError):
        R.WebSettings(**{field: value})


def test_stage_environment_uses_explicit_ascii_decimal_conversion(monkeypatch):
    monkeypatch.setenv('RAG_MODEL_MAX_OUTPUT_TOKENS', '1536')
    monkeypatch.setenv('RAG_CRITIC_MAX_OUTPUT_TOKENS', '3072')
    settings = R.WebSettings.from_env()
    assert settings.model_max_output_tokens == 1536 and settings.critic_max_output_tokens == 3072
    monkeypatch.setenv('RAG_CRITIC_MAX_OUTPUT_TOKENS', '3e3')
    with pytest.raises(ValueError):
        R.WebSettings.from_env()


def test_provider_factory_uses_model_cap_without_constructing_sdk_client(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'SYNTHETIC-TEST-ONLY')
    settings = R.WebSettings(mode='live', allow_paid=True, model='fixture-model', budget_usd=.5,
        input_price=1, output_price=5, model_max_output_tokens=1536, critic_max_output_tokens=3072)
    provider = R._provider(settings, allow_fixture=False)
    assert isinstance(provider, AnthropicProvider) and provider.client is None
    assert provider.config.max_output_tokens == 1536
    assert provider.config.budget_usd == .5 and settings.question_budget_usd == .1


def test_benchmark_agent_receives_distinct_stage_allowances():
    settings = R.WebSettings(mode='fixture', demo_access=True,
        model_max_output_tokens=1536, critic_max_output_tokens=3072)
    runtime = R.Runtime(settings, R.FixtureRetriever(), R.FixtureProvider())
    assert runtime.agent.critic.max_output_tokens == 3072
    assert runtime.agent.config.critic_max_output_tokens == 3072
    assert runtime.agent.config.generation_max_output_tokens == 1536


def test_retrieval_only_does_not_consult_or_construct_model_provider(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('Retrieval-only has no model provider')
    monkeypatch.setattr(R, 'AnthropicProvider', denied)
    runtime = R.Runtime(R.WebSettings(mode='retrieval'), R.FixtureRetriever(), None)
    assert runtime.agent is None and runtime.provider is None


def authenticated_runtime(monkeypatch, *, provider, settings=None, **attestation):
    """Keep real constructor validation; replace only database-bound services."""
    settings = settings or R.WebSettings(auth_mode='local')
    database = object()
    snapshot = SnapshotRef(snapshot_id='stage-unit', corpus_sha256='a' * 64,
        catalog_sha256='b' * 64, document_count=1)
    resolver = SimpleNamespace(database=database, snapshot=snapshot)
    retriever = SimpleNamespace(snapshot=snapshot, catalog_handle=object(), source_types=())
    monkeypatch.setattr(R.ValidatedCatalog, 'require', lambda *_a, **_k: object())
    for name, class_name in [('src.retrieval.tools', 'JiraTool'), ('src.storage.runs', 'RunRepository')]:
        module = ModuleType(name)
        setattr(module, class_name, lambda *_a, **_k: object())
        monkeypatch.setitem(sys.modules, name, module)
    runtime = R.AuthenticatedRuntime(settings, retriever, resolver, database=database,
        provider=provider, **attestation)
    # A real store-backed class identity suffices to exercise the pure scientific
    # profile; no catalogue, index, body read or model is needed for this method.
    backend = object.__new__(StoreBM25Retriever)
    backend._manifest = json.dumps({'synthetic': True})
    runtime.retriever = backend
    return runtime


def test_actual_anthropic_caps_export_adaptive_only_and_injected_base_wins(monkeypatch):
    settings = R.WebSettings(auth_mode='local', model_max_output_tokens=1024, critic_max_output_tokens=3072)
    provider = AnthropicProvider(AnthropicConfig(model='fixture-model', max_output_tokens=768))
    runtime = authenticated_runtime(monkeypatch, provider=provider, settings=settings)
    profile = runtime._scientific_profile(QueryRequest(question='Synthetic', mode='agentic'))
    assert profile['budgets']['max_output_tokens'] == 768
    assert profile['budgets']['critic_max_output_tokens'] == 3072
    assert profile['retrieval_config']['configured_stage_output_limits'] == {
        'generation_max_output_tokens': 768, 'critic_max_output_tokens': 3072}
    baseline = runtime._scientific_profile(QueryRequest(question='Synthetic', mode='baseline'))
    assert baseline['budgets']['max_output_tokens'] == 768
    assert 'critic_max_output_tokens' not in baseline['budgets']
    assert 'critic_max_output_tokens' not in baseline['retrieval_config']['configured_stage_output_limits']
    assert provider.client is None


@pytest.mark.parametrize('attested', [False, True])
def test_opaque_provider_cap_is_not_invented_but_can_be_explicitly_attested(monkeypatch, attested):
    kwargs = {'provider_critic_max_output_tokens': 2048} if attested else {}
    runtime = authenticated_runtime(monkeypatch, provider=R.FixtureProvider(),
        provider_identity=ModelProvenance(provider='scripted-fixture', model_id='stage-unit'), **kwargs)
    profile = runtime._scientific_profile(QueryRequest(question='Synthetic', mode='agentic'))
    assert ('critic_max_output_tokens' in profile['budgets']) is attested
    if attested:
        assert profile['budgets']['critic_max_output_tokens'] == 2048
    assert profile['retrieval_config']['configured_stage_output_limits']['critic_max_output_tokens'] == 2048


@pytest.mark.parametrize('value', [True, 1024, '2048'])
def test_custom_critic_attestation_must_match_configured_stage_cap(monkeypatch, value):
    with pytest.raises(ValueError, match='Attested critic output'):
        authenticated_runtime(monkeypatch, provider=R.FixtureProvider(), provider_critic_max_output_tokens=value)


def test_old_budget_documents_omit_absent_critic_cap_without_rebinding():
    old = {'top_k': 5, 'max_question_chars': 8000, 'max_context_chars': 24000,
        'max_output_tokens': 1024, 'max_run_tokens': 16000, 'max_run_seconds': 120.,
        'max_run_cost_usd': .1, 'retries': 0}
    assert GenerationBudgets.model_validate(old).model_dump(mode='json') == old
    explicit = GenerationBudgets.model_validate({**old, 'critic_max_output_tokens': 2048})
    assert explicit.model_dump(mode='json')['critic_max_output_tokens'] == 2048

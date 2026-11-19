"""Operator export uses the issued current capture and genuinely verified files.

All provider/index/model declarations here describe synthetic test doubles. The
files prove the export join and no-reference boundary, not scientific quality.
Root alone executes these disposable PostgreSQL cases.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.access.contracts import AccessDenied, EvidenceValidationError, StaleContext
from src.agents.routing import ModelRouter, ROUTER_PROMPT
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.api.runtime import AuthenticatedRuntime, RuntimeConfigurationError
from src.config import ROOT
from src.contracts import QueryRequest, ScopeBinding
from src.evaluation.experiment_contracts import ExperimentManifest
from src.evaluation.experiment_inputs import (
    ExperimentAdmission, RuntimeQuestion, VerifiedInputs, canonical_json_sha256,
    capture_code_provenance, file_sha256, question_set_sha256, verify_inputs,
)
from src.evaluation.experiment_io import RunDirectory
from src.generation.generator import (SYSTEM_PROMPT, AnthropicConfig, AnthropicProvider,
    GroundedGenerator)
from tests.db_support import pg_urls
from tests.jira_support import env, ready, source
from tests.test_experiment_inputs import GOLD_MARKER, _git as fixture_git, make_verified_inputs_fixture
from tests.test_structured_runtime import ScriptedRouter, request, runtime


QUESTION = 'Show the exact native record and complete permitted count.'


def verified_export_inputs(tmp_path, data, actual, captured=None, *, question=QUESTION, binding=None, profile=None):
    """Adapt the real tiny-file fixture to the actual imported fixture snapshot."""
    profile = profile or actual.export_profile(captured)
    manifest, verified = make_verified_inputs_fixture(tmp_path, question_ids=('q1',), workflow=profile['workflow'])
    files = replace(verified.files, corpus=data.source.corpus)
    for relative, expected in profile['source_files'].items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        source_bytes = (ROOT / path).read_bytes()
        destination = files.source_repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_bytes)
        assert file_sha256(destination) == expected
    rows = json.loads(files.questions.read_text())
    for row in rows:
        if row['question_id'] == 'q1':
            row['question'] = question
            row['expected_doc_ids'] = ['one']  # scoring-only, never forwarded
    files.questions.write_text(json.dumps(rows))
    files.catalog.write_text(''.join(json.dumps(entry.model_dump(mode='json')) + '\n' for entry in data.source.entries))
    binding = binding or (captured.binding if captured is not None else ScopeBinding.model_validate(profile['scope']))
    scope = {'binding': binding.model_dump(mode='json'), 'origin': 'synthetic'}
    files.scope.write_text(json.dumps(scope))
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    files.index_manifest.write_text(canonical(profile['index_manifest']))
    files.retrieval_config.write_text(canonical(profile['retrieval_config']))
    files.generator_prompt.write_text(SYSTEM_PROMPT)
    if files.critic_prompt is not None:
        files.critic_prompt.write_text(CRITIC_PROMPT)
    if profile['router_prompt_sha256'] is not None:
        files = replace(files, router_prompt=files.generator_prompt.with_name('router.txt'))
        files.router_prompt.write_text(ROUTER_PROMPT)
    values = manifest.model_dump()
    values.update({key: profile[key] for key in ('generation_model', 'budgets', 'temperature', 'workflow',
        'max_rounds', 'generator_prompt_sha256', 'critic_prompt_sha256', 'router_prompt_sha256', 'retrieval')})
    values.update(scope=binding, code=capture_code_provenance(files.source_repo),
        question_set_sha256=question_set_sha256((RuntimeQuestion(question_id='q1', question=question),)),
        scope_fingerprint_sha256=canonical_json_sha256(scope))
    manifest = ExperimentManifest.model_validate(values)
    files.runtime_config.write_text(json.dumps({name: manifest.model_dump(mode='json')[name] for name in (
        'generation_model', 'retrieval', 'budgets', 'temperature', 'workflow', 'max_rounds')}))
    return manifest, verify_inputs(manifest, files)


def capture(data):
    actual, provider, _ = runtime(data,
        ScriptedRouter(filters={'status': 'Exact absent status'}, lookup_too=True, mixed=True))
    captured = actual.query_with_capture(request(QUESTION), data.identities['intern'])
    return actual, provider, captured


def test_current_capture_exports_native_zero_and_exact_generation_input_without_references(ready, tmp_path):
    actual, provider, captured = capture(ready)
    manifest, verified = verified_export_inputs(tmp_path / 'inputs', ready, actual, captured)
    directory = RunDirectory.create(tmp_path / 'run', manifest, verified)
    exported = actual.export_capture(captured, manifest=manifest, verified_inputs=verified,
        question_id='q1', directory=directory)
    assert exported.run_id == captured.run_id == exported.response.trace.run_id
    assert exported.response.contract_version == '3'
    assert exported.generation_evidence == captured.generation_evidence
    assert {item.kind for item in exported.generation_evidence} == {'passage', 'record', 'aggregate'}
    assert next(item for item in exported.generation_evidence if item.kind == 'aggregate').count == 0
    serialized = exported.model_dump_json()
    assert GOLD_MARKER not in serialized and 'expected_doc_ids' not in serialized and 'gold_answer' not in serialized
    assert provider.prompts and GOLD_MARKER not in json.dumps(provider.prompts)
    assert exported.question == QUESTION
    assert directory.path.exists()


@pytest.mark.parametrize('mismatch', ['question', 'scope'])
def test_individually_verified_but_different_question_or_scope_cannot_relabel_capture(ready, tmp_path, mismatch):
    actual, _provider, captured = capture(ready)
    binding = ready.resolver.resolve(ready.identities['executive']).binding if mismatch == 'scope' else None
    manifest, verified = verified_export_inputs(tmp_path / 'inputs', ready, actual, captured,
        question='A different selected question' if mismatch == 'question' else QUESTION, binding=binding)
    with pytest.raises((AccessDenied, EvidenceValidationError, ValueError)):
        actual.export_capture(captured, manifest=manifest, verified_inputs=verified, question_id='q1')


def test_unissued_or_other_runtime_capture_cannot_export_even_when_binding_and_files_match(ready, tmp_path):
    actual, _provider, captured = capture(ready)
    manifest, verified = verified_export_inputs(tmp_path / 'inputs', ready, actual, captured)
    forged = object.__new__(type(captured))
    other, _provider, _ = runtime(ready, ScriptedRouter())
    for issuer, value in ((actual, forged), (other, captured)):
        with pytest.raises((AccessDenied, EvidenceValidationError, ValueError)):
            issuer.export_capture(value, manifest=manifest, verified_inputs=verified, question_id='q1')


def test_revoked_capture_or_changed_verified_input_cannot_be_exported_later(ready, tmp_path):
    actual, _provider, captured = capture(ready)
    manifest, verified = verified_export_inputs(tmp_path / 'inputs', ready, actual, captured)
    verified.files.generator_prompt.write_text('Changed producer prompt')
    with pytest.raises((AccessDenied, EvidenceValidationError, ValueError)):
        actual.export_capture(captured, manifest=manifest, verified_inputs=verified, question_id='q1')
    verified.files.generator_prompt.write_text(SYSTEM_PROMPT)
    ready.repository.revoke(ready.manifest)
    with pytest.raises(StaleContext):
        actual.export_capture(captured, manifest=manifest, verified_inputs=verified, question_id='q1')


def test_custom_router_without_explicit_attestation_can_query_but_cannot_export_profile(ready):
    actual, _provider, _ = runtime(ready, ScriptedRouter(), attest_router=False)
    captured = actual.query_with_capture(request(QUESTION), ready.identities['intern'])
    assert captured.response.answer.status == 'answered'
    with pytest.raises(EvidenceValidationError):
        actual.export_profile(captured)


def test_router_behavior_mutation_invalidates_existing_capture_and_next_query(ready, tmp_path):
    actual, _provider, captured = capture(ready)
    manifest, verified = verified_export_inputs(tmp_path / 'inputs', ready, actual, captured)
    actual.router.filters['status'] = 'Changed after capture'
    with pytest.raises(EvidenceValidationError):
        actual.export_capture(captured, manifest=manifest, verified_inputs=verified, question_id='q1')
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(request(QUESTION), ready.identities['intern'])


def test_model_router_must_use_the_exact_attested_runtime_provider():
    actual = object.__new__(AuthenticatedRuntime)
    actual.provider, actual._router_attestation = object(), None
    actual.router = ModelRouter(object())
    with pytest.raises(RuntimeConfigurationError):
        actual._router_profile()
    actual.router = ModelRouter(actual.provider)
    assert actual._router_profile() == {'kind': 'model_router', 'version': '1'}


@pytest.mark.parametrize('attestation', [None, {'kind': 'scripted_fixture', 'router_id': 'x', 'behavior': {}, 'extra': True},
    {'kind': 'model_router', 'router_id': 'x', 'behavior': {}},
    {'kind': 'scripted_fixture', 'router_id': 'x', 'behavior': []},
    {'kind': 'scripted_fixture', 'router_id': 'x', 'behavior': {'number': float('nan')}},
    {'kind': 'scripted_fixture', 'router_id': 'x', 'behavior': {'data': 'x' * 16385}}])
def test_fixture_attestation_rejects_unknown_keys_nonfinite_or_unbounded_values(attestation):
    with pytest.raises(RuntimeConfigurationError):
        AuthenticatedRuntime._fixture_route(attestation)


def test_distinct_attested_router_behavior_changes_persisted_effective_run_config(ready):
    import sqlalchemy as sa
    from src.db import schema
    first, _, _ = runtime(ready, ScriptedRouter(filters={'status': 'Exact absent status'}))
    second, _, _ = runtime(ready, ScriptedRouter(filters={'status': 'In Progress'}))
    run_ids = [actual.query(request(QUESTION), ready.identities['intern']).trace.run_id for actual in (first, second)]
    with ready.owner.connect() as connection:
        digests = connection.execute(sa.select(schema.query_runs.c.config_sha256).where(
            schema.query_runs.c.run_id.in_(run_ids))).scalars().all()
    assert len(digests) == 2 and len(set(digests)) == 2


def admission(data, tmp_path, *, body=None, actual=None):
    body = body or request(QUESTION)
    if actual is None:
        actual, _provider, _resolver = runtime(data, ScriptedRouter(
            filters={'status': 'Exact absent status'}, lookup_too=True, mixed=True))
    profile = actual.experiment_profile(body, data.identities['intern'])
    manifest, inputs = verified_export_inputs(tmp_path / 'experiment-inputs', data, actual,
        profile=profile, question=body.question)
    return actual, body, ExperimentAdmission(manifest, inputs, 'q1')


def test_experiment_profile_uses_current_admission_without_route_model_or_run_creation(ready):
    import sqlalchemy as sa
    from src.db import schema
    actual, provider, _resolver = runtime(ready, ScriptedRouter(mixed=True))
    profile = actual.experiment_profile(request(QUESTION), ready.identities['intern'])
    assert profile['scope'] == actual.resolver.resolve(ready.identities['intern']).binding.model_dump(mode='json')
    assert profile['workflow'] == 'fixed' and profile['max_rounds'] == 1
    assert profile['source_files'] == actual._source_files()
    assert provider.prompts == actual.router.calls == []
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one() == 0
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.aggregate_results)).scalar_one() == 0
    profile['retrieval_config']['method'] = 'invented'
    assert actual.experiment_profile(request(QUESTION), ready.identities['intern'])['retrieval_config']['method'] != 'invented'
    with pytest.raises(AccessDenied):
        actual.experiment_profile(request(source_types=('unregistered-fixture-platform',)), ready.identities['intern'])
    assert provider.prompts == []


def test_admitted_capture_exports_with_no_references_and_hashes_corpus_only_at_full_checks(ready, tmp_path, monkeypatch):
    import src.evaluation.experiment_inputs as module
    actual, body, expected = admission(ready, tmp_path / 'prepared')
    reads = []
    original = module.file_sha256
    def counted(path):
        if path == expected.inputs.files.corpus:
            reads.append(path)
        return original(path)
    monkeypatch.setattr(module, 'file_sha256', counted)
    with RunDirectory.create(tmp_path / 'run', expected.manifest, expected.inputs) as directory:
        reads.clear()
        captured = actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
        assert len(reads) == 1, 'Large inputs were rehashed at a provider/freshness boundary'
        exported = actual.export_capture(captured, manifest=expected.manifest,
            verified_inputs=expected.inputs, question_id='q1', directory=directory)
        assert len(reads) == 2  # Admission and the unchanged full export recheck.
        assert directory.pending_question_ids == ()
        assert exported.generation_evidence == captured.generation_evidence
        assert GOLD_MARKER not in exported.model_dump_json()
        assert GOLD_MARKER not in json.dumps(actual.provider.prompts)


@pytest.mark.parametrize('mismatch', ['input', 'scope', 'settings', 'question', 'raw_question', 'loaded_source', 'source_copy'])
def test_experiment_admission_rejects_stale_declarations_before_any_run_or_provider(ready, tmp_path, monkeypatch, mismatch):
    import sqlalchemy as sa
    from src.db import schema
    raw = request(' ' + QUESTION + ' ') if mismatch == 'raw_question' else None
    actual, body, expected = admission(ready, tmp_path, body=raw)
    identity = ready.identities['intern']
    if mismatch == 'input':
        expected.inputs.files.generator_prompt.write_text('Changed actual input')
    elif mismatch == 'scope':
        identity = ready.identities['executive']
    elif mismatch == 'settings':
        actual.settings = actual.settings.model_copy(update={'top_k': 2})
    elif mismatch == 'question':
        body = request('A different runtime question')
    elif mismatch == 'raw_question':
        assert body.question == expected.inputs.questions[0].question
    elif mismatch == 'loaded_source':
        changed = actual._source_stamps(); changed['src/api/runtime.py'] = None
        monkeypatch.setattr(actual, '_source_stamps', lambda: changed)
    else:
        (expected.inputs.files.source_repo / 'src/api/runtime.py').write_text('Changed copied runtime source')
    with pytest.raises((AccessDenied, EvidenceValidationError, StaleContext)):
        actual.query_with_capture(body, identity, experiment=expected)
    assert actual.provider.prompts == actual.router.calls == []
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one() == 0


@pytest.mark.parametrize('mutation', ['input', 'policy'])
def test_changed_inputs_during_full_recheck_are_caught_before_dispatch(ready, tmp_path, monkeypatch, mutation):
    actual, body, expected = admission(ready, tmp_path)
    original = VerifiedInputs.recheck
    def raced(value, manifest):
        checked = original(value, manifest)
        if mutation == 'input':
            checked.files.generator_prompt.write_text(SYSTEM_PROMPT + '\nRace after byte verification')
        else:
            ready.repository.revoke(ready.manifest)
        return checked
    monkeypatch.setattr(VerifiedInputs, 'recheck', raced)
    with pytest.raises((EvidenceValidationError, StaleContext)):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert actual.provider.prompts == actual.router.calls == []


@pytest.mark.parametrize('boundary', ['router', 'critic', 'generator'])
def test_nested_agent_dispatch_preserves_experiment_guard_before_provider_boundaries(ready, tmp_path, monkeypatch, boundary):
    from tests.test_authenticated_runtime import RecordingProvider
    provider = RecordingProvider()
    if boundary == 'router':
        actual, _, _ = runtime(ready, ModelRouter(provider), provider=provider, attest_router=False)
        body = request(QUESTION)
        target, method = ModelRouter, 'route'
    else:
        actual, _, _ = runtime(ready, ScriptedRouter(mixed=True), provider=provider)
        body = QueryRequest(question=QUESTION, mode='agentic' if boundary == 'critic' else 'baseline')
        target, method = (EvidenceCritic, 'assess') if boundary == 'critic' else (GroundedGenerator, 'generate')
    actual, body, expected = admission(ready, tmp_path, actual=actual, body=body)
    original = getattr(target, method)
    def raced(self, *args, **kwargs):
        expected.inputs.files.generator_prompt.write_text(SYSTEM_PROMPT + '\nChange before SDK boundary')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(target, method, raced)
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert provider.prompts == [], 'An SDK-facing boundary lost the additional experiment guard'


def test_experiment_guard_stops_generation_after_fake_count_tokens_changes_input(ready, tmp_path):
    calls = {'count': 0, 'create': 0}
    holder = {}
    def count_tokens(**_kwargs):
        calls['count'] += 1
        holder['inputs'].files.generator_prompt.write_text(SYSTEM_PROMPT + '\nChange during fake token counting')
        return SimpleNamespace(input_tokens=1)
    def create(**_kwargs):
        calls['create'] += 1
        pytest.fail('Generation proceeded after changed input at fake token counting')
    provider = AnthropicProvider(AnthropicConfig(model='synthetic-never-network-model', allow_paid=True,
        budget_usd=1, input_usd_per_million=1, output_usd_per_million=1),
        client=SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens, create=create)))
    base, _, resolver = runtime(ready, ScriptedRouter(mixed=True), provider=None)
    settings = base.settings.model_copy(update={'mode': 'live', 'budget_usd': 1, 'question_budget_usd': .1})
    router = ScriptedRouter(mixed=True)
    actual = AuthenticatedRuntime(settings, base.retriever, resolver, database=ready.database,
        provider=provider, router=router, router_attestation=router.export_attestation())
    actual, body, expected = admission(ready, tmp_path, actual=actual)
    holder['inputs'] = expected.inputs
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert calls == {'count': 1, 'create': 0}


def test_unattestable_experiment_profile_is_strict_while_ordinary_query_still_works(ready, tmp_path):
    actual, _, _ = runtime(ready, ScriptedRouter(), attest_router=False)
    with pytest.raises(EvidenceValidationError):
        actual.experiment_profile(request(QUESTION), ready.identities['intern'])
    assert actual.provider.prompts == actual.router.calls == []
    attested, body, expected = admission(ready, tmp_path)
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert actual.provider.prompts == actual.router.calls == []
    assert actual.query(body, ready.identities['intern']).answer.status == 'answered'


def test_prepared_runtime_ownership_change_during_route_never_reaches_a_provider(ready, tmp_path, monkeypatch):
    from tests.test_authenticated_runtime import RecordingProvider
    actual, body, expected = admission(ready, tmp_path)
    original = actual.router.route
    def swapped(*args, **kwargs):
        actual.provider = RecordingProvider()
        return original(*args, **kwargs)
    monkeypatch.setattr(actual.router, 'route', swapped)
    provider = actual.provider
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert provider.prompts == actual.provider.prompts == []


def test_external_tracked_source_symlink_target_is_pinned_before_provider_dispatch(ready, tmp_path, monkeypatch):
    actual, body, expected = admission(ready, tmp_path / 'prepared')
    source = expected.inputs.files.source_repo / 'src/api/runtime.py'
    target = tmp_path / 'external-source.py'
    target.write_bytes(source.read_bytes())
    source.unlink(); source.symlink_to(target)
    fixture_git(expected.inputs.files.source_repo, 'add', 'src/api/runtime.py')
    manifest = ExperimentManifest.model_validate({**expected.manifest.model_dump(),
        'code': capture_code_provenance(expected.inputs.files.source_repo)})
    expected = ExperimentAdmission(manifest, verify_inputs(manifest, expected.inputs.files), 'q1')
    original = GroundedGenerator.generate
    stamp_fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    link_stamp = tuple(getattr(source.lstat(), name) for name in stamp_fields)
    def changed(self, *args, **kwargs):
        target.write_bytes(target.read_bytes() + b'\n# External attested source changed\n')
        assert tuple(getattr(source.lstat(), name) for name in stamp_fields) == link_stamp
        return original(self, *args, **kwargs)
    monkeypatch.setattr(GroundedGenerator, 'generate', changed)
    with pytest.raises(EvidenceValidationError):
        actual.query_with_capture(body, ready.identities['intern'], experiment=expected)
    assert actual.provider.prompts == []


def test_source_changed_between_construction_and_ordinary_capture_cannot_be_attested(ready, tmp_path, monkeypatch):
    import src.api.runtime as module
    mirror = tmp_path / 'runtime-source'
    for path in (ROOT / 'src').rglob('*.py'):
        copy = mirror / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    monkeypatch.setattr(module, 'ROOT', mirror)
    actual, provider, _ = runtime(ready, ScriptedRouter(mixed=True), provider=None)
    changed = mirror / 'src/api/runtime.py'
    changed.write_bytes(changed.read_bytes() + b'\n# Code changed after this runtime loaded\n')
    captured = actual.query_with_capture(request(QUESTION), ready.identities['intern'])
    assert provider is None and captured.response.answer.status == 'insufficient_evidence'
    with pytest.raises(EvidenceValidationError, match='after construction'):
        actual.export_profile(captured)

"""Operator export uses the issued current capture and genuinely verified files.

All provider/index/model declarations here describe synthetic test doubles. The
files prove the export join and no-reference boundary, not scientific quality.
Root alone executes these disposable PostgreSQL cases.
"""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.access.contracts import AccessDenied, EvidenceValidationError, StaleContext
from src.agents.routing import ModelRouter
from src.api.runtime import AuthenticatedRuntime, RuntimeConfigurationError
from src.config import ROOT
from src.evaluation.experiment_contracts import ExperimentManifest
from src.evaluation.experiment_inputs import (
    RuntimeQuestion, canonical_json_sha256, capture_code_provenance, file_sha256, question_set_sha256, verify_inputs,
)
from src.evaluation.experiment_io import RunDirectory
from src.generation.generator import SYSTEM_PROMPT
from tests.db_support import pg_urls
from tests.jira_support import env, ready, source
from tests.test_experiment_inputs import GOLD_MARKER, make_verified_inputs_fixture
from tests.test_structured_runtime import ScriptedRouter, request, runtime


QUESTION = 'Show the exact native record and complete permitted count.'


def verified_export_inputs(tmp_path, data, actual, captured, *, question=QUESTION, binding=None):
    """Adapt the real tiny-file fixture to the actual imported fixture snapshot."""
    manifest, verified = make_verified_inputs_fixture(tmp_path, question_ids=('q1',))
    files = replace(verified.files, corpus=data.source.corpus)
    profile = actual.export_profile(captured)
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
    binding = binding or captured.binding
    scope = {'binding': binding.model_dump(mode='json'), 'origin': 'synthetic'}
    files.scope.write_text(json.dumps(scope))
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    files.index_manifest.write_text(canonical(profile['index_manifest']))
    files.retrieval_config.write_text(canonical(profile['retrieval_config']))
    files.generator_prompt.write_text(SYSTEM_PROMPT)
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

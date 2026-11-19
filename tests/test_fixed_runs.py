"""Fixed producer orchestration plus genuine PostgreSQL admission/export checks.

ToyRuntime tests describe only the orchestration protocol. The final cases use
the actual scoped runtime, issuer-bound capture and imported PostgreSQL store.
No paid SDK or downloaded model is called.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import QueryRequest
from src.evaluation import fixed_runs
from src.evaluation.experiment_contracts import ExperimentManifest, RuntimeCaseExport
from src.evaluation.experiment_inputs import (
    canonical_json_sha256, file_sha256, verify_inputs,
)
from src.evaluation.experiment_io import RunDirectory
from src.evaluation.fixed_runs import (
    FixedArm, FixedBudgetError, FixedRunError, UnresolvedAttempt,
    preflight_fixed_runs, prepare_fixed_arm, run_fixed_runs,
)
from tests.test_experiment_inputs import GOLD_MARKER, make_verified_inputs_fixture
from tests.test_experiment_io import case_for


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False), encoding='utf-8')


class ToyRuntime:
    def __init__(self, arm, profile, events):
        self.arm, self.profile, self.events = arm, profile, events
        self.calls, self.errors = [], set()
        self.identities = []
        self.after_export = None

    def experiment_profile(self, body, identity):
        self.events.append(('profile', self.arm.name))
        return deepcopy(self.profile)

    def query_with_capture(self, body, identity, *, experiment):
        self.events.append(('query', self.arm.name))
        self.calls.append((body, experiment))
        self.identities.append(identity)
        assert body.mode == 'baseline' and body.question == next(
            q.question for q in experiment.inputs.questions if q.question_id == experiment.question_id)
        return SimpleNamespace(run_id=self.arm.name + '-run-' + str(len(self.calls)),
            question_id=experiment.question_id)

    def export_capture(self, capture, *, manifest, verified_inputs, question_id, directory):
        value = case_for(manifest, verified_inputs, question_id, run_id=capture.run_id,
            outcome='error' if question_id in self.errors else 'answered').model_dump()
        value['response']['trace']['usage'] = {'api_calls': 0, 'estimated_cost_usd': 0}
        case = RuntimeCaseExport.model_validate(value)
        directory.append_case(case)
        if self.after_export is not None:
            self.after_export()
        return case


def toy_plan(tmp_path, *, names=('bm25', 'dense', 'hybrid'), mismatch=None, question_budget=0):
    base, verified = make_verified_inputs_fixture(tmp_path/'base')
    lexical = {'fixture': True, 'documents': 1, 'corpus_sha256': base.scope.snapshot.corpus_sha256}
    dense = {**lexical, 'model': {'name': 'fixture-encoder', 'revision': 'b'*40}}
    arms, runtimes, events = [], {}, []
    for name in names:
        folder = tmp_path/('prepared-' + name)
        folder.mkdir()
        index = deepcopy({'bm25': lexical, 'dense': dense} if name == 'hybrid'
                         else dense if name == 'dense' else lexical)
        configuration = {'method': 'store-' + name, 'top_k': 5,
            'router': {'kind': 'fixture-text'},
            'filters': {'source_types': [], 'after': None, 'before': None}}
        embedding = None
        if name in ('dense', 'hybrid'):
            configuration['dense_execution'] = {'block_rows': 4096, 'file_cache_policy': 'retain'}
            embedding = {'provider': 'sentence-transformers', 'model_id': 'fixture-encoder', 'revision': 'b'*40}
        if mismatch and name == 'hybrid':
            if mismatch == 'filters':
                configuration['filters']['source_types'] = ['jira']
            elif mismatch == 'router':
                configuration['router'] = {'kind': 'different-fixture-router'}
            elif mismatch == 'lexical':
                index['bm25']['different_build'] = True
            elif mismatch == 'dense_execution':
                configuration['dense_execution']['block_rows'] = 8192
            elif mismatch == 'embedding':
                embedding['revision'] = 'c'*40
        files = replace(verified.files, index_manifest=folder/'index.json',
            retrieval_config=folder/'retrieval.json', runtime_config=folder/'runtime.json')
        write_json(files.index_manifest, index)
        write_json(files.retrieval_config, configuration)
        values = base.model_dump()
        values.update(manifest_id='fixture-' + name,
            budgets={**values['budgets'], 'max_run_cost_usd': question_budget},
            retrieval={**values['retrieval'], 'retriever_id': 'store-' + name,
                'index_manifest_sha256': file_sha256(files.index_manifest),
                'config_sha256': canonical_json_sha256(configuration), 'embedding_model': embedding})
        manifest = ExperimentManifest.model_validate(values)
        write_json(files.runtime_config, {key: manifest.model_dump(mode='json')[key] for key in (
            'generation_model', 'retrieval', 'budgets', 'temperature', 'workflow', 'max_rounds')})
        inputs = verify_inputs(manifest, files)
        arm = FixedArm(name, manifest, inputs, tmp_path/'runs'/name)
        profile = {key: manifest.model_dump(mode='json')[key] for key in fixed_runs._PROFILE_FIELDS}
        profile.update(scope=manifest.scope.model_dump(mode='json'), source_files={},
            index_manifest=index, retrieval_config=configuration)
        arms.append(arm)
        runtimes[name] = ToyRuntime(arm, profile, events)
    @contextmanager
    def opened(arm):
        yield runtimes[arm.name], object()
    return tuple(arms), runtimes, opened, events


def test_all_actual_profiles_are_inspected_before_first_query_and_outputs_are_reference_free(tmp_path):
    arms, runtimes, opened, events = toy_plan(tmp_path)
    inspected = preflight_fixed_runs(arms, open_runtime=opened)
    assert inspected['passed'] and inspected['matched_configuration']
    assert all(not arm.directory.exists() for arm in arms)
    events.clear()
    result = run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    first_query = next(i for i, event in enumerate(events) if event[0] == 'query')
    assert {name for kind, name in events[:first_query] if kind == 'profile'} == {'bm25', 'dense', 'hybrid'}
    assert result['matched_complete'] and result['actual_cost_usd'] == result['api_calls'] == 0
    assert result['reserved_cost_usd'] == 0 and result['artifact_kind'] == 'scripted-conformance'
    for arm in arms:
        assert len(RunDirectory.load(arm.directory).load_cases(require_complete=True)) == 2
        for path in arm.directory.rglob('*.json'):
            assert GOLD_MARKER not in path.read_text()
            assert 'gold_answer' not in path.read_text() and 'expected_doc_ids' not in path.read_text()
        assert all(GOLD_MARKER not in call[0].model_dump_json() for call in runtimes[arm.name].calls)


@pytest.mark.parametrize('mismatch', ['filters', 'router', 'lexical', 'embedding', 'dense_execution'])
def test_mismatched_arm_components_or_filters_fail_before_any_provider_or_run_output(tmp_path, mismatch):
    arms, runtimes, opened, _events = toy_plan(tmp_path, mismatch=mismatch)
    with pytest.raises(FixedRunError):
        run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert all(not runtime.calls for runtime in runtimes.values())
    assert not (tmp_path/'attempts').exists()
    assert all(not arm.directory.exists() for arm in arms)


def test_later_selected_input_change_fails_before_first_arm_dispatch(tmp_path):
    arms, runtimes, opened, _events = toy_plan(tmp_path)
    question_file = arms[-1].inputs.files.questions
    rows = json.loads(question_file.read_text())
    next(row for row in rows if row['question_id'] == 'q2')['question'] += ' '
    write_json(question_file, rows)
    with pytest.raises(ValueError):
        run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert all(not runtime.calls for runtime in runtimes.values())


def test_retained_error_is_a_completed_attempt_and_canonical_resume_never_retries_it(tmp_path):
    arms, runtimes, opened, _events = toy_plan(tmp_path, names=('bm25',))
    runtimes['bm25'].errors.add('q1')
    result = run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert not result['matched_complete'] and result['arms']['bm25']['errors'] == 1
    count = len(runtimes['bm25'].calls)
    old = arms[0]
    updated = ExperimentManifest.model_validate(old.manifest.model_dump() | {
        'created_at': old.manifest.created_at + timedelta(minutes=1)})
    requested = (FixedArm(old.name, updated, old.inputs, old.directory),)
    run_fixed_runs(requested, open_runtime=opened, journal_dir=tmp_path/'attempts',
        resume=True, max_total_cost_usd=0.0)
    assert len(runtimes['bm25'].calls) == count
    assert RunDirectory.load(old.directory).manifest.created_at == old.manifest.created_at
    error = next(case for case in RunDirectory.load(old.directory).load_cases() if case.question_id == 'q1')
    assert error.error is not None and error.response.answer.status == 'error'


def test_safe_committed_checkpoint_resumes_only_pending_questions(tmp_path, monkeypatch):
    arms, runtimes, opened, _events = toy_plan(tmp_path, names=('bm25',))
    original = fixed_runs._Journal.committed
    def checkpoint(journal, item, case):
        original(journal, item, case)
        raise KeyboardInterrupt('Controlled stop after durable committed state')
    with monkeypatch.context() as patch:
        patch.setattr(fixed_runs._Journal, 'committed', checkpoint)
        with pytest.raises(KeyboardInterrupt):
            run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert len(runtimes['bm25'].calls) == 1
    run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts', resume=True)
    assert [admission.question_id for _body, admission in runtimes['bm25'].calls] == ['q1', 'q2']


def test_identity_callback_refreshes_before_every_question_without_changing_the_frozen_binding(tmp_path):
    arms, runtimes, _opened, _events = toy_plan(tmp_path, names=('bm25',))
    issued = []
    def current_identity():
        value = object()
        issued.append(value)
        return value
    @contextmanager
    def opened(_arm):
        yield runtimes['bm25'], current_identity
    run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert len(runtimes['bm25'].identities) == 2
    assert runtimes['bm25'].identities[0] is not runtimes['bm25'].identities[1]
    assert runtimes['bm25'].identities == issued[-2:]
    assert all(admission.manifest.scope == arms[0].manifest.scope
               for _body, admission in runtimes['bm25'].calls)


def test_case_written_without_final_export_return_stays_unresolved_and_never_retries(tmp_path):
    arms, runtimes, opened, _events = toy_plan(tmp_path, names=('bm25',))
    def uncertain():
        raise OSError('Controlled failure after case publication but before export returned')
    runtimes['bm25'].after_export = uncertain
    with pytest.raises(OSError):
        run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert len(RunDirectory.load(arms[0].directory).load_cases()) == 1
    assert not (arms[0].directory/'.writer.lock').exists()
    runtimes['bm25'].after_export = None
    with pytest.raises(UnresolvedAttempt):
        run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts', resume=True)
    assert len(runtimes['bm25'].calls) == 1
    state = json.loads((tmp_path/'attempts/state.json').read_text())
    assert state['attempts'][0]['state'] == 'captured'


def test_original_preflight_plan_can_start_via_resume_but_missing_journal_never_resets_existing_work(tmp_path):
    arms, runtimes, opened, _events = toy_plan(tmp_path, names=('bm25',))
    preflight_fixed_runs(arms, open_runtime=opened)
    result = run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts', resume=True)
    assert result['passed'] and len(runtimes['bm25'].calls) == 2
    (tmp_path/'attempts/state.json').unlink()
    with pytest.raises(UnresolvedAttempt):
        run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts', resume=True)
    assert len(runtimes['bm25'].calls) == 2


def test_empty_journal_recovers_header_setup_without_restarting_existing_work(tmp_path):
    arms, _runtimes, opened, _events = toy_plan(tmp_path)
    plan = fixed_runs._plan(arms, fixed_runs._amount(0), False)
    with fixed_runs._Journal(tmp_path/'attempts', plan, resume=False):
        with RunDirectory.create(arms[0].directory, arms[0].manifest, arms[0].inputs):
            pass
    result = run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts', resume=True)
    assert result['matched_complete']


def test_conservative_reservations_survive_restart_and_span_arms(tmp_path):
    arms, _runtimes, _opened, _events = toy_plan(tmp_path)
    charged = []
    for arm in arms:
        manifest = ExperimentManifest.model_validate(arm.manifest.model_dump() | {
            'generation_model': {'provider': 'anthropic', 'model_id': 'not-executed'},
            'budgets': arm.manifest.budgets.model_dump() | {'max_run_cost_usd': .1}})
        charged.append(FixedArm(arm.name, manifest, arm.inputs, arm.directory))
    plan = fixed_runs._plan(charged, fixed_runs._amount(.15), True)
    with fixed_runs._Journal(tmp_path/'attempts', plan, resume=False) as journal:
        journal.reserve(charged[0], 'q1')
    with fixed_runs._Journal(tmp_path/'attempts', plan, resume=True) as journal:
        with pytest.raises(FixedBudgetError):
            journal.reserve(charged[1], 'q1')
        assert fixed_runs._amount(journal.state['reserved_usd']) == fixed_runs._amount(.1)


def test_scripted_profiles_with_positive_question_ceiling_still_reserve_zero(tmp_path):
    arms, _runtimes, opened, _events = toy_plan(tmp_path, names=('bm25',), question_budget=.1)
    result = run_fixed_runs(arms, open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert result['reserved_cost_usd'] == result['actual_cost_usd'] == result['api_calls'] == 0
    assert result['artifact_kind'] == 'scripted-conformance'


def test_unapproved_real_generation_is_rejected_before_runtime_construction(tmp_path):
    arms, _runtimes, _opened, _events = toy_plan(tmp_path, names=('bm25',))
    arm = arms[0]
    values = arm.manifest.model_dump()
    values.update(generation_model={'provider': 'anthropic', 'model_id': 'not-executed'},
                  budgets=values['budgets'] | {'max_run_cost_usd': .1})
    manifest = ExperimentManifest.model_validate(values)
    write_json(arm.inputs.files.runtime_config, {key: manifest.model_dump(mode='json')[key] for key in (
        'generation_model', 'retrieval', 'budgets', 'temperature', 'workflow', 'max_rounds')})
    arm = FixedArm(arm.name, manifest, verify_inputs(manifest, arm.inputs.files), arm.directory)
    @contextmanager
    def forbidden(_arm):
        pytest.fail('Unapproved paid runtime was constructed')
        yield
    with pytest.raises(FixedBudgetError):
        run_fixed_runs((arm,), open_runtime=forbidden, journal_dir=tmp_path/'attempts', max_total_cost_usd=.2)
    assert not (tmp_path/'attempts').exists()


# Genuine-PG fixtures are imported only for their corresponding opt-in cases.
from tests.db_support import pg_urls
from tests.jira_support import PASSWORD, env, ready, source


def real_arm(tmp_path, data, *, provider=True, questions=('q1', 'q2')):
    from src.config import ROOT
    from tests.test_structured_runtime import ScriptedRouter, runtime
    actual, configured, _resolver = runtime(data, ScriptedRouter(mixed=True, lookup_too=True), provider=provider)
    grant = data.auth.login('intern', PASSWORD, client_ip='127.0.0.1')
    identity = data.auth.authenticate(grant.token, touch=False).identity
    question_file, split_file, catalog = (tmp_path/name for name in ('questions.json', 'splits.json', 'catalog.jsonl'))
    write_json(question_file, [{'question_id': qid, 'question': 'Show permitted notes for ' + qid,
        'gold_answer': GOLD_MARKER, 'expected_doc_ids': ['hidden']} for qid in (*questions, 'qc', 'qf')])
    write_json(split_file, {'version': 1, 'splits': {'development': list(questions),
        'calibration': ['qc'], 'final': ['qf']}})
    catalog.write_text(''.join(json.dumps(entry.model_dump(mode='json'))+'\n' for entry in data.source.entries))
    arm = prepare_fixed_arm('bm25', actual, identity, sources={'questions': question_file,
        'splits': split_file, 'corpus': data.source.corpus, 'catalog': catalog, 'source_repo': ROOT},
        prepared=tmp_path/'prepared', directory=tmp_path/'run', comparison={'experiment_id': 'real-fixed-fixture'})
    @contextmanager
    def opened(_arm):
        yield actual, lambda: data.auth.authenticate(grant.token, touch=True).identity
    return arm, actual, configured, opened


def test_real_scope_bound_prepare_and_producer_export_no_labels_or_invented_generation(ready, tmp_path):
    import sqlalchemy as sa
    from src.db import schema
    arm, actual, provider, opened = real_arm(tmp_path, ready)
    assert provider.prompts == []
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one() == 0
    assert file_sha256(arm.inputs.files.index_manifest) == arm.manifest.retrieval.index_manifest_sha256
    result = run_fixed_runs((arm,), open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert result['passed'] and not result['matched_complete']
    assert result['api_calls'] == result['actual_cost_usd'] == 0
    assert provider.prompts and GOLD_MARKER not in json.dumps(provider.prompts)
    cases = RunDirectory.load(arm.directory).load_cases(require_complete=True)
    assert len(cases) == 2 and all(case.generation_evidence for case in cases)
    assert all(GOLD_MARKER not in case.model_dump_json() for case in cases)


def test_real_retrieval_only_results_are_explicit_validation_not_generated_baselines(ready, tmp_path):
    arm, actual, provider, opened = real_arm(tmp_path, ready, provider=None, questions=('q1',))
    assert provider is None
    result = run_fixed_runs((arm,), open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert result['artifact_kind'] == 'retrieval-only-validation' and not result['matched_complete']
    case = RunDirectory.load(arm.directory).load_cases(require_complete=True)[0]
    assert case.generation_evidence == () and case.response.answer.status == 'insufficient_evidence'
    assert case.response.trace.usage.api_calls == 0


def test_real_scope_change_after_preparation_blocks_every_run_and_provider(ready, tmp_path):
    import sqlalchemy as sa
    from src.db import schema
    arm, actual, provider, opened = real_arm(tmp_path, ready)
    ready.repository.revoke(ready.manifest)
    with pytest.raises(ValueError):
        run_fixed_runs((arm,), open_runtime=opened, journal_dir=tmp_path/'attempts')
    assert provider.prompts == [] and not arm.directory.exists()
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one() == 0

"""Pure fixed CLI boundaries; no runtime, SQL, provider SDK, or model loading."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.evaluation import run_fixed as cli


def config_fixture(tmp_path, *, provider='disabled'):
    questions = [{'question_id': name, 'question': 'Question ' + name,
        'reference_answer': 'GOLD_MUST_NEVER_REACH_RUNTIME', 'expected_doc_ids': ['private-label']}
        for name in ('q1', 'q2', 'q3')]
    (tmp_path / 'questions.json').write_text(json.dumps(questions))
    (tmp_path / 'splits.json').write_text(json.dumps({'splits': {
        'development': ['q1'], 'calibration': ['q2'], 'final': ['q3']}}))
    generation = {'provider': 'disabled'} if provider == 'disabled' else {
        'provider': 'anthropic', 'model': 'explicit-fixture-model', 'budget_usd': 2,
        'input_price_per_million': 1, 'output_price_per_million': 2}
    value = {'format': 'rag-fixed-cli-v1', 'sources': {'questions': 'questions.json',
        'splits': 'splits.json', 'corpus': 'absent-corpus.jsonl', 'catalog': 'absent-catalog.jsonl',
        'source_repo': '.'}, 'authentication': {'snapshot_file': 'absent-snapshot.json',
        'credentials_file': 'absent-private-credentials.json', 'principal_kind': 'benchmark'},
        'retrieval': {'bm25_index': 'absent-bm25', 'dense_index': 'absent-dense'},
        'generation': generation, 'selection': {'split': 'development', 'count': 1},
        'output_root': 'output', 'comparison': {'experiment_id': 'cli-pure-fixture'}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(value))
    return path, value


def guarded_process(argv, *, environment=None):
    """A module-import firewall proves draft/help never reach external stacks."""
    script = r'''
import builtins, json, sys
original = builtins.__import__
blocked = {'sqlalchemy','psycopg','anthropic','numpy','torch','transformers',
    'sentence_transformers','bm25s','pyarrow','dotenv'}
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('EXTERNAL_STACK_IMPORTED')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from src.evaluation.run_fixed import main
raise SystemExit(main(json.loads(sys.argv[1])))
'''
    return subprocess.run([sys.executable, '-c', script, json.dumps(argv)], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, **(environment or {})}, text=True, capture_output=True, check=False)


def test_help_imports_no_external_stacks():
    result = guarded_process(['--help'])
    assert result.returncode == 0 and 'draft' in result.stdout
    assert 'EXTERNAL_STACK_IMPORTED' not in result.stderr


@pytest.mark.parametrize('provider', ['disabled', 'anthropic'])
def test_draft_is_unbound_file_only_and_does_not_read_credentials_or_reference_labels(tmp_path, provider):
    path, _ = config_fixture(tmp_path, provider=provider)
    target = tmp_path / 'draft.json'
    result = guarded_process(['draft', '--config', str(path), '--output', str(target)],
        environment={'RAG_ALLOW_PAID': 'true', 'RAG_MODE': 'live', 'ANTHROPIC_API_KEY': 'DO_NOT_PRINT_THIS_KEY'})
    assert result.returncode == 0, (result.stdout, result.stderr)
    value = json.loads(target.read_text())
    assert value['runtime_authority_verified'] is False and value['scope'] is None
    assert value['paid_model_calls'] == 0 and value['answers_generated'] is False
    assert value['question_ids'] == ['q1']
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert 'GOLD_MUST_NEVER_REACH_RUNTIME' not in target.read_text()
    assert 'private-label' not in target.read_text()
    assert 'DO_NOT_PRINT_THIS_KEY' not in result.stdout + result.stderr + target.read_text()
    assert not (tmp_path / 'output').exists()


def test_draft_refuses_to_overwrite_an_existing_file(tmp_path):
    path, _ = config_fixture(tmp_path)
    target = tmp_path / 'draft.json'
    target.write_text('ORIGINAL')
    assert cli.main(['draft', '--config', str(path), '--output', str(target)]) == 1
    assert target.read_text() == 'ORIGINAL'


@pytest.mark.parametrize('field', ['allow_paid', 'mode', 'auth_mode', 'model', 'input_price', 'output_price', 'budget_usd'])
def test_web_config_cannot_override_managed_authority_or_provider_controls(tmp_path, field):
    path, value = config_fixture(tmp_path)
    value['web'] = {field: True}
    path.write_text(json.dumps(value))
    with pytest.raises(cli.FixedCLIError):
        cli.load_config(path)


@pytest.mark.parametrize('mutation', [
    lambda value: value['generation'].update(provider='unknown'),
    lambda value: value['generation'].update(allow_paid=True),
    lambda value: value['generation'].update(model='cannot-label-disabled-as-generated'),
    lambda value: value['selection'].update(question_ids=['q1'], count=1),
    lambda value: value['selection'].update(split='final', allow_final=False),
    lambda value: value['authentication'].update(credentials_file=None),
    lambda value: value.update(comparison={'generation': {'allow_paid': True}}),
    lambda value: value.update(request_filters={'user_id': 'forged-authority'}),
    lambda value: value.update(web={'allowed_sources': ['jira']}, request_filters={'source_types': ['gmail']}),
])
def test_invalid_declarations_fail_before_runtime(tmp_path, mutation):
    path, value = config_fixture(tmp_path)
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises((cli.FixedCLIError, ValueError)):
        cli.load_config(path)


@pytest.mark.parametrize('raw', ['{"format":"one","format":"two"}', '{"value":NaN}', '{"value":1e999}'])
def test_configuration_rejects_ambiguous_or_nonfinite_json(tmp_path, raw):
    path = tmp_path / 'bad.json'
    path.write_text(raw)
    with pytest.raises(cli.FixedCLIError):
        cli.load_config(path)


@pytest.mark.parametrize('args', [[], ['--allow-paid'], ['--max-total-cost-usd', '2'],
    ['--allow-paid', '--max-total-cost-usd', '1'], ['--allow-paid', '--max-total-cost-usd', 'nan']])
def test_environment_cannot_authorize_paid_execution_and_gate_precedes_imports(tmp_path, args):
    path, _ = config_fixture(tmp_path, provider='anthropic')
    result = guarded_process(['run', '--config', str(path), *args], environment={
        'RAG_ALLOW_PAID': 'true', 'RAG_MODE': 'live', 'ANTHROPIC_API_KEY': 'DO_NOT_PRINT_THIS_KEY'})
    assert result.returncode == 1
    assert json.loads(result.stdout)['error_kind'] == 'FixedCLIError'
    assert 'EXTERNAL_STACK_IMPORTED' not in result.stderr
    assert 'DO_NOT_PRINT_THIS_KEY' not in result.stdout + result.stderr
    assert not (tmp_path / 'output').exists()


def test_live_run_requires_key_and_exact_explicit_cap(tmp_path, monkeypatch):
    path, _ = config_fixture(tmp_path, provider='anthropic')
    config = cli.load_config(path)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(cli.FixedCLIError, match='API key'):
        cli._authorization(config, 'run', allow_paid=True, max_total_cost_usd=2)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'unused-test-key')
    assert cli._authorization(config, 'run', allow_paid=True, max_total_cost_usd=2) is True
    assert cli._authorization(config, 'preflight') is False
    assert cli._authorization(config, 'draft') is False


@pytest.mark.parametrize('allow_paid, cap', [(True, 0), (False, 1), (True, 1)])
def test_disabled_validation_never_accepts_paid_authorization(tmp_path, allow_paid, cap):
    path, _ = config_fixture(tmp_path)
    with pytest.raises(cli.FixedCLIError):
        cli._authorization(cli.load_config(path), 'run', allow_paid=allow_paid, max_total_cost_usd=cap)


def test_output_override_and_relative_paths_are_resolved_without_reading_targets(tmp_path):
    path, _ = config_fixture(tmp_path)
    target = tmp_path / 'separate-preflight'
    config = cli.load_config(path, output_root=target)
    assert config['output_root'] == str(target.resolve())
    assert config['sources']['questions'] == str((tmp_path / 'questions.json').resolve())
    assert config['authentication']['credentials_file'] == str((tmp_path / 'absent-private-credentials.json').resolve())
    assert config['comparison']['generation']['provider'] == 'disabled'


@pytest.mark.parametrize('command', ['preflight', 'run', 'resume'])
def test_cli_prepares_or_loads_exact_plan_then_preflights_all_arms_before_execution(tmp_path, monkeypatch, command):
    path, _ = config_fixture(tmp_path)
    config = cli.load_config(path)
    events = []
    arms = tuple(SimpleNamespace(name=name, manifest=SimpleNamespace(question_ids=('q1',))) for name in cli.ARMS)
    @contextmanager
    def suite(_config, *, paid):
        events.append(('suite', paid))
        yield 'SHARED_RUNTIME_FACTORY'
    def prepare(_config, factory):
        assert factory == 'SHARED_RUNTIME_FACTORY'
        events.append('prepare-all')
        return arms
    def resume(_config):
        events.append('load-original-plan')
        return arms
    def preflight(received, *, open_runtime):
        assert received is arms and open_runtime == 'SHARED_RUNTIME_FACTORY'
        events.append('preflight-all')
    def run(received, **kwargs):
        assert received is arms and kwargs['open_runtime'] == 'SHARED_RUNTIME_FACTORY'
        assert kwargs['journal_dir'] == Path(config['output_root']) / 'attempts'
        assert kwargs['allow_paid'] is False and kwargs['max_total_cost_usd'] == 0
        assert kwargs['resume'] is (command == 'resume')
        events.append('execute-after-preflight')
        return {'passed': True, 'arms': {'bm25': {'cases': 1, 'errors': 1}},
            'reserved_cost_usd': 0, 'reported_cost_usd': 0, 'api_calls': 0,
            'artifact_kind': 'retrieval-only-validation', 'quality_scored': False}
    monkeypatch.setattr(cli, '_runtime_suite', suite)
    monkeypatch.setattr(cli, '_prepare', prepare)
    monkeypatch.setattr(cli, '_resume_arms', resume)
    monkeypatch.setitem(sys.modules, 'src.evaluation.fixed_runs', SimpleNamespace(
        preflight_fixed_runs=preflight, run_fixed_runs=run))
    result = cli.execute(config, command)
    expected = [('suite', False), 'load-original-plan' if command == 'resume' else 'prepare-all', 'preflight-all']
    if command != 'preflight':
        expected.append('execute-after-preflight')
    assert events == expected
    assert result['result_label'] == 'validation_only' and result['paid_model_calls'] == 0
    if command != 'preflight':
        assert result['execution']['arms']['bm25']['errors'] == 1
        assert result['execution']['artifact_kind'] == 'retrieval-only-validation'


@pytest.fixture
def fixed_cli_pg_urls():
    # Keep optional DB/model imports out of the pure CLI tests' collection path.
    from tests.db_support import pg_urls
    return pg_urls.__wrapped__()


@pytest.fixture
def fixed_cli_source(tmp_path):
    from tests.jira_support import source
    return source.__wrapped__(tmp_path)


@pytest.fixture
def fixed_cli_env(fixed_cli_pg_urls, fixed_cli_source):
    from tests.jira_support import env
    yield from env.__wrapped__(fixed_cli_pg_urls, fixed_cli_source)


@pytest.fixture
def fixed_cli_ready(fixed_cli_env):
    from tests.jira_support import ready
    return ready.__wrapped__(fixed_cli_env)


def test_real_three_arm_cli_keeps_one_backend_exact_human_scope_and_touches_each_question(
        fixed_cli_ready, tmp_path, monkeypatch, capsys):
    """Real SQL/RLS/store/index/capture/export; only the tiny encoder is stubbed."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sqlalchemy as sa
    from src.api.runtime import AuthenticatedRuntime
    from src.auth.sessions import AuthService
    from src.config import ROOT
    from src.db import schema
    from src.evaluation.experiment_io import RunDirectory
    from src.evaluation.scoped_retrieval import ScopedRetrievalRunner
    from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25
    from src.retrieval.corpus_dense import EmbeddingModel
    from src.storage.postgres_store import PostgresDocumentStore
    from tests.jira_support import PASSWORD

    data = fixed_cli_ready
    store = PostgresDocumentStore(data.database, data.source.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    texts = [row['title'] + '\n' + row['content'] for row in data.source.rows]
    lexical, dense = tmp_path / 'bm25', tmp_path / 'dense'
    index_config = BM25Config()
    index, statistics = build_bm25(texts, index_config)
    CorpusBM25(index, list(proof.doc_ids), index_config, manifest={'version': 1,
        'config': index_config.model_dump(), 'corpus_sha256': data.source.snapshot.corpus_sha256,
        'stats': statistics}).save(lexical)
    dense.mkdir()
    model = {'name': 'fixture-encoder', 'revision': 'c' * 40, 'normalize': True,
        'fp16': False, 'max_seq_length': 512, 'query_prefix': ''}
    (dense / 'manifest.json').write_text(json.dumps({'version': 1, 'complete': True,
        'documents': len(proof.entries), 'shards': 1, 'model': model,
        'corpus_sha256': data.source.snapshot.corpus_sha256}))
    (dense / 'doc_ids.json').write_text(json.dumps(list(proof.doc_ids)))
    np.save(dense / 'shard_00000.npy', np.array([[1., index / 20]
        for index in range(len(proof.entries))], dtype=np.float32), allow_pickle=False)
    pq.write_table(pa.table({'doc_index': list(range(len(proof.entries))),
        'start_char': [0] * len(proof.entries), 'end_char': [8] * len(proof.entries)}),
        dense / 'shard_00000.parquet')
    encoder_loads = []
    class Encoder:
        identity = model
        def encode(self, texts, query=False):
            return np.array([[1., 0.]], dtype=np.float32)
    def encoder(cls, *args, **kwargs):
        encoder_loads.append(1)
        return Encoder()
    monkeypatch.setattr(EmbeddingModel, 'for_index', classmethod(encoder))

    token = data.auth.login('intern', PASSWORD, client_ip='127.0.0.1').token.get_secret_value()
    credential, snapshot, catalog = (tmp_path / name for name in ('session.json', 'snapshot.json', 'catalog.jsonl'))
    credential.write_text(json.dumps({'format': 'rag-session-v1', 'token': token}))
    credential.chmod(0o600)
    snapshot.write_text(data.source.snapshot.model_dump_json())
    catalog.write_text(''.join(json.dumps(entry.model_dump(mode='json')) + '\n' for entry in data.source.entries))
    questions, splits = tmp_path / 'questions.json', tmp_path / 'splits.json'
    questions.write_text(json.dumps([{'question_id': qid, 'question': 'Exact source prose ' + qid,
        'reference_answer': 'GOLD_NEVER_ENTERS_FIXED_CLI', 'expected_doc_ids': ['hidden']}
        for qid in ('q1', 'q2', 'qc', 'qf')]))
    splits.write_text(json.dumps({'splits': {'development': ['q1', 'q2'], 'calibration': ['qc'], 'final': ['qf']}}))
    output = tmp_path / 'fixed-output'
    configuration = {'format': 'rag-fixed-cli-v1', 'sources': {'questions': str(questions),
        'splits': str(splits), 'corpus': str(data.source.corpus), 'catalog': str(catalog), 'source_repo': str(ROOT)},
        'authentication': {'snapshot_file': str(snapshot), 'credentials_file': str(credential), 'principal_kind': 'session'},
        'retrieval': {'bm25_index': str(lexical), 'dense_index': str(dense), 'candidate_k': 5,
            'dense_block_rows': 3}, 'web': {'allowed_sources': ['jira']},
        'generation': {'provider': 'disabled'}, 'selection': {'split': 'development'},
        'output_root': str(output), 'comparison': {'experiment_id': 'real-cli-validation'}}
    config_path = tmp_path / 'cli-config.json'
    config_path.write_text(json.dumps(configuration))

    actual_open = ScopedRetrievalRunner.open
    opened = []
    def open_shared(cls, settings, **kwargs):
        opened.append(1)
        return actual_open(settings, database=data.database, **kwargs)
    monkeypatch.setattr(ScopedRetrievalRunner, 'open', classmethod(open_shared))
    touches, calls = [], []
    actual_authenticate = AuthService.authenticate
    def authenticate(service, supplied, *, touch=True):
        result = actual_authenticate(service, supplied, touch=touch)
        if touch:
            touches.append(result.identity.session_id)
        return result
    monkeypatch.setattr(AuthService, 'authenticate', authenticate)
    actual_query = AuthenticatedRuntime.query_with_capture
    def query(runtime, body, identity, **kwargs):
        previous = calls[-1]['touches'] if calls else 0
        assert len(touches) > previous, 'Human identity was not refreshed before this question'
        assert runtime.provider is None and body.mode == 'baseline'
        calls.append({'touches': len(touches), 'session': identity.session_id})
        return actual_query(runtime, body, identity, **kwargs)
    monkeypatch.setattr(AuthenticatedRuntime, 'query_with_capture', query)
    def run_count():
        with data.owner.connect() as connection:
            return connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one()

    assert cli.main(['preflight', '--config', str(config_path)]) == 0
    assert len(opened) == len(encoder_loads) == 1 and run_count() == 0 and calls == []
    original_plan = (output / 'plan.json').read_bytes()
    prepared = {str(path.relative_to(output)): path.read_bytes()
        for path in (output / 'prepared').rglob('*') if path.is_file()}
    assert cli.main(['resume', '--config', str(config_path)]) == 0
    assert len(opened) == len(encoder_loads) == 2 and len(calls) == run_count() == 6
    assert len({call['session'] for call in calls}) == 1
    cases = {}
    scope = None
    for arm in cli.ARMS:
        run = RunDirectory.load(output / 'runs' / arm)
        scope = scope or run.manifest.scope
        assert run.manifest.scope == scope
        assert run.manifest.scope.snapshot == data.source.snapshot
        assert run.manifest.generation_model.provider == 'disabled'
        rows = run.load_cases(require_complete=True)
        assert len(rows) == 2 and all(row.generation_evidence == () and row.error is None for row in rows)
        for row in rows:
            assert row.response.sources and row.response.trace.usage.api_calls == 0
            assert {item.chunk.doc_id for item in row.response.sources} <= {'one', 'two'}
            assert 'GOLD_NEVER_ENTERS_FIXED_CLI' not in row.model_dump_json()
        cases.update({str(path.relative_to(output)): path.read_bytes()
            for path in (output / 'runs' / arm / 'cases').iterdir() if path.is_file()})
    assert (output / 'plan.json').read_bytes() == original_plan
    assert prepared == {str(path.relative_to(output)): path.read_bytes()
        for path in (output / 'prepared').rglob('*') if path.is_file()}
    assert cli.main(['resume', '--config', str(config_path)]) == 0
    assert len(opened) == len(encoder_loads) == 3 and len(calls) == run_count() == 6
    assert cases == {str(path.relative_to(output)): path.read_bytes()
        for path in (output / 'runs').rglob('case_*.json')}
    text = capsys.readouterr().out
    assert token not in text and 'GOLD_NEVER_ENTERS_FIXED_CLI' not in text

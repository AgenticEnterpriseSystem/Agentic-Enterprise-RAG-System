"""Tiny authenticated retrieval evaluations; no downloaded or live models.

The PostgreSQL fixture enforces real runtime-role RLS. Only the query encoder
is a deterministic two-dimensional test double; lexical/dense/store/fusion
code and both command entry points are exercised directly.
"""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sa = pytest.importorskip('sqlalchemy', reason='Scoped evaluation requires DB dependencies')
pytest.importorskip('psycopg', reason='Scoped evaluation requires psycopg')
pytest.importorskip('alembic', reason='Scoped evaluation requires migrations')

from src.access.contracts import EvidenceValidationError, PolicyUnavailable, SessionExpired, StaleContext
from src.access.policy import PolicyResolver
from src.config import RetrievalSettings
from src.db import schema
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.scoped_retrieval import (
    RunOutput, ScopedEvaluationError, ScopedRetrievalRunner, label_summary,
    result_row, sanitized_error,
)
from src.retrieval.factory import load_store_retriever
from src.storage.contracts import ReadFilters
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import pg_urls
from tests.test_access_policy import (
    ENTRIES, TEXTS, acl_db, identity, raw_token, seed_explicit_grants,
)


@pytest.fixture
def evaluation_backend(acl_db, tmp_path, monkeypatch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25
    from src.retrieval.corpus_dense import EmbeddingModel

    seed_explicit_grants(acl_db)
    store = PostgresDocumentStore(acl_db.database, acl_db.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    lexical, dense = tmp_path / 'bm25', tmp_path / 'dense'
    config = BM25Config()
    model, statistics = build_bm25(TEXTS, config)
    CorpusBM25(model, list(proof.doc_ids), config, manifest={'version': 1,
        'config': config.model_dump(), 'corpus_sha256': acl_db.snapshot.corpus_sha256,
        'stats': statistics}).save(lexical)
    dense.mkdir()
    model_identity = {'name': 'fixture-encoder', 'revision': 'c' * 40,
        'normalize': True, 'fp16': False, 'max_seq_length': 512, 'query_prefix': ''}
    (dense / 'manifest.json').write_text(json.dumps({'version': 1, 'complete': True,
        'documents': len(proof.entries), 'shards': 1,
        'corpus_sha256': acl_db.snapshot.corpus_sha256, 'model': model_identity}))
    (dense / 'doc_ids.json').write_text(json.dumps(proof.doc_ids))
    np.save(dense / 'shard_00000.npy', np.array([[1., index / 20]
        for index in range(len(proof.entries))], dtype=np.float32), allow_pickle=False)
    pq.write_table(pa.table({'doc_index': list(range(len(proof.entries))),
        'start_char': [0] * len(proof.entries),
        'end_char': [8] * len(proof.entries)}), dense / 'shard_00000.parquet')
    calls = []

    class Encoder:
        identity = model_identity

        def encode(self, texts, query=False):
            calls.append((tuple(texts), query))
            return np.array([[1., 0.]], dtype=np.float32)

    monkeypatch.setattr(EmbeddingModel, 'for_index', classmethod(lambda cls, *args, **kwargs: Encoder()))
    settings = RetrievalSettings(bm25_index=lexical, dense_index=dense, candidate_k=20,
        dense_block_rows=3, dense_file_cache_policy='retain')
    retriever = load_store_retriever(settings, store)
    touches = []

    def runner(principal='intern', *, filters=ReadFilters(), doc_ids=None, read_more_chars=0):
        resolver = PolicyResolver(acl_db.database, acl_db.snapshot, catalog=proof,
            source_types=filters.source_types, doc_ids=doc_ids)

        def authenticate(touch):
            touches.append((principal, touch))
            return acl_db.auth.authenticate(raw_token(acl_db, principal), touch=touch).identity

        return ScopedRetrievalRunner(retriever, resolver, database=acl_db.database,
            identity_provider=authenticate, principal_kind='session', settings=settings,
            filters=filters, read_more_chars=read_more_chars)

    return SimpleNamespace(fixture=acl_db, settings=settings, retriever=retriever,
        store=store, proof=proof, runner=runner, touches=touches, encodes=calls)


def question(identifier='q1', references=('general', 'finance-exec'), *, sources=('gmail',)):
    return {'question_id': identifier, 'question': 'Synthetic document text',
        'question_type': 'single_hop', 'source_types': list(sources),
        'expected_doc_ids': list(references)}


def assert_exact_source(observation, allowed):
    assert {hit.chunk.doc_id for hit in observation.hits} == set(allowed)
    for hit in observation.hits + observation.read_more:
        entry = next(value for value in ENTRIES if value.doc_id == hit.chunk.doc_id)
        text = TEXTS[entry.position]
        assert hit.revision_id == entry.revision_id
        assert hit.chunk.text == text[hit.chunk.start_char:hit.chunk.end_char]
        assert hit.citation_id == f'{entry.doc_id}:{hit.chunk.start_char}:{hit.chunk.end_char}'


@pytest.mark.parametrize('principal,allowed', [
    ('intern', {'general', 'atlas'}), ('executive', {'general', 'finance-exec'}),
])
def test_real_hybrid_and_dense_evaluations_enforce_each_human_scope(evaluation_backend, principal, allowed):
    bundle = evaluation_backend
    runner = bundle.runner(principal, read_more_chars=80)
    assert bundle.touches == [(principal, False)]
    context = runner.context()
    assert bundle.touches[-1] == (principal, True)
    for method in ('bm25', 'dense', 'hybrid'):
        observation = runner.search(context, question()['question'], method=method, top_k=20)
        assert_exact_source(observation, allowed)
        assert observation.context is context
        assert observation.seconds >= 0
        for hit in observation.hits:
            assert hit.snapshot_id == bundle.fixture.snapshot.snapshot_id
    manifest = runner.manifest()
    assert manifest['scope'] == context.binding.model_dump(mode='json')
    assert manifest['snapshot'] == bundle.fixture.snapshot.model_dump(mode='json')
    assert manifest['principal']['user_id'] == bundle.fixture.users[principal].user_id
    assert manifest['allowed_documents'] == len(allowed)
    assert manifest['block_rows'] == 3 and manifest['file_cache_policy'] == 'retain'
    assert raw_token(bundle.fixture, principal) not in json.dumps(manifest)


@pytest.mark.parametrize('filters,doc_ids,expected', [
    (ReadFilters(source_types=('jira',)), None, {'general'}),
    (ReadFilters(after=datetime(2020, 1, 1, tzinfo=timezone.utc)), None, set()),
    (ReadFilters(), [], set()),
    (ReadFilters(), ['atlas'], {'atlas'}),
])
def test_operator_source_date_and_explicit_empty_restrictions_are_applied(evaluation_backend, filters, doc_ids, expected):
    bundle = evaluation_backend
    runner = bundle.runner(filters=filters, doc_ids=doc_ids)
    context = runner.context()
    found = runner.search(context, 'Synthetic', method='hybrid', top_k=20)
    assert_exact_source(found, expected)
    labels = runner.labels(question(references=('general', 'atlas')), context)
    assert set(labels['relevant_doc_ids']) == expected
    assert labels['original_reference_count'] == 2
    assert labels['eligible_reference_count'] == len(expected)
    assert labels['scoring_status'] == ('scored' if expected else 'unanswerable')
    if not expected:
        assert bundle.encodes == []


def test_role_metadata_without_grants_returns_no_evidence_or_denied_reference_ids(evaluation_backend):
    runner = evaluation_backend.runner('labels_only')
    context = runner.context()
    observation = runner.search(context, 'Synthetic', method='hybrid', top_k=20)
    labels = runner.labels(question(), context)
    row = result_row(question(), observation, labels=labels)
    assert observation.hits == () and row['retrieved_doc_ids'] == []
    assert row['relevant_doc_ids'] == [] and row['scoring_status'] == 'unanswerable'
    assert 'finance-exec' not in json.dumps(row)


def test_reference_intersection_changes_only_declared_metric_denominators(evaluation_backend):
    runner = evaluation_backend.runner()
    context = runner.context()
    observation = runner.search(context, 'Synthetic', method='hybrid', top_k=20)
    questions = (question(), question('q2', ('finance-exec',)), question('q3', ()),
        question('q4', ('general', 'missing-reference')))
    rows = [result_row(q, observation, labels=runner.labels(q, context)) for q in questions]
    assert [row['scoring_status'] for row in rows] == ['scored', 'unanswerable', 'unlabeled', 'unknown_reference']
    assert rows[0]['relevant_doc_ids'] == ['general']
    assert rows[3]['eligible_reference_count'] == 1 and rows[3]['unknown_reference_count'] == 1
    assert rows[3]['relevant_doc_ids'] == []
    quality = evaluate_retrieval(rows, (10,))['overall']
    assert quality['questions'] == 4 and quality['recall@10'] == {'mean': 1., 'scored': 1, 'excluded': 3}
    summary = label_summary(rows)
    assert summary['original_references'] == 5 and summary['eligible_references'] == 2
    assert summary['unknown_references'] == 1
    assert summary['matches_original_reference_denominators'] is False
    assert 'finance-exec' not in json.dumps(rows)


def test_one_question_metadata_is_replayed_without_reencoding_or_cross_context_reuse(evaluation_backend):
    bundle = evaluation_backend
    runner = bundle.runner()
    context = runner.context()
    prepared = runner.prepare(context, 'Synthetic', depth=20)
    assert len(bundle.encodes) == 1
    assert all(not hasattr(candidate, 'chunk') for ranking in prepared.rankings for candidate in ranking)
    for weight in (0., .5, 1.):
        config = bundle.settings.model_copy(update={'bm25_weight': weight}).hybrid_config()
        result = runner.hybrid(prepared, config=config, top_k=20)
        assert_exact_source(result, {'general', 'atlas'})
    for method in ('bm25', 'dense'):
        assert_exact_source(runner.baseline(prepared, method=method, top_k=20), {'general', 'atlas'})
    assert len(bundle.encodes) == 1
    with pytest.raises(EvidenceValidationError):
        runner.hybrid(replace(prepared, depth=1), config=bundle.settings.hybrid_config(), top_k=20)
    other = runner.resolver.resolve(identity(bundle.fixture, 'executive'))
    with pytest.raises(StaleContext):
        runner.current(other)


@pytest.mark.parametrize('mutation', ['grant', 'logout', 'database'])
def test_prepared_candidates_cannot_bypass_live_authority_failure(evaluation_backend, monkeypatch, mutation):
    bundle = evaluation_backend
    runner = bundle.runner()
    prepared = runner.prepare(runner.context(), 'Synthetic', depth=20)
    if mutation == 'grant':
        with bundle.fixture.owner.begin() as connection:
            connection.execute(schema.document_grants.delete().where(
                schema.document_grants.c.snapshot_id == bundle.fixture.snapshot.snapshot_id,
                schema.document_grants.c.doc_id == 'general'))
        expected = StaleContext
    elif mutation == 'logout':
        bundle.fixture.auth.logout(raw_token(bundle.fixture, 'intern'),
            bundle.fixture.sessions['intern'].response.csrf_token)
        expected = SessionExpired
    else:
        def unavailable(context):
            raise PolicyUnavailable('Synthetic outage: never print this detail')
        monkeypatch.setattr(runner.resolver, 'assert_current', unavailable)
        expected = PolicyUnavailable
    for release in (lambda: runner.hybrid(prepared, config=bundle.settings.hybrid_config(), top_k=20),
                    lambda: runner.baseline(prepared, method='bm25', top_k=20), runner.manifest):
        with pytest.raises(expected):
            release()
    assert len(bundle.encodes) == 1


def test_malformed_adapter_revision_is_rejected_before_export(evaluation_backend, monkeypatch):
    runner = evaluation_backend.runner()
    context = runner.context()
    actual = runner.adapter('bm25').search(context, 'Synthetic', guard=runner.resolver, top_k=20)
    monkeypatch.setattr(runner.adapter('bm25'), 'search', lambda *args, **kwargs:
        (actual[0].model_copy(update={'revision_id': 'f' * 64}),))
    with pytest.raises(EvidenceValidationError):
        runner.search(context, 'Synthetic', method='bm25', top_k=20)


def test_failure_exports_keep_question_counts_and_never_exception_text(tmp_path):
    marker = 'PRIVATE SQL OR CREDENTIAL CONTENT'
    row = result_row(question(), error=sanitized_error(RuntimeError(marker)))
    assert row['error'] == 'retrieval_failed' and row['scoring_status'] == 'access_unavailable'
    assert row['eligible_reference_count'] is None
    output = RunOutput(tmp_path / 'attempt')
    try:
        output.append(row, .25)
        assert json.loads((output.path / 'predictions.jsonl').read_text()) == row
        assert not (output.path / 'manifest.json').exists()
        assert marker not in (output.path / 'predictions.jsonl').read_text()
        with pytest.raises(FileExistsError):
            RunOutput(output.path)
    finally:
        output.close()


def cli_inputs(bundle, tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(bundle.fixture.snapshot.model_dump_json())
    credential = tmp_path / 'session.json'
    credential.write_text(json.dumps({'format': 'rag-session-v1', 'token': raw_token(bundle.fixture, 'intern')}))
    credential.chmod(0o600)
    questions = tmp_path / 'questions.jsonl'
    rows = [question('q2', ('atlas',)), question()]
    questions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'splits': {'development': ['q1', 'q2'], 'calibration': [], 'final': []}}))
    return snapshot, credential, questions, splits


@pytest.mark.parametrize('command', ['baseline', 'hybrid'])
def test_real_cli_uses_private_session_exact_snapshot_filters_and_exclusive_outputs(evaluation_backend, tmp_path, monkeypatch, capsys, command):
    from src.evaluation import run_hybrid, run_retrieval
    bundle = evaluation_backend
    snapshot, credential, questions, splits = cli_inputs(bundle, tmp_path)
    module = run_retrieval if command == 'baseline' else run_hybrid
    actual_open = ScopedRetrievalRunner.open
    opened = []

    def open_shared(cls, settings, **kwargs):
        opened.append(kwargs)
        return actual_open(settings, database=bundle.fixture.database, **kwargs)

    monkeypatch.setattr(ScopedRetrievalRunner, 'open', classmethod(open_shared))
    monkeypatch.setattr(module, 'git_state', lambda: {'commit': 'e' * 40, 'dirty': False})
    monkeypatch.setattr(module, 'runtime_versions', lambda: {'test_fixture': True})
    monkeypatch.setattr(run_retrieval, 'git_state', lambda: {'commit': 'e' * 40, 'dirty': False})
    monkeypatch.setattr(run_retrieval, 'runtime_versions', lambda: {'test_fixture': True})
    output = tmp_path / 'results'
    args = ['scoped-evaluation', '--snapshot-file', str(snapshot), '--credentials-file', str(credential),
        '--principal-kind', 'session', '--questions', str(questions), '--splits', str(splits),
        '--bm25-index', str(bundle.settings.bm25_index), '--dense-index', str(bundle.settings.dense_index),
        '--source-types', 'jira', '--dense-block-rows', '3', '--read-more-chars', '80', '--limit', '1']
    args += ['--output', str(output)] if command == 'baseline' else [
        '--output-dir', str(output), '--bm25-weights', '0.5', '--write-baselines']
    monkeypatch.setattr('sys.argv', args)
    module.main()
    folders = [output] if command == 'baseline' else sorted(path for path in output.iterdir() if path.is_dir())
    assert len(folders) == (1 if command == 'baseline' else 3)
    for folder in folders:
        manifest = json.loads((folder / 'manifest.json').read_text())
        row = json.loads((folder / 'predictions.jsonl').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        assert manifest['complete'] is True and manifest['principal']['kind'] == 'session'
        assert manifest['snapshot'] == bundle.fixture.snapshot.model_dump(mode='json')
        assert manifest['filters']['source_types'] == ['jira']
        assert manifest['inputs']['partial_evaluation'] is True
        assert manifest['inputs']['split_questions'] == 2
        assert manifest['inputs']['questions_sha256'] == hashlib.sha256(questions.read_bytes()).hexdigest()
        assert row['question_id'] == 'q1' and row['source_types'] == ['gmail']
        assert row['retrieved_doc_ids'] == row['relevant_doc_ids'] == ['general']
        assert row['original_reference_count'] == 2 and row['eligible_reference_count'] == 1
        assert summary['overall']['recall@10']['mean'] == 1.
        serialized = ''.join(path.read_text() for path in folder.iterdir() if path.is_file())
        assert raw_token(bundle.fixture, 'intern') not in serialized
        assert 'finance-exec' not in serialized and TEXTS[0] not in serialized
        assert all('text' not in value and value['text_sha256'] for value in row['read_more'])
        if command == 'hybrid' and not folder.name.startswith('bm25_'):
            assert row['read_more']
    assert len(opened) == 1
    before = {str(path): path.read_bytes() for folder in folders for path in folder.iterdir() if path.is_file()}
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2 and len(opened) == 1
    assert before == {str(path): path.read_bytes() for folder in folders for path in folder.iterdir() if path.is_file()}
    assert raw_token(bundle.fixture, 'intern') not in capsys.readouterr().out


@pytest.mark.parametrize('mode', [0o644, 0o400])
def test_session_credentials_require_private_exact_mode_before_opening_backend(evaluation_backend, tmp_path, mode):
    bundle = evaluation_backend
    snapshot, credential, _questions, _splits = cli_inputs(bundle, tmp_path)
    credential.chmod(mode)
    with pytest.raises(ScopedEvaluationError):
        ScopedRetrievalRunner.open(bundle.settings, snapshot_file=snapshot,
            credentials_file=credential, principal_kind='session', database=bundle.fixture.database)


def test_session_input_symlink_and_extra_fields_fail_without_secret_text(evaluation_backend, tmp_path):
    bundle = evaluation_backend
    snapshot, credential, _questions, _splits = cli_inputs(bundle, tmp_path)
    link = tmp_path / 'session-link.json'
    link.symlink_to(credential)
    for path in (link, credential):
        if path == credential:
            value = json.loads(credential.read_text())
            value['password'] = 'SYNTHETIC_MUST_NOT_ECHO'
            credential.write_text(json.dumps(value))
        with pytest.raises(ScopedEvaluationError) as failure:
            ScopedRetrievalRunner.open(bundle.settings, snapshot_file=snapshot,
                credentials_file=path, principal_kind='session', database=bundle.fixture.database)
        assert raw_token(bundle.fixture, 'intern') not in str(failure.value)
        assert 'SYNTHETIC_MUST_NOT_ECHO' not in str(failure.value)


def test_source_changes_invalidate_completion_provenance(evaluation_backend, monkeypatch):
    import src.evaluation.scoped_retrieval as scoped
    runner = evaluation_backend.runner()
    actual = runner.manifest()['source_sha256']
    assert actual and all(len(digest) == 64 for digest in actual.values())
    assert 'src/evaluation/scoped_retrieval.py' in actual
    changed = dict(actual)
    changed['src/evaluation/scoped_retrieval.py'] = 'f' * 64
    monkeypatch.setattr(scoped, 'source_hashes', lambda: changed)
    with pytest.raises(ScopedEvaluationError):
        runner.manifest()


def test_summary_uses_exact_snapshot_override_without_git_or_legacy_corpus_file(tmp_path, monkeypatch):
    from src.evaluation import run_retrieval
    monkeypatch.setattr(run_retrieval, 'ROOT', tmp_path)
    monkeypatch.setattr(run_retrieval, 'runtime_versions', lambda: {'test_fixture': True})

    def unavailable(*args, **kwargs):
        raise FileNotFoundError('Synthetic lean image does not include Git')

    monkeypatch.setattr(run_retrieval.subprocess, 'run', unavailable)
    corpus = {'documents': 17, 'content_sha256': 'a' * 64, 'snapshot_id': 'tiny-exact-snapshot'}
    row = {'question_id': 'q1', 'question_type': 'single_hop', 'source_types': [],
        'relevant_doc_ids': ['general'], 'retrieved_doc_ids': ['general'], 'error': None}
    summary = run_retrieval.build_summary([row], [.2], method='bm25', split='development',
        top_k=20, index={'method': 'tiny'}, load_seconds=.1, corpus=corpus)
    assert summary['corpus'] == corpus
    assert summary['code'] == {'commit': None, 'dirty': None, 'available': False}
    assert summary['overall']['recall@10']['mean'] == 1.
    assert not (tmp_path / 'evaluation').exists()


@pytest.mark.parametrize('module', ['src.evaluation.run_retrieval', 'src.evaluation.run_hybrid'])
def test_cli_help_loads_no_native_model_database_or_full_corpus_modules(module):
    code = '''import builtins, runpy, sys
original = builtins.__import__
blocked = {'numpy', 'pyarrow', 'faiss', 'torch', 'transformers', 'sentence_transformers', 'bm25s', 'sqlalchemy', 'psycopg', 'alembic'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise RuntimeError('Heavy dependency imported by --help: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--help']
runpy.run_module(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code, module], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--snapshot-file' in result.stdout and '--credentials-file' in result.stdout
    assert '--allowed-doc-ids-file' in result.stdout and '--read-more-chars' in result.stdout

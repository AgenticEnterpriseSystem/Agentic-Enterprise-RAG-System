"""Small offline delivery checks; Docker/SQL/network calls are mocked or absent."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import docker_dependencies, local_data, local_stack


def test_direct_preparation_script_finds_repository_package_without_pythonpath(tmp_path):
    script = Path(local_data.__file__).resolve()
    program = '''
import importlib.util, pathlib, runpy, sys
script=pathlib.Path(sys.argv[1])
# Match direct script execution, from an unrelated working directory and with
# no PYTHONPATH. --help performs no database/model imports or connection work.
sys.path[0]=str(script.parent)
assert importlib.util.find_spec('src') is None
sys.argv=[str(script),'import-corpus','--help']
try:
    runpy.run_path(str(script),run_name='__main__')
except SystemExit as result:
    assert result.code == 0
spec=importlib.util.find_spec('src')
assert spec is not None and pathlib.Path(spec.origin).resolve() == script.parents[1]/'src/__init__.py'
assert not any(name in sys.modules for name in ('sqlalchemy','psycopg','torch','pyarrow'))
print('repository-package-discovered')
'''
    result = subprocess.run([sys.executable, '-c', program, str(script)], cwd=tmp_path,
        env={'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'repository-package-discovered' in result.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'documents.parquet').write_bytes(b'Fixture bytes for staging, not an inference corpus.')
    corpus_sha = digest(source / 'documents.parquet')
    dense = source / 'indexes/dense-bge-base'
    lexical = source / 'indexes/bm25'
    dense.mkdir(parents=True)
    lexical.mkdir(parents=True)
    header = {'version': 1, 'complete': True, 'documents': 2, 'shards': 2,
        'corpus_sha256': corpus_sha, 'model': {'name': 'BAAI/bge-base-en-v1.5', 'revision': 'a' * 40}}
    (dense / 'manifest.json').write_text(json.dumps(header))
    for name in {'doc_ids.json', 'shard_00000.npy', 'shard_00000.parquet',
                 'shard_00001.npy', 'shard_00001.parquet'}:
        (dense / name).write_bytes(('Fixture: ' + name).encode())
    (lexical / 'manifest.json').write_text(json.dumps({'version': 1,
        'corpus_sha256': corpus_sha, 'stats': {'documents': 2}}))
    for name in local_data._BM25_FILES - {'manifest.json'}:
        (lexical / name).write_bytes(('Fixture: ' + name).encode())
    return source


def stage(bundle, tmp_path, **options):
    return local_data.stage_artifacts(bundle, tmp_path / 'corpus', tmp_path / 'indexes',
                                     tmp_path / 'state', min_free_bytes=0, **options)


def test_complete_stage_keeps_exact_inventory_and_receipts_only_in_state(bundle, tmp_path):
    source_bytes = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob('*') if path.is_file()}
    receipt = stage(bundle, tmp_path)
    assert receipt['documents'] == 2 and receipt['dense_shards'] == 2
    assert (tmp_path / 'state/artifacts.json').is_file()
    assert not (tmp_path / 'state/artifacts.lock').exists()
    for relative, expected in receipt['files'].items():
        path = tmp_path / relative
        assert digest(path) == expected and stat.S_IMODE(path.stat().st_mode) == 0o444
    assert local_data._inventory(tmp_path / 'indexes') == local_data._inventory(bundle / 'indexes')
    assert local_data._inventory(tmp_path / 'corpus') == {'documents.parquet'}
    assert {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob('*') if path.is_file()} == source_bytes


def test_unpublished_identical_target_needs_explicit_adoption_and_is_never_recopied(bundle, tmp_path, monkeypatch):
    (tmp_path / 'corpus').mkdir()
    shutil.copyfile(bundle / 'documents.parquet', tmp_path / 'corpus/documents.parquet')
    shutil.copytree(bundle / 'indexes', tmp_path / 'indexes')
    with pytest.raises(local_data.PreparationError, match='explicitly adopt'):
        stage(bundle, tmp_path)
    monkeypatch.setattr(local_data, '_copy', lambda *_: pytest.fail('Identical adoption must not copy/replace data'))
    stage(bundle, tmp_path, adopt_existing=True)
    assert (tmp_path / 'state/artifacts.json').exists()


@pytest.mark.parametrize('corruption', ['changed', 'partial', 'extra'])
def test_published_targets_reject_mutation_without_repair(bundle, tmp_path, corruption):
    stage(bundle, tmp_path)
    path = tmp_path / 'indexes/dense-bge-base/shard_00000.npy'
    if corruption == 'changed':
        path.chmod(0o644)
        path.write_bytes(b'Changed target')
    elif corruption == 'partial':
        path.parent.chmod(0o755)
        path.unlink()
    else:
        path.parent.chmod(0o755)
        (path.parent / 'extra').write_bytes(b'Unexpected target')
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(local_data.PreparationError):
        stage(bundle, tmp_path, adopt_existing=True)
    assert (path.read_bytes() if path.exists() else None) == before


def legacy_copy_outputs(bundle, tmp_path):
    corpus, indexes = tmp_path / 'corpus', tmp_path / 'indexes'
    corpus.mkdir()
    shutil.copyfile(bundle / 'documents.parquet', corpus / 'documents.parquet')
    shutil.copytree(bundle / 'indexes', indexes)
    rows = [{'relative_path': path.relative_to(indexes).as_posix(), 'bytes': path.stat().st_size,
             'sha256': digest(path)} for path in sorted(indexes.rglob('*')) if path.is_file()]
    index_value = {'format': 'verified-search-indexes-1', 'complete': True,
        'file_count': len(rows), 'files': rows, 'bytes': sum(row['bytes'] for row in rows)}
    corpus_value = {'format': 'verified-raw-corpus-1', 'complete': True,
        'corpus_sha256': digest(corpus / 'documents.parquet'), 'bytes': (corpus / 'documents.parquet').stat().st_size}
    index_path, corpus_path = indexes / '.artifact-copy.json', corpus / '.verified-copy.json'
    index_path.write_text(json.dumps(index_value))
    corpus_path.write_text(json.dumps(corpus_value))
    return index_path, corpus_path, index_value, corpus_value


def test_explicit_legacy_receipt_adoption_preserves_every_byte_and_binds_metadata(bundle, tmp_path, monkeypatch):
    index, corpus, _iv, _cv = legacy_copy_outputs(bundle, tmp_path)
    original = {path: (path.read_bytes(), path.stat().st_ino) for directory in
        (tmp_path / 'corpus', tmp_path / 'indexes') for path in directory.rglob('*') if path.is_file()}
    monkeypatch.setattr(local_data, '_copy', lambda *a: pytest.fail('Accepted identical data must not be duplicated'))
    with pytest.raises(local_data.PreparationError, match='explicitly adopt'):
        stage(bundle, tmp_path)
    result = stage(bundle, tmp_path, adopt_existing=True)
    assert result['adopted_metadata_sha256'] == {
        'indexes/.artifact-copy.json': digest(index), 'corpus/.verified-copy.json': digest(corpus)}
    assert not any(name.endswith(('.artifact-copy.json', '.verified-copy.json')) for name in result['files'])
    assert {path: (path.read_bytes(), path.stat().st_ino) for path in original} == original
    assert stage(bundle, tmp_path) == result


@pytest.mark.parametrize('corruption', ['format', 'complete-int', 'count-bool', 'bytes-bool', 'total-size',
    'hash', 'file-size', 'duplicate', 'traversal', 'missing-record', 'extra-record'])
def test_legacy_index_receipt_requires_exact_strict_core_mapping(bundle, tmp_path, corruption):
    index, _corpus, value, _cv = legacy_copy_outputs(bundle, tmp_path)
    if corruption == 'format': value['format'] = 'another-copy-format'
    elif corruption == 'complete-int': value['complete'] = 1
    elif corruption == 'count-bool': value['file_count'] = True
    elif corruption == 'bytes-bool': value['bytes'] = True
    elif corruption == 'total-size': value['bytes'] += 1
    elif corruption == 'hash': value['files'][0]['sha256'] = 'f' * 64
    elif corruption == 'file-size': value['files'][0]['bytes'] += 1
    elif corruption == 'duplicate': value['files'][1] = dict(value['files'][0])
    elif corruption == 'traversal': value['files'][0]['relative_path'] = '../outside'
    elif corruption == 'missing-record': value['files'].pop()
    else: value['files'].append({'relative_path': 'extra', 'bytes': 0, 'sha256': 'f' * 64})
    index.write_text(json.dumps(value))
    before = index.read_bytes()
    with pytest.raises(local_data.PreparationError, match='Legacy index'):
        stage(bundle, tmp_path, adopt_existing=True)
    assert index.read_bytes() == before and not (tmp_path / 'state/artifacts.json').exists()


@pytest.mark.parametrize('corruption', ['format', 'complete-int', 'bytes-bool', 'size', 'hash'])
def test_legacy_corpus_receipt_requires_actual_corpus_hash_and_strict_size(bundle, tmp_path, corruption):
    _index, corpus, _iv, value = legacy_copy_outputs(bundle, tmp_path)
    if corruption == 'format': value['format'] = 'another-copy-format'
    elif corruption == 'complete-int': value['complete'] = 1
    elif corruption == 'bytes-bool': value['bytes'] = True
    elif corruption == 'size': value['bytes'] += 1
    else: value['corpus_sha256'] = 'f' * 64
    corpus.write_text(json.dumps(value))
    with pytest.raises(local_data.PreparationError, match='Legacy corpus'):
        stage(bundle, tmp_path, adopt_existing=True)
    assert not (tmp_path / 'state/artifacts.json').exists()


@pytest.mark.parametrize('corruption', ['metadata-change', 'metadata-removal', 'unknown-extra', 'nested-known-name', 'core-change'])
def test_adopted_receipts_never_hide_reuse_mutations_or_unknown_extras(bundle, tmp_path, corruption):
    index, _corpus, value, _cv = legacy_copy_outputs(bundle, tmp_path)
    stage(bundle, tmp_path, adopt_existing=True)
    if corruption == 'metadata-change':
        index.chmod(0o644)
        index.write_text(json.dumps({**value, 'notes': 'Changed after acceptance'}))
    elif corruption == 'metadata-removal':
        index.parent.chmod(0o755)
        index.unlink()
    elif corruption == 'unknown-extra':
        index.parent.chmod(0o755)
        (index.parent / '.unverified-copy.json').write_text('{}')
    elif corruption == 'nested-known-name':
        (index.parent / 'bm25').chmod(0o755)
        (index.parent / 'bm25/.artifact-copy.json').write_text(json.dumps(value))
    else:
        target = index.parent / 'bm25/data.csc.index.npy'
        target.chmod(0o644)
        target.write_bytes(b'Unverified replacement')
    accepted = (tmp_path / 'state/artifacts.json').read_bytes()
    with pytest.raises(local_data.PreparationError):
        stage(bundle, tmp_path, adopt_existing=True)
    assert (tmp_path / 'state/artifacts.json').read_bytes() == accepted


@pytest.mark.parametrize('violation', ['symlink', 'oversized', 'directory'])
def test_legacy_metadata_must_be_an_ordinary_bounded_root_file(bundle, tmp_path, violation):
    index, _corpus, value, _cv = legacy_copy_outputs(bundle, tmp_path)
    index.unlink()
    if violation == 'symlink':
        outside = tmp_path / 'outside.json'
        outside.write_text(json.dumps(value))
        index.symlink_to(outside)
    elif violation == 'oversized': index.write_text(' ' * (256 * 1024 + 1))
    else: index.mkdir()
    with pytest.raises((local_data.PreparationError, OSError)):
        stage(bundle, tmp_path, adopt_existing=True)
    assert not (tmp_path / 'state/artifacts.json').exists()


def test_legacy_metadata_is_rechecked_after_core_hashes_before_publication(bundle, tmp_path, monkeypatch):
    index, _corpus, value, _cv = legacy_copy_outputs(bundle, tmp_path)
    original = local_data._digest
    changed = False

    def change_metadata(path):
        nonlocal changed
        result = original(path)
        if Path(path) == tmp_path / 'indexes/bm25/manifest.json' and not changed:
            changed = True
            index.write_text(json.dumps({**value, 'notes': 'Changed during verification'}))
        return result

    monkeypatch.setattr(local_data, '_digest', change_metadata)
    with pytest.raises(local_data.PreparationError, match='legacy metadata changed'):
        stage(bundle, tmp_path, adopt_existing=True)
    assert changed and not (tmp_path / 'state/artifacts.json').exists()


def test_same_run_source_mutation_cannot_publish_complete_receipt(bundle, tmp_path, monkeypatch):
    actual = local_data._copy
    changed = False
    def copy_then_change(*args):
        nonlocal changed
        actual(*args)
        if not changed:
            changed = True
            (bundle / 'documents.parquet').write_bytes(b'Changed after its successful copy')
    monkeypatch.setattr(local_data, '_copy', copy_then_change)
    with pytest.raises(local_data.PreparationError, match='Source set changed'):
        stage(bundle, tmp_path)
    assert not (tmp_path / 'state/artifacts.json').exists()


def test_missing_or_extra_shard_and_symlink_inputs_fail_before_publication(bundle, tmp_path):
    path = bundle / 'indexes/dense-bge-base/shard_00000.npy'
    path.unlink()
    path.symlink_to(bundle / 'documents.parquet')
    with pytest.raises(local_data.PreparationError, match='symbolic links'):
        stage(bundle, tmp_path)
    assert not (tmp_path / 'state/artifacts.json').exists()


def test_disk_reserve_failure_and_stale_lock_are_not_ignored(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(local_data.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(local_data.PreparationError, match='disk headroom'):
        local_data.stage_artifacts(bundle, tmp_path / 'corpus', tmp_path / 'indexes', tmp_path / 'state', min_free_bytes=1)
    assert local_data._inventory(tmp_path / 'corpus') == set()
    lock = tmp_path / 'state/artifacts.lock'
    lock.write_text('Prior interrupted staging')
    with pytest.raises(local_data.PreparationError, match='lock exists'):
        stage(bundle, tmp_path)
    assert lock.read_text() == 'Prior interrupted staging'


@pytest.mark.parametrize('reserve,accepted', [(4, True), (7, True), (3, False)])
def test_explicit_operator_staging_reserve_has_four_gib_minimum(tmp_path, monkeypatch, capsys, reserve, accepted):
    observed = []
    monkeypatch.setattr(local_data, 'stage_models', lambda **kwargs: observed.append(kwargs) or {'prepared': True})
    monkeypatch.setattr(sys, 'argv', ['local_data.py', 'stage-models', '--source', str(tmp_path / 'source'),
        '--dense-directory', str(tmp_path / 'dense'), '--output', str(tmp_path / 'models'),
        '--state', str(tmp_path / 'state'), '--min-free-gib', str(reserve)])
    if accepted:
        local_data.main()
        assert observed[0]['min_free_bytes'] == reserve * 1024**3
        assert json.loads(capsys.readouterr().out) == {'prepared': True}
    else:
        with pytest.raises(SystemExit) as failed:
            local_data.main()
        assert failed.value.code == 2 and observed == []
        assert 'at least4GiB' in capsys.readouterr().err


def test_default_staging_reserve_remains_eight_gib(tmp_path, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(local_data, 'stage_artifacts', lambda **kwargs: observed.append(kwargs) or {'prepared': True})
    monkeypatch.setattr(sys, 'argv', ['local_data.py', 'stage-artifacts', '--source', str(tmp_path / 'source'),
        '--corpus-output', str(tmp_path / 'corpus'), '--indexes-output', str(tmp_path / 'indexes'),
        '--state', str(tmp_path / 'state')])
    local_data.main()
    assert observed[0]['min_free_bytes'] == 8 * 1024**3
    capsys.readouterr()


def test_reservation_requires_pending_copy_plus_explicit_retained_floor(tmp_path, monkeypatch):
    needed, reserve = 439_000_000, 7 * 1024**3
    monkeypatch.setattr(local_data.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=needed + reserve - 1))
    with pytest.raises(local_data.PreparationError, match='copy plus the retained reserve'):
        local_data._reserve(tmp_path, needed, reserve)
    monkeypatch.setattr(local_data.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=needed + reserve))
    assert local_data._reserve(tmp_path, needed, reserve) is None
    monkeypatch.setattr(local_data.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=8 * 1024**3 - 1))
    with pytest.raises(local_data.PreparationError):
        local_data._reserve(tmp_path, 0, 8 * 1024**3)


def model_cache(root):
    model = root / 'models--BAAI--bge-base-en-v1.5'
    snapshot = model / 'snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    for name in ('config.json', 'modules.json', 'tokenizer.json'):
        (snapshot / name).write_text('{}')
    (model / 'blobs').mkdir()
    (model / 'blobs/weights').write_bytes(b'Synthetic model bytes; never loaded')
    (snapshot / 'model.safetensors').symlink_to('../../blobs/weights')
    return snapshot


def test_model_staging_follows_only_internal_blob_links_and_does_not_change_source(bundle, tmp_path):
    source = tmp_path / 'model-source'
    snapshot = model_cache(source)
    original = (snapshot / 'model.safetensors').resolve()
    mode = original.stat().st_mode
    output = tmp_path / 'models'
    receipt = local_data.stage_models(source, bundle / 'indexes/dense-bge-base', output,
                                      tmp_path / 'state', min_free_bytes=0)
    assert receipt['revision'] == 'a' * 40
    assert (tmp_path / 'state/model.json').exists()
    assert not (output / local_data.MODEL_RECEIPT).exists()
    for relative, expected in receipt['files'].items():
        assert digest(output / relative) == expected and not (output / relative).is_symlink()
    assert original.stat().st_mode == mode


def test_model_links_cannot_escape_and_source_cannot_be_writable_target(bundle, tmp_path):
    source = tmp_path / 'model-source'
    snapshot = model_cache(source)
    with pytest.raises(local_data.PreparationError, match='writable staging'):
        local_data.stage_models(source, bundle / 'indexes/dense-bge-base', source,
                               tmp_path / 'state', adopt_existing=True, min_free_bytes=0)
    link = snapshot / 'model.safetensors'
    link.unlink()
    link.symlink_to(bundle / 'documents.parquet')
    with pytest.raises(local_data.PreparationError, match='inside that model cache'):
        local_data.stage_models(source, bundle / 'indexes/dense-bge-base', tmp_path / 'models',
                               tmp_path / 'state', min_free_bytes=0)


@pytest.mark.parametrize('published', [False, True])
def test_selected_model_snapshot_rejects_unattested_extra_config_or_weights(bundle, tmp_path, published):
    source = tmp_path / 'model-source'
    snapshot = model_cache(source)
    output, state = tmp_path / 'models', tmp_path / 'state'
    relative = snapshot.relative_to(source)
    if published:
        local_data.stage_models(source, bundle / 'indexes/dense-bge-base', output, state, min_free_bytes=0)
        (output / relative).chmod(0o755)
        receipt = (state / 'model.json').read_bytes()
    else:
        shutil.copytree(snapshot, output / relative, symlinks=False)
    extra = output / relative / ('pytorch_model.bin' if published else 'adapter_config.json')
    extra.write_bytes(b'Unexpected alternative model input')
    with pytest.raises(local_data.PreparationError, match='unexpected'):
        local_data.stage_models(source, bundle / 'indexes/dense-bge-base', output, state,
            adopt_existing=True, min_free_bytes=0)
    assert extra.read_bytes() == b'Unexpected alternative model input'
    if published:
        assert (state / 'model.json').read_bytes() == receipt
    else:
        assert not (state / 'model.json').exists()


def test_identical_adopted_model_cache_supports_internal_blob_links(bundle, tmp_path):
    source = tmp_path / 'model-source'
    snapshot = model_cache(source)
    output = tmp_path / 'models'
    model_root = output / 'models--BAAI--bge-base-en-v1.5'
    target = output / snapshot.relative_to(source)
    target.mkdir(parents=True)
    for name in ('config.json', 'modules.json', 'tokenizer.json'):
        shutil.copyfile(snapshot / name, target / name)
    (model_root / 'blobs').mkdir()
    shutil.copyfile((snapshot / 'model.safetensors').resolve(), model_root / 'blobs/weights')
    (target / 'model.safetensors').symlink_to('../../blobs/weights')
    local_data.stage_models(source, bundle / 'indexes/dense-bge-base', output,
        tmp_path / 'state', adopt_existing=True, min_free_bytes=0)
    assert (target / 'model.safetensors').is_symlink()


def test_root_staging_prepares_only_state_directory_owner_and_preserves_receipts(tmp_path, monkeypatch):
    state, source = tmp_path / 'state', tmp_path / 'private-source'
    state.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    private = source / 'input.json'
    private.write_bytes(b'Private source bytes')
    private.chmod(0o600)
    receipt = state / 'artifacts.json'
    receipt.write_text('{"format":"synthetic-nonsecret-receipt"}')
    before = (receipt.read_bytes(), receipt.stat().st_mode, private.read_bytes(), private.stat().st_mode,
              source.stat().st_mode)
    owners = []
    monkeypatch.setattr(local_data.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(local_data.os, 'fchown', lambda descriptor, uid, gid:
        owners.append((os.fstat(descriptor).st_ino, uid, gid)))
    local_data._prepare_state(state, forbidden=(source,))
    assert owners == [(state.stat().st_ino, 10001, 10001)]
    assert stat.S_IMODE(state.stat().st_mode) == 0o755
    assert before == (receipt.read_bytes(), receipt.stat().st_mode, private.read_bytes(), private.stat().st_mode,
                      source.stat().st_mode)


@pytest.mark.parametrize('violation', ['symlink', 'file', 'partial-receipt', 'linked-receipt', 'source', 'inside-source'])
def test_state_preparation_refuses_ambiguous_or_input_paths_before_owner_changes(tmp_path, monkeypatch, violation):
    state, source = tmp_path / 'state', tmp_path / 'source'
    source.mkdir()
    protected = source / 'private.json'
    protected.write_text('{"private":"unchanged"}')
    if violation == 'symlink': state.symlink_to(source, target_is_directory=True)
    elif violation == 'file': state.write_text('Not a directory')
    elif violation in ('partial-receipt', 'linked-receipt'):
        state.mkdir()
        if violation == 'partial-receipt': (state / 'deployment.json').write_text('{')
        else: (state / 'deployment.json').symlink_to(protected)
    elif violation == 'source': state = source
    else: state = source / 'nested-state'
    monkeypatch.setattr(local_data.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(local_data.os, 'fchown', lambda *args: pytest.fail('Refused state must not change ownership'))
    with pytest.raises((local_data.PreparationError, OSError, ValueError)):
        local_data._prepare_state(state, forbidden=(source,))
    assert protected.read_text() == '{"private":"unchanged"}'


def test_exclusive_metadata_publication_does_not_leave_a_partial_receipt(tmp_path, monkeypatch):
    def fail(*_):
        raise OSError('Injected publication failure')
    monkeypatch.setattr(local_data.os, 'link', fail)
    with pytest.raises(OSError):
        local_data._write_same(tmp_path / 'receipt.json', {'complete': True})
    assert list(tmp_path.iterdir()) == []


def test_private_init_is_exclusive_generates_distinct_secrets_and_prints_none(bundle, tmp_path):
    path = tmp_path / 'private/stack.env'
    result = local_stack.initialize(path, artifacts_dir=bundle, models_dir=bundle,
        source_repo_dir=bundle, operator_dir=tmp_path / 'operator', project='fixture', port=18080)
    assert result == {'initialized': True, 'project': 'fixture', 'port': 18080,
                      'accounts_seeded': False, 'credentials_printed': False}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    keys = {'RAG_DB_BOOTSTRAP_PASSWORD', 'RAG_DB_MIGRATOR_PASSWORD', 'RAG_DB_RUNTIME_PASSWORD'}
    secrets = {line.partition('=')[2] for line in path.read_text().splitlines() if line.partition('=')[0] in keys}
    assert len(secrets) == 3 and all(len(value) >= 40 for value in secrets)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        local_stack.initialize(path, artifacts_dir=bundle, models_dir=bundle,
            source_repo_dir=bundle, operator_dir=tmp_path / 'operator')
    assert path.read_bytes() == before


def configuration(volume='fixture_postgres-data'):
    return {'name': 'fixture', 'volumes': {'postgres-data': {'name': volume}}, 'services': {
        'db': {'environment': {'POSTGRES_PASSWORD': 'fixture-a', 'RAG_DB_MIGRATOR_PASSWORD': 'fixture-b',
            'RAG_DB_RUNTIME_PASSWORD': 'fixture-c'}, 'volumes': [
                {'target': '/var/lib/postgresql/data', 'type': 'volume', 'source': 'postgres-data'},
                {'target': '/docker-entrypoint-initdb.d/10-roles.sh', 'type': 'bind', 'read_only': True,
                 'source': str(local_stack.ROOT / 'docker/postgres/init.sh')}]},
        'app': {'environment': {'RAG_DB_USER': 'rag_runtime', 'RAG_DB_PASSWORD': 'fixture-c'}, 'volumes': [
            {'target': value, 'type': 'volume', 'read_only': True} for value in
            ('/data/corpus', '/data/indexes', '/app/.cache/models', '/state')]}}}


def stack_with_docker_mock(tmp_path, monkeypatch, config, containers=()):
    path = tmp_path / 'private.env'
    path.write_text('Synthetic env; never parsed by Docker in this test')
    path.chmod(0o600)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if 'config' in command:
            output = json.dumps(config)
        elif command[:2] == ['docker', 'ps']:
            output = 'container-id' if containers else ''
        elif command[:2] == ['docker', 'inspect']:
            output = json.dumps(containers)
        else:
            pytest.fail('Unexpected unmocked Docker action')
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(local_stack.subprocess, 'run', run)
    return local_stack.Stack(path), calls


def test_protected_canonical_database_is_never_a_target(tmp_path, monkeypatch):
    stack, calls = stack_with_docker_mock(tmp_path, monkeypatch,
                                        configuration('rag-doc01-data-a283c181'))
    with pytest.raises(local_stack.StackError, match='protected'):
        stack.guard_database()
    assert len(calls) == 1


def test_protected_volume_alias_is_rejected_at_every_mount_target(tmp_path, monkeypatch):
    config = configuration()
    config['volumes']['alternate'] = {'name': 'rag-doc01-data-a283c181'}
    config['services']['db']['volumes'].append({'type': 'volume', 'source': 'alternate', 'target': '/alternate'})
    stack, calls = stack_with_docker_mock(tmp_path, monkeypatch, config)
    with pytest.raises(local_stack.StackError, match='protected'):
        stack.guard_database()
    assert len(calls) == 1


@pytest.mark.parametrize('service,target', [('stage-artifacts', '/data/corpus'),
    ('stage-models', '/app/.cache/models'), ('prepare-state', '/state'), ('app', '/state')])
def test_protected_database_cannot_be_aliased_into_any_job_output(tmp_path, monkeypatch, service, target):
    config = configuration()
    config['volumes']['source-alias'] = {'name': 'rag-doc01-data-a283c181'}
    config['services'].setdefault(service, {})['volumes'] = [
        {'type': 'volume', 'source': 'source-alias', 'target': target, 'read_only': True}]
    # App still needs all four managed RO targets for constructor validation.
    if service == 'app':
        config['services']['app']['volumes'] += [{'type': 'volume', 'source': 'ordinary',
            'target': value, 'read_only': True} for value in ('/data/corpus', '/data/indexes', '/app/.cache/models')]
    stack, calls = stack_with_docker_mock(tmp_path, monkeypatch, config)
    with pytest.raises(local_stack.StackError, match='protected'):
        stack.guard_database()
    assert len(calls) == 1


@pytest.mark.parametrize('violation', ['pgdata', 'extra-volume', 'extra-bind', 'writable-init',
    'different-init', 'missing-init', 'duplicate-data', 'bind-driver'])
def test_only_pinned_pgdata_and_exact_database_mounts_are_allowed(tmp_path, monkeypatch, violation):
    config = configuration()
    mounts = config['services']['db']['volumes']
    if violation == 'pgdata': config['services']['db']['environment']['PGDATA'] = '/alternate'
    elif violation == 'extra-volume': mounts.append({'type': 'volume', 'source': 'another', 'target': '/other'})
    elif violation == 'extra-bind': mounts.append({'type': 'bind', 'source': '/other', 'target': '/other'})
    elif violation == 'writable-init': mounts[1]['read_only'] = False
    elif violation == 'different-init': mounts[1]['source'] = str(tmp_path / 'unreviewed-init.sh')
    elif violation == 'missing-init': mounts.pop()
    elif violation == 'duplicate-data': mounts.append(copy.deepcopy(mounts[0]))
    else: config['volumes']['postgres-data']['driver_opts'] = {'type': 'none', 'o': 'bind', 'device': '/source-data'}
    stack, calls = stack_with_docker_mock(tmp_path, monkeypatch, config)
    with pytest.raises(local_stack.StackError):
        stack.guard_database()
    assert len(calls) == 1


def test_explicit_standard_pgdata_preserves_valid_stack_configuration(tmp_path, monkeypatch):
    config = configuration()
    config['services']['db']['environment']['PGDATA'] = '/var/lib/postgresql/data'
    stack, _ = stack_with_docker_mock(tmp_path, monkeypatch, config)
    assert stack.guard_database() == 'fixture_postgres-data'


@pytest.mark.parametrize('field,value', [('tmpfs', ['/var/lib/postgresql/data']),
    ('volumes_from', ['foreign-container']), ('configs', [{'source': 'another-init'}]),
    ('secrets', [{'source': 'another-init'}])])
def test_database_mount_inheritance_or_alternate_storage_cannot_skip_guard(tmp_path, monkeypatch, field, value):
    config = configuration()
    config['services']['db'][field] = value
    stack, calls = stack_with_docker_mock(tmp_path, monkeypatch, config)
    with pytest.raises(local_stack.StackError, match='storage mounts'):
        stack.guard_database()
    assert len(calls) == 1


@pytest.mark.parametrize('service,key', [('app', 'RAG_DATABASE_URL'), ('app', 'RAG_MIGRATION_DATABASE_URL'),
    ('operator', 'RAG_DATABASE_URL'), ('operator', 'RAG_MIGRATION_DATABASE_URL')])
def test_direct_dsns_cannot_override_guarded_structured_credentials(tmp_path, monkeypatch, service, key):
    config = configuration()
    config['services'].setdefault(service, {'environment': {}})['environment'][key] = 'synthetic-private-owner-dsn'
    with pytest.raises(local_stack.StackError, match='direct DSNs') as failed:
        stack_with_docker_mock(tmp_path, monkeypatch, config)
    assert 'synthetic-private-owner-dsn' not in str(failed.value)


@pytest.mark.parametrize('running,project,allowed', [(True, 'other', False), (True, 'fixture', True),
                                                  (False, 'other', True)])
def test_named_volume_never_starts_a_second_running_owner(tmp_path, monkeypatch, running, project, allowed):
    containers = [{'State': {'Running': running}, 'Config': {'Labels': {
        'com.docker.compose.project': project, 'com.docker.compose.service': 'db'}}}]
    stack, _ = stack_with_docker_mock(tmp_path, monkeypatch, configuration(), containers)
    if allowed:
        assert stack.guard_database() == 'fixture_postgres-data'
    else:
        with pytest.raises(local_stack.StackError, match='Another running container'):
            stack.guard_database()


def test_operator_database_override_is_validated_and_never_changes_saved_configuration(tmp_path, monkeypatch):
    config = configuration()
    stack, _ = stack_with_docker_mock(tmp_path, monkeypatch, config)
    before = copy.deepcopy(stack.configuration)
    commands = []
    monkeypatch.setattr(stack, '_run', lambda *args: commands.append(args))
    stack._job('operator', command=['python', '-m', 'src.db.recovery', '--expected-database', 'rag_restore_fixture'],
               database='rag_restore_fixture')
    assert 'RAG_DB_NAME=rag_restore_fixture' in commands[0]
    assert stack.configuration == before
    for target in ('rag; DROP DATABASE rag', '', 'with space'):
        with pytest.raises(local_stack.StackError):
            stack._job('operator', command=['true'], database=target)
    with pytest.raises(local_stack.StackError):
        stack._job('app', command=['true'], database='rag_restore_fixture')


def test_up_uses_supported_compose_wait_and_a_host_process_deadline(monkeypatch):
    stack = object.__new__(local_stack.Stack)
    stack.command = ['docker', 'compose', '--env-file', 'synthetic-private.env']
    guards, calls = [], []
    monkeypatch.setattr(stack, 'guard_database', lambda: guards.append(True))
    monkeypatch.setattr(local_stack.subprocess, 'run', lambda command, **kwargs:
        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    stack.up()
    assert guards == [True]
    command, options = calls[0]
    assert '--wait' in command and '--wait-timeout' not in command
    assert command[-2:] == ['verify', 'app']
    assert options == {'check': False, 'timeout': 180}


def test_startup_timeout_is_sanitized_and_never_attempts_cleanup(monkeypatch):
    stack = object.__new__(local_stack.Stack)
    stack.command = ['docker', 'compose']
    calls = []
    monkeypatch.setattr(stack, 'guard_database', lambda: None)

    def expired(command, **kwargs):
        calls.append((command, kwargs))
        raise local_stack.subprocess.TimeoutExpired(command, 180, output='synthetic-private-output')

    monkeypatch.setattr(local_stack.subprocess, 'run', expired)
    with pytest.raises(local_stack.StackError, match='deadline') as failed:
        stack.up()
    assert len(calls) == 1 and 'synthetic-private-output' not in str(failed.value)
    assert not any(value in calls[0][0] for value in ('down', 'stop', '--volumes'))


def test_long_preparation_jobs_do_not_inherit_startup_deadline(monkeypatch):
    stack = object.__new__(local_stack.Stack)
    stack.command = ['docker', 'compose']
    calls = []
    monkeypatch.setattr(stack, 'guard_database', lambda: None)
    monkeypatch.setattr(local_stack.subprocess, 'run', lambda command, **kwargs:
        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    for service in ('stage-artifacts', 'stage-models', 'import-corpus', 'import-jira'):
        stack._job(service)
    assert len(calls) == 4 and all(options == {'check': False} for _command, options in calls)


def test_verified_linux_lock_and_lean_selection_preserve_mac_pins():
    path = local_stack.ROOT / 'requirements.lock.txt'
    full = docker_dependencies.selected_pins(path, hybrid=True)
    lean = docker_dependencies.selected_pins(path, hybrid=False)
    assert len(full) == 68 and 'torch==2.14.1+cpu' in full
    assert 'transformers==5.20.2' in full
    assert not any(item.startswith(('torch==', 'sentence-transformers==', 'faiss-cpu==')) for item in lean)
    assert 'torch==2.14.0; sys_platform == "darwin"' in path.read_text()


def test_compose_resolves_explicit_cpu_quotas_for_every_job_without_nanocpu_conflicts():
    import yaml

    # Safe YAML resolution expands the real shared operator anchor; checking
    # just text occurrences would miss inherited/overridden one-off limits.
    configured = yaml.safe_load((local_stack.ROOT / 'compose.yaml').read_text())
    expected = {'db': 1, 'verify': 1, 'app': 2, 'stage-artifacts': 2,
        'stage-models': 2, 'migrate': 1, 'import-corpus': 2, 'import-jira': 2,
        'prepare-state': 1, 'operator': 2}
    assert 'cpus' not in configured['x-operator']
    for name, cores in expected.items():
        service = configured['services'][name]
        assert 'cpus' not in service, name
        assert type(service['cpu_period']) is int and service['cpu_period'] == 100000, name
        assert type(service['cpu_quota']) is int and service['cpu_quota'] == cores * 100000, name
        assert service['memswap_limit'] == service['mem_limit'], name
    assert configured['x-operator']['cpu_period'] == 100000
    assert configured['x-operator']['cpu_quota'] == 200000
    fixture = yaml.safe_load((local_stack.ROOT / 'tests/deployment/compose.yaml').read_text())['services']['fixture-load']
    assert fixture['extends'] == {'file': 'compose.yaml', 'service': 'stage-artifacts'}
    inherited = {**configured['services']['stage-artifacts'], **fixture}
    assert 'cpus' not in inherited
    assert inherited['cpu_period'] == 100000 and inherited['cpu_quota'] == 200000
    assert inherited['memswap_limit'] == inherited['mem_limit']

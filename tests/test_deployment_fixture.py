"""Pure filesystem/index checks; service/Git/native import execution is root/CI."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.deployment import fixture


@pytest.fixture
def generated(tmp_path, monkeypatch):
    # These unit checks validate the actual artifact formats. The separate
    # Compose smoke executes genuine local Git/native imports; no Git command
    # or service is executed by this filesystem-only fixture.
    def fake_git(command, **kwargs):
        assert command[0] == 'git'
        return SimpleNamespace(stdout='a' * 40 + '\n', returncode=0)
    monkeypatch.setattr(fixture.subprocess, 'run', fake_git)
    root = tmp_path / 'deployment'
    result = fixture.prepare(root)
    return root, result


def test_fixture_has_genuine_canonical_bm25_and_bounded_import_span_files(generated):
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.storage.contracts import CatalogEntry
    from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
    root, result = generated
    operator = root / 'operator'
    plan = fixture.private(operator / 'fixture-plan.json')
    artifacts = operator / 'artifacts'
    rows = pq.read_table(artifacts / 'documents.parquet').to_pylist()
    assert result == {'fixture_documents': 5, 'synthetic': True, 'model_downloads': 0}
    assert [row['doc_id'] for row in rows] == list(fixture.DOC_IDS)
    canonical = [row['title'] + '\n' + row['content'] for row in rows]
    entries = tuple(CatalogEntry(doc_id=row['doc_id'], position=index, source_type=row['source_type'],
        revision_id=canonical_revision_id(canonical[index]), text_length=len(canonical[index]), metadata_origin='synthetic')
        for index, row in enumerate(rows))
    assert plan['snapshot']['corpus_sha256'] == hashlib.sha256((artifacts / 'documents.parquet').read_bytes()).hexdigest()
    assert plan['snapshot']['catalog_sha256'] == catalog_fingerprint(entries, expected_count=5)
    # Exercise the real native-import identity reader without Git or a DB.
    from src.contracts import SnapshotRef
    from src.data.structured import _canonical_records
    assert _canonical_records(artifacts / 'documents.parquet', SnapshotRef.model_validate(plan['snapshot'])) == {
        doc: ('jira/fixture/' + doc + '.json', canonical_revision_id(canonical[fixture.DOC_IDS.index(doc)]),
              len(canonical[fixture.DOC_IDS.index(doc)])) for doc in ('general', 'executive')}
    index = CorpusBM25.load(artifacts / 'indexes/bm25', corpus_path=None, mmap=True)
    assert index.corpus is None
    assert {doc for doc, _score in index.rank('scopefixture', 5)} == set(fixture.DOC_IDS)
    assert {doc for doc, _score in index.rank('scopefixture', 5, mask=np.array([True, True, False, False, False]))} == {'general', 'atlas'}
    spans = pq.read_table(artifacts / 'indexes/dense-bge-base/shard_00000.parquet').to_pydict()
    assert spans == {'doc_index': list(range(5)), 'start_char': [0] * 5, 'end_char': [len(text) for text in canonical]}
    manifest = json.loads((artifacts / 'indexes/dense-bge-base/manifest.json').read_text())
    assert manifest['fixture'] is True and manifest['model']['name'] == 'ci-import-spans-not-a-model'
    assert 'not a searchable embedding index' in manifest['purpose']
    assert all(fixture.sha(artifacts / name) == digest for name, digest in plan['files'].items())


def test_fixture_configuration_is_private_separate_and_real_auth_only(generated):
    root, _ = generated
    env = root / 'compose.env'
    assert env.stat().st_mode & 0o777 == 0o600
    values = dict(line.split('=', 1) for line in env.read_text().splitlines())
    assert values['COMPOSE_PROJECT_NAME'].startswith('rag-ci-fixture-')
    assert values['RAG_AUTH_MODE'] == 'local' and values['RAG_DEMO_ACCESS'] == 'false'
    assert values['INSTALL_HYBRID'] == 'false' and values['RAG_RETRIEVAL'] == 'bm25' and values['RAG_ALLOW_PAID'] == 'false'
    assert len({values[key] for key in ('RAG_DB_BOOTSTRAP_PASSWORD', 'RAG_DB_MIGRATOR_PASSWORD', 'RAG_DB_RUNTIME_PASSWORD')}) == 3
    assert env.parent != Path(values['OPERATOR_DIR'])
    assert not (root / 'operator/qa-state.json').exists(), 'No account/default password may be preseeded by artifact preparation'


def test_preparing_existing_fixture_never_overwrites_its_data_or_configuration(generated):
    root, _ = generated
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        fixture.prepare(root)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_fixture_extends_an_explicit_service_in_the_actual_base_compose_file():
    import yaml
    root = Path(__file__).resolve().parents[1]
    override_file = root / 'tests/deployment/compose.yaml'
    override = yaml.safe_load(override_file.read_text())
    extension = override['services']['fixture-load']['extends']
    # Compose resolves extends.file relative to the base project file, not
    # the nested override's directory. Omitting file tries the override alone.
    target = (root / extension['file']).resolve()
    assert target == root / 'compose.yaml'
    base = yaml.safe_load(target.read_text())
    assert extension['service'] not in override['services']
    inherited = base['services'][extension['service']]
    assert inherited['build']['target'] == 'operator'
    assert 'corpus-data:/data/corpus' in inherited['volumes']
    assert 'index-data:/data/indexes' in inherited['volumes']
    assert inherited['depends_on'] == {}
    workflow = (root / '.github/workflows/ci.yml').read_text()
    assert '-f compose.yaml -f tests/deployment/compose.yaml' in workflow
    assert '--wait-timeout' not in workflow, 'The supported Compose2.10 CLI lacks that newer flag'


@pytest.mark.parametrize('script_path', ['/operator/fixture.py', '/fixture.py'])
def test_copied_helper_bootstraps_from_container_working_directory_at_any_depth(tmp_path, monkeypatch, script_path):
    # Execute the real stdlib-only module bootstrap with its container __file__;
    # no Git, model, database, main command or service is invoked.
    source = Path(fixture.__file__).read_text()
    app = tmp_path / 'app'
    (app / 'src').mkdir(parents=True)
    monkeypatch.chdir(app)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    namespace = {'__file__': script_path, '__name__': 'fixture_import_only'}
    exec(compile(source, script_path, 'exec'), namespace)
    assert sys.path[0] == str(app)
    assert namespace['DOC_IDS'] == fixture.DOC_IDS

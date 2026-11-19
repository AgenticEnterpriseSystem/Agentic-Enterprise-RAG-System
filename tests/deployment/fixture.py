"""Explicit five-document CI deployment fixture; never a production fallback.

prepare/load are filesystem-only. native/seed are operator-only commands. HTTP
checks use actual human login/RLS and persistent cookies; no model/provider is
called. Root executes every service/database/network command in this session.
"""
import argparse
import hashlib
from http.cookiejar import MozillaCookieJar
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPCookieProcessor, HTTPRedirectHandler

for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
    if (candidate / 'src').is_dir():
        sys.path.insert(0, str(candidate))
        break

SNAPSHOT_ID = 'ci-five-document-auth-fixture'
REPOSITORY = 'https://example.invalid/ci-deployment-fixture.git'
DOC_IDS = ('general', 'atlas', 'executive', 'unclassified', 'ungranted')
SCOPES = {'intern': ('general', 'atlas'), 'executive': ('general', 'executive')}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), 'wb') as stream:
        stream.write(canonical(value) + b'\n')


def private(path):
    import stat
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size < 262144,
            'Expected bounded private fixture input')
        return json.loads(stream.read(262144))


def prepare(directory):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.data.corpus import extract_document
    from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25
    from src.storage.contracts import CatalogEntry
    from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
    root = directory.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    operator, artifacts = root / 'operator', root / 'operator/artifacts'
    operator.mkdir(mode=0o700)
    artifacts.mkdir()
    (operator / 'models').mkdir()
    repo = operator / 'source'
    source = repo / 'generated_data/sources/jira/fixture'
    source.mkdir(parents=True)
    rows, entries, texts = [], [], []
    for position, doc in enumerate(DOC_IDS):
        title = 'CI synthetic ' + doc
        content = 'scopefixture ' + doc + ' synthetic permitted text. Café 🧭.'
        platform = 'jira' if doc in {'general', 'executive'} else 'confluence' if doc in {'atlas', 'ungranted'} else 'slack'
        if platform == 'jira':
            record = {'dataset_doc_uuid': doc, 'key': 'CI-' + str(position), 'summary': title,
                'description': content, 'title_field_name': 'summary', 'content_field_names': ['description'],
                'status': 'Open', 'priority': 'P1', 'project': 'synthetic-ci'}
            write(source / (doc + '.json'), record, mode=0o644)
            title, content = extract_document(record)
        text = title + '\n' + content
        rows.append({'doc_id': doc, 'source_type': platform, 'title': title, 'content': content,
            'path': platform + '/fixture/' + doc + '.json', 'metadata_origin': 'synthetic'})
        texts.append(text)
        entries.append(CatalogEntry(doc_id=doc, position=position, source_type=platform,
            revision_id=canonical_revision_id(text), text_length=len(text), metadata_origin='synthetic'))
    git_env = {**os.environ, 'GIT_AUTHOR_NAME': 'CI fixture', 'GIT_COMMITTER_NAME': 'CI fixture',
        'GIT_AUTHOR_EMAIL': 'fixture@example.invalid', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'}
    for command in (['init', '-q'], ['remote', 'add', 'origin', REPOSITORY], ['add', '.'],
                    ['-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'Synthetic deployment fixture']):
        subprocess.run(['git', '-C', str(repo), *command], check=True, capture_output=True, env=git_env)
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
        capture_output=True, text=True).stdout.strip()
    corpus = artifacts / 'documents.parquet'
    pq.write_table(pa.Table.from_pylist(rows), corpus, row_group_size=2)
    snapshot = {'snapshot_id': SNAPSHOT_ID, 'corpus_sha256': sha(corpus),
        'catalog_sha256': catalog_fingerprint(entries, expected_count=5), 'document_count': 5}
    config = BM25Config()
    model, statistics = build_bm25(texts, config)
    CorpusBM25(model, list(DOC_IDS), config, manifest={'version': 1, 'corpus_sha256': snapshot['corpus_sha256'],
        'config': config.model_dump(), 'stats': statistics}).save(artifacts / 'indexes/bm25')
    dense = artifacts / 'indexes/dense-bge-base'
    dense.mkdir(parents=True)
    write(dense / 'manifest.json', {'version': 1, 'complete': True, 'documents': 5, 'shards': 1,
        'corpus_sha256': snapshot['corpus_sha256'], 'fixture': True,
        'purpose': 'Import span format only; not a searchable embedding index. CI application MUST use bm25.',
        'model': {'name': 'ci-import-spans-not-a-model', 'revision': '0' * 40, 'normalize': True}}, mode=0o644)
    write(dense / 'doc_ids.json', list(DOC_IDS), mode=0o644)
    np.save(dense / 'shard_00000.npy', np.ones((5, 2), dtype=np.float32), allow_pickle=False)
    pq.write_table(pa.table({'doc_index': list(range(5)), 'start_char': [0] * 5,
        'end_char': [len(text) for text in texts]}), dense / 'shard_00000.parquet')
    ledger = {path.relative_to(artifacts).as_posix(): sha(path) for path in artifacts.rglob('*') if path.is_file()}
    write(operator / 'fixture-plan.json', {'format': 'ci-real-auth-five-doc-v1', 'synthetic': True,
        'snapshot': snapshot, 'commit': commit, 'files': ledger})
    shutil.copyfile(__file__, operator / 'fixture.py')
    values = {'COMPOSE_PROJECT_NAME': 'rag-ci-fixture-' + secrets.token_hex(4), 'OPERATOR_DIR': str(operator),
        'LOCAL_UID': str(os.getuid()), 'LOCAL_GID': str(os.getgid()), 'ARTIFACTS_DIR': str(artifacts),
        'MODELS_DIR': str(operator / 'models'), 'SOURCE_REPO_DIR': str(repo),
        'INSTALL_HYBRID': 'false', 'RAG_MODE': 'retrieval', 'RAG_RETRIEVAL': 'bm25',
        'RAG_AUTH_MODE': 'local', 'RAG_DEMO_ACCESS': 'false', 'RAG_ALLOW_PAID': 'false',
        'RAG_AUTH_SECURE_COOKIES': 'false', 'RAG_AUTH_ALLOW_INSECURE_LOOPBACK': 'true',
        'RAG_AUTH_ALLOWED_ORIGINS': 'http://127.0.0.1:18080', 'WEB_PORT': '18080', 'RAG_SNAPSHOT_ID': SNAPSHOT_ID,
        'RAG_DB_BOOTSTRAP_PASSWORD': secrets.token_urlsafe(32), 'RAG_DB_MIGRATOR_PASSWORD': secrets.token_urlsafe(32),
        'RAG_DB_RUNTIME_PASSWORD': secrets.token_urlsafe(32)}
    with os.fdopen(os.open(root / 'compose.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        stream.write('\n'.join(key + '=' + value for key, value in values.items()) + '\n')
    return {'fixture_documents': 5, 'synthetic': True, 'model_downloads': 0}


def load(operator):
    need(os.environ.get('RAG_CI_FIXTURE') == 'explicit-five-doc', 'This loader is CI-only')
    plan = private(operator / 'fixture-plan.json')
    need(plan['format'] == 'ci-real-auth-five-doc-v1' and plan['snapshot']['document_count'] == 5, 'Not the explicit tiny fixture')
    artifacts = operator / 'artifacts'
    need(set(plan['files']) == {path.relative_to(artifacts).as_posix() for path in artifacts.rglob('*') if path.is_file()}, 'Fixture inventory differs')
    need(all(sha(artifacts / name) == value for name, value in plan['files'].items()), 'Fixture bytes changed')
    for destination in (Path('/data/corpus'), Path('/data/indexes')):
        destination.mkdir(parents=True, exist_ok=True)
        need(not any(destination.iterdir()), 'CI loader requires fresh empty project volumes')
    shutil.copyfile(artifacts / 'documents.parquet', '/data/corpus/documents.parquet')
    shutil.copytree(artifacts / 'indexes', '/data/indexes', dirs_exist_ok=True)
    for root in (Path('/data/corpus'), Path('/data/indexes')):
        for path in root.rglob('*'):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(root, 0o555)
    state = Path('/state')
    state.mkdir(parents=True, exist_ok=True)
    need(not any(state.iterdir()), 'CI state volume must be fresh')
    os.chown(state, 10001, 10001); os.chmod(state, 0o755)
    return {'fixture_documents_staged': 5, 'synthetic': True}


def native(operator):
    from src.contracts import SnapshotRef
    from src.data.structured import import_jira, prepare_jira_inputs
    plan = private(operator / 'fixture-plan.json')
    snapshot = SnapshotRef.model_validate_json(Path('/state/snapshot.json').read_bytes())
    need(snapshot.model_dump(mode='json') == plan['snapshot'], 'Actual canonical import differs from fixture plan')
    inputs = prepare_jira_inputs(operator / 'source', Path('/data/corpus/documents.parquet'), snapshot,
        repository=REPOSITORY, commit=plan['commit'], coverage='synthetic_fixture')
    result = import_jira(inputs)
    need(result['state'] == 'ready' and result['records'] == 2 and result['coverage'] == 'synthetic_fixture', 'Native fixture import incomplete')
    return {'native_records': 2, 'synthetic': True}


def seed(operator):
    from src.access.repository import GrantSpec, PolicyManifest, PolicyRepository
    from src.auth.accounts import create_account
    from src.contracts import SnapshotRef
    snapshot = SnapshotRef.model_validate_json(Path('/state/snapshot.json').read_bytes())
    need(snapshot.model_dump(mode='json') == private(operator / 'fixture-plan.json')['snapshot'], 'Wrong fixture snapshot')
    nonce = secrets.token_hex(16)
    accounts = {}
    for name, documents in SCOPES.items():
        account = {'login': 'ci_' + name + '_' + nonce, 'password': secrets.token_urlsafe(32),
            'user_id': 'ci-' + name + '-' + nonce, 'documents': list(documents)}
        accounts[name] = account
    # Reserve private credentials before any account mutation; no seeded/default
    # password exists in the image, template, application or public CI report.
    write(operator / 'qa-state.json', {'format': 'ci-auth-private-v1', 'accounts': accounts})
    for name, account in accounts.items():
        create_account(account['login'], account['password'], user_id=account['user_id'],
            display_name='Synthetic CI ' + name, role='intern' if name == 'intern' else 'ceo')
    manifest = PolicyManifest(manifest_id='ci-five-doc:' + nonce, snapshot=snapshot, origin='synthetic',
        reviewed_by='disposable-ci-fixture', provenance={'purpose': 'Actual human auth/RLS deployment test, not production policy'},
        grants=tuple(GrantSpec(user_id=value['user_id'], doc_ids=tuple(value['documents'])) for value in accounts.values()))
    result = PolicyRepository().apply(manifest)
    need(result.changed_grants == 4 and result.changed_memberships == 0, 'Unexpected CI grant mutation')
    return {'human_accounts': 2, 'direct_grants': 4, 'synthetic': True}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, base, path):
        self.base, self.jar = base, MozillaCookieJar(str(path))
        self.opener = build_opener(NoRedirect(), HTTPCookieProcessor(self.jar))

    def request(self, path, body=None, csrf=None):
        headers = {'Origin': self.base}
        if csrf is not None:
            headers['X-CSRF-Token'] = csrf
        if body is not None:
            headers['Content-Type'] = 'application/json'
        try:
            response = self.opener.open(Request(self.base + path, data=canonical(body) if body is not None else None, headers=headers), timeout=30)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(2 * 1024 * 1024 + 1)
            need(len(raw) <= 2 * 1024 * 1024, 'HTTP fixture response exceeds bound')
            need('no-store' in response.headers.get('Cache-Control', ''), 'Missing private response caching policy')
            return response.code, json.loads(raw) if raw else None


def http_check(operator, phase, base):
    need(base == 'http://127.0.0.1:18080', 'CI smoke is loopback-only')
    state = private(operator / 'qa-state.json')
    clients = {name: Client(base, operator / (name + '.cookies')) for name in state['accounts']}
    deadline = time.monotonic() + 120
    wanted = 503 if phase == 'unavailable' else 200
    while True:
        try:
            status, _ = clients['intern'].request('/api/ready')
            if status == wanted:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        need(time.monotonic() < deadline, 'Expected readiness state was not reached')
        time.sleep(.5)
    need(clients['intern'].request('/api/health')[0] == 200, 'Liveness must remain available')
    if phase == 'before':
        need(clients['intern'].request('/api/query', {'question': 'scopefixture'})[0] == 401, 'Anonymous query was not denied')
        saved = {}
        for name, account in state['accounts'].items():
            client = clients[name]
            status, challenge = client.request('/api/auth/csrf'); need(status == 200, 'Bootstrap failed')
            status, result = client.request('/api/auth/login', {'login': account['login'], 'password': account['password']}, challenge['csrf_token'])
            need(status == 200 and result['user']['user_id'] == account['user_id'], 'Actual human login failed')
            cookie = next(value for value in client.jar if value.name == 'rag_session')
            need(not cookie.domain_specified and cookie.path == '/' and 'HttpOnly' in cookie._rest
                and cookie._rest.get('SameSite', '').lower() == 'strict', 'Real session cookie flags differ')
            saved[name] = result['csrf_token']
            need(client.request('/api/query', {'question': 'scopefixture'})[0] == 403, 'Missing CSRF was not rejected')
        write(operator / 'session-state.json', saved)
    else:
        saved = private(operator / 'session-state.json')
        for client in clients.values():
            import stat
            descriptor = os.open(client.jar.filename, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
            with os.fdopen(descriptor, 'rb') as stream:
                info = os.fstat(stream.fileno())
                need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size < 65536,
                    'Private persisted cookie is missing')
                raw = stream.read(65536)
                need(len(raw) < 65536, 'Private cookie grew beyond its bound')
            client.jar._really_load(StringIO(raw.decode()), client.jar.filename, True, True)
    for name, account in state['accounts'].items():
        client = clients[name]
        if phase == 'unavailable':
            status, body = client.request('/api/query', {'question': 'scopefixture'}, saved[name])
            need(status == 503 and 'sources' not in body, 'DB loss must fail closed')
            continue
        status, who = client.request('/api/auth/me')
        need(status == 200 and who['user']['user_id'] == account['user_id'], 'Persisted session/account did not survive')
        for mode in ('baseline', 'agentic'):
            status, result = client.request('/api/query', {'question': 'scopefixture', 'mode': mode}, saved[name])
            need(status == 200 and result['trace']['usage']['api_calls'] == 0, 'BM25 fixture query failed or called a model')
            need(result['answer']['status'] == 'insufficient_evidence', 'Retrieval-only fixture generated an answer')
            need({item['chunk']['doc_id'] for item in result['sources']} == set(account['documents']), 'RLS persisted grant scope differs')
            need(all(item['kind'] == 'passage' and item['chunk']['metadata_origin'] == 'synthetic' for item in result['sources']), 'Not canonical fixture excerpts')
        if phase == 'before':
            fd = os.open(client.jar.filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd); client.jar.save(ignore_discard=True, ignore_expires=True)
    return {'phase': phase, 'human_sessions_checked': 2, 'synthetic': True, 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'load', 'native', 'seed', 'before', 'resume', 'unavailable'))
    parser.add_argument('--directory', type=Path, default=Path('/operator'))
    parser.add_argument('--base', default='http://127.0.0.1:18080')
    args = parser.parse_args()
    try:
        result = prepare(args.directory) if args.command == 'prepare' else load(args.directory) if args.command == 'load' else native(args.directory) if args.command == 'native' else seed(args.directory) if args.command == 'seed' else http_check(args.directory, args.command, args.base)
    except Exception as error:
        # CI should publish only a sanitized failure. Keep this private file out
        # of action artifacts; root can inspect it without forwarding secrets.
        root = args.directory if args.directory.is_dir() else args.directory.parent
        diagnostic = root / ('diagnostic-' + args.command + '-' + secrets.token_hex(4) + '.json')
        try:
            write(diagnostic, {'exception_type': type(error).__name__, 'traceback': traceback.format_exc()})
        except OSError:
            pass  # Read-only loader inputs must never turn failure into a raw traceback.
        print(json.dumps({'passed': False, 'phase': args.command, 'error_type': type(error).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({'passed': True, **result}))


if __name__ == '__main__':
    main()

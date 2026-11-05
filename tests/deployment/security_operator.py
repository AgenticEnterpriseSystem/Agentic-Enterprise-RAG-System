"""Root-operated full-corpus SEC fixture; credentials stay in a private ledger.

Only the independently attested local OPS project is supported. This helper
creates two nonce-owned humans and two reviewed direct-grant manifests. It never
changes source documents, native feeds, existing users, or unrelated grants.
Root mounts this script read-only and supplies the dedicated migrator login.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys

for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
    if (candidate / 'src').is_dir():
        sys.path.insert(0, str(candidate))
        break

FORMAT = 'rag-sec-human-v1'
PROJECT = 'rag-ops01-full-20261104'
BASE = 'http://127.0.0.1:18081'
COUNT = 511958
MAX_STATE = 262144
PURPOSE = 'Explicit synthetic full-corpus SEC-01 isolation QA; no production permission claim'
NAMES = ('intern', 'executive')
QUESTION = 'What information is recorded when investigating a service incident?'


class SecurityOperatorError(ValueError):
    """Content-free fixture administration failure."""


def need(condition, message):
    if not condition:
        raise SecurityOperatorError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate private ledger field')
        result[key] = value
    return result


def read_json(path, *, private=True, limit=MAX_STATE):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_size <= limit
             and (not private or stat.S_IMODE(info.st_mode) == 0o600), 'Expected bounded regular private input')
        raw = stream.read(limit + 1)
    need(len(raw) <= limit, 'Private input grew beyond its bound')
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(SecurityOperatorError('Invalid JSON constant')))


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_state(path, value, *, first=False):
    raw = canonical(value) + b'\n'
    need(len(raw) <= MAX_STATE, 'Private ledger exceeds its bound')
    if first:
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    else:
        read_json(path)
        temporary = path.with_name('.sec-state-' + secrets.token_hex(12))
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
             'Expected private regular ledger lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)  # Kernel lock release also permits recovery after a crash.


def manifests_for(state):
    from src.access.repository import GrantSpec, PolicyManifest
    return {name: PolicyManifest(manifest_id='sec01:' + state['nonce'] + ':' + name,
        snapshot=state['snapshot'], origin='synthetic', reviewed_by=state['reviewed_by'],
        provenance={'purpose': PURPOSE, 'compose_project': state['project'], 'nonce': state['nonce'],
                    'account': name},
        grants=(GrantSpec(user_id=state['accounts'][name]['user_id'],
                          doc_ids=tuple(state['accounts'][name]['allowed_doc_ids'])),)) for name in NAMES}


def make_state(project, snapshot, reviewed_by, documents):
    nonce = secrets.token_hex(16)
    accounts = {}
    for name, allowed in (('intern', ('A',)), ('executive', ('A', 'B'))):
        forbidden = tuple(key for key in ('A', 'B', 'C') if key not in allowed)
        accounts[name] = {'user_id': 'sec01-' + nonce + '-' + name,
            'login': 'sec01_' + nonce + '_' + name,
            'password': secrets.token_urlsafe(32), 'display_name': 'Synthetic SEC-01 ' + nonce + ' ' + name,
            'role': 'executive' if name == 'intern' else 'intern', 'department': 'synthetic-sec01',
            'allowed_doc_ids': [documents[key]['doc_id'] for key in allowed],
            'forbidden_markers': [documents[key][field] for key in forbidden for field in ('doc_id', 'title', 'text_marker')]}
    state = {'format': FORMAT, 'nonce': nonce, 'project': project, 'expected_database': 'rag',
        'base': BASE, 'snapshot': snapshot.model_dump(mode='json'), 'reviewed_by': reviewed_by,
        'status': 'reserved', 'created_at': datetime.now(timezone.utc).isoformat(),
        'question': QUESTION, 'documents': documents, 'accounts': accounts,
        'ledger': {name: {'account_created': False, 'apply_requested': False, 'revoke_requested': False}
                   for name in NAMES}}
    manifests = manifests_for(state)
    state['manifests'] = {name: value.model_dump(mode='json') for name, value in manifests.items()}
    state['manifest_sha256'] = {name: digest(value) for name, value in state['manifests'].items()}
    validate_state(state, project, snapshot)
    return state


def validate_state(state, project, snapshot):
    need(isinstance(state, dict) and state.get('format') == FORMAT and project == PROJECT
         and state.get('project') == project and state.get('base') == BASE
         and state.get('expected_database') == 'rag'
         and state.get('snapshot') == snapshot.model_dump(mode='json') and snapshot.document_count == COUNT,
         'Private fixture target differs')
    nonce = state.get('nonce')
    need(isinstance(nonce, str) and re.fullmatch(r'[0-9a-f]{32}', nonce), 'Invalid owned fixture namespace')
    reviewer = state.get('reviewed_by')
    need(isinstance(reviewer, str) and reviewer == reviewer.strip() and 1 <= len(reviewer) <= 254
         and all(ord(c) >= 32 and ord(c) != 127 for c in reviewer), 'Invalid original review attestation')
    need(state.get('status') in {'reserved', 'seeding', 'ready', 'intern-revoked', 'cleaning', 'cleaned'}
         and state.get('question') == QUESTION, 'Invalid fixture lifecycle')
    documents = state.get('documents', {})
    need(set(documents) == {'A', 'B', 'C'} and len({v.get('doc_id') for v in documents.values()}) == 3,
         'Expected three distinct canonical documents')
    for value in documents.values():
        need(isinstance(value.get('doc_id'), str) and value['doc_id']
             and isinstance(value.get('title'), str) and 20 <= len(value['title']) <= 240
             and isinstance(value.get('revision_id'), str) and re.fullmatch(r'[0-9a-f]{64}', value['revision_id'])
             and isinstance(value.get('text_marker'), str) and 80 <= len(value['text_marker']) <= 120
             and len(value['text_marker'].strip()) >= 80
             and type(value.get('text_marker_start')) is int and value['text_marker_start'] >= len(value['title'])
             and value.get('source_type') == 'jira', 'Invalid canonical document declaration')
    need(len({v['title'].casefold() for v in documents.values()}) == 3, 'Document titles are not distinct')
    need(set(state.get('accounts', {})) == set(NAMES) and set(state.get('ledger', {})) == set(NAMES),
         'Expected the two exact owned humans')
    for name, labels in (('intern', ('A',)), ('executive', ('A', 'B'))):
        account = state['accounts'][name]
        denied = tuple(label for label in ('A', 'B', 'C') if label not in labels)
        need(account.get('user_id') == 'sec01-' + nonce + '-' + name
             and account.get('login') == 'sec01_' + nonce + '_' + name
             and account.get('display_name') == 'Synthetic SEC-01 ' + nonce + ' ' + name
             and account.get('department') == 'synthetic-sec01'
             and account.get('role') == ('executive' if name == 'intern' else 'intern')
             and isinstance(account.get('password'), str) and 32 <= len(account['password']) <= 128
             and account.get('allowed_doc_ids') == [documents[label]['doc_id'] for label in labels]
             and account.get('forbidden_markers') == [documents[label][field] for label in denied for field in ('doc_id', 'title', 'text_marker')],
             'Owned human or explicit scope differs')
        entry = state['ledger'][name]
        need(isinstance(entry, dict) and all(type(entry.get(key)) is bool
             for key in ('account_created', 'apply_requested', 'revoke_requested')), 'Invalid durable mutation ledger')
    expected = {name: value.model_dump(mode='json') for name, value in manifests_for(state).items()}
    need(state.get('manifests') == expected and state.get('manifest_sha256') ==
         {name: digest(value) for name, value in expected.items()}, 'Reviewed manifest or ownership binding differs')
    return manifests_for(state)


def canonical_body(row):
    text = row['canonical_text']
    if text is None:
        text = bytes(row['canonical_utf8']).decode('utf-8')
    need(hashlib.sha256(text.encode('utf-8')).hexdigest() == row['revision_id'],
         'Canonical revision identity differs')
    return text


def body_marker(text, title, other_texts):
    """Return one bounded verbatim body slice, preferring a single text line."""
    start = len(title)
    while start < len(text) and text[start].isspace():
        start += 1
    folded = tuple(value.casefold() for value in other_texts)
    # Single-line candidates avoid escaping differences when scanning response
    # text. A multiline fallback still preserves the exact canonical substring.
    for single_line in (True, False):
        for offset in range(start, len(text) - 79, 40):
            marker = text[offset:offset + 100]
            if len(marker.strip()) < 80 or (single_line and ('\n' in marker or '\r' in marker)):
                continue
            if all(marker.casefold() not in value for value in folded):
                return marker, offset
    return None


def choose_documents(rows):
    """Select bounded real Jira records whose denied markers do not collide."""
    chosen, texts = {}, []
    for row in rows:
        title = row['title']
        if not isinstance(title, str) or not 20 <= len(title) <= 240 or any(ord(c) < 32 for c in title):
            continue
        text = canonical_body(row)
        markers = (row['doc_id'], title)
        if not text.startswith(title) or any(marker.casefold() in old.casefold() for marker in markers for old in texts):
            continue
        if any(old[field].casefold() in text.casefold() for old in chosen.values()
               for field in ('doc_id', 'title', 'text_marker')):
            continue
        selected_marker = body_marker(text, title, texts)
        if selected_marker is None:
            continue
        document = {key: row[key] for key in
            ('doc_id', 'revision_id', 'title', 'source_type', 'native_key', 'status', 'priority', 'project_id')}
        document['text_marker'], document['text_marker_start'] = selected_marker
        chosen[('A', 'B', 'C')[len(chosen)]] = document
        texts.append(text)
        if len(chosen) == 3:
            return chosen
    raise SecurityOperatorError('Could not select three collision-free native Jira records')


class Operator:
    def __init__(self, project, snapshot):
        need(project == PROJECT and os.environ.get('RAG_SEC_COMPOSE_PROJECT') == project,
             'Root must attest the exact independently guarded Compose project')
        import sqlalchemy as sa
        from sqlalchemy.pool import NullPool
        from src.db.runtime import database_url
        self.sa, self.project, self.snapshot = sa, project, snapshot
        self.url = database_url(migration=True)
        need(self.url.database == 'rag' and self.url.username == 'rag_migrator', 'Expected dedicated rag migrator target')
        self.engine = sa.create_engine(self.url, poolclass=NullPool, hide_parameters=True,
            connect_args={'prepare_threshold': None, 'connect_timeout': 5, 'application_name': 'rag-sec01-fixture'})
        self.guard()

    def guard(self):
        from src.db import schema
        from src.db.migrate import EXPECTED_HEAD
        with self.engine.connect() as connection:
            row = connection.execute(self.sa.text('''SELECT current_database(), current_user, session_user,
                rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, rolreplication
                FROM pg_roles WHERE rolname=current_user''')).one()
            need(tuple(row[:3]) == ('rag', 'rag_migrator', 'rag_migrator') and all(v is False for v in row[3:])
                 and connection.connection.driver_connection.info.user == 'rag_migrator', 'Wrong database administrator role')
            need(connection.execute(self.sa.text("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname='rag'")).scalar_one()
                 == 'rag_migrator', 'Unexpected application schema owner')
            need(EXPECTED_HEAD == '0004_jira_feed' and connection.execute(self.sa.text(
                'SELECT version_num FROM rag.alembic_version')).scalars().all() == [EXPECTED_HEAD], 'Incompatible migration head')
            actual = connection.execute(self.sa.select(schema.dataset_snapshots).where(
                schema.dataset_snapshots.c.snapshot_id == self.snapshot.snapshot_id)).mappings().one_or_none()
            need(actual is not None and actual['state'] == 'ready' and all(actual[k] == v
                 for k, v in self.snapshot.model_dump().items()), 'Accepted ready snapshot differs')
            imported = connection.execute(self.sa.select(schema.catalog_imports.c.phase,
                schema.catalog_imports.c.expected_document_count).where(
                schema.catalog_imports.c.snapshot_id == self.snapshot.snapshot_id)).one_or_none()
            need(imported is not None and tuple(imported) == ('complete', COUNT), 'Canonical import is incomplete')
            feed = connection.execute(self.sa.select(schema.ticket_feeds).where(
                schema.ticket_feeds.c.snapshot_id == self.snapshot.snapshot_id,
                schema.ticket_feeds.c.source_type == 'jira')).mappings().one_or_none()
            need(feed is not None and feed['state'] == 'ready' and feed['coverage'] == 'verified_records'
                 and feed['corpus_sha256'] == self.snapshot.corpus_sha256
                 and feed['expected_records'] == feed['imported_records'] == 6118, 'Expected accepted native Jira feed')
            self.feed_id = feed['feed_id']

    def documents(self, ids=None):
        from src.db import schema
        d, r, t = schema.documents, schema.document_revisions, schema.tickets
        join = d.join(r, self.sa.and_(d.c.snapshot_id == r.c.snapshot_id, d.c.doc_id == r.c.doc_id,
            d.c.revision_id == r.c.revision_id)).join(t, self.sa.and_(d.c.snapshot_id == t.c.snapshot_id,
                d.c.doc_id == t.c.doc_id, d.c.revision_id == t.c.revision_id))
        statement = self.sa.select(d.c.doc_id, d.c.revision_id, d.c.source_type, r.c.title,
            r.c.canonical_text, r.c.canonical_utf8, t.c.native_key, t.c.status, t.c.priority, t.c.project_id).select_from(join).where(
                d.c.snapshot_id == self.snapshot.snapshot_id, d.c.source_type == 'jira',
                t.c.feed_id == self.feed_id, t.c.coverage == 'verified_records').order_by(d.c.position)
        statement = statement.where(d.c.doc_id.in_(ids)) if ids is not None else statement.limit(64)
        with self.engine.connect() as connection:
            return connection.execute(statement).mappings().all()

    def verify_documents(self, state):
        expected = {value['doc_id']: value for value in state['documents'].values()}
        rows = self.documents(tuple(expected))
        need(len(rows) == 3, 'Selected canonical records are unavailable')
        for row in rows:
            document = expected[row['doc_id']]
            need(all(row[k] == v for k, v in document.items() if k not in ('text_marker', 'text_marker_start')),
                 'Selected native record metadata changed')
            text = canonical_body(row); start = document['text_marker_start']; marker = document['text_marker']
            need(start >= len(document['title']) and text[start:start + len(marker)] == marker,
                 'Selected body marker differs from its actual canonical revision')
        need(choose_documents(rows) == state['documents'], 'Selected forbidden markers no longer bind the canonical bodies')

    @staticmethod
    def check_account(row, account):
        need(row['user_id'] == account['user_id'] and row['login'] == account['login']
             and row['account_kind'] == 'human' and row['display_name'] == account['display_name']
             and row['role'] == account['role'] and row['department'] == account['department']
             and tuple(row['project_ids']) == (), 'Database human is outside exact owned fixture identity')

    def account(self, account):
        from src.db import schema
        with self.engine.connect() as connection:
            rows = connection.execute(self.sa.select(schema.users).where(self.sa.or_(
                schema.users.c.user_id == account['user_id'], schema.users.c.login == account['login']))).mappings().all()
        need(len(rows) <= 1, 'Owned login or user ID collides')
        if rows:
            self.check_account(rows[0], account)
            return rows[0]
        return None

    def audit(self, manifest):
        from src.db import schema
        with self.engine.connect() as connection:
            rows = connection.execute(self.sa.select(schema.policy_changes).where(
                schema.policy_changes.c.manifest_id == manifest.manifest_id)).mappings().all()
        body = manifest.model_dump(mode='json'); sha = digest(body); result = {}
        for row in rows:
            need(row['action'] in {'apply', 'revoke'} and row['manifest'] == body and row['manifest_sha256'] == sha
                 and row['origin'] == 'synthetic' and row['reviewed_by'] == manifest.reviewed_by
                 and row['snapshot_id'] == self.snapshot.snapshot_id
                 and row['change_id'] == hashlib.sha256((row['action'] + ':' + manifest.manifest_id + ':' + sha).encode()).hexdigest(),
                 'Owned policy audit differs')
            result[row['action']] = row['change_id']
        need('revoke' not in result or 'apply' in result, 'Owned policy revocation has no original apply')
        return result

    def subject(self, account):
        from src.db import schema
        with self.engine.connect() as connection:
            grants = connection.execute(self.sa.select(schema.document_grants).where(
                schema.document_grants.c.user_id == account['user_id'])).mappings().all()
            groups = connection.execute(self.sa.select(self.sa.func.count()).select_from(schema.group_memberships).where(
                schema.group_memberships.c.user_id == account['user_id'])).scalar_one()
            sessions = connection.execute(self.sa.select(self.sa.func.count()).select_from(schema.sessions).where(
                schema.sessions.c.user_id == account['user_id'], schema.sessions.c.revoked_at.is_(None))).scalar_one()
        return grants, groups, sessions

    def revoke_sessions(self, account):
        from src.db import schema
        from src.db.clock import session_time
        with self.engine.begin() as connection:
            row = connection.execute(self.sa.select(schema.users).where(
                schema.users.c.user_id == account['user_id'], schema.users.c.login == account['login']).with_for_update()).mappings().one()
            self.check_account(row, account)
            need(row['is_active'] is False, 'Disable the exact owned account before revoking its sessions')
            now = session_time(connection)
            return connection.execute(schema.sessions.update().where(schema.sessions.c.user_id == account['user_id'],
                schema.sessions.c.revoked_at.is_(None)).values(revoked_at=self.sa.func.greatest(now, schema.sessions.c.created_at))).rowcount


def run(args):
    from src.contracts import SnapshotRef
    from src.access.repository import PolicyRepository
    from src.auth.accounts import PasswordHasher, create_account, disable_account
    snapshot = SnapshotRef.model_validate(read_json(args.snapshot, private=False, limit=8192))
    need(snapshot.document_count == COUNT, 'Expected the accepted full corpus')
    with exclusive(args.state):
        operator = Operator(args.project, snapshot)
        try:
            fresh = not os.path.lexists(args.state)
            if fresh:
                need(args.command == 'seed' and isinstance(args.reviewed_by, str)
                     and bool(args.reviewed_by.strip()), 'Fresh seed requires explicit review')
                state = make_state(args.project, snapshot, args.reviewed_by.strip(), choose_documents(operator.documents()))
                write_state(args.state, state, first=True)  # All credentials/manifests before any DB mutation.
            else:
                state = read_json(args.state)
            manifests = validate_state(state, args.project, snapshot)
            need(args.reviewed_by is None or args.reviewed_by.strip() == state['reviewed_by'], 'Original review identity differs')
            operator.verify_documents(state)
            repository = PolicyRepository(url=operator.url)
            changed_grants = disabled = sessions_revoked = 0
            if args.command == 'seed':
                need(state['status'] in {'reserved', 'seeding', 'ready'} and
                     not any(value['revoke_requested'] for value in state['ledger'].values()), 'Revoked fixture cannot be reseeded')
                state['status'] = 'seeding'; write_state(args.state, state)
                for name in NAMES:
                    account = state['accounts'][name]; row = operator.account(account)
                    audit = operator.audit(manifests[name])
                    need('revoke' not in audit, 'Revoked manifest cannot be reapplied')
                    if row is None:
                        need(not state['ledger'][name]['account_created'] and not audit, 'Previously created human is missing')
                        create_account(account['login'], account['password'], user_id=account['user_id'],
                            display_name=account['display_name'], role=account['role'], department=account['department'], url=operator.url)
                    else:
                        need(row['is_active'] and PasswordHasher().verify(row['password_hash'], account['password']),
                             'Previously seeded human is disabled or credentials changed')
                    state['ledger'][name]['account_created'] = True
                    state['ledger'][name]['apply_requested'] = True; write_state(args.state, state)
                    grants, groups, _ = operator.subject(account)
                    need(groups == 0 and all(value['policy_change_id'] == audit.get('apply') and value['origin'] == 'synthetic'
                         and value['snapshot_id'] == snapshot.snapshot_id and value['doc_id'] in account['allowed_doc_ids'] for value in grants),
                         'Owned fixture received outside policy')
                    result = repository.apply(manifests[name]); changed_grants += result.changed_grants
                    audit = operator.audit(manifests[name]); grants, groups, _ = operator.subject(account)
                    need(groups == 0 and len(grants) == len(account['allowed_doc_ids'])
                         and {value['doc_id'] for value in grants} == set(account['allowed_doc_ids'])
                         and all(value['policy_change_id'] == audit['apply'] for value in grants), 'Explicit scope is incomplete')
                    state['ledger'][name]['apply_change_id'] = audit['apply']; write_state(args.state, state)
                state['status'] = 'ready'
            elif args.command in {'revoke-intern', 'cleanup'}:
                if args.command == 'revoke-intern':
                    need(state['status'] in {'ready', 'intern-revoked'}, 'Intern revocation requires seeded accounts')
                    targets = ('intern',)
                else:
                    state['status'] = 'cleaning'; targets = NAMES
                for name in targets:
                    account = state['accounts'][name]; row = operator.account(account)
                    audit = operator.audit(manifests[name])
                    state['ledger'][name]['revoke_requested'] = True; write_state(args.state, state)
                    if args.command == 'cleanup' and row is not None:
                        if row['is_active']:
                            disable_account(account['login'], url=operator.url); disabled += 1
                        sessions_revoked += operator.revoke_sessions(account)
                    if 'apply' in audit:
                        changed_grants += repository.revoke(manifests[name]).changed_grants
                        audit = operator.audit(manifests[name]); state['ledger'][name]['revoke_change_id'] = audit['revoke']
                    elif args.command == 'revoke-intern':
                        raise SecurityOperatorError('Intern manifest has never been applied')
                    write_state(args.state, state)
                state['status'] = 'intern-revoked' if args.command == 'revoke-intern' else 'cleaned'
            operator.guard()
            accounts = {}
            for name in NAMES:
                account = state['accounts'][name]; row = operator.account(account); audit = operator.audit(manifests[name])
                grants, groups, sessions = operator.subject(account)
                owned = sum('apply' in audit and value['policy_change_id'] == audit['apply'] for value in grants)
                accounts[name] = {'present': row is not None, 'active': bool(row and row['is_active']),
                    'owned_grants': owned, 'subject_grants': len(grants), 'group_memberships': groups,
                    'unrevoked_session_rows': sessions, 'manifest_applied': 'apply' in audit, 'manifest_revoked': 'revoke' in audit}
                if state['status'] == 'cleaned':
                    need(not accounts[name]['active'] and owned == 0 and sessions == 0, 'Exact owned cleanup is incomplete')
                if state['status'] == 'intern-revoked' and name == 'intern':
                    need('revoke' in audit and owned == len(grants) == groups == 0,
                         'Revoked intern still has policy outside the exact fixture')
            if args.command != 'inspect':
                write_state(args.state, state)
            return {'passed': True, 'operation': args.command, 'status': state['status'], 'synthetic': True,
                'document_count': COUNT, 'fixture_document_count': 3, 'accounts': accounts,
                'changed_owned_grants': changed_grants, 'disabled_owned_accounts': disabled,
                'revoked_owned_sessions': sessions_revoked, 'paid_model_calls': 0}
        finally:
            operator.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seed', 'inspect', 'revoke-intern', 'cleanup'))
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=Path('/state/snapshot.json'))
    parser.add_argument('--project', required=True)
    parser.add_argument('--reviewed-by')
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        # No SQL/DSN, credentials, document IDs/titles, or exception repr on stdout.
        print(json.dumps({'passed': False, 'operation': args.command, 'error_type': type(error).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

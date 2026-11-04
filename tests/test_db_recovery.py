"""Restore quarantine refuses ordinary databases and preserves policy/data.

Root alone executes the opt-in PostgreSQL cases. Their helper creates and drops
only its own fresh rag_restore_UUID database through the already guarded
dedicated test-admin server; it never renames or modifies the base database.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
import sqlalchemy as sa
from pydantic import SecretStr

from src.db import recovery, schema
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import engine_for, pg_urls, seed_catalog


TARGET = 'rag_restore_unit'
NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


class Result:
    def __init__(self, value=None, count=0):
        self.value, self.rowcount = value, count

    def mappings(self): return self
    def one_or_none(self): return self.value
    def scalar_one_or_none(self): return self.value
    def scalar_one(self): return self.value
    def scalars(self): return self
    def all(self): return self.value


class FakeEngine:
    def __init__(self, *, row=None, authenticated='rag_migrator', owner='rag_migrator',
                 heads=('0004_jira_feed',), fail_sessions=False):
        self.row = row or dict(database_name=TARGET, current_role='rag_migrator',
            login_role='rag_migrator', rolsuper=False, rolbypassrls=False,
            rolcreaterole=False, rolcreatedb=False, rolreplication=False)
        self.connection = SimpleNamespace(driver_connection=SimpleNamespace(info=SimpleNamespace(user=authenticated)))
        self.owner, self.heads, self.fail_sessions = owner, list(heads), fail_sessions
        self.updates, self.committed, self.disposed = [], False, False

    def execute(self, statement):
        if getattr(statement, 'is_update', False):
            self.updates.append(statement.table.name)
            if self.fail_sessions and statement.table.name == 'sessions':
                raise sa.exc.SQLAlchemyError('synthetic private database marker')
            return Result(count=2 if statement.table.name == 'users' else 3)
        sql = str(statement)
        if 'FROM pg_roles' in sql: return Result(self.row)
        if 'pg_get_userbyid' in sql: return Result(self.owner)
        if 'version_num' in sql: return Result(self.heads)
        if 'clock_timestamp' in sql: return Result(NOW)
        raise AssertionError('Unexpected recovery SQL')

    @contextmanager
    def begin(self):
        yield self
        self.committed = True

    def dispose(self): self.disposed = True


@pytest.mark.parametrize('name', ['rag', 'rag_test_base', 'production', 'rag_restore_',
    'rag_restore_x;DELETE', 'rag_restore_Ü', 'RAG_RESTORE_x'])
def test_ordinary_or_ambiguous_target_is_refused_before_connect(name, monkeypatch):
    monkeypatch.setattr(recovery, '_engine', lambda *a: pytest.fail('Unexpected connection attempt'))
    with pytest.raises(recovery.RecoveryRefused):
        recovery.quarantine_restored_database(expected_database=name)


def test_configured_url_must_match_explicit_restore_name_before_connect(monkeypatch):
    monkeypatch.setattr(sa, 'create_engine', lambda *a, **kw: pytest.fail('Unexpected connection attempt'))
    value = sa.URL.create('postgresql+psycopg', username='synthetic', password='private-marker',
        host='localhost', database='rag_test_base')
    with pytest.raises(recovery.RecoveryRefused) as failed:
        recovery._engine(value, TARGET)
    assert 'private-marker' not in str(failed.value)


@pytest.mark.parametrize('change', ['database', 'current', 'session', 'authenticated',
    'superuser', 'bypass', 'owner', 'missing-head', 'wrong-head', 'multiple-heads'])
def test_identity_owner_and_head_refusals_precede_all_mutations(change, monkeypatch):
    engine = FakeEngine()
    if change == 'database': engine.row['database_name'] = 'rag'
    elif change == 'current': engine.row['current_role'] = 'rag_runtime'
    elif change == 'session': engine.row['login_role'] = 'postgres'
    elif change == 'authenticated': engine.connection.driver_connection.info.user = 'postgres'
    elif change == 'superuser': engine.row['rolsuper'] = True
    elif change == 'bypass': engine.row['rolbypassrls'] = True
    elif change == 'owner': engine.owner = 'postgres'
    elif change == 'missing-head': engine.heads = []
    elif change == 'wrong-head': engine.heads = ['0003_document_policy']
    else: engine.heads.append('another_head')
    monkeypatch.setattr(recovery, '_engine', lambda *a: engine)
    with pytest.raises(recovery.RecoveryRefused):
        recovery.quarantine_restored_database(expected_database=TARGET)
    assert engine.updates == [] and not engine.committed and engine.disposed


def test_success_updates_only_account_lifecycle_and_uses_one_transaction(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(recovery, '_engine', lambda *a: engine)
    result = recovery.quarantine_restored_database(expected_database=TARGET)
    assert result.model_dump() == {'accounts_disabled': 2, 'sessions_revoked': 3}
    assert engine.updates == ['users', 'sessions'] and engine.committed and engine.disposed


def test_error_in_second_update_rolls_back_and_is_sanitized(monkeypatch):
    engine = FakeEngine(fail_sessions=True)
    monkeypatch.setattr(recovery, '_engine', lambda *a: engine)
    with pytest.raises(recovery.RecoveryRefused) as failed:
        recovery.quarantine_restored_database(expected_database=TARGET)
    assert not engine.committed and engine.disposed
    assert 'private database marker' not in str(failed.value)


def test_cli_prints_only_counters_or_generic_refusal(monkeypatch, capsys):
    monkeypatch.setattr(recovery, 'quarantine_restored_database',
        lambda **kw: recovery.QuarantineResult(accounts_disabled=2, sessions_revoked=3))
    assert recovery.main(['--expected-database', TARGET]) == 0
    assert json.loads(capsys.readouterr().out) == {'accounts_disabled': 2, 'sessions_revoked': 3}

    def failed(**kw): raise RuntimeError('synthetic private credential marker')
    monkeypatch.setattr(recovery, 'quarantine_restored_database', failed)
    assert recovery.main(['--expected-database', TARGET]) == 1
    captured = capsys.readouterr()
    assert captured.out == 'Restore quarantine refused.\n' and captured.err == ''


def test_cli_help_needs_no_database_driver_or_connection():
    program = '''
import importlib.abc, runpy, sys
class NoDatabase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sqlalchemy','psycopg','alembic'}:
            raise AssertionError('Unexpected database import')
sys.meta_path.insert(0,NoDatabase())
sys.argv=['recovery','--help']
runpy.run_module('src.db.recovery',run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--expected-database' in result.stdout and '--allow-live' not in result.stdout


@contextmanager
def restored_database(urls):
    """Admin URL is guarded rag_test_*; target is a newly generated exact name."""
    name = 'rag_restore_' + uuid4().hex
    assert name.startswith('rag_restore_') and name != urls.admin.database
    admin = engine_for(urls.admin)
    created = False
    quoted = admin.dialect.identifier_preparer.quote(name)
    try:
        with admin.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
            connection.exec_driver_sql('CREATE DATABASE ' + quoted + ' OWNER rag_migrator')
            created = True
            connection.exec_driver_sql('REVOKE ALL ON DATABASE ' + quoted + ' FROM PUBLIC')
            connection.exec_driver_sql('GRANT CONNECT ON DATABASE ' + quoted + ' TO rag_runtime')
        yield SimpleNamespace(name=name, migration=urls.migration.set(database=name),
            runtime=urls.runtime.set(database=name), admin=urls.admin.set(database=name))
    finally:
        if created:
            assert name.startswith('rag_restore_') and quoted == admin.dialect.identifier_preparer.quote(name)
            with admin.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
                connection.exec_driver_sql('DROP DATABASE ' + quoted + ' WITH (FORCE)')
        admin.dispose()


def digest_tables(connection, *, omit=()):
    values = {}
    for name in sorted(sa.inspect(connection).get_table_names(schema='rag')):
        if name in omit: continue
        quoted = connection.dialect.identifier_preparer.quote(name)
        payload = connection.exec_driver_sql("SELECT COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text),'[]'::jsonb)::text FROM rag."
            + quoted + ' AS t').scalar_one()
        values[name] = hashlib.sha256(payload.encode()).hexdigest()
    return values


@pytest.fixture
def recovery_db(pg_urls):
    from src.auth.accounts import create_account
    from src.auth.sessions import AuthService
    from src.auth.settings import AuthSettings
    from src.db.migrate import upgrade_database

    class Hasher:
        def hash(self, value): return 'fixture-only$' + hashlib.sha256(value.encode()).hexdigest()
        def verify(self, encoded, value): return encoded == self.hash(value)

    with restored_database(pg_urls) as target:
        upgrade_database(target.migration)
        owner = sa.create_engine(target.migration, poolclass=sa.pool.NullPool, hide_parameters=True)
        runtime = Database(DatabaseSettings(dsn=SecretStr(target.runtime.render_as_string(hide_password=False))))
        try:
            with owner.begin() as connection:
                data = seed_catalog(connection, ready=True)
            hasher, password = Hasher(), 'Synthetic restored reader password'
            reader = create_account('restored-reader', password, display_name='Restored reader',
                url=target.migration, hasher=hasher)
            inactive = create_account('already-disabled', password, display_name='Inactive reader',
                url=target.migration, hasher=hasher)
            service = AuthService(runtime, AuthSettings(enabled=True), hasher=hasher)
            grant = service.login('restored-reader', password, client_ip='127.0.0.1')
            assert service.authenticate(grant.token, touch=False).identity.user_id == reader.user_id
            with owner.begin() as connection:
                now = connection.execute(sa.text('SELECT clock_timestamp()')).scalar_one()
                connection.execute(schema.users.update().where(schema.users.c.user_id == inactive.user_id)
                    .values(is_active=False, session_version=7))
                for label, created, expires, revoked, user_id, version in (
                    ('future', now + timedelta(seconds=120), now + timedelta(hours=1), None, reader.user_id, 0),
                    ('expired', now - timedelta(days=2), now - timedelta(days=1), None, reader.user_id, 0),
                    ('inactive', now - timedelta(hours=1), now + timedelta(hours=1), None, inactive.user_id, 7),
                    ('revoked', now - timedelta(days=2), now + timedelta(days=1), now - timedelta(hours=1), reader.user_id, 0)):
                    connection.execute(schema.sessions.insert().values(session_id='fixture-' + label,
                        user_id=user_id, session_version=version, token_hash=hashlib.sha256(label.encode()).hexdigest(),
                        csrf_hash=hashlib.sha256(('csrf-' + label).encode()).hexdigest(),
                        created_at=created, expires_at=expires, revoked_at=revoked))
                connection.execute(schema.groups.insert().values(group_id='restore-group', label='Restored group'))
                connection.execute(schema.group_memberships.insert().values(user_id=reader.user_id,
                    group_id='restore-group', origin='synthetic'))
                connection.execute(schema.document_grants.insert().values(snapshot_id=data['snapshot_id'],
                    doc_id=data['doc_id'], group_id='restore-group', origin='synthetic'))
                connection.execute(schema.query_runs.insert().values(run_id='restore-run', owner_user_id=reader.user_id,
                    snapshot_id=data['snapshot_id'], scope_id='restored-historical-scope', policy_revision=0,
                    session_version=0, config_sha256='c' * 64, mode='baseline', question='sha256:' + 'd' * 64))
                connection.execute(schema.run_steps.insert().values(run_id='restore-run', step_number=1,
                    payload={'tool': 'text_search', 'evidence_ids': [data['doc_id'] + ':0:8']}))
                connection.execute(schema.aggregate_results.insert().values(artifact_id='restore-aggregate',
                    citation_id='aggregate:restore-aggregate', run_id='restore-run', owner_user_id=reader.user_id,
                    snapshot_id=data['snapshot_id'], scope_id='restored-historical-scope', policy_revision=0,
                    session_version=0, template_id='synthetic-recovery-fixture', template_version='1', count=0,
                    lineage_sha256='e' * 64, coverage='synthetic_fixture'))
            yield SimpleNamespace(target=target, owner=owner, runtime=runtime, auth=service,
                token=grant.token, reader=reader, inactive=inactive, data=data)
        finally:
            runtime.close()
            owner.dispose()


def test_real_postgres_quarantine_revokes_old_tokens_and_preserves_every_other_table(recovery_db):
    from src.auth.sessions import AuthError
    fixture = recovery_db
    with fixture.owner.connect() as connection:
        preserved = digest_tables(connection, omit=('users', 'sessions'))
        users = {row['user_id']: dict(row) for row in connection.execute(sa.select(schema.users)).mappings()}
        sessions = {row['session_id']: dict(row) for row in connection.execute(sa.select(schema.sessions)).mappings()}
    result = recovery.quarantine_restored_database(expected_database=fixture.target.name, url=fixture.target.migration)
    assert result.accounts_disabled == sum(value['is_active'] for value in users.values())
    assert result.sessions_revoked == sum(value['revoked_at'] is None for value in sessions.values())
    with pytest.raises(AuthError) as failed:
        fixture.auth.authenticate(fixture.token, touch=False)
    assert failed.value.status == 401
    with fixture.owner.connect() as connection:
        assert digest_tables(connection, omit=('users', 'sessions')) == preserved
        for row in connection.execute(sa.select(schema.users)).mappings():
            original = users[row['user_id']]
            assert row['is_active'] is False
            assert row['session_version'] == original['session_version'] + int(original['is_active'])
            assert {k: v for k, v in row.items() if k not in ('is_active', 'session_version')} == {
                k: v for k, v in original.items() if k not in ('is_active', 'session_version')}
        for row in connection.execute(sa.select(schema.sessions)).mappings():
            original = sessions[row['session_id']]
            assert row['revoked_at'] is not None and row['revoked_at'] >= row['created_at']
            if original['revoked_at'] is not None: assert row['revoked_at'] == original['revoked_at']
            assert {k: v for k, v in row.items() if k != 'revoked_at'} == {k: v for k, v in original.items() if k != 'revoked_at'}
        assert connection.execute(sa.select(schema.sessions.c.revoked_at == schema.sessions.c.created_at)
            .where(schema.sessions.c.session_id == 'fixture-future')).scalar_one()
        completed = digest_tables(connection)
    second = recovery.quarantine_restored_database(expected_database=fixture.target.name, url=fixture.target.migration)
    assert second.model_dump() == {'accounts_disabled': 0, 'sessions_revoked': 0}
    with fixture.owner.connect() as connection:
        assert digest_tables(connection) == completed


@pytest.mark.parametrize('refusal', ['ordinary-name', 'mismatch', 'runtime-role', 'masked-admin', 'wrong-head'])
def test_real_postgres_refusals_leave_all_restored_rows_unchanged(recovery_db, refusal):
    fixture = recovery_db
    expected, url = fixture.target.name, fixture.target.migration
    if refusal == 'ordinary-name': expected = 'rag'
    elif refusal == 'mismatch': expected = 'rag_restore_another'
    elif refusal == 'runtime-role': url = fixture.target.runtime
    elif refusal == 'masked-admin':
        url = fixture.target.admin.update_query_dict({'options': '-c session_authorization=rag_migrator'})
    else:
        with fixture.owner.begin() as connection:
            connection.execute(sa.text("UPDATE rag.alembic_version SET version_num='incompatible_restore'"))
    with fixture.owner.connect() as connection:
        before = digest_tables(connection)
    with pytest.raises(recovery.RecoveryRefused):
        recovery.quarantine_restored_database(expected_database=expected, url=url)
    with fixture.owner.connect() as connection:
        assert digest_tables(connection) == before


def test_real_postgres_second_update_failure_rolls_back_account_disabling(recovery_db):
    fixture = recovery_db
    with fixture.owner.begin() as connection:
        connection.exec_driver_sql('''CREATE FUNCTION rag.reject_recovery_fixture_session() RETURNS trigger
LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Synthetic rollback fixture'; END; $$''')
        connection.exec_driver_sql('''CREATE TRIGGER reject_recovery_fixture_session
BEFORE UPDATE OF revoked_at ON rag.sessions FOR EACH ROW
EXECUTE FUNCTION rag.reject_recovery_fixture_session()''')
        before = digest_tables(connection)
    with pytest.raises(recovery.RecoveryRefused):
        recovery.quarantine_restored_database(expected_database=fixture.target.name, url=fixture.target.migration)
    with fixture.owner.connect() as connection:
        assert digest_tables(connection) == before

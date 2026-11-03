"""Real PostgreSQL pool/RLS checks; opt in only to dedicated rag_test_* DBs."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Database checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Database checks require the psycopg driver")

from src.access.contracts import LegacySetScope, RequestContext, ResolvedIdentity
from src.contracts import ScopeBinding, SnapshotRef
from src.db.runtime import Database, DatabaseConfigurationError, DatabaseRoleError, DatabaseSettings
from tests.db_support import admin_engine, db_engine, pg_urls, seed_catalog, url_string


AUTHORITY_KEYS = ("user_id", "session_id", "session_version", "policy_revision", "snapshot_id", "scope_id")


def identity_context(label="alice"):
    snapshot = SnapshotRef(snapshot_id="snapshot-" + label, corpus_sha256="a" * 64,
                           catalog_sha256="b" * 64, document_count=1)
    binding = ScopeBinding(scope_id="scope-" + label, user_id=label, snapshot=snapshot,
                           policy_revision=7, session_version=3)
    identity = ResolvedIdentity(user_id=label, session_id="session-row-" + label,
        session_version=3, expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    return RequestContext(identity, LegacySetScope(binding=binding, allowed_doc_ids={"doc"}))


def authority(connection):
    return {key: connection.execute(sa.text("SELECT current_setting(:key, true)"),
            {"key": "rag." + key}).scalar_one() for key in AUTHORITY_KEYS}


@pytest.fixture
def runtime_database(pg_urls, db_engine):
    database = Database(DatabaseSettings(dsn=url_string(pg_urls.runtime), pool_size=1,
                                         pool_timeout_seconds=.1))
    try:
        yield database
    finally:
        database.close()


def test_runtime_context_is_transaction_local_and_neutral_reuse_has_no_identity(runtime_database):
    context = identity_context()
    with runtime_database.transaction(context) as connection:
        pid = connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one()
        assert authority(connection) == {
            "user_id": "alice", "session_id": "session-row-alice", "session_version": "3",
            "policy_revision": "7", "snapshot_id": "snapshot-alice", "scope_id": "scope-alice"}
        assert connection.connection.driver_connection.prepare_threshold is None
        assert connection.execute(sa.text("SHOW statement_timeout")).scalar_one() == "5s"
        assert connection.execute(sa.text("SHOW idle_in_transaction_session_timeout")).scalar_one() == "5s"
    with runtime_database.transaction() as connection:
        assert connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one() == pid
        assert all(value in (None, "") for value in authority(connection).values())
    with runtime_database.transaction(identity_context("bob")) as connection:
        assert authority(connection)["user_id"] == "bob"


@pytest.mark.parametrize("failure", ["python_error", "sql_error"])
def test_failed_transactions_rollback_and_do_not_poison_next_identity(runtime_database, failure):
    pid = None
    with pytest.raises((RuntimeError, sa.exc.DBAPIError)):
        with runtime_database.transaction(identity_context("alice")) as connection:
            pid = connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one()
            connection.execute(sa.text("SELECT set_config('rag.user_id', 'forged-session-value', false)"))
            if failure == "sql_error":
                connection.execute(sa.text("SELECT 1 / 0"))
            raise RuntimeError("injected application failure")
    with runtime_database.transaction(identity_context("bob")) as connection:
        assert connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one() == pid
        assert authority(connection)["user_id"] == "bob"
        assert connection.execute(sa.text("SELECT 1")).scalar_one() == 1


def test_discard_all_clears_session_level_authority_settings_and_advisory_locks(
        runtime_database, admin_engine):
    lock_id = uuid4().int % (2**31)
    with runtime_database.transaction(identity_context()) as connection:
        pid = connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one()
        connection.execute(sa.text("SELECT set_config('rag.user_id', 'session-poison', false)"))
        connection.execute(sa.text("SELECT set_config('rag.test_poison', 'must-not-survive', false)"))
        connection.execute(sa.text("SELECT set_config('statement_timeout', '0', false)"))
        connection.execute(sa.text("SELECT pg_advisory_lock(:key)"), {"key": lock_id})
    with admin_engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT count(*) FROM pg_locks WHERE pid=:pid AND locktype='advisory'"), {"pid": pid}).scalar_one() == 0
    with runtime_database.transaction() as connection:
        assert connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one() == pid
        assert all(value in (None, "") for value in authority(connection).values())
        assert connection.execute(sa.text("SELECT current_setting('rag.test_poison', true)")).scalar_one() in (None, "")
        assert connection.execute(sa.text("SHOW statement_timeout")).scalar_one() == "5s"


def test_pool_has_no_overflow_and_exhaustion_does_not_lose_the_first_transaction(runtime_database):
    assert isinstance(runtime_database.engine.pool, sa.pool.QueuePool)
    with runtime_database.transaction(identity_context("alice")) as first:
        with pytest.raises(sa.exc.TimeoutError):
            with runtime_database.transaction(identity_context("bob")):
                pytest.fail("A one-connection pool overflowed")
        assert authority(first)["user_id"] == "alice"
    with runtime_database.transaction(identity_context("bob")) as second:
        assert authority(second)["user_id"] == "bob"


@pytest.mark.parametrize("credential", ["migration", "admin"])
def test_runtime_rejects_owner_and_superuser_credentials_on_checkout(pg_urls, db_engine, credential):
    database = Database(DatabaseSettings(dsn=url_string(getattr(pg_urls, credential)), pool_size=1))
    try:
        with pytest.raises(DatabaseRoleError, match="unprivileged and nonowner"):
            with database.transaction():
                pytest.fail("Privileged credentials entered a runtime transaction")
    finally:
        database.close()


@pytest.mark.parametrize("credential", ["migration", "admin"])
@pytest.mark.parametrize("setting", ["role", "session_authorization"])
def test_privileged_login_cannot_hide_behind_startup_identity_options(pg_urls, db_engine, credential, setting):
    url = getattr(pg_urls, credential)
    masked = url.set(query=dict(url.query) | {"options": f"-c {setting}=rag_runtime"})
    database = Database(DatabaseSettings(dsn=url_string(masked), pool_size=1))
    try:
        try:
            with database.transaction():
                pytest.fail("A masked privileged login entered the runtime pool")
        except DatabaseRoleError:
            pass
        except sa.exc.OperationalError as error:
            # A non-superuser migrator can be rejected by PostgreSQL before the
            # checkout guard. psycopg can lose SQLSTATE during connection startup,
            # so constrain its FATAL reason in that case; ordinary connectivity
            # or password failures must not turn this regression green. The
            # unmasked migration login was already validated by db_engine.
            assert credential == "migration"
            if error.orig.sqlstate is None:
                action = "role" if setting == "role" else "session authorization"
                assert f'permission denied to set {action} "rag_runtime"' in str(error.orig)
            else:
                assert error.orig.sqlstate in {"42501", "55P02"}
    finally:
        database.close()


def test_pooled_runtime_is_rechecked_after_temporary_owner_membership(pg_urls, admin_engine, db_engine):
    assert pg_urls.runtime.username == "rag_runtime" and pg_urls.migration.username == "rag_migrator"
    database = Database(DatabaseSettings(dsn=url_string(pg_urls.runtime), pool_size=1))
    with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin:
        names = admin.execute(sa.text("SELECT datname FROM pg_database WHERE NOT datistemplate AND datname <> 'postgres'")).scalars().all()
        assert names and all(name.startswith("rag_test_") for name in names), "Role mutation requires the dedicated test-only server"
        admin.execute(sa.text("SELECT pg_advisory_lock(59800002)"))
        granted = False
        try:
            assert admin.execute(sa.text("SELECT pg_has_role('rag_runtime','rag_migrator','MEMBER')")).scalar_one() is False
            with database.transaction() as connection:
                connection.execute(sa.text("SELECT 1"))
            granted = True
            admin.execute(sa.text("GRANT rag_migrator TO rag_runtime"))
            with pytest.raises(DatabaseRoleError):
                with database.transaction():
                    pytest.fail("An existing pooled connection bypassed new owner membership")
        finally:
            try:
                if granted:
                    admin.execute(sa.text("REVOKE rag_migrator FROM rag_runtime"))
            finally:
                admin.execute(sa.text("SELECT pg_advisory_unlock(59800002)"))
                database.close()
        assert admin.execute(sa.text("SELECT pg_has_role('rag_runtime','rag_migrator','MEMBER')")).scalar_one() is False


@pytest.mark.parametrize("query_key", ["password", "sslpassword"])
def test_secret_dsn_query_values_are_rejected_before_an_engine_can_repr_them(query_key):
    secret = "query-secret-must-stay-private"
    settings = DatabaseSettings(dsn=f"postgresql://user@localhost/rag_test_redaction?{query_key}={secret}")
    with pytest.raises(DatabaseConfigurationError) as error:
        Database(settings)
    assert secret not in str(error.value)


def test_initial_rls_denies_protected_reads_even_with_forged_transaction_settings(
        runtime_database, db_engine):
    from src.db import schema
    with db_engine.begin() as connection:
        data = seed_catalog(connection)
        assert connection.execute(sa.select(sa.func.count()).select_from(
            schema.document_revisions).where(schema.document_revisions.c.snapshot_id == data["snapshot_id"])).scalar_one() == 1
    for context in (None, identity_context(data["user_id"])):
        with runtime_database.transaction(context) as connection:
            for name in schema.PROTECTED_TABLES:
                assert connection.execute(sa.text(f'SELECT count(*) FROM rag."{name}"')).scalar_one() == 0
            with pytest.raises(sa.exc.DBAPIError) as denied:
                with connection.begin_nested():
                    connection.execute(schema.document_revisions.insert().values(
                        snapshot_id=data["snapshot_id"], doc_id=data["doc_id"],
                        revision_id=data["revision_id"], title="Runbook", canonical_text=data["text"]))
            assert denied.value.orig.sqlstate == "42501"  # RLS, not FK/uniqueness validation.


def test_database_settings_redact_credentials_without_authenticating():
    secret = "must-never-appear-in-diagnostics"
    settings = DatabaseSettings(dsn="postgresql://user:" + secret + "@localhost/rag_test_redaction")
    assert secret not in repr(settings) and secret not in settings.model_dump_json()
    with pytest.raises(TypeError, match="server-resolved"):
        database = Database(settings)
        try:
            with database.transaction({"user_id": "forged"}):
                pytest.fail("Client mapping was accepted as identity")
        finally:
            database.close()

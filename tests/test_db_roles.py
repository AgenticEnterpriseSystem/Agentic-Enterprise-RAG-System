"""New-volume role bootstrap, including secret boundaries and opt-in real roles."""

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
INIT_SCRIPT = ROOT / "docker/postgres/init.sh"


def bootstrap_environment(tmp_path):
    """Synthetic credentials only; the fake psql captures SQL/argv, not env."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake = binaries / "psql"
    fake.write_text('#!/bin/sh\ncat > "$INIT_CAPTURE_SQL"\nprintf "%s\\n" "$@" > "$INIT_CAPTURE_ARGV"\n: > "$INIT_PSQL_CALLED"\n')
    fake.chmod(0o700)
    return {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
        "POSTGRES_USER": "bootstrap_admin", "POSTGRES_DB": "rag_test_roles",
        "RAG_DB_MIGRATOR_PASSWORD": "synthetic-migrator", "RAG_DB_RUNTIME_PASSWORD": "synthetic-runtime",
        "INIT_CAPTURE_SQL": str(tmp_path / "input.sql"), "INIT_CAPTURE_ARGV": str(tmp_path / "arguments.txt"),
        "INIT_PSQL_CALLED": str(tmp_path / "psql-called")}


@pytest.mark.parametrize("name", ["POSTGRES_USER", "POSTGRES_DB", "RAG_DB_MIGRATOR_PASSWORD", "RAG_DB_RUNTIME_PASSWORD"])
@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_missing_bootstrap_inputs_fail_before_psql(tmp_path, name, empty):
    environment = bootstrap_environment(tmp_path)
    if empty:
        environment[name] = ""
    else:
        environment.pop(name)
    result = subprocess.run(["bash", str(INIT_SCRIPT)], env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path / "psql-called").exists()
    assert "synthetic-migrator" not in result.stdout + result.stderr
    assert "synthetic-runtime" not in result.stdout + result.stderr


def test_secret_characters_never_become_shell_commands_argv_or_logged_sql(tmp_path):
    environment = bootstrap_environment(tmp_path)
    marker = tmp_path / "injection-was-executed"
    password = f"synthetic'\\;$(touch {marker})`touch {marker}`\nsecond-line$HOME"
    environment["RAG_DB_MIGRATOR_PASSWORD"] = password
    environment["RAG_DB_RUNTIME_PASSWORD"] = password + "-other"
    # Even explicitly requested shell tracing is disabled before any expansion.
    result = subprocess.run(["bash", "-x", str(INIT_SCRIPT)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0 and not marker.exists()
    argv = (tmp_path / "arguments.txt").read_text()
    sql = (tmp_path / "input.sql").read_text()
    assert password not in result.stdout + result.stderr + argv + sql
    assert "\\getenv migrator_password RAG_DB_MIGRATOR_PASSWORD" in sql
    assert ":'migrator_password'" in sql and ":'runtime_password'" in sql
    assert "PASSWORD %L" in sql and "DATABASE %I" in sql and "\\gexec" in sql
    assert "--no-psqlrc" in argv and "--no-password" in argv


def test_bootstrap_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(INIT_SCRIPT)], check=True, capture_output=True)


@pytest.fixture
def role_database_urls():
    """Keep shell-only checks usable even without optional DB dependencies."""
    names = ("RAG_TEST_ADMIN_URL", "RAG_TEST_MIGRATION_URL", "RAG_TEST_RUNTIME_URL")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Role integration requires the three explicit RAG_TEST_* URLs")
    from tests.db_support import DatabaseUrls, require_test_url
    urls = DatabaseUrls(*(require_test_url(os.environ[name]) for name in names))
    if len({(url.host, url.port, url.database) for url in (urls.admin, urls.migration, urls.runtime)}) != 1:
        pytest.fail("Role test URLs must identify one dedicated test database", pytrace=False)
    return urls


def test_real_roles_are_separate_least_privilege_logins_and_migrator_owns_database(role_database_urls):
    from tests.db_support import engine_for
    admin = engine_for(role_database_urls.admin)
    try:
        with admin.connect() as connection:
            rows = connection.exec_driver_sql("""
                SELECT rolname, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls
                FROM pg_catalog.pg_roles WHERE rolname IN ('rag_migrator','rag_runtime') ORDER BY rolname
            """).all()
            assert [row[0] for row in rows] == ["rag_migrator", "rag_runtime"]
            assert all(row[1] and not any(row[2:]) for row in rows)
            assert connection.exec_driver_sql("""
                SELECT NOT pg_has_role('rag_runtime','rag_migrator','MEMBER')
                   AND NOT pg_has_role('rag_migrator','rag_runtime','MEMBER')
            """).scalar_one()
            assert connection.exec_driver_sql("""
                SELECT r.rolname FROM pg_catalog.pg_database d
                JOIN pg_catalog.pg_roles r ON r.oid=d.datdba WHERE d.datname=current_database()
            """).scalar_one() == "rag_migrator"
            assert connection.exec_driver_sql("""
                SELECT NOT EXISTS (SELECT 1 FROM pg_catalog.pg_database d,
                    LATERAL aclexplode(d.datacl) acl
                    WHERE d.datname=current_database() AND acl.grantee=0)
            """).scalar_one()
    finally:
        admin.dispose()


def test_real_runtime_defaults_and_permissions_are_bounded(role_database_urls):
    from tests.db_support import engine_for
    runtime = engine_for(role_database_urls.runtime)
    migrator = engine_for(role_database_urls.migration)
    try:
        with runtime.connect() as connection:
            assert connection.exec_driver_sql("SELECT current_user").scalar_one() == "rag_runtime"
            assert connection.exec_driver_sql("SHOW search_path").scalar_one() == "pg_catalog, rag"
            assert connection.exec_driver_sql("SHOW statement_timeout").scalar_one() == "15s"
            assert connection.exec_driver_sql("SHOW idle_in_transaction_session_timeout").scalar_one() == "15s"
            assert connection.exec_driver_sql("""
                SELECT has_database_privilege(current_database(),'CONNECT')
                   AND NOT has_database_privilege(current_database(),'CREATE')
                   AND NOT has_database_privilege(current_database(),'TEMP')
                   AND NOT has_schema_privilege('public','CREATE')
            """).scalar_one()
        with migrator.connect() as connection:
            assert connection.exec_driver_sql("SELECT current_user").scalar_one() == "rag_migrator"
            assert connection.exec_driver_sql("SELECT has_database_privilege(current_database(),'CREATE')").scalar_one()
    finally:
        runtime.dispose()
        migrator.dispose()

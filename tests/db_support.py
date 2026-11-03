"""Opt-in helpers restricted to explicitly named disposable PostgreSQL databases.

No service is started and no default developer database is used. Connection URLs
come from the three RAG_TEST_* variables and are never included in diagnostics.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from uuid import uuid4

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Database checks require the optional DB test dependencies")


def require_test_url(value):
    """Fail closed before connecting when a URL targets an ordinary database."""
    try:
        url = sa.engine.make_url(value)
    except (ValueError, sa.exc.ArgumentError):
        raise ValueError("Invalid dedicated PostgreSQL test URL") from None
    if url.get_backend_name() != "postgresql" or not (url.database or "").startswith("rag_test_"):
        raise ValueError("Database tests require an explicitly named rag_test_ PostgreSQL database")
    return url.set(drivername="postgresql+psycopg")


@dataclass(frozen=True)
class DatabaseUrls:
    admin: object = field(repr=False)
    migration: object = field(repr=False)
    runtime: object = field(repr=False)


@pytest.fixture(scope="session")
def pg_urls():
    names = ("RAG_TEST_ADMIN_URL", "RAG_TEST_MIGRATION_URL", "RAG_TEST_RUNTIME_URL")
    values = [os.environ.get(name) for name in names]
    if not all(values):
        pytest.skip("Real PostgreSQL checks require RAG_TEST_ADMIN_URL, RAG_TEST_MIGRATION_URL and RAG_TEST_RUNTIME_URL")
    try:
        urls = DatabaseUrls(*(require_test_url(value) for value in values))
    except ValueError as error:
        pytest.fail(str(error), pytrace=False)
    targets = {(url.host, url.port, url.database) for url in (urls.admin, urls.migration, urls.runtime)}
    if len(targets) != 1:
        pytest.fail("All database-test URLs must identify the same dedicated test server/database", pytrace=False)
    return urls


def engine_for(url):
    return sa.create_engine(require_test_url(url), poolclass=sa.pool.NullPool,
                            connect_args={"prepare_threshold": None})


def url_string(url):
    """For library configuration only; never print the returned credential URL."""
    return require_test_url(url).render_as_string(hide_password=False)


def seed_catalog(connection, *, ready=False):
    """Insert unique synthetic owner-import rows, never real corpus content."""
    from src.db import schema
    suffix = uuid4().hex
    canonical = "Runbook\nEscalate after 8 minutes. Café 🧭."
    values = {"user_id": "user-" + suffix, "login": "login-" + suffix,
              "snapshot_id": "snapshot-" + suffix, "doc_id": "doc-" + suffix,
              "chunk_id": "chunk-" + suffix, "ticket_id": "ticket-" + suffix,
              "revision_id": hashlib.sha256(canonical.encode()).hexdigest(), "text": canonical}
    connection.execute(schema.users.insert().values(user_id=values["user_id"],
        login=values["login"], password_hash="synthetic-hash-not-a-login-credential", display_name="Synthetic user"))
    connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=values["snapshot_id"],
        corpus_sha256="a" * 64, document_count=1))
    connection.execute(schema.documents.insert().values(snapshot_id=values["snapshot_id"],
        doc_id=values["doc_id"], position=0, revision_id=values["revision_id"],
        source_type="jira", text_length=len(canonical)))
    connection.execute(schema.document_revisions.insert().values(snapshot_id=values["snapshot_id"],
        doc_id=values["doc_id"], revision_id=values["revision_id"], title="Runbook", canonical_text=canonical))
    connection.execute(schema.chunks.insert().values(snapshot_id=values["snapshot_id"],
        chunk_id=values["chunk_id"], doc_id=values["doc_id"], revision_id=values["revision_id"],
        start_char=0, end_char=len(canonical), chunker_id="synthetic-fixture"))
    connection.execute(schema.tickets.insert().values(snapshot_id=values["snapshot_id"],
        ticket_id=values["ticket_id"], doc_id=values["doc_id"], revision_id=values["revision_id"],
        coverage="synthetic_fixture"))
    if ready:
        connection.execute(schema.dataset_snapshots.update().where(
            schema.dataset_snapshots.c.snapshot_id == values["snapshot_id"]).values(
            state="ready", catalog_sha256="b" * 64))
    return values


@pytest.fixture
def admin_engine(pg_urls):
    engine = engine_for(pg_urls.admin)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_engine(pg_urls):
    from src.db.migrate import upgrade_database
    upgrade_database(url_string(pg_urls.migration))
    engine = engine_for(pg_urls.migration)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def fresh_db_engine(pg_urls):
    """Exercise the current initial revision in a genuinely fresh database."""
    from src.db.migrate import upgrade_database
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        engine = engine_for(fresh.migration)
        try:
            yield engine
        finally:
            engine.dispose()


@contextmanager
def disposable_database(urls):
    """Create/drop only our newly generated rag_test_* database, never the base.

    The admin connection itself is also restricted to the dedicated test target.
    DROP ... FORCE can terminate only connections to the random DB created here.
    """
    admin = engine_for(urls.admin)
    name = "rag_test_" + uuid4().hex
    owner = require_test_url(urls.migration).username
    created = False
    try:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            quote = connection.dialect.identifier_preparer.quote
            connection.exec_driver_sql(f"CREATE DATABASE {quote(name)} OWNER {quote(owner)}")
            created = True
        fresh = DatabaseUrls(*(require_test_url(url.set(database=name))
                               for url in (urls.admin, urls.migration, urls.runtime)))
        yield fresh
    finally:
        if created:
            with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                connection.exec_driver_sql(
                    f"DROP DATABASE {connection.dialect.identifier_preparer.quote(name)} WITH (FORCE)")
        admin.dispose()

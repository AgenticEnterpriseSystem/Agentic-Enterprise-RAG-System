"""A real small PostgreSQL snapshot restore, restricted to our disposable server."""

import os
import subprocess

import pytest
import sqlalchemy as sa

from src.db import schema
from src.db.migrate import current_revision, upgrade_database
from src.db.runtime import Database, DatabaseSettings
from tests.db_support import (
    disposable_database, engine_for, pg_urls, seed_catalog, url_string,
)


def test_snapshot_restores_data_revision_and_default_denial(pg_urls):
    container = os.environ.get("RAG_TEST_PG_CONTAINER")
    if not container:
        pytest.skip("Snapshot restore requires the explicitly labeled disposable Docker server")
    label = subprocess.run(["docker", "inspect", "--format",
        '{{index .Config.Labels "rag.task"}}', container], capture_output=True, text=True, check=True)
    # Both labels identify explicitly created disposable test servers. The
    # authoritative full-corpus DOC-01 service is deliberately excluded.
    assert label.stdout.strip() in {"DB-01", "TOOL-01"}, "Refusing a non-test database container"

    with disposable_database(pg_urls) as source, disposable_database(pg_urls) as target:
        upgrade_database(url_string(source.migration))
        owner = engine_for(source.migration)
        try:
            with owner.begin() as connection:
                values = seed_catalog(connection, ready=True)
        finally:
            owner.dispose()
        dump = subprocess.run(["docker", "exec", container, "pg_dump", "--no-password",
            "-U", "rag_migrator", "-d", source.migration.database, "--format=custom"],
            capture_output=True, check=True).stdout
        assert dump.startswith(b"PGDMP")
        subprocess.run(["docker", "exec", "-i", container, "pg_restore", "--no-password",
            "--exit-on-error", "-U", "rag_migrator", "-d", target.migration.database],
            input=dump, capture_output=True, check=True)
        assert current_revision(url_string(target.migration)) == "0004_jira_feed"
        upgrade_database(url_string(target.migration))
        owner = engine_for(target.migration)
        try:
            with owner.connect() as connection:
                restored = connection.execute(sa.select(schema.document_revisions.c.canonical_text)
                    .where(schema.document_revisions.c.doc_id == values["doc_id"])).scalar_one()
                assert restored == values["text"]
                assert connection.execute(sa.select(schema.dataset_snapshots.c.state)
                    .where(schema.dataset_snapshots.c.snapshot_id == values["snapshot_id"])).scalar_one() == "ready"
        finally:
            owner.dispose()
        runtime = Database(DatabaseSettings(dsn=url_string(target.runtime)))
        try:
            with runtime.transaction() as connection:
                assert connection.execute(sa.select(sa.func.count()).select_from(
                    schema.document_revisions)).scalar_one() == 0
        finally:
            runtime.close()

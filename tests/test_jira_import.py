"""Pinned Git/Parquet identity and independently published typed coverage."""

import hashlib
import json

import pytest
import sqlalchemy as sa

from src.data.structured import StructuredImportError, canonical, import_jira, prepare_jira_inputs
from src.db import schema
from tests.db_support import pg_urls, url_string
from tests.jira_support import REPOSITORY, commit, env, ready, source


def prepare(data, **changes):
    values = dict(repository=REPOSITORY, commit=data.inputs.commit, coverage="synthetic_fixture")
    values.update(changes)
    return prepare_jira_inputs(data.repo, data.corpus, data.snapshot, **values)


def test_actual_git_blob_canonical_uuid_path_revision_and_native_field_provenance(source):
    manifest = source.inputs.manifest
    assert len(source.inputs.rows_json) == 4
    assert manifest["expected_records"] == 4 and manifest["duplicate_native_keys"] == 1
    assert manifest["excluded_noncanonical_paths"] == ["jira/directory-is-not-project/noncanonical-alias.json"]
    assert manifest["field_coverage"]["status"] == {"verified": 3, "missing": 1, "invalid": 0}
    row = json.loads(source.inputs.rows_json[0])
    assert row["ticket_id"] == row["doc_id"]
    assert row["native_key"] == "SUP-100" and row["project_id"] == "customer-support"
    assert row["field_provenance"]["source"]["path"].startswith("generated_data/sources/jira/directory-is-not-project/")
    assert row["field_provenance"]["source"]["git_commit"] == source.inputs.commit
    assert row["field_provenance"]["source"]["raw_sha256"] == hashlib.sha256(
        (source.repo / row["field_provenance"]["source"]["path"]).read_bytes()).hexdigest()
    source.inputs.recheck()


@pytest.mark.parametrize("change", ["raw", "corpus", "remote", "commit", "symlink"])
def test_changed_actual_artifact_or_git_identity_rejects_before_import(source, change):
    if change == "raw":
        (source.raw_dir / "one.json").write_text("{}")
    elif change == "corpus":
        source.corpus.write_bytes(source.corpus.read_bytes() + b"changed")
    elif change == "remote":
        with pytest.raises(StructuredImportError):
            prepare(source, repository="https://example.invalid/different.git")
        return
    elif change == "commit":
        with pytest.raises(StructuredImportError):
            prepare(source, commit="a" * 40)
        return
    else:
        path = source.raw_dir / "one.json"
        target = source.repo.parent / "saved.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    with pytest.raises((StructuredImportError, OSError)):
        prepare(source)


def test_synthetic_git_cannot_be_mislabeled_as_verified_upstream(source):
    with pytest.raises(StructuredImportError):
        prepare(source, coverage="verified_records")


@pytest.mark.parametrize("value", ["bad\0status", ["In Progress"], "\ud800"])
def test_native_field_invalidity_is_recorded_without_inventing_a_value(source, value):
    path = source.raw_dir / "one.json"
    record = json.loads(path.read_text())
    record["status"] = value
    path.write_text(json.dumps(record))
    inputs = prepare(source, commit=commit(source.repo))
    row = next(json.loads(value) for value in inputs.rows_json if json.loads(value)["doc_id"] == "one")
    assert row["status"] is None
    assert next(field for field in row["field_provenance"]["fields"] if field["field"] == "status")["state"] == "invalid"
    assert inputs.manifest["field_coverage"]["status"] == {"verified": 2, "missing": 1, "invalid": 1}


@pytest.mark.parametrize("field,value", [("dataset_doc_uuid", "different"), ("summary", "Changed title"), ("description", "Changed prose")])
def test_valid_git_commit_still_must_match_canonical_uuid_and_exact_body(source, field, value):
    path = source.raw_dir / "one.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    revision = commit(source.repo)
    with pytest.raises(StructuredImportError):
        prepare(source, commit=revision)


def test_real_migration_import_restart_and_coverage_are_published_once(ready):
    data = ready
    report = import_jira(data.source.inputs, url=url_string(data.urls.migration), batch_size=1)
    assert report == data.report and report["state"] == "ready" and report["records"] == 4
    with data.owner.connect() as connection:
        feed = connection.execute(sa.select(schema.ticket_feeds)).mappings().one()
        assert feed["expected_records"] == feed["imported_records"] == 4
        assert feed["field_coverage"]["status"]["missing"] == 1
        assert feed["feed_id"] == hashlib.sha256(b"rag-jira-feed-v1\n" + canonical(feed["source_manifest"])).hexdigest()
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.tickets)).scalar_one() == 4


def test_new_feed_migration_matches_current_core_metadata(env):
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    with env.owner.connect() as connection:
        def include_object(obj, name, kind, reflected, compare_to):
            return name != "alembic_version" and (kind != "table" or obj.schema == "rag")
        migration = MigrationContext.configure(connection, opts={"include_schemas": True,
            "include_object": include_object, "compare_type": True, "compare_server_default": True})
        assert compare_metadata(migration, schema.metadata) == []


def test_import_rejects_a_different_catalog_binding_even_when_corpus_matches(env):
    snapshot = env.source.snapshot.model_copy(update={"catalog_sha256": "f" * 64})
    changed = prepare_jira_inputs(env.source.repo, env.source.corpus, snapshot, repository=REPOSITORY,
        commit=env.source.inputs.commit, coverage="synthetic_fixture")
    with pytest.raises(StructuredImportError):
        import_jira(changed, url=url_string(env.urls.migration))
    with env.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.ticket_feeds)).scalar_one() == 0


def test_interrupted_batches_stay_unavailable_and_resume_without_duplicate_records(env):
    inserts = 0

    def interrupt(connection, cursor, statement, parameters, context, executemany):
        nonlocal inserts
        if statement.startswith("INSERT INTO rag.tickets"):
            inserts += 1
            if inserts == 2:
                raise StructuredImportError("Simulated bounded import interruption")

    sa.event.listen(sa.engine.Engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(StructuredImportError):
            import_jira(env.source.inputs, url=url_string(env.urls.migration), batch_size=1)
    finally:
        sa.event.remove(sa.engine.Engine, "before_cursor_execute", interrupt)
    with env.owner.begin() as connection:
        feed = connection.execute(sa.select(schema.ticket_feeds)).mappings().one()
        assert feed["state"] == "importing" and feed["imported_records"] == 1
        with pytest.raises(sa.exc.DBAPIError):
            with connection.begin_nested():
                connection.execute(schema.ticket_feeds.update().values(state="ready"))
    from src.contracts import ToolRequest
    result = env.tool.execute(ToolRequest(source_type="jira", operation="count"),
        context=env.contexts["intern"], guard=env.resolver)
    assert result.coverage == "unknown" and result.count is None
    report = import_jira(env.source.inputs, url=url_string(env.urls.migration), batch_size=1)
    assert report["records"] == 4 and report["state"] == "ready"
    with env.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.tickets)).scalar_one() == 4


def test_ready_source_cannot_be_replaced_when_only_native_field_changes(ready):
    data = ready
    path = data.source.raw_dir / "one.json"
    record = json.loads(path.read_text())
    record["status"] = "New"
    path.write_text(json.dumps(record))
    revision = commit(data.source.repo)
    changed = prepare(data.source, commit=revision)
    with pytest.raises(StructuredImportError):
        import_jira(changed, url=url_string(data.urls.migration))
    with data.owner.connect() as connection:
        assert connection.execute(sa.select(schema.tickets.c.status).where(schema.tickets.c.doc_id == "one")).scalar_one() == "In Progress"


@pytest.mark.parametrize("mutation", ["feed", "record", "delete", "truncate"])
def test_ready_records_and_feed_are_immutable_in_postgresql(ready, mutation):
    with ready.owner.begin() as connection:
        with pytest.raises(sa.exc.DBAPIError):
            with connection.begin_nested():
                if mutation == "feed":
                    connection.execute(schema.ticket_feeds.update().values(expected_records=5))
                elif mutation == "record":
                    connection.execute(schema.tickets.update().values(status="Changed"))
                elif mutation == "delete":
                    connection.execute(schema.tickets.delete())
                else:
                    connection.exec_driver_sql("TRUNCATE rag.tickets")


def test_runtime_cannot_import_or_rewrite_verified_records(ready):
    with ready.database.transaction() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.tickets)).scalar_one() == 0
        with pytest.raises(sa.exc.DBAPIError) as error:
            with connection.begin_nested():
                connection.execute(schema.ticket_feeds.update().values(state="importing"))
        assert error.value.orig.sqlstate == "42501"
        for column in ("source_manifest", "field_coverage"):
            with pytest.raises(sa.exc.DBAPIError) as error:
                with connection.begin_nested():
                    connection.execute(sa.select(schema.ticket_feeds.c[column]))
            assert error.value.orig.sqlstate == "42501"

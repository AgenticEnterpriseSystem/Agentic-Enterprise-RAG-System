"""Real tiny-file/PostgreSQL import checks, never a full-corpus acceptance run."""

import hashlib
import importlib
import json
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sa = pytest.importorskip("sqlalchemy", reason="Import checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Import checks require the psycopg driver")
pytest.importorskip("alembic", reason="Import checks require Alembic")

from src.data.import_catalog import ImportIntegrityError, import_catalog, verify_import
from src.db import schema
from src.db.migrate import upgrade_database
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
from tests.db_support import disposable_database, engine_for, pg_urls, url_string


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def tiny_artifacts(tmp_path):
    """Pinned-by-content artifacts with duplicate spans and exact Unicode text."""
    corpus = tmp_path / "documents.parquet"
    directory = tmp_path / "dense"
    directory.mkdir()
    rows = [
        {"doc_id": "doc-é", "source_type": "jira", "title": "  Café 🧭  ",
         "content": "naïve🙂 e\u0301\nline\r\nspaces  " * 8, "path": "jira/a.json"},
        {"doc_id": "doc-b", "source_type": "slack", "title": "",
         "content": "Second\nunknown date.  ", "path": "slack/b.json"},
        {"doc_id": "doc-c", "source_type": "jira", "title": "Third",
         "content": "Exact third text.", "path": "jira/c.json"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), corpus, row_group_size=1)
    texts = tuple(f"{row['title']}\n{row['content']}" if row["title"] else row["content"] for row in rows)
    entries = tuple(CatalogEntry(doc_id=row["doc_id"], revision_id=canonical_revision_id(text),
        position=position, source_type=row["source_type"], text_length=len(text))
        for position, (row, text) in enumerate(zip(rows, texts)))
    records = (((0, 0, 20), (1, 0, len(texts[1]))),
               ((0, 0, 20), (0, 20, 40), (2, 0, len(texts[2]))))
    manifest = {"version": 1, "complete": True, "documents": len(rows), "shards": 2,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "model": {"name": "tiny-offline-fixture", "revision": "b" * 40,
                  "normalize": True}}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "doc_ids.json", [row["doc_id"] for row in rows])
    for index, values in enumerate(records):
        np.save(directory / f"shard_{index:05d}.npy",
                np.ones((len(values), 2), dtype=np.float16), allow_pickle=False)
        pq.write_table(pa.table({name: pa.array([row[column] for row in values], type=pa.int64())
            for column, name in enumerate(("doc_index", "start_char", "end_char"))}),
            directory / f"shard_{index:05d}.parquet", row_group_size=1)
    return SimpleNamespace(corpus=corpus, directory=directory, rows=rows, texts=texts,
        entries=entries, records=records, manifest=manifest,
        catalog_sha256=catalog_fingerprint(entries, expected_count=len(rows)))


@pytest.fixture
def import_db(pg_urls):
    """A new migrated DB for each test; no policy or data touches the base DB."""
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        engine = engine_for(fresh.migration)
        try:
            yield SimpleNamespace(urls=fresh, engine=engine)
        finally:
            engine.dispose()


def run_import(data, database, **changes):
    return import_catalog(data.corpus, data.directory, snapshot_id="tiny-import",
                          url=url_string(database.urls.migration), batch_size=1, **changes)


def verify(data, database, **changes):
    return verify_import(data.corpus, data.directory, snapshot_id="tiny-import",
                         url=url_string(database.urls.migration), batch_size=1, **changes)


def imported_rows(database):
    with database.engine.connect() as connection:
        return {
            "snapshots": connection.execute(sa.select(schema.dataset_snapshots)).mappings().all(),
            "documents": connection.execute(sa.select(schema.documents).order_by(schema.documents.c.position)).mappings().all(),
            "bodies": connection.execute(sa.select(schema.document_revisions).order_by(schema.document_revisions.c.doc_id)).mappings().all(),
            "chunks": connection.execute(sa.select(schema.chunks).order_by(schema.chunks.c.chunk_id)).mappings().all(),
        }


def assert_not_ready(database):
    with database.engine.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.dataset_snapshots)
            .where(schema.dataset_snapshots.c.state == "ready")).scalar_one() == 0


def test_exact_catalog_text_offsets_and_duplicate_spans_are_imported_once(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    report = run_import(data, import_db)
    assert report.state == "ready" and report.document_count == 3
    assert report.passage_count == 5 and report.unique_chunk_count == 4
    assert report.binary_document_count == 0
    assert report.document_cursor == 3
    assert report.snapshot.corpus_sha256 == data.manifest["corpus_sha256"]
    assert report.snapshot.catalog_sha256 == data.catalog_sha256
    expected_hashes = {"corpus": hashlib.sha256(data.corpus.read_bytes()).hexdigest(),
        **{f"dense/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
           for path in data.directory.iterdir()}}
    assert dict(report.artifact_hashes) == expected_hashes
    assert len(report.source_set_sha256) == 64
    with pytest.raises(TypeError):
        report.artifact_hashes["corpus"] = "f" * 64
    rows = imported_rows(import_db)
    assert [row["doc_id"] for row in rows["documents"]] == [row["doc_id"] for row in data.rows]
    assert [row["position"] for row in rows["documents"]] == [0, 1, 2]
    assert all(row["timestamp"] is None and row["metadata_origin"] == "upstream"
               for row in rows["documents"])
    body_by_id = {row["doc_id"]: row for row in rows["bodies"]}
    for row, text, entry in zip(data.rows, data.texts, data.entries):
        assert body_by_id[row["doc_id"]]["title"] == row["title"]
        assert body_by_id[row["doc_id"]]["canonical_text"] == text
        assert body_by_id[row["doc_id"]]["revision_id"] == entry.revision_id
    actual = {(row["doc_id"], row["start_char"], row["end_char"]) for row in rows["chunks"]}
    expected = {(data.rows[index]["doc_id"], start, end) for shard in data.records for index, start, end in shard}
    assert actual == expected and len(rows["chunks"]) == 4
    # Plain import must not create grants, groups, tickets, or invented dates.
    with import_db.engine.connect() as connection:
        for table in (schema.document_grants, schema.groups, schema.tickets):
            assert connection.execute(sa.select(sa.func.count()).select_from(table)).scalar_one() == 0


def test_ready_repeat_is_idempotent_and_verify_only_performs_reads(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    first = run_import(data, import_db)
    before = imported_rows(import_db)
    second = run_import(data, import_db)
    assert second.snapshot == first.snapshot
    assert imported_rows(import_db) == before
    statements = []

    def record(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lstrip().upper())

    sa.event.listen(sa.engine.Engine, "before_cursor_execute", record)
    try:
        readonly_url = import_db.urls.migration.set(query={
            **import_db.urls.migration.query, "options": "-c default_transaction_read_only=on"})
        checked = verify_import(data.corpus, data.directory, snapshot_id="tiny-import",
            url=url_string(readonly_url), batch_size=1)
    finally:
        sa.event.remove(sa.engine.Engine, "before_cursor_execute", record)
    assert checked.snapshot == first.snapshot and checked.state == "ready"
    assert statements and not any(statement.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE"))
                                  for statement in statements)
    assert imported_rows(import_db) == before


def test_verify_unknown_snapshot_raises_typed_error_without_creating_records(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    with pytest.raises(ImportIntegrityError):
        verify(data, import_db)
    assert all(not rows for rows in imported_rows(import_db).values())
    with import_db.engine.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.catalog_imports)).scalar_one() == 0


def test_initial_manifest_rewrite_cannot_pair_old_declarations_with_new_file_hash(tmp_path, import_db, monkeypatch):
    data = tiny_artifacts(tmp_path)
    module = importlib.import_module("src.data.import_catalog")
    original_json = module._json
    mutated = False

    def rewrite_after_parse(path):
        nonlocal mutated
        parsed = original_json(path)
        if path.name == "manifest.json" and not mutated:
            mutated = True
            # The returned declarations are complete=True, while the artifact
            # subsequently hashed on disk says complete=False. No timing loop or
            # thread is needed to exercise precisely this capture boundary.
            write_json(path, {**parsed, "complete": False})
        return parsed

    monkeypatch.setattr(module, "_json", rewrite_after_parse)
    with pytest.raises(ImportIntegrityError):
        run_import(data, import_db)
    assert mutated
    assert all(not rows for rows in imported_rows(import_db).values())
    with import_db.engine.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.catalog_imports)).scalar_one() == 0


@pytest.mark.parametrize("phase", ["documents", "spans"])
def test_interrupted_checkpoint_is_durable_incomplete_and_resumes_without_duplicates(tmp_path, import_db, phase):
    data = tiny_artifacts(tmp_path)
    events = []

    def interrupt(event):
        events.append(dict(event))
        if event["phase"] == phase and (phase == "documents" or event["span_cursor"] > 0 or event["span_shard"] > 0):
            raise RuntimeError("deliberate interruption after durable checkpoint")

    with pytest.raises(RuntimeError, match="durable checkpoint"):
        run_import(data, import_db, on_checkpoint=interrupt)
    assert events and any(event["phase"] == phase for event in events)
    assert_not_ready(import_db)
    with import_db.engine.connect() as connection:
        checkpoint = connection.execute(sa.select(schema.catalog_imports)).mappings().one()
        for table in (schema.documents, schema.document_revisions):
            assert connection.execute(sa.select(sa.func.count()).select_from(table)).scalar_one() == checkpoint["document_cursor"]
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.chunks)).scalar_one() == checkpoint["unique_chunk_count"]
        if phase == "spans":
            assert checkpoint["passage_cursor"] > 0
    with pytest.raises(ImportIntegrityError):
        verify(data, import_db)
    report = run_import(data, import_db)
    assert report.state == "ready" and report.unique_chunk_count == 4
    rows = imported_rows(import_db)
    assert len(rows["documents"]) == len(rows["bodies"]) == 3 and len(rows["chunks"]) == 4


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("changed", ["corpus", "vector", "metadata", "manifest"])
def test_changed_sources_cannot_resume_or_overwrite_a_snapshot(tmp_path, import_db, ready, changed):
    data = tiny_artifacts(tmp_path)
    if ready:
        run_import(data, import_db)
    else:
        with pytest.raises(RuntimeError, match="interrupt"):
            run_import(data, import_db, on_checkpoint=lambda _event: (_ for _ in ()).throw(RuntimeError("interrupt")))
    before = imported_rows(import_db)
    if changed == "corpus":
        data.rows[0]["content"] += "changed"
        pq.write_table(pa.Table.from_pylist(data.rows), data.corpus)
        data.manifest["corpus_sha256"] = hashlib.sha256(data.corpus.read_bytes()).hexdigest()
        write_json(data.directory / "manifest.json", data.manifest)
    elif changed == "vector":
        np.save(data.directory / "shard_00000.npy", np.zeros((2, 2), dtype=np.float16), allow_pickle=False)
    elif changed == "metadata":
        path = data.directory / "shard_00000.parquet"
        pq.write_table(pa.table({"doc_index": [0, 1], "start_char": [1, 0], "end_char": [20, len(data.texts[1])]}), path)
    else:
        data.manifest["model"]["name"] = "different-source-model"
        write_json(data.directory / "manifest.json", data.manifest)
    with pytest.raises(ImportIntegrityError):
        run_import(data, import_db)
    with pytest.raises(ImportIntegrityError):
        verify(data, import_db)
    assert imported_rows(import_db) == before


def test_source_changed_during_import_cannot_publish_a_ready_snapshot(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    changed = False

    def mutate_after_commit(event):
        nonlocal changed
        if not changed and event["phase"] == "documents":
            changed = True
            # Same shape, dtype and row count; only the real artifact bytes change.
            np.save(data.directory / "shard_00001.npy", np.zeros((3, 2), dtype=np.float16), allow_pickle=False)

    with pytest.raises(ImportIntegrityError):
        run_import(data, import_db, on_checkpoint=mutate_after_commit)
    assert changed
    assert_not_ready(import_db)


@pytest.mark.parametrize("corruption", ["duplicate", "order", "count", "span", "unknown_doc", "incomplete", "extra_shard", "corpus_hash"])
def test_invalid_source_identity_or_spans_never_become_ready(tmp_path, import_db, corruption):
    data = tiny_artifacts(tmp_path)
    if corruption in {"duplicate", "order"}:
        ids = [row["doc_id"] for row in data.rows]
        ids = [ids[0], ids[0], ids[2]] if corruption == "duplicate" else list(reversed(ids))
        write_json(data.directory / "doc_ids.json", ids)
    elif corruption in {"span", "unknown_doc"}:
        pq.write_table(pa.table({"doc_index": [0 if corruption == "span" else 99, 1],
            "start_char": [0, 0], "end_char": [len(data.texts[0]) + 1 if corruption == "span" else 20, len(data.texts[1])]}),
            data.directory / "shard_00000.parquet")
    elif corruption == "extra_shard":
        np.save(data.directory / "shard_00009.npy", np.ones((1, 2), dtype=np.float16), allow_pickle=False)
    else:
        data.manifest[{"count": "documents", "incomplete": "complete", "corpus_hash": "corpus_sha256"}[corruption]] = {
            "count": 4, "incomplete": False, "corpus_hash": "f" * 64}[corruption]
        write_json(data.directory / "manifest.json", data.manifest)
    with pytest.raises(ImportIntegrityError):
        run_import(data, import_db)
    assert_not_ready(import_db)


def test_duplicate_canonical_document_is_rejected_even_when_dense_ids_repeat(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    data.rows[1]["doc_id"] = data.rows[0]["doc_id"]
    pq.write_table(pa.Table.from_pylist(data.rows), data.corpus)
    write_json(data.directory / "doc_ids.json", [row["doc_id"] for row in data.rows])
    data.manifest["corpus_sha256"] = hashlib.sha256(data.corpus.read_bytes()).hexdigest()
    write_json(data.directory / "manifest.json", data.manifest)
    with pytest.raises(ImportIntegrityError):
        run_import(data, import_db)
    assert_not_ready(import_db)


def test_explicit_timestamps_and_metadata_origin_are_preserved_without_inference(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    timestamp = datetime(2026, 11, 3, 14, 17, 9, 123456,
                         tzinfo=timezone(timedelta(hours=2)))
    for index, row in enumerate(data.rows):
        row.update(timestamp=timestamp if index == 0 else None, metadata_origin="synthetic")
    pq.write_table(pa.Table.from_pylist(data.rows), data.corpus)
    data.manifest["corpus_sha256"] = hashlib.sha256(data.corpus.read_bytes()).hexdigest()
    write_json(data.directory / "manifest.json", data.manifest)
    expected = tuple(entry.model_copy(update={"timestamp": timestamp if index == 0 else None,
        "metadata_origin": "synthetic"}) for index, entry in enumerate(data.entries))
    report = run_import(data, import_db)
    assert report.snapshot.catalog_sha256 == catalog_fingerprint(expected, expected_count=3)
    documents = imported_rows(import_db)["documents"]
    assert documents[0]["timestamp"] == timestamp
    assert [row["timestamp"] for row in documents[1:]] == [None, None]
    assert all(row["metadata_origin"] == "synthetic" for row in documents)


def test_canonical_nul_uses_exact_utf8_fallback_and_survives_ready_restart(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    data.rows[0]["content"] += "\x00literal \\000 and 🧭e\u0301\x00 tail"
    pq.write_table(pa.Table.from_pylist(data.rows), data.corpus)
    data.manifest["corpus_sha256"] = hashlib.sha256(data.corpus.read_bytes()).hexdigest()
    write_json(data.directory / "manifest.json", data.manifest)
    text = data.rows[0]["title"] + "\n" + data.rows[0]["content"]
    expected_entries = (data.entries[0].model_copy(update={"revision_id": canonical_revision_id(text),
        "text_length": len(text)}), *data.entries[1:])
    first = run_import(data, import_db)
    assert first.state == "ready"
    assert first.binary_document_count == 1
    assert first.snapshot.catalog_sha256 == catalog_fingerprint(expected_entries, expected_count=3)
    bodies = imported_rows(import_db)["bodies"]
    binary = next(row for row in bodies if row["doc_id"] == data.rows[0]["doc_id"])
    assert binary["canonical_text"] is None
    assert bytes(binary["canonical_utf8"]) == text.encode("utf-8")
    assert binary["revision_id"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert all(row["canonical_utf8"] is None for row in bodies if row["doc_id"] != binary["doc_id"])
    assert run_import(data, import_db).snapshot == first.snapshot
    assert verify(data, import_db).snapshot == first.snapshot
    assert imported_rows(import_db)["bodies"] == bodies


@pytest.mark.parametrize("name,value", [("batch_size", 0), ("batch_size", True),
    ("batch_size", 1.5), ("max_batch_bytes", 0), ("max_batch_bytes", True)])
def test_import_bounds_are_strict_before_work(tmp_path, import_db, name, value):
    data = tiny_artifacts(tmp_path)
    kwargs = {"batch_size": 1, name: value}
    with pytest.raises((ValueError, TypeError)):
        import_catalog(data.corpus, data.directory, snapshot_id="tiny-import",
                       url=url_string(import_db.urls.migration), **kwargs)
    assert_not_ready(import_db)


def test_document_checkpoints_are_bounded_and_oversized_single_row_is_reported(tmp_path, import_db):
    data = tiny_artifacts(tmp_path)
    events = []
    report = run_import(data, import_db, on_checkpoint=lambda event: events.append(dict(event)))
    cursors = [event["document_cursor"] for event in events if event["phase"] == "documents"]
    assert cursors == [1, 2, 3] and report.document_cursor == 3
    oversized = import_catalog(data.corpus, data.directory, snapshot_id="tiny-oversized",
        url=url_string(import_db.urls.migration), batch_size=1, max_batch_bytes=16)
    # The byte setting is a batching target; one indivisible row may exceed it.
    # That exception must be measured rather than claimed as a hard memory cap.
    assert oversized.state == "ready" and oversized.max_document_bytes > 16
    assert oversized.max_batch_bytes_observed >= oversized.max_document_bytes

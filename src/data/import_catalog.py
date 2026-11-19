"""Bounded, resumable canonical Parquet and dense-span import.

Operator credentials use one separate NullPool connection, never the runtime
pool. Every batch commits its rows and checkpoint together. Source hashes cover
the corpus and every declared dense artifact; paths are not persisted. The byte
budget counts canonical UTF-8 text, not RSS. Arrow prefetch is at most128 rows;
one oversized document is admitted unchanged and its size is reported. There is
no model, FAISS, lexical index or complete body-table load in this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.pool import NullPool

from src.contracts import SnapshotRef
from src.db import schema
from src.db.migrate import EXPECTED_HEAD
from src.db.runtime import database_url, normalize_url
from src.storage.contracts import CatalogEntry, validate_catalog_batch_size
from src.storage.fingerprints import catalog_fingerprint

DEFAULT_MAX_BATCH_BYTES = 8 * 1024 * 1024
_SPAN_COLUMNS = ("doc_index", "start_char", "end_char")
_ENTRY_FIELDS = tuple(CatalogEntry.model_fields)


class ImportIntegrityError(ValueError):
    """Source/checkpoint/canonical data disagree; no snapshot is published."""


@dataclass(frozen=True)
class ImportReport:
    snapshot: SnapshotRef | None
    state: str
    document_count: int
    passage_count: int
    unique_chunk_count: int
    source_set_sha256: str
    artifact_hashes: Mapping[str, str]
    document_cursor: int
    span_shard: int
    span_cursor: int
    max_document_bytes: int
    max_batch_bytes_observed: int
    binary_document_count: int = 0

    def as_dict(self):
        return {"snapshot": self.snapshot.model_dump(mode="json") if self.snapshot else None,
            **{name: getattr(self, name) for name in self.__dataclass_fields__
               if name not in {"snapshot", "artifact_hashes"}}, "artifact_hashes": dict(self.artifact_hashes)}


@dataclass(frozen=True)
class _Sources:
    corpus: Path
    dense: Path
    hashes: Mapping[str, str]
    signatures: Mapping[str, tuple]
    source_set_sha256: str
    doc_ids: tuple[str, ...]
    shard_rows: tuple[int, ...]
    chunker_id: str

    @property
    def documents(self):
        return len(self.doc_ids)

    @property
    def passages(self):
        return sum(self.shard_rows)

    def path(self, name):
        return self.corpus if name == "corpus" else self.dense / name.removeprefix("dense/")


@dataclass(frozen=True)
class _Document:
    entry: CatalogEntry
    title: str
    text: str
    byte_size: int


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImportIntegrityError("Duplicate artifact JSON fields")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ImportIntegrityError("Non-finite artifact JSON values")


def _json(path):
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError):
        raise ImportIntegrityError("Invalid artifact JSON") from None


def _signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ImportIntegrityError("Import artifacts must be ordinary files")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            with suppress(OSError):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def _unchanged(source, names=None):
    for name in source.signatures if names is None else names:
        if _signature(source.path(name)) != source.signatures[name]:
            raise ImportIntegrityError("Import source changed during the operation")


def _recheck_hashes(source):
    _unchanged(source)
    if any(_hash_file(source.path(name)) != value for name, value in source.hashes.items()):
        raise ImportIntegrityError("Import source bytes changed during the operation")
    _unchanged(source)


def _sources(corpus, dense_directory):
    corpus, dense = Path(corpus).resolve(), Path(dense_directory).resolve()
    manifest_path = dense / "manifest.json"
    manifest_signature = _signature(manifest_path)
    manifest = _json(manifest_path)
    if _signature(manifest_path) != manifest_signature:
        raise ImportIntegrityError("Dense manifest changed while reading its declarations")
    if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
            or manifest["version"] != 1 or manifest.get("complete") is not True
            or type(manifest.get("documents")) is not int or manifest["documents"] < 0
            or type(manifest.get("shards")) is not int or manifest["shards"] < 1):
        raise ImportIntegrityError("Dense manifest is incomplete or unsupported")
    files = {path.name for path in dense.iterdir() if path.name.startswith("shard_")}
    if len(files) != 2 * manifest["shards"]:
        raise ImportIntegrityError("Dense artifacts do not match the declared shard count")
    names = tuple(f"shard_{index:05d}" for index in range(manifest["shards"]))
    if files != {name + suffix for name in names for suffix in (".npy", ".parquet")}:
        raise ImportIntegrityError("Dense artifacts do not match the declared shards")
    roles = ("corpus", "dense/manifest.json", "dense/doc_ids.json") + tuple(
        "dense/" + name + suffix for name in names for suffix in (".npy", ".parquet"))
    paths = {name: corpus if name == "corpus" else dense / name.removeprefix("dense/") for name in roles}
    signatures = {name: _signature(path) for name, path in paths.items()}
    if signatures["dense/manifest.json"] != manifest_signature:
        raise ImportIntegrityError("Dense manifest changed before source hashing")
    hashes = {name: _hash_file(path) for name, path in paths.items()}
    if hashes["corpus"] != manifest.get("corpus_sha256"):
        raise ImportIntegrityError("Dense manifest corpus hash does not match the actual corpus")
    ids = _json(dense / "doc_ids.json")
    if (not isinstance(ids, list) or len(ids) != manifest["documents"]
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ImportIntegrityError("Dense document IDs are invalid or duplicated")
    parquet = pq.ParquetFile(corpus, memory_map=False, pre_buffer=False)
    try:
        if parquet.metadata.num_rows != len(ids) or not {"doc_id", "source_type", "title", "content"} <= set(parquet.schema_arrow.names):
            raise ImportIntegrityError("Corpus shape does not match the dense catalogue")
    finally:
        parquet.close()
    rows, shapes = [], set()
    for name in names:
        vectors = np.load(dense / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        try:
            if (not isinstance(vectors, np.memmap) or vectors.ndim != 2 or vectors.shape[1] < 1
                    or vectors.dtype not in (np.dtype("float16"), np.dtype("float32"))
                    or not vectors.flags.c_contiguous):
                raise ImportIntegrityError("Dense vectors require C-order native fp16/fp32 rows")
            count = len(vectors)
            shapes.add((vectors.shape[1], vectors.dtype.str))
        finally:
            if isinstance(vectors, np.memmap):
                vectors._mmap.close()
            elif hasattr(vectors, "close"):
                vectors.close()
        metadata = pq.ParquetFile(dense / (name + ".parquet"), memory_map=False, pre_buffer=False)
        try:
            if (tuple(metadata.schema_arrow.names) != _SPAN_COLUMNS or metadata.metadata.num_rows != count
                    or any(not pa.types.is_signed_integer(metadata.schema_arrow.field(column).type) for column in _SPAN_COLUMNS)):
                raise ImportIntegrityError("Dense metadata schema or row count differs from vectors")
        finally:
            metadata.close()
        rows.append(count)
    if len(shapes) != 1:
        raise ImportIntegrityError("Dense vector dimensions or dtypes differ across shards")
    digest = hashlib.sha256(_canonical_json({"format": "rag-catalog-import-v1",
        "canonical_text": "exact-title-newline-content-v1", "artifacts": hashes})).hexdigest()
    result = _Sources(corpus, dense, MappingProxyType(hashes), MappingProxyType(signatures), digest,
                      tuple(ids), tuple(rows), "dense-v1:" + hashes["dense/manifest.json"])
    _unchanged(result)
    return result


def _bounds(batch_size, max_batch_bytes):
    validate_catalog_batch_size(batch_size)
    if type(max_batch_bytes) is not int or max_batch_bytes < 1:
        raise ValueError("max_batch_bytes must be a positive integer")


def _document_batches(source, batch_size, max_batch_bytes):
    parquet = pq.ParquetFile(source.corpus, memory_map=False, pre_buffer=False)
    columns = [name for name in ("doc_id", "source_type", "title", "content", "timestamp", "metadata_origin")
               if name in parquet.schema_arrow.names]
    position, documents, byte_size = 0, [], 0
    try:
        for arrow_batch in parquet.iter_batches(batch_size=min(batch_size, 128), columns=columns, use_threads=False):
            _unchanged(source, ("corpus",))
            for row in arrow_batch.to_pylist():
                if not isinstance(row["title"], str) or not isinstance(row["content"], str):
                    raise ImportIntegrityError(f"Canonical title/content must be strings at position {position}")
                title, content = row["title"], row["content"]
                if "\0" in title:
                    raise ImportIntegrityError(f"Canonical title contains PostgreSQL-incompatible NUL at position {position}")
                text = title + "\n" + content if title else content
                try:
                    encoded = text.encode("utf-8")
                except UnicodeError:
                    raise ImportIntegrityError(f"Canonical text is not valid UTF-8 at position {position}") from None
                if "\0" in text and len(encoded) > 1024 * 1024:
                    raise ImportIntegrityError(f"Canonical binary fallback exceeds 1MiB at position {position}")
                entry = CatalogEntry(doc_id=row["doc_id"], revision_id=hashlib.sha256(encoded).hexdigest(),
                    position=position, source_type=row["source_type"], text_length=len(text),
                    timestamp=row.get("timestamp"), metadata_origin=row.get("metadata_origin", "upstream"))
                if position >= source.documents or entry.doc_id != source.doc_ids[position]:
                    raise ImportIntegrityError(f"Corpus document order differs from dense IDs at position {position}")
                document = _Document(entry, title, text, len(encoded))
                if documents and (len(documents) >= batch_size or byte_size + document.byte_size > max_batch_bytes):
                    yield tuple(documents)
                    documents, byte_size = [], 0
                documents.append(document)
                byte_size += document.byte_size
                position += 1
                if len(documents) >= batch_size or byte_size >= max_batch_bytes:
                    yield tuple(documents)
                    documents, byte_size = [], 0
        if documents:
            yield tuple(documents)
        if position != source.documents:
            raise ImportIntegrityError("Corpus document count differs from its declared snapshot")
    finally:
        parquet.close()


@contextmanager
def _operator(url, snapshot_id):
    if not isinstance(snapshot_id, str) or not snapshot_id.strip() or "\0" in snapshot_id:
        raise ValueError("A nonblank snapshot_id is required")
    lock = int.from_bytes(hashlib.sha256(snapshot_id.encode("utf-8")).digest()[:8], "big", signed=True)
    engine = sa.create_engine(normalize_url(url) if url is not None else database_url(migration=True),
        poolclass=NullPool, hide_parameters=True,
        connect_args={"prepare_threshold": None, "connect_timeout": 5, "application_name": "rag-catalog-import"})
    try:
        with engine.connect() as connection:
            acquired = False
            try:
                with connection.begin():
                    current, session = connection.execute(sa.text("SELECT current_user, session_user")).one()
                    actual = connection.connection.driver_connection.info.user
                    if current != "rag_migrator" or session != current or actual != current:
                        raise ImportIntegrityError("Canonical imports require the dedicated migrator login")
                    head = connection.execute(sa.text("SELECT version_num FROM rag.alembic_version")).scalar_one()
                    if head != EXPECTED_HEAD:
                        raise ImportIntegrityError("Database migration head does not support canonical imports")
                    acquired = connection.execute(sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": lock}).scalar_one()
                    if not acquired:
                        raise ImportIntegrityError("Another operator is importing or verifying this snapshot")
                yield connection
            finally:
                if connection.in_transaction():
                    connection.rollback()
                if acquired and not connection.invalidated:
                    with suppress(sa.exc.SQLAlchemyError):
                        with connection.begin():
                            connection.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": lock})
    finally:
        engine.dispose()


def _checkpoint(connection, snapshot_id, *, lock=False):
    statement = sa.select(schema.catalog_imports).where(schema.catalog_imports.c.snapshot_id == snapshot_id)
    if lock:
        statement = statement.with_for_update()
    return connection.execute(statement).mappings().one_or_none()


def _check_source(checkpoint, source):
    if (not checkpoint or checkpoint["source_set_sha256"] != source.source_set_sha256
            or checkpoint["artifact_hashes"] != dict(source.hashes)
            or checkpoint["expected_document_count"] != source.documents
            or checkpoint["expected_passage_count"] != source.passages
            or checkpoint["expected_shards"] != len(source.shard_rows)):
        raise ImportIntegrityError("Checkpoint does not match the actual complete source artifact set")


def _start(connection, snapshot_id, source):
    with connection.begin():
        checkpoint = _checkpoint(connection, snapshot_id, lock=True)
        snapshot = connection.execute(sa.select(schema.dataset_snapshots).where(
            schema.dataset_snapshots.c.snapshot_id == snapshot_id)).mappings().one_or_none()
        if checkpoint is None:
            if snapshot is not None:
                raise ImportIntegrityError("Snapshot already exists without this import checkpoint")
            connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=snapshot_id,
                corpus_sha256=source.hashes["corpus"], document_count=source.documents))
            connection.execute(schema.catalog_imports.insert().values(snapshot_id=snapshot_id,
                source_set_sha256=source.source_set_sha256, artifact_hashes=dict(source.hashes),
                expected_document_count=source.documents, expected_passage_count=source.passages,
                expected_shards=len(source.shard_rows)))
        else:
            _check_source(checkpoint, source)
            if (snapshot is None or snapshot["corpus_sha256"] != source.hashes["corpus"]
                    or snapshot["document_count"] != source.documents or snapshot["state"] == "failed"):
                raise ImportIntegrityError("Import snapshot identity or state differs")


def _event(checkpoint):
    return {"phase": "ready" if checkpoint["phase"] == "complete" else checkpoint["phase"],
        "state": "ready" if checkpoint["phase"] == "complete" else "importing",
        "document_cursor": checkpoint["document_cursor"], "span_shard": checkpoint["span_shard"],
        "span_cursor": checkpoint["span_cursor"], "passage_count": checkpoint["passage_cursor"],
        "unique_chunk_count": checkpoint["unique_chunk_count"]}


def _update(connection, snapshot_id, **values):
    connection.execute(schema.catalog_imports.update().where(
        schema.catalog_imports.c.snapshot_id == snapshot_id).values(**values, updated_at=sa.func.now()))
    return dict(_checkpoint(connection, snapshot_id))


def _compare_documents(connection, snapshot_id, documents):
    if not documents:
        return
    table, body = schema.documents, schema.document_revisions
    rows = connection.execute(sa.select(*(table.c[name] for name in _ENTRY_FIELDS), body.c.title,
        body.c.canonical_text, body.c.canonical_utf8)
        .select_from(table.join(body, sa.and_(table.c.snapshot_id == body.c.snapshot_id,
            table.c.doc_id == body.c.doc_id, table.c.revision_id == body.c.revision_id)))
        .where(table.c.snapshot_id == snapshot_id, table.c.position >= documents[0].entry.position,
               table.c.position <= documents[-1].entry.position).order_by(table.c.position)).mappings().all()
    if len(rows) != len(documents):
        raise ImportIntegrityError("Committed canonical document batch is missing rows")
    for row, expected in zip(rows, documents):
        entry = CatalogEntry.model_validate({name: row[name] for name in _ENTRY_FIELDS})
        actual_text = row["canonical_text"]
        if row["canonical_utf8"] is not None:
            try:
                actual_text = bytes(row["canonical_utf8"]).decode("utf-8")
            except UnicodeError:
                raise ImportIntegrityError("Committed canonical binary document is not UTF-8") from None
        if entry != expected.entry or row["title"] != expected.title or actual_text != expected.text:
            raise ImportIntegrityError(f"Committed canonical document differs at position {expected.entry.position}")


def _import_documents(connection, snapshot_id, source, batch_size, max_batch_bytes, callback):
    def entries():
        for batch in _document_batches(source, batch_size, max_batch_bytes):
            committed = None
            with connection.begin():
                checkpoint = _checkpoint(connection, snapshot_id, lock=True)
                _check_source(checkpoint, source)
                cursor = checkpoint["document_cursor"]
                existing = tuple(document for document in batch if document.entry.position < cursor)
                fresh = tuple(document for document in batch if document.entry.position >= cursor)
                _compare_documents(connection, snapshot_id, existing)
                if fresh:
                    if checkpoint["phase"] != "documents" or fresh[0].entry.position != cursor:
                        raise ImportIntegrityError("Document checkpoint is not contiguous")
                    connection.execute(schema.documents.insert(), [dict(snapshot_id=snapshot_id,
                        **document.entry.model_dump()) for document in fresh])
                    connection.execute(schema.document_revisions.insert(), [{"snapshot_id": snapshot_id,
                        "doc_id": document.entry.doc_id, "revision_id": document.entry.revision_id,
                        "title": document.title,
                        "canonical_text": None if "\0" in document.text else document.text,
                        "canonical_utf8": document.text.encode("utf-8") if "\0" in document.text else None}
                        for document in fresh])
                    committed = _update(connection, snapshot_id,
                        document_cursor=fresh[-1].entry.position + 1,
                        max_document_bytes=max(checkpoint["max_document_bytes"], max(document.byte_size for document in batch)),
                        max_batch_bytes_observed=max(checkpoint["max_batch_bytes_observed"], sum(document.byte_size for document in batch)))
            if committed is not None and callback:
                callback(_event(committed))
            yield from (document.entry for document in batch)
    digest = catalog_fingerprint(entries(), expected_count=source.documents)
    with connection.begin():
        checkpoint = _checkpoint(connection, snapshot_id, lock=True)
        _check_source(checkpoint, source)
        if checkpoint["document_cursor"] != source.documents:
            raise ImportIntegrityError("Canonical document import is incomplete")
        if checkpoint["phase"] == "documents":
            connection.execute(schema.dataset_snapshots.update().where(
                schema.dataset_snapshots.c.snapshot_id == snapshot_id).values(catalog_sha256=digest))
            committed = _update(connection, snapshot_id, phase="spans")
        else:
            actual = connection.execute(sa.select(schema.dataset_snapshots.c.catalog_sha256).where(
                schema.dataset_snapshots.c.snapshot_id == snapshot_id)).scalar_one()
            if actual != digest:
                raise ImportIntegrityError("Committed catalogue fingerprint differs from canonical source")
            committed = None
    if committed is not None and callback:
        callback(_event(committed))


def _span_batches(source, batch_size):
    for shard, expected_rows in enumerate(source.shard_rows):
        role = f"dense/shard_{shard:05d}.parquet"
        parquet = pq.ParquetFile(source.path(role), memory_map=False, pre_buffer=False)
        cursor = 0
        try:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(_SPAN_COLUMNS), use_threads=False):
                _unchanged(source, (role,))
                if any(column.null_count for column in batch.columns):
                    raise ImportIntegrityError("Dense passage metadata contains null values")
                columns = [batch.column(index).to_numpy(zero_copy_only=False) for index in range(3)]
                rows = tuple(tuple(int(value) for value in row) for row in zip(*columns))
                if any(not 0 <= position < source.documents or not 0 <= start < end
                       for position, start, end in rows):
                    raise ImportIntegrityError("Dense passage metadata is outside the canonical catalogue")
                yield shard, cursor, rows
                cursor += len(rows)
            if cursor != expected_rows:
                raise ImportIntegrityError("Dense passage metadata row count changed")
            if expected_rows == 0:
                yield shard, 0, ()
        finally:
            parquet.close()


def _chunk_rows(connection, snapshot_id, source, spans):
    if not spans:
        return []
    table = schema.documents
    positions = sorted({position for position, _start, _end in spans})
    rows = connection.execute(sa.select(table.c.position, table.c.doc_id, table.c.revision_id, table.c.text_length).where(
        table.c.snapshot_id == snapshot_id, table.c.position.in_(positions))).mappings().all()
    catalog = {row["position"]: row for row in rows}
    if len(catalog) != len(positions):
        raise ImportIntegrityError("Dense passage refers to a missing canonical document")
    result = []
    for position, start, end in spans:
        entry = catalog[position]
        if entry["doc_id"] != source.doc_ids[position] or end > entry["text_length"]:
            raise ImportIntegrityError("Dense passage differs from canonical document bounds")
        result.append({"snapshot_id": snapshot_id, "chunk_id": f"{entry['doc_id']}:{start}:{end}",
            "doc_id": entry["doc_id"], "revision_id": entry["revision_id"], "start_char": start, "end_char": end,
            "chunker_id": source.chunker_id, "provenance": {"source_set_sha256": source.source_set_sha256,
                "index_manifest_sha256": source.hashes["dense/manifest.json"]}})
    return result


def _compare_chunks(connection, snapshot_id, expected):
    if not expected:
        return
    table = schema.chunks
    rows = connection.execute(sa.select(table).where(table.c.snapshot_id == snapshot_id,
        table.c.chunk_id.in_(tuple({row["chunk_id"] for row in expected})))).mappings().all()
    actual = {row["chunk_id"]: dict(row) for row in rows}
    if any(actual.get(row["chunk_id"]) != row for row in expected):
        raise ImportIntegrityError("Committed indexed passage differs from the source spans")


def _import_spans(connection, snapshot_id, source, batch_size, callback):
    for shard, start, spans in _span_batches(source, batch_size):
        stop, committed = start + len(spans), None
        with connection.begin():
            checkpoint = _checkpoint(connection, snapshot_id, lock=True)
            _check_source(checkpoint, source)
            cursor = (checkpoint["span_shard"], checkpoint["span_cursor"])
            if cursor < (shard, start):
                raise ImportIntegrityError("Passage checkpoint is not contiguous")
            expected = _chunk_rows(connection, snapshot_id, source, spans)
            if checkpoint["phase"] != "complete" and cursor < (shard, stop):
                if checkpoint["phase"] != "spans" or cursor[0] != shard:
                    raise ImportIntegrityError("Passage checkpoint phase is inconsistent")
                offset = max(0, cursor[1] - start)
                fresh = expected[offset:]
                inserted = connection.execute(pg_insert(schema.chunks).values(fresh).on_conflict_do_nothing(
                    index_elements=["snapshot_id", "chunk_id"]).returning(schema.chunks.c.chunk_id)).scalars().all() if fresh else []
                finished = stop == source.shard_rows[shard]
                committed = _update(connection, snapshot_id,
                    span_shard=shard + 1 if finished else shard, span_cursor=0 if finished else stop,
                    passage_cursor=checkpoint["passage_cursor"] + len(fresh),
                    unique_chunk_count=checkpoint["unique_chunk_count"] + len(inserted))
            elif not spans and cursor == (shard, 0) and checkpoint["phase"] == "spans":
                committed = _update(connection, snapshot_id, span_shard=shard + 1, span_cursor=0)
            _compare_chunks(connection, snapshot_id, expected)
        if committed is not None and callback:
            callback(_event(committed))


def _verify(connection, snapshot_id, source, batch_size, max_batch_bytes, *, allow_importing=False):
    with connection.begin():
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        checkpoint = dict(_checkpoint(connection, snapshot_id) or {})
        _check_source(checkpoint, source)
        snapshot_row = connection.execute(sa.select(schema.dataset_snapshots).where(
            schema.dataset_snapshots.c.snapshot_id == snapshot_id)).mappings().one_or_none()
        if (snapshot_row is None or snapshot_row["corpus_sha256"] != source.hashes["corpus"]
                or snapshot_row["document_count"] != source.documents
                or snapshot_row["state"] not in (("importing", "ready") if allow_importing else ("ready",))):
            raise ImportIntegrityError("Snapshot is unavailable or incomplete")
        for table in (schema.documents, schema.document_revisions):
            count = connection.execute(sa.select(sa.func.count()).select_from(table).where(
                table.c.snapshot_id == snapshot_id)).scalar_one()
            if count != source.documents:
                raise ImportIntegrityError("Canonical table count differs from the complete corpus")
        unique_chunks = connection.execute(sa.select(sa.func.count()).select_from(schema.chunks).where(
            schema.chunks.c.snapshot_id == snapshot_id, schema.chunks.c.chunker_id == source.chunker_id)).scalar_one()
        binary_count = connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions).where(
            schema.document_revisions.c.snapshot_id == snapshot_id,
            schema.document_revisions.c.canonical_utf8.is_not(None))).scalar_one()
    def entries():
        for batch in _document_batches(source, batch_size, max_batch_bytes):
            with connection.begin():
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                _compare_documents(connection, snapshot_id, batch)
            yield from (document.entry for document in batch)
    digest = catalog_fingerprint(entries(), expected_count=source.documents)
    if snapshot_row["catalog_sha256"] != digest:
        raise ImportIntegrityError("Stored catalogue fingerprint differs from canonical source")
    passages = 0
    for _shard, _start, spans in _span_batches(source, batch_size):
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            _compare_chunks(connection, snapshot_id, _chunk_rows(connection, snapshot_id, source, spans))
        passages += len(spans)
    if (checkpoint["document_cursor"] != source.documents or checkpoint["passage_cursor"] != source.passages
            or passages != source.passages or checkpoint["span_shard"] != len(source.shard_rows)
            or checkpoint["span_cursor"] != 0 or checkpoint["unique_chunk_count"] != unique_chunks
            or (not allow_importing and checkpoint["phase"] != "complete")):
        raise ImportIntegrityError("Import checkpoint or indexed passage coverage is incomplete")
    _recheck_hashes(source)
    snapshot = SnapshotRef(snapshot_id=snapshot_id, corpus_sha256=source.hashes["corpus"],
        catalog_sha256=digest, document_count=source.documents)
    return ImportReport(snapshot, snapshot_row["state"], source.documents, passages, unique_chunks,
        source.source_set_sha256, source.hashes, checkpoint["document_cursor"], checkpoint["span_shard"],
        checkpoint["span_cursor"], checkpoint["max_document_bytes"], checkpoint["max_batch_bytes_observed"], binary_count)


def import_catalog(corpus: Path, dense_directory: Path, *, snapshot_id: str, url=None,
                   batch_size: int = 500, max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
                   on_checkpoint=None) -> ImportReport:
    """Import/resume unchanged files; callbacks observe only committed progress."""
    _bounds(batch_size, max_batch_bytes)
    source = _sources(corpus, dense_directory)
    with _operator(url, snapshot_id) as connection:
        _start(connection, snapshot_id, source)
        with connection.begin():
            checkpoint = _checkpoint(connection, snapshot_id)
            complete = checkpoint["phase"] == "complete"
        if complete:
            return _verify(connection, snapshot_id, source, batch_size, max_batch_bytes)
        _import_documents(connection, snapshot_id, source, batch_size, max_batch_bytes, on_checkpoint)
        _import_spans(connection, snapshot_id, source, batch_size, on_checkpoint)
        report = _verify(connection, snapshot_id, source, batch_size, max_batch_bytes, allow_importing=True)
        with connection.begin():
            checkpoint = _checkpoint(connection, snapshot_id, lock=True)
            _check_source(checkpoint, source)
            committed = _update(connection, snapshot_id, phase="complete")
            connection.execute(schema.dataset_snapshots.update().where(
                schema.dataset_snapshots.c.snapshot_id == snapshot_id).values(state="ready"))
        if on_checkpoint:
            on_checkpoint(_event(committed))
        return ImportReport(**{**report.__dict__, "state": "ready"})


def verify_import(corpus: Path, dense_directory: Path, *, snapshot_id: str, url=None,
                  batch_size: int = 500, max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES) -> ImportReport:
    """Read-only actual-file, ordered-body, metadata and indexed-span verification."""
    _bounds(batch_size, max_batch_bytes)
    source = _sources(corpus, dense_directory)
    with _operator(url, snapshot_id) as connection:
        return _verify(connection, snapshot_id, source, batch_size, max_batch_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--dense-directory", required=True, type=Path)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-batch-bytes", type=int, default=DEFAULT_MAX_BATCH_BYTES)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        parser.error("Report output already exists; choose a new path")
    try:
        operation = verify_import if args.verify_only else import_catalog
        result = operation(args.corpus, args.dense_directory, snapshot_id=args.snapshot_id,
                           batch_size=args.batch_size, max_batch_bytes=args.max_batch_bytes)
        payload = json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        if args.report is not None:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        print(payload, end="")
    except (ValueError, OSError, sa.exc.SQLAlchemyError) as error:
        parser.error(str(error) if isinstance(error, ImportIntegrityError) else
                     "Canonical import/verification failed; no incomplete snapshot is published")


if __name__ == "__main__":
    main()

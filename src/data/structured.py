"""Verified typed Jira source records, separate from canonical prose.

The pinned Git tree and the canonical Parquet UUID/path/body set select the
records. Native keys are not unique and never identify a physical record.
No Jira field, permission or definition of 'open' is inferred from prose.
Runtime adapters do no downloading. This operator import uses the migrator and
short bounded transactions, then publishes an immutable complete feed.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

import pyarrow.parquet as pq
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

from src.config import CORPUS_COMMIT, CORPUS_REPO_URL
from src.contracts import SnapshotRef
from src.data.corpus import extract_document
from src.db import schema
from src.db.migrate import EXPECTED_HEAD
from src.db.runtime import database_url, normalize_url

PARSER_VERSION = "jira-native-fields-v1"
FIELDS = {"native_key": "key", "status": "status", "priority": "priority", "project_id": "project"}
MAX_JSON_BYTES = 1024 * 1024
PREFIX = "generated_data/sources/"


class StructuredImportError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise StructuredImportError("Pinned Jira Git provenance is unavailable") from None


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise StructuredImportError("Jira source must be a bounded ordinary JSON file")
        raw = handle.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise StructuredImportError("Jira source grew beyond its bound")
        return raw


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StructuredImportError("Duplicate Jira JSON fields")
        value[key] = item
    return value


@dataclass(frozen=True)
class JiraInputs:
    snapshot: SnapshotRef
    repository: str
    commit: str
    coverage: str
    source_set_sha256: str
    manifest_json: bytes
    rows_json: tuple[bytes, ...]
    source_repo: Path
    corpus: Path

    @property
    def feed_id(self):
        return digest(b"rag-jira-feed-v1\n" + self.manifest_json)

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    def recheck(self):
        current = prepare_jira_inputs(self.source_repo, self.corpus, self.snapshot,
            repository=self.repository, commit=self.commit, coverage=self.coverage)
        if current != self:
            raise StructuredImportError("Verified Jira inputs changed")


def _canonical_records(corpus, snapshot):
    if file_digest(corpus) != snapshot.corpus_sha256:
        raise StructuredImportError("Canonical Parquet differs from the snapshot")
    result = {}
    parquet = pq.ParquetFile(corpus)
    try:
        if parquet.metadata.num_rows != snapshot.document_count or not {"doc_id", "source_type", "title", "content", "path"} <= set(parquet.schema_arrow.names):
            raise StructuredImportError("Canonical Parquet lacks its exact source identity")
        index = parquet.schema_arrow.get_field_index("source_type")
        for number in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(number).column(index).statistics
            if statistics is not None and statistics.has_min_max and not statistics.min <= "jira" <= statistics.max:
                continue
            for batch in parquet.iter_batches(batch_size=128, row_groups=[number], columns=["doc_id", "source_type", "title", "content", "path"], use_threads=False):
                for row in batch.to_pylist():
                    if row["source_type"] != "jira":
                        continue
                    if any(not isinstance(row[key], str) for key in ("doc_id", "title", "content", "path")):
                        raise StructuredImportError("Canonical Jira identity is invalid")
                    path = PurePosixPath(row["path"])
                    if path.is_absolute() or ".." in path.parts or not row["path"].startswith("jira/") or path.suffix != ".json":
                        raise StructuredImportError("Canonical Jira path is invalid")
                    if row["doc_id"] in result:
                        raise StructuredImportError("Canonical Jira identity is duplicated or invalid")
                    text = row["title"] + "\n" + row["content"] if row["title"] else row["content"]
                    result[row["doc_id"]] = (row["path"], digest(text.encode("utf-8")), len(text))
    finally:
        parquet.close()
    return result


def prepare_jira_inputs(source_repo, corpus, snapshot: SnapshotRef, *, repository=CORPUS_REPO_URL,
                        commit=CORPUS_COMMIT, coverage="verified_records") -> JiraInputs:
    """Verify every canonical record against its actual pinned Git blob.

    A synthetic_fixture source must be explicitly selected and remains labeled
    through import, lookup and aggregates. Production verification only accepts
    the project's pinned upstream repository/commit.
    """
    source_repo, corpus = Path(source_repo).resolve(), Path(corpus).resolve()
    snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
    if coverage not in {"verified_records", "synthetic_fixture"} or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise StructuredImportError("Invalid Jira source declaration")
    if coverage == "verified_records" and (repository != CORPUS_REPO_URL or commit != CORPUS_COMMIT):
        raise StructuredImportError("Verified Jira records require the pinned upstream source")
    if _git(source_repo, "rev-parse", "HEAD").decode().strip() != commit or _git(source_repo, "remote", "get-url", "origin").decode().strip() != repository:
        raise StructuredImportError("Jira checkout or remote differs from its declaration")
    tree = {}
    for record in _git(source_repo, "ls-tree", "-rz", commit, "--", PREFIX + "jira").split(b"\0"):
        if not record:
            continue
        header, name = record.decode("utf-8").split("\t", 1)
        mode, kind, blob = header.split()
        # Other tracked repository files are not record inputs or instructions.
        if not name.endswith(".json"):
            continue
        if mode != "100644" or kind != "blob":
            raise StructuredImportError("Pinned Jira source tree must contain ordinary JSON blobs")
        tree[name.removeprefix(PREFIX)] = blob
    expected = _canonical_records(corpus, snapshot)
    paths = {value[0] for value in expected.values()}
    if len(paths) != len(expected) or not paths <= tree.keys():
        raise StructuredImportError("Canonical Jira paths differ from the pinned tree")
    records, inventory, coverage_counts, keys = [], [], {field: Counter(verified=0, missing=0, invalid=0) for field in FIELDS}, Counter()
    for doc_id, (relative, revision, length) in sorted(expected.items()):
        raw = _read(source_repo / PREFIX / relative)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != tree[relative]:
            raise StructuredImportError("Jira file differs from its pinned Git blob")
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            title, content = extract_document(record)
            text = title + "\n" + content if title else content
            if record["dataset_doc_uuid"] != doc_id or digest(text.encode("utf-8")) != revision or len(text) != length:
                raise StructuredImportError("Jira UUID or canonical body revision differs")
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise StructuredImportError("Invalid or mismatching canonical Jira source") from None
        provenance = {"repository": repository, "git_commit": commit, "git_blob": blob,
            "path": PREFIX + relative, "raw_sha256": digest(raw), "parser_version": PARSER_VERSION}
        fields, field_provenance = {}, []
        for field, key in FIELDS.items():
            value = record.get(key)
            valid_text = type(value) is str and value.strip() and len(value) <= 512 and "\0" not in value
            if valid_text:
                try:
                    value.encode("utf-8")
                except UnicodeError:
                    valid_text = False
            state = ("verified" if valid_text
                     else "missing" if value is None else "invalid")
            fields[field] = value if state == "verified" else None
            coverage_counts[field][state] += 1
            field_provenance.append({"field": field, "json_pointer": "/" + key, "state": state})
        if fields["native_key"]:
            keys[fields["native_key"]] += 1
        rows = {"snapshot_id": snapshot.snapshot_id, "ticket_id": doc_id, "doc_id": doc_id,
            "revision_id": revision, "coverage": coverage, **fields,
            "field_provenance": {"source": provenance, "fields": field_provenance}}
        records.append(canonical(rows))
        inventory.append({"doc_id": doc_id, "path": relative, "git_blob": blob, "raw_sha256": digest(raw), "revision_id": revision})
    source_set_sha256 = digest(canonical(inventory))
    manifest = {"format": PARSER_VERSION, "snapshot": snapshot.model_dump(mode="json"), "repository": repository,
        "git_commit": commit, "coverage": coverage, "source_set_sha256": source_set_sha256,
        "records": inventory, "expected_records": len(records),
        "excluded_noncanonical_paths": sorted(tree.keys() - paths),
        "field_coverage": {key: dict(value) for key, value in coverage_counts.items()},
        "duplicate_native_keys": sum(value > 1 for value in keys.values()), "cardinality": "canonical_source_records"}
    return JiraInputs(snapshot, repository, commit, coverage, source_set_sha256, canonical(manifest),
                      tuple(records), source_repo, corpus)


@contextmanager
def _operator(url):
    engine = sa.create_engine(normalize_url(url) if url else database_url(migration=True), poolclass=NullPool,
        hide_parameters=True, connect_args={"prepare_threshold": None, "connect_timeout": 5, "application_name": "rag-jira-import"})
    try:
        with engine.connect() as connection:
            with connection.begin():
                current, session = connection.execute(sa.text("SELECT current_user,session_user")).one()
                if current != "rag_migrator" or session != current or connection.connection.driver_connection.info.user != current:
                    raise StructuredImportError("Jira import requires the dedicated migrator login")
                if connection.execute(sa.text("SELECT version_num FROM rag.alembic_version")).scalar_one() != EXPECTED_HEAD:
                    raise StructuredImportError("Jira import requires the current migration head")
                connection.execute(sa.text("SELECT pg_advisory_lock(59800005)"))
            try:
                yield connection
            finally:
                connection.rollback()
                with connection.begin():
                    connection.execute(sa.text("SELECT pg_advisory_unlock(59800005)"))
    except sa.exc.SQLAlchemyError:
        raise StructuredImportError("Jira database import failed") from None
    finally:
        engine.dispose()


def import_jira(inputs: JiraInputs, *, url=None, batch_size=128):
    if not isinstance(inputs, JiraInputs) or type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise StructuredImportError("Jira import requires verified inputs and a bounded batch")
    inputs.recheck()
    manifest, feed_id = inputs.manifest, inputs.feed_id
    header = {"feed_id": feed_id, "snapshot_id": inputs.snapshot.snapshot_id, "source_type": "jira",
        "coverage": inputs.coverage, "repository": inputs.repository, "git_commit": inputs.commit,
        "corpus_sha256": inputs.snapshot.corpus_sha256, "source_set_sha256": inputs.source_set_sha256,
        "expected_records": len(inputs.rows_json), "field_coverage": manifest["field_coverage"], "source_manifest": manifest}
    with _operator(url) as connection:
        with connection.begin():
            snapshot = connection.execute(sa.select(schema.dataset_snapshots).where(
                schema.dataset_snapshots.c.snapshot_id == inputs.snapshot.snapshot_id)).mappings().one_or_none()
            if snapshot is None or snapshot["state"] != "ready" or any(snapshot[key] != value
                    for key, value in inputs.snapshot.model_dump().items()):
                raise StructuredImportError("Jira source differs from the ready canonical snapshot")
            row = connection.execute(sa.select(schema.ticket_feeds).where(schema.ticket_feeds.c.snapshot_id == inputs.snapshot.snapshot_id,
                schema.ticket_feeds.c.source_type == "jira")).mappings().one_or_none()
            if row is None:
                connection.execute(schema.ticket_feeds.insert().values(**header))
            elif any(row[key] != value for key, value in header.items()):
                raise StructuredImportError("Existing Jira feed differs from the verified source")
        for start in range(0, len(inputs.rows_json), batch_size):
            rows = [dict(json.loads(value), feed_id=feed_id) for value in inputs.rows_json[start:start + batch_size]]
            with connection.begin():
                for row in rows:
                    document = connection.execute(sa.select(schema.documents.c.revision_id, schema.documents.c.source_type).where(
                        schema.documents.c.snapshot_id == row["snapshot_id"], schema.documents.c.doc_id == row["doc_id"])).one_or_none()
                    if document is None or document.revision_id != row["revision_id"] or document.source_type != "jira":
                        raise StructuredImportError("Jira record differs from the stored canonical document")
                existing = {value["ticket_id"]: dict(value) for value in connection.execute(sa.select(schema.tickets).where(
                    schema.tickets.c.snapshot_id == inputs.snapshot.snapshot_id, schema.tickets.c.ticket_id.in_([row["ticket_id"] for row in rows]))).mappings()}
                fresh = [row for row in rows if row["ticket_id"] not in existing]
                if any(existing[row["ticket_id"]] != row for row in rows if row["ticket_id"] in existing):
                    raise StructuredImportError("Committed Jira records differ from the source")
                if fresh:
                    connection.execute(schema.tickets.insert(), fresh)
                total = connection.execute(sa.select(sa.func.count()).select_from(schema.tickets).where(schema.tickets.c.feed_id == feed_id)).scalar_one()
                connection.execute(schema.ticket_feeds.update().where(schema.ticket_feeds.c.feed_id == feed_id).values(imported_records=total))
        inputs.recheck()
        with connection.begin():
            connection.execute(schema.ticket_feeds.update().where(schema.ticket_feeds.c.feed_id == feed_id).values(
                imported_records=len(inputs.rows_json), state="ready"))
    return {"feed_id": feed_id, "state": "ready", "records": len(inputs.rows_json),
            "coverage": inputs.coverage, "field_coverage": manifest["field_coverage"],
            "duplicate_native_keys": manifest["duplicate_native_keys"], "excluded_noncanonical_paths": manifest["excluded_noncanonical_paths"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        inputs = prepare_jira_inputs(args.source_repo, args.corpus, SnapshotRef.model_validate_json(args.snapshot.read_bytes()))
        report = inputs.manifest if args.preflight else import_jira(inputs)
    except (ValueError, OSError, sa.exc.SQLAlchemyError):
        parser.error("Verified Jira operation failed; check source identity and operator configuration")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

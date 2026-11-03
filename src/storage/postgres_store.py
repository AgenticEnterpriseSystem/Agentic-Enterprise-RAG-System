"""Lazy canonical excerpts through the existing nonowner runtime pool.

Catalogue methods are internal metadata operations. Body reads require a bound
server context, current authorization and PostgreSQL RLS. Until ACL installs its
reviewed policies, protected reads deny by default. No privileged connection or
second pool is created here, and no transaction survives a yielded catalogue
batch or an external freshness check.
"""

from collections.abc import Iterator
import hashlib

import sqlalchemy as sa

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, FreshnessGuard, RequestContext, SnapshotMismatch,
)
from src.contracts import Chunk, PassageEvidence, SnapshotRef
from src.db import schema
from src.db.runtime import Database
from src.storage.contracts import (
    CatalogEntry, DEFAULT_READ_MORE_CHARS, MAX_CATALOG_BATCH, RankedCandidate,
    ReadFilters, validate_catalog_batch_size, validate_read_more_bounds,
)
from src.storage.fingerprints import catalog_fingerprint

_ENTRY_FIELDS = tuple(CatalogEntry.model_fields)
MAX_BINARY_DOCUMENT_BYTES = 1024 * 1024


def _entry(row) -> CatalogEntry:
    return CatalogEntry.model_validate({name: row[name] for name in _ENTRY_FIELDS})


class PostgresDocumentStore:
    """One ready immutable snapshot; retain its identity, never its body table."""

    def __init__(self, database: Database, snapshot: SnapshotRef):
        if not isinstance(database, Database) or not isinstance(snapshot, SnapshotRef):
            raise TypeError("Document storage requires the runtime Database and SnapshotRef")
        self._database = database
        self._snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        with database.transaction() as connection:
            self._ready(connection)
        digest = catalog_fingerprint((entry for batch in self.iter_catalog() for entry in batch),
                                     expected_count=self._snapshot.document_count)
        if digest != self._snapshot.catalog_sha256:
            raise SnapshotMismatch("Stored catalogue differs from the requested snapshot")

    def snapshot(self) -> SnapshotRef:
        return self._snapshot

    def _ready(self, connection) -> None:
        table = schema.dataset_snapshots
        row = connection.execute(sa.select(table).where(
            table.c.snapshot_id == self._snapshot.snapshot_id)).mappings().one_or_none()
        if (row is None or row["state"] != "ready"
                or any(row[name] != getattr(self._snapshot, name) for name in
                       ("corpus_sha256", "catalog_sha256", "document_count"))):
            raise SnapshotMismatch("Requested document snapshot is unavailable or incomplete")

    def iter_catalog(self, *, batch_size: int = MAX_CATALOG_BATCH) -> Iterator[tuple[CatalogEntry, ...]]:
        validate_catalog_batch_size(batch_size)
        position, total = -1, 0
        table = schema.documents
        while True:
            with self._database.transaction() as connection:
                self._ready(connection)
                rows = connection.execute(sa.select(*(table.c[name] for name in _ENTRY_FIELDS)).where(
                    table.c.snapshot_id == self._snapshot.snapshot_id,
                    table.c.position > position).order_by(table.c.position).limit(batch_size)).mappings().all()
            if not rows:
                if total != self._snapshot.document_count:
                    raise SnapshotMismatch("Stored catalogue is incomplete")
                return
            batch = tuple(_entry(row) for row in rows)
            if any(entry.position != total + offset for offset, entry in enumerate(batch)):
                raise SnapshotMismatch("Stored catalogue positions are inconsistent")
            total += len(batch)
            if total > self._snapshot.document_count:
                raise SnapshotMismatch("Stored catalogue exceeds its snapshot")
            position = batch[-1].position
            yield batch

    def source_types(self) -> tuple[str, ...]:
        table = schema.documents
        with self._database.transaction() as connection:
            self._ready(connection)
            return tuple(connection.execute(sa.select(table.c.source_type).where(
                table.c.snapshot_id == self._snapshot.snapshot_id).distinct().order_by(
                table.c.source_type)).scalars())

    def _current(self, context: RequestContext, guard: FreshnessGuard) -> None:
        if not isinstance(context, RequestContext):
            raise EvidenceValidationError("Document reads require a resolved request context")
        guard.assert_current(context)
        context.assert_bound()
        if context.binding.snapshot != self._snapshot:
            raise SnapshotMismatch("Read context belongs to another snapshot")

    def _authorize(self, context, doc_id, guard) -> None:
        self._current(context, guard)
        if context.scope.permits(doc_id) is not True:
            raise AccessDenied("Document is unavailable in the current scope")
        context.assert_bound()

    def _record(self, context, doc_id, guard) -> CatalogEntry:
        self._authorize(context, doc_id, guard)
        table = schema.documents
        with self._database.transaction(context) as connection:
            self._ready(connection)
            row = connection.execute(sa.select(*(table.c[name] for name in _ENTRY_FIELDS)).where(
                table.c.snapshot_id == self._snapshot.snapshot_id, table.c.doc_id == doc_id)).mappings().one_or_none()
        if row is None:
            raise AccessDenied("Document is unavailable in the current scope")
        return _entry(row)

    def _spans(self, context, entry, spans, guard):
        """One protected SELECT; SQL returns only requested Unicode slices."""
        self._authorize(context, entry.doc_id, guard)
        table = schema.document_revisions
        columns = [table.c.title, table.c.revision_id,
                   sa.func.char_length(table.c.canonical_text).label("text_length"),
                   sa.case((sa.func.octet_length(table.c.canonical_utf8) <= MAX_BINARY_DOCUMENT_BYTES,
                            table.c.canonical_utf8), else_=None).label("canonical_utf8")]
        for index, (start, end) in enumerate(spans):
            if not 0 <= start < end <= entry.text_length:
                raise EvidenceValidationError("Requested passage is outside canonical text")
            columns.append(sa.func.substr(table.c.canonical_text, start + 1, end - start).label(f"text_{index}"))
        with self._database.transaction(context) as connection:
            self._ready(connection)
            row = connection.execute(sa.select(*columns).where(table.c.snapshot_id == self._snapshot.snapshot_id,
                table.c.doc_id == entry.doc_id, table.c.revision_id == entry.revision_id)).mappings().one_or_none()
        self._authorize(context, entry.doc_id, guard)
        if row is None:
            raise AccessDenied("Document is unavailable in the current scope")
        if row["canonical_utf8"] is not None:
            if row["text_length"] is not None:
                raise EvidenceValidationError("Canonical revision has ambiguous body storage")
            raw = bytes(row["canonical_utf8"])
            try:
                text = raw.decode("utf-8")
            except UnicodeError:
                raise EvidenceValidationError("Canonical binary revision is not valid UTF-8") from None
            if (len(raw) > MAX_BINARY_DOCUMENT_BYTES or len(text) != entry.text_length
                    or hashlib.sha256(raw).hexdigest() != entry.revision_id):
                raise EvidenceValidationError("Canonical binary revision differs from its catalogue")
            row = dict(row)
            row["text_length"] = len(text)
            for index, (start, end) in enumerate(spans):
                row[f"text_{index}"] = text[start:end]
        if row["revision_id"] != entry.revision_id or row["text_length"] != entry.text_length:
            raise EvidenceValidationError("Canonical revision differs from its catalogue")
        if any(len(row[f"text_{index}"]) != end - start for index, (start, end) in enumerate(spans)):
            raise EvidenceValidationError("Canonical passage length differs from its requested span")
        return row

    def _evidence(self, entry, title, text, start, end, score, method) -> PassageEvidence:
        key = f"{entry.doc_id}:{start}:{end}"
        return PassageEvidence(snapshot_id=self._snapshot.snapshot_id, revision_id=entry.revision_id,
            citation_id=key, score=score, retrieval_method=method,
            chunk=Chunk(chunk_id=key, doc_id=entry.doc_id, source_type=entry.source_type,
                title=title, text=text, start_char=start, end_char=end,
                timestamp=entry.timestamp, metadata_origin=entry.metadata_origin))

    @staticmethod
    def _filters(filters):
        try:
            return ReadFilters.model_validate(filters.model_dump(warnings=False))
        except (ValueError, AttributeError):
            raise EvidenceValidationError("Invalid document read restrictions") from None

    def read_passage(self, context: RequestContext, candidate: RankedCandidate, *,
                     guard: FreshnessGuard, filters: ReadFilters = ReadFilters()) -> PassageEvidence:
        self._current(context, guard)
        try:
            candidate = RankedCandidate.model_validate(candidate.model_dump(warnings=False))
        except (ValueError, AttributeError):
            raise EvidenceValidationError("Invalid passage candidate") from None
        filters = self._filters(filters)
        if candidate.snapshot != self._snapshot:
            raise SnapshotMismatch("Candidate belongs to another snapshot")
        entry = self._record(context, candidate.doc_id, guard)
        try:
            candidate.validate_catalog(self._snapshot, entry)
        except ValueError:
            raise EvidenceValidationError("Candidate differs from the canonical catalogue") from None
        if not filters.matches(entry):
            raise AccessDenied("Document is outside the requested source/date filters")
        row = self._spans(context, entry, ((candidate.start_char, candidate.end_char),), guard)
        result = self._evidence(entry, row["title"], row["text_0"], candidate.start_char,
                                candidate.end_char, candidate.score, candidate.retrieval_method)
        self._authorize(context, entry.doc_id, guard)
        return result

    def read_more(self, context: RequestContext, anchors: tuple[PassageEvidence, ...], *,
                  guard: FreshnessGuard, max_chars: int = DEFAULT_READ_MORE_CHARS,
                  filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        self._current(context, guard)
        validate_read_more_bounds(len(anchors), max_chars)
        filters = self._filters(filters)
        try:
            anchors = tuple(PassageEvidence.model_validate(anchor.model_dump(warnings=False)) for anchor in anchors)
        except (ValueError, AttributeError):
            raise EvidenceValidationError("Invalid expansion anchor") from None
        records = []
        # Every anchor is authorized and structurally checked before any body SELECT.
        for anchor in anchors:
            if anchor.snapshot_id != self._snapshot.snapshot_id:
                raise SnapshotMismatch("Expansion anchor belongs to another snapshot")
            entry = self._record(context, anchor.chunk.doc_id, guard)
            chunk = anchor.chunk
            key = f"{entry.doc_id}:{chunk.start_char}:{chunk.end_char}"
            if (anchor.revision_id != entry.revision_id or chunk.source_type != entry.source_type
                    or chunk.timestamp != entry.timestamp or chunk.metadata_origin != entry.metadata_origin
                    or chunk.end_char > entry.text_length or anchor.citation_id != key or chunk.chunk_id != key):
                raise EvidenceValidationError("Expansion anchor differs from canonical metadata")
            records.append((anchor, entry))
        results, seen = [], set()
        for anchor, entry in records:
            self._authorize(context, entry.doc_id, guard)
            if not filters.matches(entry):
                continue
            chunk = anchor.chunk
            overlap = min(256, max_chars // 4)
            if chunk.end_char < entry.text_length:
                start = max(0, chunk.end_char - overlap)
                end = min(entry.text_length, start + max_chars)
            else:
                end = min(entry.text_length, chunk.start_char + overlap)
                start = max(0, end - max_chars)
            expands = (entry.doc_id not in seen and start < end
                       and not (start >= chunk.start_char and end <= chunk.end_char))
            spans = ((chunk.start_char, chunk.end_char),) + (((start, end),) if expands else ())
            row = self._spans(context, entry, spans, guard)
            if row["title"] != chunk.title or row["text_0"] != chunk.text:
                raise EvidenceValidationError("Expansion anchor differs from canonical text")
            seen.add(entry.doc_id)
            if expands:
                results.append(self._evidence(entry, row["title"], row["text_1"], start, end,
                                               anchor.score, anchor.retrieval_method))
        for anchor, _entry in records:
            self._authorize(context, anchor.chunk.doc_id, guard)
        self._current(context, guard)
        return tuple(results)

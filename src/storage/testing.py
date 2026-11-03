"""Small eager conformance fixture; NOT a production document store.

This reusable fixture intentionally keeps a tiny set of strings in memory for
contract consumers. It implements no indexing, ranking, SQL, identity resolution
or current-policy lookup. Supply a trusted test scope and a freshness guard.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, FreshnessGuard, RequestContext, SnapshotMismatch,
)
from src.contracts import Chunk
from src.contracts_v2 import PassageEvidence, SnapshotRef
from src.storage.contracts import (
    CatalogEntry, DEFAULT_READ_MORE_CHARS, MAX_CATALOG_BATCH, RankedCandidate,
    ReadFilters, validate_catalog_batch_size, validate_read_more_bounds,
)


@dataclass(frozen=True)
class FixtureDocument:
    entry: CatalogEntry
    text: str
    title: str = ""


class InMemoryDocumentStore:
    """Executable tiny-store contract reference, limited to 1000 documents.

    body_reads is diagnostic test state: it makes pre-read denial/freshness
    checks observable. A returned result is also checked before release.
    """

    def __init__(self, snapshot: SnapshotRef, documents: Iterable[FixtureDocument]):
        self._snapshot = SnapshotRef.model_validate(snapshot.model_dump())
        records = tuple(documents)
        if len(records) > MAX_CATALOG_BATCH:
            raise ValueError("This eager fixture is limited to 1000 documents")
        if len(records) != self._snapshot.document_count:
            raise ValueError("Fixture document count differs from snapshot")
        checked = []
        for record in records:
            entry = CatalogEntry.model_validate(record.entry.model_dump())
            if not isinstance(record.text, str) or len(record.text) != entry.text_length:
                raise ValueError("Fixture text length differs from canonical catalogue")
            checked.append(FixtureDocument(entry, record.text, record.title))
        checked.sort(key=lambda item: item.entry.position)
        if [r.entry.position for r in checked] != list(range(len(checked))):
            raise ValueError("Fixture catalogue positions must be contiguous and unique")
        if len({r.entry.doc_id for r in checked}) != len(checked):
            raise ValueError("Duplicate fixture document IDs")
        self._documents = {r.entry.doc_id: r for r in checked}
        self.body_reads: list[str] = []

    def snapshot(self) -> SnapshotRef:
        return self._snapshot

    def iter_catalog(self, *, batch_size: int = MAX_CATALOG_BATCH) -> Iterator[tuple[CatalogEntry, ...]]:
        validate_catalog_batch_size(batch_size)
        entries = iter(record.entry for record in self._documents.values())
        while batch := tuple(islice(entries, batch_size)):
            yield batch

    def source_types(self) -> tuple[str, ...]:
        return tuple(sorted({r.entry.source_type for r in self._documents.values()}))

    def _current(self, context: RequestContext, guard: FreshnessGuard) -> None:
        guard.assert_current(context)
        context.assert_bound()
        if context.binding.snapshot != self._snapshot:
            raise SnapshotMismatch("Read context belongs to another snapshot")

    def _authorized_record(self, context: RequestContext, doc_id: str, guard: FreshnessGuard) -> FixtureDocument:
        self._current(context, guard)
        if not context.scope.permits(doc_id) or doc_id not in self._documents:
            raise AccessDenied("Document is unavailable in the current scope")
        context.assert_bound()
        return self._documents[doc_id]

    def _body(self, record: FixtureDocument) -> str:
        self.body_reads.append(record.entry.doc_id)
        return record.text

    def _passage(self, record, text, start, end, score, method) -> PassageEvidence:
        entry = record.entry
        key = f"{entry.doc_id}:{start}:{end}"
        return PassageEvidence(snapshot_id=self._snapshot.snapshot_id,
            revision_id=entry.revision_id, citation_id=key, score=score,
            retrieval_method=method, chunk=Chunk(chunk_id=key, doc_id=entry.doc_id,
                source_type=entry.source_type, title=record.title,
                metadata_origin=entry.metadata_origin, timestamp=entry.timestamp,
                start_char=start, end_char=end, text=text[start:end]))

    def read_passage(self, context: RequestContext, candidate: RankedCandidate, *,
                     guard: FreshnessGuard, filters: ReadFilters = ReadFilters()) -> PassageEvidence:
        self._current(context, guard)
        try:
            candidate = RankedCandidate.model_validate(candidate.model_dump())
            filters = ReadFilters.model_validate(filters.model_dump())
        except (ValueError, AttributeError):
            raise EvidenceValidationError("Invalid candidate or read filters") from None
        if candidate.snapshot != self._snapshot:
            raise SnapshotMismatch("Candidate belongs to another snapshot")
        record = self._authorized_record(context, candidate.doc_id, guard)
        try:
            candidate.validate_catalog(self._snapshot, record.entry)
        except ValueError:
            raise EvidenceValidationError("Candidate differs from the canonical catalogue") from None
        if not filters.matches(record.entry):
            raise AccessDenied("Document is outside the requested source/date filters")
        text = self._body(record)
        result = self._passage(record, text, candidate.start_char, candidate.end_char,
                               candidate.score, candidate.retrieval_method)
        self._current(context, guard)
        return result

    def read_more(self, context: RequestContext, anchors: tuple[PassageEvidence, ...], *,
                  guard: FreshnessGuard, max_chars: int = DEFAULT_READ_MORE_CHARS,
                  filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        self._current(context, guard)
        validate_read_more_bounds(len(anchors), max_chars)
        try:
            filters = ReadFilters.model_validate(filters.model_dump())
            anchors = tuple(PassageEvidence.model_validate(a.model_dump()) for a in anchors)
        except (ValueError, AttributeError):
            raise EvidenceValidationError("Invalid expansion anchor or read filters") from None
        records = []
        # Check every anchor's permission and metadata before reading any body.
        for anchor in anchors:
            if anchor.snapshot_id != self._snapshot.snapshot_id:
                raise SnapshotMismatch("Expansion anchor belongs to another snapshot")
            record = self._authorized_record(context, anchor.chunk.doc_id, guard)
            entry, chunk = record.entry, anchor.chunk
            key = f"{entry.doc_id}:{chunk.start_char}:{chunk.end_char}"
            if (anchor.revision_id != entry.revision_id or chunk.source_type != entry.source_type
                    or chunk.timestamp != entry.timestamp or chunk.metadata_origin != entry.metadata_origin
                    or chunk.title != record.title or chunk.end_char > entry.text_length
                    or anchor.citation_id != key or chunk.chunk_id != key):
                raise EvidenceValidationError("Expansion anchor differs from canonical metadata")
            records.append((anchor, record))
        result, seen = [], set()
        for anchor, record in records:
            self._current(context, guard)
            if not filters.matches(record.entry):
                continue
            text = self._body(record)
            chunk = anchor.chunk
            if text[chunk.start_char:chunk.end_char] != chunk.text:
                raise EvidenceValidationError("Expansion anchor differs from canonical text")
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            overlap = min(256, max_chars // 4)
            if chunk.end_char < len(text):
                start = max(0, chunk.end_char - overlap)
                end = min(len(text), start + max_chars)
            else:
                end = min(len(text), chunk.start_char + overlap)
                start = max(0, end - max_chars)
            if start < end and not (start >= chunk.start_char and end <= chunk.end_char):
                result.append(self._passage(record, text, start, end, anchor.score, anchor.retrieval_method))
        self._current(context, guard)
        return tuple(result)

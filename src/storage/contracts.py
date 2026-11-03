"""Metadata-only interfaces for future lazy, authorized canonical document reads.

Catalog iteration is an internal server/index operation, never a public source
listing. Each batch is bounded; no contract requires a full catalogue or corpus
body in one model. These contracts do not implement SQL, policy, or ranking.
"""

from collections.abc import Iterator
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from src.access.contracts import FreshnessGuard, RequestContext
from src.contracts import Contract, Identifier
from src.contracts_v2 import PassageEvidence, SnapshotRef

MAX_CATALOG_BATCH = 1000
MAX_READ_MORE_ANCHORS = 2
MAX_READ_MORE_CHARS = 6000
DEFAULT_READ_MORE_CHARS = 3000
Position = Annotated[int, Field(strict=True, ge=0)]
SourceType = Annotated[str, Field(min_length=1, max_length=64, pattern=r"\S")]


class CatalogEntry(Contract):
    """One immutable document revision; text_length counts Unicode characters.

    position is the stable zero-based index position within the store snapshot.
    metadata_origin describes provenance, not permission or verified freshness.
    """

    doc_id: Identifier
    revision_id: Identifier
    position: Position
    source_type: SourceType
    text_length: Position
    timestamp: AwareDatetime | None = None
    metadata_origin: Literal["upstream", "extracted", "synthetic"] = "upstream"


class RankedCandidate(Contract):
    """A ranked exact span without its text; score is finite and larger is better.

    Snapshot and revision are mandatory, even for whole-document lexical hits.
    Such a hit uses [0, catalog.text_length) until a passage selector narrows it.
    scoring_provenance identifies the scorer/index configuration; it does not
    authorize the document or prove that the span supports a generated claim.
    Adjacent reads retain the anchor's score/method as provenance, not a newly
    computed relevance score. No tie-breaking or ranking algorithm lives here.
    """

    snapshot: SnapshotRef
    doc_id: Identifier
    revision_id: Identifier
    source_type: SourceType
    start_char: Position
    end_char: Annotated[int, Field(strict=True, gt=0)]
    score: Annotated[float, Field(strict=True)]
    retrieval_method: Literal["bm25", "dense", "hybrid", "structured"]
    scoring_provenance: Identifier

    @model_validator(mode="after")
    def nonempty_span(self):
        if self.start_char >= self.end_char:
            raise ValueError("Candidate span must be zero-based and end-exclusive")
        return self

    def validate_catalog(self, snapshot: SnapshotRef, entry: CatalogEntry) -> None:
        """Check metadata integrity before accessing canonical text.

        Adapters still check current identity, scope, source/date restrictions,
        canonical text and revision. This check alone is not authorization.
        """
        if self.snapshot != snapshot:
            raise ValueError("Candidate belongs to another snapshot")
        if (self.doc_id != entry.doc_id or self.revision_id != entry.revision_id
                or self.source_type != entry.source_type):
            raise ValueError("Candidate differs from the canonical catalogue")
        if entry.position >= snapshot.document_count or self.end_char > entry.text_length:
            raise ValueError("Candidate span or position is outside the catalogue")


class ReadFilters(Contract):
    """Additional restrictions; filters can only narrow a server-resolved scope.

    Unknown timestamps fail any date filter. Date bounds are inclusive, matching
    the existing SearchRequest semantics. These fields never confer authority.
    """

    source_types: tuple[SourceType, ...] = Field(default=(), max_length=9)
    after: AwareDatetime | None = None
    before: AwareDatetime | None = None

    @model_validator(mode="after")
    def valid_restrictions(self):
        if self.after and self.before and self.after > self.before:
            raise ValueError("after must not exceed before")
        if len(self.source_types) != len(set(self.source_types)):
            raise ValueError("Duplicate source filters")
        return self

    def matches(self, entry: CatalogEntry) -> bool:
        if self.source_types and entry.source_type not in self.source_types:
            return False
        if self.after and (entry.timestamp is None or entry.timestamp < self.after):
            return False
        if self.before and (entry.timestamp is None or entry.timestamp > self.before):
            return False
        return True


def validate_catalog_batch_size(batch_size: int) -> None:
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_CATALOG_BATCH:
        raise ValueError(f"Catalogue batch size must be between 1 and {MAX_CATALOG_BATCH}")


def validate_read_more_bounds(anchor_count: int, max_chars: int) -> None:
    if anchor_count > MAX_READ_MORE_ANCHORS:
        raise ValueError("Document expansion accepts at most two anchors")
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_READ_MORE_CHARS:
        raise ValueError("Document expansion requires 1 to 6000 characters per document")


@runtime_checkable
class DocumentStore(Protocol):
    """A trusted adapter for a single immutable canonical snapshot.

    Metadata methods are server-only. Concrete adapters must reject mismatching
    corpus/catalog identities and verify canonical revision/text before serving
    excerpts. Reads call guard.assert_current(context) and context.assert_bound()
    before text access and again before release; unavailable policy fails closed.
    PostgreSQL, RLS and migration behavior are deliberately outside this ticket.
    """

    def snapshot(self) -> SnapshotRef: ...

    def iter_catalog(self, *, batch_size: int = MAX_CATALOG_BATCH) -> Iterator[tuple[CatalogEntry, ...]]:
        """Yield position-ordered metadata batches of 1..batch_size entries.

        batch_size is a strict integer in [1, 1000]; an empty catalogue yields
        no batches. Positions are contiguous, unique and match the snapshot's
        document_count. No complete document bodies are returned.
        """
        ...

    def source_types(self) -> tuple[str, ...]:
        """Return unique sorted internal platform identifiers, without text."""
        ...

    def read_passage(
        self, context: RequestContext, candidate: RankedCandidate, *,
        guard: FreshnessGuard, filters: ReadFilters = ReadFilters(),
    ) -> PassageEvidence:
        """Read one exact authorized Unicode-character span.

        Fail closed on denied/missing documents, stale scopes, snapshot/revision
        mismatch, excluded source/date or invalid span. citation_id and chunk_id
        are exactly doc_id:start_char:end_char; snapshot/revision are separate.
        """
        ...

    def read_more(
        self, context: RequestContext, anchors: tuple[PassageEvidence, ...], *,
        guard: FreshnessGuard, max_chars: int = DEFAULT_READ_MORE_CHARS,
        filters: ReadFilters = ReadFilters(),
    ) -> tuple[PassageEvidence, ...]:
        """Read one adjacent exact window per matched authorized document.

        At most two anchors; strict max_chars in [1, 6000], default 3000, per
        returned document. Validate every anchor's scope/snapshot/revision;
        verify eligible anchors against canonical text before expansion. Never
        discover new IDs. Excluded source/date anchors yield no passage and do
        not require a text read. Deduplicate documents in input order. Return ()
        if no adjacent text remains; overlap is allowed but must make progress.
        Preserve anchor score/method as provenance, not a new relevance score.
        """
        ...

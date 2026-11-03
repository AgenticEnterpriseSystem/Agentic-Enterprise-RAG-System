"""Conformance examples for immutable metadata and pre-text authorization.

The eager fake is deliberately tiny. These tests establish interface semantics,
not PostgreSQL integration, ranking parity, policy accuracy or memory feasibility.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, RequestContext,
    ResolvedIdentity, SnapshotMismatch, StaleContext,
)
from src.contracts_v2 import ScopeBinding, SnapshotRef
from src.storage.contracts import CatalogEntry, DocumentStore, RankedCandidate, ReadFilters
from src.storage.testing import FixtureDocument, InMemoryDocumentStore

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def snapshot(count=2, **changes):
    return SnapshotRef(snapshot_id="fixture-snapshot", corpus_sha256="a" * 64,
        catalog_sha256="b" * 64, document_count=count, **changes)


def document(doc_id, text, position, source="jira", timestamp=None):
    return FixtureDocument(CatalogEntry(doc_id=doc_id, revision_id="rev-" + doc_id,
        position=position, source_type=source, text_length=len(text), timestamp=timestamp,
        metadata_origin="synthetic"), text, title="Title " + doc_id)


def context(snap, allowed=("a", "b")):
    binding = ScopeBinding(scope_id="scope", user_id="intern", snapshot=snap,
                           policy_revision=3, session_version=2)
    return RequestContext(identity=ResolvedIdentity(user_id="intern", session_id="row-id",
        session_version=2, expires_at=NOW + timedelta(hours=1)),
        scope=LegacySetScope(binding=binding, allowed_doc_ids=frozenset(allowed)))


class Guard:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def assert_current(self, request):
        self.calls += 1
        if self.calls == self.fail_at:
            raise StaleContext("Session changed")


@pytest.fixture
def store():
    return InMemoryDocumentStore(snapshot(), (
        document("a", "Title a\nnaïve🙂 e\u0301 and canonical\nline endings. " * 20, 0),
        document("b", "Title b\nA different document.", 1, source="gmail", timestamp=NOW),
    ))


def candidate(store, doc_id="a", start=0, end=12, **updates):
    entry = next(e for batch in store.iter_catalog() for e in batch if e.doc_id == doc_id)
    values = dict(snapshot=store.snapshot(), doc_id=entry.doc_id, revision_id=entry.revision_id,
        source_type=entry.source_type, start_char=start, end_char=end, score=0.75,
        retrieval_method="dense", scoring_provenance="fixture-index-v1")
    values.update(updates)
    return RankedCandidate(**values)


def test_metadata_contracts_are_frozen_and_reject_invalid_numeric_or_span_values(store):
    entry = next(iter(store.iter_catalog()))[0]
    with pytest.raises(ValidationError):
        entry.position = 2
    with pytest.raises(ValidationError):
        CatalogEntry(**{**entry.model_dump(), "position": -1})
    for field, value in (("start_char", -1), ("start_char", True), ("end_char", 0),
                         ("end_char", 1.5), ("end_char", 0), ("score", "0.75"),
                         ("score", float("nan")), ("score", float("inf"))):
        with pytest.raises(ValidationError):
            candidate(store, **{field: value})
    with pytest.raises(ValidationError):
        candidate(store, start=20, end=10)


def test_unicode_character_spans_citations_and_provenance_are_exact(store):
    request, guard = context(store.snapshot()), Guard()
    result = store.read_passage(request, candidate(store, start=8, end=21), guard=guard)
    original = store._documents["a"].text
    assert result.chunk.text == original[8:21]
    assert len(result.chunk.text) == 13
    assert "🙂" in result.chunk.text and "e\u0301" in result.chunk.text
    assert result.citation_id == result.chunk.chunk_id == "a:8:21"
    assert result.snapshot_id == store.snapshot().snapshot_id and result.revision_id == "rev-a"
    assert result.chunk.metadata_origin == "synthetic" and result.retrieval_method == "dense"
    assert result.score == .75 and guard.calls >= 2


@pytest.mark.parametrize("allowed", [(), ("b",)])
def test_denied_read_never_loads_body(store, allowed):
    with pytest.raises(AccessDenied):
        store.read_passage(context(store.snapshot(), allowed), candidate(store), guard=Guard())
    assert store.body_reads == []


def test_freshness_check_precedes_body_and_is_repeated_before_release(store):
    with pytest.raises(StaleContext):
        store.read_passage(context(store.snapshot()), candidate(store), guard=Guard(fail_at=1))
    assert store.body_reads == []
    # Entry and authorization checks use the first two guard calls. The third
    # checks that loaded text is still releasable after a possible I/O interval.
    with pytest.raises(StaleContext):
        store.read_passage(context(store.snapshot()), candidate(store), guard=Guard(fail_at=3))
    assert store.body_reads == ["a"]


@pytest.mark.parametrize("changes", [
    {"revision_id": "other-revision"}, {"source_type": "slack"}, {"end_char": 100000},
])
def test_revision_source_and_length_mismatch_fail_before_body(store, changes):
    with pytest.raises(EvidenceValidationError):
        store.read_passage(context(store.snapshot()), candidate(store, **changes), guard=Guard())
    assert not store.body_reads


def test_snapshot_checks_compare_fingerprints_not_only_snapshot_name(store):
    altered = store.snapshot().model_copy(update={"catalog_sha256": "c" * 64})
    with pytest.raises(SnapshotMismatch):
        store.read_passage(context(store.snapshot()), candidate(store, snapshot=altered), guard=Guard())
    with pytest.raises(SnapshotMismatch):
        store.read_passage(context(altered), candidate(store), guard=Guard())
    assert not store.body_reads


def test_model_copy_cannot_bypass_candidate_or_filter_validation(store):
    invalid = candidate(store).model_copy(update={"score": float("nan")})
    with pytest.raises(EvidenceValidationError):
        store.read_passage(context(store.snapshot()), invalid, guard=Guard())
    invalid_filters = ReadFilters().model_copy(update={"source_types": ("jira",) * 10})
    with pytest.raises(EvidenceValidationError):
        store.read_passage(context(store.snapshot()), candidate(store), guard=Guard(), filters=invalid_filters)
    assert not store.body_reads


@pytest.mark.parametrize("filters", [ReadFilters(source_types=("gmail",)),
    ReadFilters(after=NOW), ReadFilters(before=NOW)])
def test_unknown_dates_and_excluded_sources_fail_before_body(store, filters):
    with pytest.raises(AccessDenied):
        store.read_passage(context(store.snapshot()), candidate(store), guard=Guard(), filters=filters)
    assert not store.body_reads


def test_date_bounds_are_inclusive_and_filters_are_frozen(store):
    limits = ReadFilters(source_types=("gmail",), after=NOW, before=NOW)
    result = store.read_passage(context(store.snapshot()), candidate(store, doc_id="b"),
                               guard=Guard(), filters=limits)
    assert result.chunk.doc_id == "b"
    with pytest.raises(ValidationError):
        ReadFilters(after=NOW, before=NOW - timedelta(seconds=1))
    with pytest.raises(ValidationError):
        ReadFilters(source_types=("jira", "jira"))
    with pytest.raises(ValidationError):
        ReadFilters(source_types=tuple(str(i) for i in range(10)))


def test_bounded_catalogue_has_stable_order_no_bodies_and_empty_case(store):
    assert isinstance(store, DocumentStore)
    batches = tuple(store.iter_catalog(batch_size=1))
    assert len(batches) == 2 and all(len(batch) == 1 for batch in batches)
    assert [batch[0].position for batch in batches] == [0, 1]
    assert all("text" not in batch[0].model_dump() for batch in batches)
    assert store.source_types() == ("gmail", "jira")
    for value in (0, -1, 1001, True, 1.5):
        with pytest.raises(ValueError):
            tuple(store.iter_catalog(batch_size=value))
    empty = InMemoryDocumentStore(snapshot(0), ())
    assert tuple(empty.iter_catalog()) == () and empty.source_types() == ()
    assert not store.body_reads


def test_fixture_rejects_count_position_duplicates_and_wrong_text_length():
    a, b = document("a", "a", 0), document("b", "b", 1)
    with pytest.raises(ValueError, match="count"):
        InMemoryDocumentStore(snapshot(), (a,))
    with pytest.raises(ValueError, match="positions"):
        InMemoryDocumentStore(snapshot(), (a, FixtureDocument(b.entry.model_copy(update={"position": 0}), "b")))
    with pytest.raises(ValueError, match="Duplicate"):
        InMemoryDocumentStore(snapshot(), (a, document("a", "other", 1)))
    with pytest.raises(ValueError, match="length"):
        InMemoryDocumentStore(snapshot(1), (FixtureDocument(a.entry, "different"),))


def test_read_more_is_bounded_adjacent_and_keeps_exact_offsets_and_anchor_score(store):
    request, guard = context(store.snapshot()), Guard()
    anchor = store.read_passage(request, candidate(store, start=0, end=30), guard=guard)
    result = store.read_more(request, (anchor,), guard=guard, max_chars=80)
    assert len(result) == 1
    item = result[0]
    assert len(item.chunk.text) <= 80 and item.chunk.end_char > anchor.chunk.end_char
    assert item.chunk.text == store._documents["a"].text[item.chunk.start_char:item.chunk.end_char]
    assert item.citation_id == f"a:{item.chunk.start_char}:{item.chunk.end_char}"
    assert item.revision_id == anchor.revision_id and item.score == anchor.score
    assert item.retrieval_method == anchor.retrieval_method
    assert len(store.read_more(request, (anchor, anchor), guard=guard, max_chars=80)) == 1


def test_read_more_authorizes_all_anchors_before_any_body(store):
    permitted = context(store.snapshot())
    anchors = tuple(store.read_passage(permitted, candidate(store, doc_id=doc), guard=Guard())
                    for doc in ("a", "b"))
    store.body_reads.clear()
    with pytest.raises(AccessDenied):
        store.read_more(context(store.snapshot(), ("a",)), anchors, guard=Guard())
    assert not store.body_reads


def test_read_more_rejects_tampered_anchor_and_bounds(store):
    request = context(store.snapshot())
    anchor = store.read_passage(request, candidate(store), guard=Guard())
    for changes in ({"snapshot_id": "other-snapshot"}, {"revision_id": "other-revision"},
                    {"citation_id": "invented"}):
        with pytest.raises((SnapshotMismatch, EvidenceValidationError)):
            store.read_more(request, (anchor.model_copy(update=changes),), guard=Guard())
    bad_text = anchor.model_copy(update={"chunk": anchor.chunk.model_copy(update={"text": "?" * len(anchor.chunk.text)})})
    with pytest.raises(EvidenceValidationError, match="canonical text"):
        store.read_more(request, (bad_text,), guard=Guard())
    for value in (0, -1, 6001, True, 1.5):
        with pytest.raises(ValueError):
            store.read_more(request, (anchor,), guard=Guard(), max_chars=value)
    with pytest.raises(ValueError):
        store.read_more(request, (anchor,) * 3, guard=Guard())


def test_read_more_unknown_dates_skip_bodies_and_whole_document_has_no_expansion(store):
    request = context(store.snapshot())
    anchor = store.read_passage(request, candidate(store), guard=Guard())
    store.body_reads.clear()
    assert store.read_more(request, (anchor,), guard=Guard(), filters=ReadFilters(after=NOW)) == ()
    assert not store.body_reads
    full = store.read_passage(request, candidate(store, doc_id="b", end=len(store._documents["b"].text)), guard=Guard())
    assert store.read_more(request, (full,), guard=Guard()) == ()

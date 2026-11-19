"""Storage conformance and exact streaming fingerprint examples.

The eager fake is deliberately tiny. These tests establish interface semantics,
not PostgreSQL integration, ranking parity, policy accuracy or memory feasibility.
"""

from datetime import datetime, timedelta, timezone
from itertools import chain

import pytest
from pydantic import ValidationError

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, RequestContext,
    ResolvedIdentity, SnapshotMismatch, StaleContext,
)
from src.contracts import ScopeBinding, SnapshotRef
from src.storage.contracts import CatalogEntry, DocumentStore, RankedCandidate, ReadFilters
from src.storage.testing import FixtureDocument, InMemoryDocumentStore
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint

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


# Frozen payload (UTF-8, LF after prefix and after each array):
# rag-catalog-v2
# [0,"doc-a","rev-a","jira",7,"2026-11-02T09:30:00.123456Z","synthetic"]
# [1,"doc-é","rev-b","confluence",0,null,"upstream"]
GOLDEN_CATALOG_SHA256 = "1a95fb73b7814b1dd1063964dc37fd9c15616ffe05090a37095a942f6da5ea40"


def fixture_entries():
    return (
        CatalogEntry(position=0, doc_id="doc-a", revision_id="rev-a", source_type="jira",
            text_length=7, timestamp=datetime(2026, 11, 2, 9, 30, 0, 123456, tzinfo=timezone.utc),
            metadata_origin="synthetic"),
        CatalogEntry(position=1, doc_id="doc-é", revision_id="rev-b", source_type="confluence",
            text_length=0),
    )


def test_golden_catalogue_digest_freezes_utf8_array_field_order_and_timestamp_codec():
    assert catalog_fingerprint(fixture_entries(), expected_count=2) == GOLDEN_CATALOG_SHA256


def test_single_pass_iterable_does_not_require_length_or_a_second_traversal():
    class SinglePass:
        used = False

        def __len__(self):
            raise AssertionError("Streaming codec must not request catalogue length")

        def __iter__(self):
            if self.used:
                raise AssertionError("Streaming codec traversed the catalogue twice")
            self.used = True
            yield from fixture_entries()

    assert catalog_fingerprint(SinglePass(), expected_count=2) == GOLDEN_CATALOG_SHA256


def test_partitioning_batches_does_not_change_the_fingerprint():
    a, b = fixture_entries()
    for batches in (((a,), (b,)), ((a, b),), ((), (a,), (), (b,), ())):
        stream = chain.from_iterable(iter(batch) for batch in batches)
        assert catalog_fingerprint(stream, expected_count=2) == GOLDEN_CATALOG_SHA256


def test_equivalent_timestamp_offsets_encode_identical_utc_microseconds():
    a, b = fixture_entries()
    local = datetime(2026, 11, 2, 1, 30, 0, 123456, tzinfo=timezone(timedelta(hours=-8)))
    equivalent = a.model_copy(update={"timestamp": local})
    assert catalog_fingerprint((equivalent, b), expected_count=2) == GOLDEN_CATALOG_SHA256


@pytest.mark.parametrize("changes", [
    {"doc_id": "doc-other"}, {"revision_id": "rev-other"}, {"source_type": "slack"},
    {"text_length": 8}, {"metadata_origin": "upstream"}, {"timestamp": None},
    {"timestamp": datetime(2026, 11, 2, 9, 30, 0, 123457, tzinfo=timezone.utc)},
])
def test_authoritative_metadata_changes_change_the_digest(changes):
    a, b = fixture_entries()
    assert catalog_fingerprint((a.model_copy(update=changes), b), expected_count=2) != GOLDEN_CATALOG_SHA256


def test_reordered_document_positions_change_digest_and_unsorted_positions_reject():
    a, b = fixture_entries()
    with pytest.raises(ValueError, match="ordered"):
        catalog_fingerprint((b, a), expected_count=2)
    reordered = (b.model_copy(update={"position": 0}), a.model_copy(update={"position": 1}))
    assert catalog_fingerprint(reordered, expected_count=2) != GOLDEN_CATALOG_SHA256


@pytest.mark.parametrize("count", [-1, True, False, 2.0, "2", None])
def test_expected_count_is_a_strict_nonnegative_integer(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        catalog_fingerprint((), expected_count=count)


def test_missing_excess_and_noncontiguous_catalogues_reject():
    a, b = fixture_entries()
    with pytest.raises(ValueError, match="missing"):
        catalog_fingerprint((a,), expected_count=2)
    with pytest.raises(ValueError, match="excess"):
        catalog_fingerprint((a,), expected_count=0)
    for positions in ((1, 2), (0, 0), (0, 2)):
        entries = tuple(e.model_copy(update={"position": p}) for e, p in zip((a, b), positions))
        with pytest.raises(ValueError, match="contiguous"):
            catalog_fingerprint(entries, expected_count=2)


def test_excess_record_stops_consumption_without_buffering_the_remaining_stream():
    def too_many():
        yield from fixture_entries()
        yield fixture_entries()[0].model_copy(update={"position": 2})
        raise AssertionError("Codec consumed past the first excess row")

    with pytest.raises(ValueError, match="excess"):
        catalog_fingerprint(too_many(), expected_count=2)


def test_copied_invalid_entries_are_revalidated():
    a, b = fixture_entries()
    for changes in ({"position": True}, {"text_length": -1}, {"metadata_origin": "untrusted"},
                    {"timestamp": datetime(2026, 11, 2)}):
        with pytest.raises(ValidationError):
            catalog_fingerprint((a.model_copy(update=changes), b), expected_count=2)
    with pytest.raises(ValueError, match="CatalogEntry"):
        catalog_fingerprint((a.model_dump(), b), expected_count=2)


def test_duplicate_id_uniqueness_is_a_separate_storage_constraint():
    a, b = fixture_entries()
    duplicate = b.model_copy(update={"doc_id": a.doc_id})
    assert len(catalog_fingerprint((a, duplicate), expected_count=2)) == 64


def test_revision_hash_preserves_unicode_normalization_newlines_whitespace_and_empty_text():
    assert canonical_revision_id("é") == "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c"
    assert canonical_revision_id("e\u0301") == "bf12767b0f2a56b2190075bae8169f656e3ce8d6357d4aff184bc6c7ea48f9f6"
    assert canonical_revision_id("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    for different in (" text", "text ", "text\n", "text\r\n", "\ufefftext"):
        assert canonical_revision_id(different) != canonical_revision_id("text")
    with pytest.raises(TypeError):
        canonical_revision_id(b"text")
    with pytest.raises(UnicodeEncodeError):
        canonical_revision_id("\ud800")


def test_catalogue_preserves_unicode_ids_and_empty_catalogue_has_versioned_digest():
    a, b = fixture_entries()
    decomposed = b.model_copy(update={"doc_id": "doc-e\u0301"})
    assert catalog_fingerprint((a, decomposed), expected_count=2) != GOLDEN_CATALOG_SHA256
    assert catalog_fingerprint((), expected_count=0) == "b42c8d52450c8f3b2e96a2db3aff36942b9cfdfda995b9746b87c9f86043ddbc"

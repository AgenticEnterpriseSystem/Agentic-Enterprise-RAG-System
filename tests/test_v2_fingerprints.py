"""Golden codec examples: exact bytes and one-pass bounded catalogue traversal."""

from datetime import datetime, timedelta, timezone
from itertools import chain

import pytest
from pydantic import ValidationError

from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint

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

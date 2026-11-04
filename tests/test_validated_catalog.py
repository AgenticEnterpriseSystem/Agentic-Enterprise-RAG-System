"""Metadata-capability integrity; a validated catalog never grants body access."""

import copy
from datetime import datetime, timezone
import pickle
from types import MappingProxyType, SimpleNamespace

import pytest

from src.access.contracts import EvidenceValidationError, SnapshotMismatch
from src.contracts import SnapshotRef
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import ValidatedCatalog, canonical_revision_id, catalog_fingerprint


ENTRIES = (
    CatalogEntry(doc_id="zeta", position=0, revision_id=canonical_revision_id("  Café 🧭\n"),
        source_type="jira", text_length=len("  Café 🧭\n"), metadata_origin="synthetic"),
    CatalogEntry(doc_id="café-e\u0301", position=1, revision_id=canonical_revision_id("Second\r\n"),
        source_type="slack", text_length=len("Second\r\n"),
        timestamp=datetime(2026, 11, 3, 12, tzinfo=timezone.utc)),
    CatalogEntry(doc_id="last", position=2, revision_id=canonical_revision_id(""),
        source_type="jira", text_length=0, metadata_origin="extracted"),
)


def snapshot_for(entries=ENTRIES, **changes):
    values = dict(snapshot_id="validated-catalog-fixture", corpus_sha256="a" * 64,
        catalog_sha256=catalog_fingerprint(entries, expected_count=len(entries)), document_count=len(entries))
    values.update(changes)
    return SnapshotRef(**values)


def test_build_preserves_exact_order_metadata_and_single_validation_per_entry(monkeypatch):
    issuer, snapshot, seen = object(), snapshot_for(), []
    original = CatalogEntry.model_validate

    def tracked(cls, value, *args, **kwargs):
        seen.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(CatalogEntry, "model_validate", classmethod(tracked))
    proof = ValidatedCatalog.build(snapshot, iter(ENTRIES), issuer=issuer)
    assert len(seen) == len(ENTRIES)
    assert type(proof) is ValidatedCatalog
    assert proof.require(snapshot, issuer=issuer) is proof
    assert proof.require(snapshot) is proof
    assert proof.snapshot == snapshot and proof.snapshot is not snapshot
    assert type(proof.entries) is tuple and proof.entries == ENTRIES
    assert all(stored is not supplied for stored, supplied in zip(proof.entries, ENTRIES))
    assert proof.doc_ids == ("zeta", "café-e\u0301", "last")
    assert proof.sources == ("jira", "slack", "jira")
    assert proof.source_types == ("jira", "slack")
    assert type(proof.positions) is MappingProxyType
    assert dict(proof.positions) == {"zeta": 0, "café-e\u0301": 1, "last": 2}
    assert proof.positions.get("unknown") is None


def test_builder_owns_clones_of_mutable_input_container_models_and_snapshot():
    supplied = [CatalogEntry.model_validate(entry.model_dump()) for entry in ENTRIES]
    snapshot, issuer = snapshot_for(), object()
    expected = SnapshotRef.model_validate(snapshot.model_dump())
    proof = ValidatedCatalog.build(snapshot, supplied, issuer=issuer)
    # Bypass the caller model's frozen setter to demonstrate that retaining the
    # supplied model object itself would leave a mutable alias into the proof.
    object.__setattr__(supplied[0], "doc_id", "caller-changed-id")
    object.__setattr__(snapshot, "corpus_sha256", "f" * 64)
    supplied.reverse()
    supplied.append(ENTRIES[0])
    assert proof.entries == ENTRIES and proof.doc_ids == tuple(entry.doc_id for entry in ENTRIES)
    assert dict(proof.positions) == {entry.doc_id: entry.position for entry in ENTRIES}
    assert proof.require(expected, issuer=issuer) is proof


@pytest.mark.parametrize("attribute,value", [
    ("entries", ()), ("doc_ids", ()), ("sources", ()), ("source_types", ()),
    ("positions", {}), ("snapshot", snapshot_for(())),
])
def test_capability_attributes_and_owned_mapping_are_read_only(attribute, value):
    proof = ValidatedCatalog.build(snapshot_for(), ENTRIES, issuer=object())
    with pytest.raises(AttributeError):
        setattr(proof, attribute, value)
    with pytest.raises(TypeError):
        proof.positions["injected"] = 0
    with pytest.raises(ValueError):
        proof.entries[0].doc_id = "injected"


def test_constructor_subclass_and_unissued_exact_class_cannot_attest_validity():
    snapshot, issuer = snapshot_for(), object()
    with pytest.raises(TypeError):
        ValidatedCatalog(snapshot, ENTRIES, issuer=issuer)
    with pytest.raises(TypeError):
        type("ForgedCatalogSubclass", (ValidatedCatalog,), {})
    unissued = object.__new__(ValidatedCatalog)
    issued = ValidatedCatalog.build(snapshot, ENTRIES, issuer=issuer)
    # Even a field-for-field forged exact-class instance is not an issued
    # capability; a copied private boolean or token must not establish trust.
    for slot in ValidatedCatalog.__slots__:
        if slot != "__weakref__":
            object.__setattr__(unissued, slot, getattr(issued, slot))
    with pytest.raises(EvidenceValidationError):
        ValidatedCatalog.require(unissued, snapshot, issuer=issuer)
    lookalike = SimpleNamespace(snapshot=snapshot, entries=ENTRIES,
        doc_ids=tuple(entry.doc_id for entry in ENTRIES), positions={entry.doc_id: entry.position for entry in ENTRIES},
        require=lambda *args, **kwargs: lookalike)
    with pytest.raises(EvidenceValidationError):
        ValidatedCatalog.require(lookalike, snapshot, issuer=issuer)


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy, pickle.dumps], ids=["copy", "deepcopy", "pickle"])
def test_issued_capability_cannot_be_copied_or_serialized_as_a_new_attestation(copier):
    proof = ValidatedCatalog.build(snapshot_for(), ENTRIES, issuer=object())
    with pytest.raises(TypeError):
        copier(proof)


def test_exact_issuer_identity_is_required_even_for_equivalent_objects():
    class Issuer:
        def __eq__(self, other):
            return isinstance(other, Issuer)

    issuer, other = Issuer(), Issuer()
    assert issuer == other and issuer is not other
    proof = ValidatedCatalog.build(snapshot_for(), ENTRIES, issuer=issuer)
    with pytest.raises(EvidenceValidationError):
        proof.require(snapshot_for(), issuer=other)
    assert proof.require(snapshot_for(), issuer=issuer) is proof
    with pytest.raises((ValueError, TypeError)):
        ValidatedCatalog.build(snapshot_for(), ENTRIES, issuer=None)


@pytest.mark.parametrize("changes", [
    {"snapshot_id": "another-snapshot"}, {"corpus_sha256": "b" * 64},
    {"catalog_sha256": "c" * 64}, {"document_count": 4},
])
def test_require_binds_every_snapshot_field(changes):
    snapshot, issuer = snapshot_for(), object()
    proof = ValidatedCatalog.build(snapshot, ENTRIES, issuer=issuer)
    with pytest.raises(SnapshotMismatch):
        proof.require(snapshot.model_copy(update=changes), issuer=issuer)


@pytest.mark.parametrize("changes", [
    {"document_count": True}, {"document_count": -1}, {"catalog_sha256": "invalid"}, {"snapshot_id": ""},
])
def test_require_and_build_revalidate_invalid_snapshot_model_copies_before_consuming_rows(changes):
    snapshot, issuer = snapshot_for(), object()
    proof = ValidatedCatalog.build(snapshot, ENTRIES, issuer=issuer)
    invalid = snapshot.model_copy(update=changes)
    with pytest.raises(ValueError):
        proof.require(invalid, issuer=issuer)
    consumed = []
    def rows():
        consumed.append(True)
        yield from ENTRIES
    with pytest.raises(ValueError):
        ValidatedCatalog.build(invalid, rows(), issuer=issuer)
    assert consumed == []


@pytest.mark.parametrize("changes", [
    {"position": -1}, {"position": True}, {"doc_id": ""}, {"source_type": ""},
    {"text_length": True}, {"timestamp": datetime(2026, 11, 3)}, {"metadata_origin": "inferred-authority"},
])
def test_invalid_catalog_model_copy_cannot_bypass_entry_validation(changes):
    rows = (ENTRIES[0].model_copy(update=changes), *ENTRIES[1:])
    with pytest.raises(ValueError):
        ValidatedCatalog.build(snapshot_for(), rows, issuer=object())


@pytest.mark.parametrize("rows", [ENTRIES[1:], ENTRIES + (ENTRIES[0],), tuple(reversed(ENTRIES)),
    (ENTRIES[0], ENTRIES[1].model_copy(update={"position": 2}), ENTRIES[2])],
    ids=["missing", "excess", "reversed", "position-gap"])
def test_missing_excess_or_reordered_catalog_cannot_be_issued(rows):
    with pytest.raises(ValueError):
        ValidatedCatalog.build(snapshot_for(), rows, issuer=object())


def test_duplicate_document_ids_reject_even_when_the_codec_digest_matches():
    rows = (ENTRIES[0], ENTRIES[1].model_copy(update={"doc_id": ENTRIES[0].doc_id}), ENTRIES[2])
    matching_digest_snapshot = snapshot_for(rows)
    with pytest.raises(ValueError):
        ValidatedCatalog.build(matching_digest_snapshot, rows, issuer=object())


def test_hash_mismatch_rejects_an_otherwise_valid_complete_catalog():
    with pytest.raises(SnapshotMismatch):
        ValidatedCatalog.build(snapshot_for(catalog_sha256="f" * 64), ENTRIES, issuer=object())


def test_zero_document_snapshot_has_a_valid_empty_immutable_capability():
    snapshot, issuer = snapshot_for(()), object()
    proof = ValidatedCatalog.build(snapshot, iter(()), issuer=issuer)
    assert proof.require(snapshot, issuer=issuer) is proof
    assert proof.entries == proof.doc_ids == proof.sources == proof.source_types == ()
    assert dict(proof.positions) == {} and proof.snapshot.document_count == 0
    with pytest.raises(ValueError):
        ValidatedCatalog.build(snapshot, ENTRIES[:1], issuer=issuer)

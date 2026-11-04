"""Frozen V2 canonical-text and catalogue fingerprint codecs.

These functions serialize existing metadata; they perform no import, policy,
ranking, or database work. A future importer must additionally enforce unique
document IDs. Keeping a global ID set here would defeat bounded streaming.
"""

from collections.abc import Iterable
from datetime import timezone
from hashlib import sha256
import json
from threading import Lock
from types import MappingProxyType
from weakref import WeakSet

from src.access.contracts import EvidenceValidationError, SnapshotMismatch
from src.contracts import SnapshotRef
from src.storage.contracts import CatalogEntry

CATALOG_FINGERPRINT_PREFIX = b"rag-catalog-v2\n"
_ISSUED_CATALOGS = WeakSet()
_ISSUANCE_LOCK = Lock()


def canonical_revision_id(text: str) -> str:
    """SHA256 hex of exact UTF-8 canonical text, without any normalization.

    Newlines, BOMs, whitespace and Unicode normalization forms are significant.
    Empty text is valid. Text containing invalid UTF-8 surrogate code points is
    rejected by strict encoding rather than being silently replaced.
    """
    if not isinstance(text, str):
        raise TypeError("Canonical text must be a string")
    return sha256(text.encode("utf-8")).hexdigest()


def catalog_fingerprint(entries: Iterable[CatalogEntry], *, expected_count: int) -> str:
    """Stream the versioned, position-ordered V2 catalogue to a SHA256 hex digest.

    Start with b'rag-catalog-v2\\n'. Each entry is a compact UTF-8 JSON array:
    [position, doc_id, revision_id, source_type, text_length, timestamp,
     metadata_origin], followed by one LF byte. JSON uses ensure_ascii=False,
    separators=(',', ':'), and disallows non-finite numbers. A timestamp is UTC
    ISO 8601 with exactly six fractional digits and a final Z, or JSON null.

    expected_count is a strict nonnegative integer. Positions must be exactly
    0..expected_count-1; missing/excess records reject. Revalidate model copies
    before serialization. The iterable is consumed once, without buffering the
    catalogue or tracking global IDs; duplicate-ID constraints belong to storage.
    """
    return _fingerprint_validated(_validated_entries(entries), expected_count=expected_count)


def _validated_entries(entries):
    for supplied in entries:
        if not isinstance(supplied, CatalogEntry):
            raise ValueError("Catalogue entries must be CatalogEntry records")
        yield CatalogEntry.model_validate(supplied.model_dump())


def _fingerprint_validated(entries, *, expected_count):
    """The original codec over already cloned/validated entries; no trust flag."""
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("Expected catalogue count must be a nonnegative integer")
    digest = sha256(CATALOG_FINGERPRINT_PREFIX)
    count = 0
    for position, entry in enumerate(entries):
        if position >= expected_count:
            raise ValueError("Catalogue contains excess records")
        if entry.position != position:
            raise ValueError("Catalogue positions must be contiguous and ordered from zero")
        timestamp = (entry.timestamp.astimezone(timezone.utc).isoformat(timespec="microseconds")[:-6] + "Z"
                     if entry.timestamp is not None else None)
        record = [entry.position, entry.doc_id, entry.revision_id, entry.source_type,
                  entry.text_length, timestamp, entry.metadata_origin]
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    if count != expected_count:
        raise ValueError("Catalogue is missing expected records")
    return digest.hexdigest()


class ValidatedCatalog:
    """Process-local immutable metadata proof, never permission or freshness.

    build clones and verifies the complete ordered catalogue with the unchanged
    fingerprint codec. Consumers can then share its tuple/map without another
    validation pass. Issuance identity prevents constructed/copied lookalikes
    from entering the trusted path. This is not a serialization format.
    """

    __slots__ = ("_snapshot", "_entries", "_doc_ids", "_sources", "_positions",
                 "_source_types", "_issuer", "__weakref__")

    def __new__(cls, *arguments, **keywords):
        raise TypeError("Validated catalogues must be built from verified metadata")

    def __init_subclass__(cls, **keywords):
        raise TypeError("Validated catalogues cannot be subclassed")

    def __setattr__(self, name, value):
        raise AttributeError("Validated catalogue metadata is immutable")

    def __delattr__(self, name):
        raise AttributeError("Validated catalogue metadata is immutable")

    def __copy__(self):
        raise TypeError("Validated catalogue capabilities cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Validated catalogue capabilities cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("Validated catalogue capabilities cannot be serialized")

    @classmethod
    def build(cls, snapshot: SnapshotRef, entries: Iterable[CatalogEntry], *, issuer: object):
        if cls is not ValidatedCatalog:
            raise TypeError("Validated catalogue requires the concrete proof type")
        if not isinstance(snapshot, SnapshotRef) or issuer is None:
            raise ValueError("Validated catalogue requires a snapshot and issuing store")
        snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        retained, ids, sources, positions, canonical_sources = [], [], [], {}, {}

        def checked():
            for entry in _validated_entries(entries):
                if entry.doc_id in positions:
                    raise ValueError("Catalogue document IDs must be unique")
                positions[entry.doc_id] = len(retained)
                retained.append(entry)
                ids.append(entry.doc_id)
                sources.append(canonical_sources.setdefault(entry.source_type, entry.source_type))
                yield entry

        digest = _fingerprint_validated(checked(), expected_count=snapshot.document_count)
        if digest != snapshot.catalog_sha256:
            raise SnapshotMismatch("Validated catalogue differs from its snapshot")
        result = object.__new__(ValidatedCatalog)
        for name, value in (
            ("_snapshot", snapshot), ("_entries", tuple(retained)), ("_doc_ids", tuple(ids)),
            ("_sources", tuple(sources)), ("_positions", MappingProxyType(positions)),
            ("_source_types", tuple(sorted(canonical_sources))), ("_issuer", issuer),
        ):
            object.__setattr__(result, name, value)
        with _ISSUANCE_LOCK:
            _ISSUED_CATALOGS.add(result)
        return result

    def require(self, snapshot: SnapshotRef, *, issuer=None):
        if type(self) is not ValidatedCatalog:
            raise EvidenceValidationError("An issued immutable catalogue is required")
        with _ISSUANCE_LOCK:
            issued = self in _ISSUED_CATALOGS
        if not issued:
            raise EvidenceValidationError("An issued immutable catalogue is required")
        if not isinstance(snapshot, SnapshotRef):
            raise EvidenceValidationError("Validated catalogue requires an exact snapshot")
        snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        if snapshot != self._snapshot:
            raise SnapshotMismatch("Validated catalogue belongs to another snapshot")
        if issuer is not None and issuer is not self._issuer:
            raise EvidenceValidationError("Validated catalogue belongs to another issuing store")
        return self

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def entries(self):
        return self._entries

    @property
    def doc_ids(self):
        return self._doc_ids

    @property
    def sources(self):
        return self._sources

    @property
    def positions(self):
        return self._positions

    @property
    def source_types(self):
        return self._source_types

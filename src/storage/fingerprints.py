"""Frozen V2 canonical-text and catalogue fingerprint codecs.

These functions serialize existing metadata; they perform no import, policy,
ranking, or database work. A future importer must additionally enforce unique
document IDs. Keeping a global ID set here would defeat bounded streaming.
"""

from collections.abc import Iterable
from datetime import timezone
from hashlib import sha256
import json

from src.storage.contracts import CatalogEntry

CATALOG_FINGERPRINT_PREFIX = b"rag-catalog-v2\n"


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
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("Expected catalogue count must be a nonnegative integer")
    digest = sha256(CATALOG_FINGERPRINT_PREFIX)
    count = 0
    for position, supplied in enumerate(entries):
        if position >= expected_count:
            raise ValueError("Catalogue contains excess records")
        if not isinstance(supplied, CatalogEntry):
            raise ValueError("Catalogue entries must be CatalogEntry records")
        entry = CatalogEntry.model_validate(supplied.model_dump())
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

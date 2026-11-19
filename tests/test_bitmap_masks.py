"""Exact bitmap decoding against independent per-position bit tests; no database."""

from dataclasses import replace
from types import MappingProxyType

import pytest

from src.access.contracts import EvidenceValidationError, SnapshotMismatch
from src.access.policy import BitmapScope, _Catalog
from src.contracts import ScopeBinding, SnapshotRef


def make_scope(count, allowed=(), *, scope_type=BitmapScope):
    snapshot = SnapshotRef(snapshot_id='bitmap-unit', corpus_sha256='a' * 64,
                           catalog_sha256='b' * 64, document_count=count)
    ids = tuple('document-' + str(index) for index in range(count))
    catalog = _Catalog(snapshot, ids, ('jira',) * count,
                       MappingProxyType({doc: index for index, doc in enumerate(ids)}))
    bits = bytearray((count + 7) // 8)
    for index in allowed:
        bits[index // 8] |= 1 << (index % 8)
    binding = ScopeBinding(scope_id='scope:unit', user_id='synthetic-reader', snapshot=snapshot,
                           policy_revision=0, session_version=0)
    return scope_type(binding, catalog, bytes(bits), 'c' * 64)


def oracle(scope, ids):
    # Per-ID position lookup and independent integer division/remainder.
    result = []
    for doc in ids:
        position = scope._catalog.positions.get(doc)
        result.append(False if position is None else bool((scope.bitmap[position // 8] >> (position % 8)) & 1))
    return tuple(result)


@pytest.mark.parametrize('byte', range(256))
def test_every_byte_matches_independent_little_endian_bit_oracle(byte):
    scope = make_scope(8, (index for index in range(8) if (byte >> index) & 1))
    result = scope.mask_for(scope.binding.snapshot, scope._catalog.doc_ids)
    assert result == oracle(scope, scope._catalog.doc_ids)
    assert type(result) is tuple and all(type(value) is bool for value in result)


@pytest.mark.parametrize('count', [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127])
@pytest.mark.parametrize('pattern', ['none', 'full', 'sparse', 'last'])
def test_count_boundaries_trim_padding_and_keep_exact_tuple_order(count, pattern):
    allowed = (range(count) if pattern == 'full' else range(0, count, 3) if pattern == 'sparse'
               else (count - 1,) if pattern == 'last' and count else ())
    scope = make_scope(count, allowed)
    result = scope.mask_for(scope.binding.snapshot, scope._catalog.doc_ids)
    assert result == oracle(scope, scope._catalog.doc_ids)
    assert len(result) == count and type(result) is tuple
    assert all(type(value) is bool for value in result)


def test_exact_aligned_builtin_scope_avoids_per_id_permits_without_retaining_an_expanded_mask(monkeypatch):
    scope = make_scope(17, (0, 7, 8, 15, 16))
    before = vars(scope).copy()
    def forbidden(*_):
        pytest.fail('Aligned built-in mask performed a per-document lookup')
    monkeypatch.setattr(BitmapScope, 'permits', forbidden)
    expected = (True,) + (False,) * 6 + (True, True) + (False,) * 6 + (True, True)
    assert scope.mask_for(scope.binding.snapshot, scope._catalog.doc_ids) == expected
    assert vars(scope) == before


@pytest.mark.parametrize('selection', ['equal-copy', 'subset', 'reordered', 'duplicates', 'unknown', 'mutable'])
def test_nonidentical_collections_preserve_permits_fallback(selection, monkeypatch):
    scope = make_scope(9, (0, 7, 8))
    ids = scope._catalog.doc_ids
    requested = {'equal-copy': tuple(list(ids)), 'subset': ids[2:8], 'reordered': tuple(reversed(ids)),
                 'duplicates': (ids[8], ids[8], ids[1]), 'unknown': ('unknown', ids[0], 'another'),
                 'mutable': list(ids)}[selection]
    assert requested is not ids
    expected = oracle(scope, requested)
    calls, original = [], BitmapScope.permits
    def observed(self, doc):
        calls.append(doc)
        return original(self, doc)
    monkeypatch.setattr(BitmapScope, 'permits', observed)
    assert scope.mask_for(scope.binding.snapshot, requested) == expected
    assert calls == list(requested)


def test_subclass_custom_permits_is_not_bypassed_even_for_same_catalogue_tuple():
    class Narrower(BitmapScope):
        def permits(self, doc):
            return super().permits(doc) and doc.endswith('-1')
    scope = make_scope(8, range(8), scope_type=Narrower)
    assert scope.mask_for(scope.binding.snapshot, scope._catalog.doc_ids) == (False, True) + (False,) * 6


def test_mutable_requested_ids_and_copied_bitmap_cannot_change_an_earlier_result():
    scope = make_scope(9, (0,))
    requested = list(scope._catalog.doc_ids)
    initial = scope.mask_for(scope.binding.snapshot, requested)
    requested[0] = 'unknown'
    assert scope.mask_for(scope.binding.snapshot, requested) == (False,) * 9
    assert initial == (True,) + (False,) * 8
    with pytest.raises(TypeError):
        initial[0] = False
    changed = replace(scope, bitmap=b'\x00\x01')
    assert changed.mask_for(changed.binding.snapshot, changed._catalog.doc_ids) == (False,) * 8 + (True,)
    assert scope.mask_for(scope.binding.snapshot, scope._catalog.doc_ids) == initial


@pytest.mark.parametrize('changes', [
    {'corpus_sha256': 'd' * 64}, {'catalog_sha256': 'e' * 64}, {'snapshot_id': 'another'}, {'document_count': 8},
])
def test_valid_but_different_snapshot_is_rejected_before_aligned_decode(changes, monkeypatch):
    scope = make_scope(9, range(9))
    monkeypatch.setattr(BitmapScope, 'permits', lambda *_: pytest.fail('Snapshot mismatch reached fallback'))
    different = scope.binding.snapshot.model_copy(update=changes)
    with pytest.raises(SnapshotMismatch):
        scope.mask_for(different, scope._catalog.doc_ids)


@pytest.mark.parametrize('changes', [{'document_count': True}, {'catalog_sha256': 'invalid'}, {'corpus_sha256': None}])
def test_invalid_snapshot_model_copy_is_revalidated_before_fast_path(changes):
    scope = make_scope(9, range(9))
    malformed = scope.binding.snapshot.model_copy(update=changes)
    with pytest.raises(ValueError):
        scope.mask_for(malformed, scope._catalog.doc_ids)


def test_non_snapshot_and_invalid_bitmap_padding_fail_closed():
    scope = make_scope(9, (0, 8))
    with pytest.raises(SnapshotMismatch):
        scope.mask_for(None, scope._catalog.doc_ids)
    with pytest.raises(EvidenceValidationError):
        replace(scope, bitmap=b'\x01\xff')
    with pytest.raises(EvidenceValidationError):
        replace(scope, bitmap=bytearray(scope.bitmap))

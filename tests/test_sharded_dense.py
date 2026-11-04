"""Tiny real-artifact regressions for bounded exact dense retrieval.

The FAISS oracle reproduces the old float32 normalization and max-passage score,
without loading a model, canonical corpus bodies, or any full-size artifact.
"""

from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.access.contracts import LegacySetScope, StaleContext
from src.contracts import ScopeBinding, SnapshotRef
from src.storage.contracts import CatalogEntry, ReadFilters
from src.storage.fingerprints import catalog_fingerprint
from src.retrieval.sharded_dense import ShardedDense, _top_candidates


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _write_metadata(path, records):
    pq.write_table(pa.table({name: [row[index] for row in records]
                            for index, name in enumerate(("doc_index", "start_char", "end_char"))}), path)


def artifact(tmp_path, *, dtype=np.float16, block_rows=2):
    directory = tmp_path / "dense"
    directory.mkdir()
    doc_ids = ("zeta", "alpha", "middle", "secret")
    times = (None, datetime(2026, 10, 1, tzinfo=timezone.utc),
             datetime(2026, 10, 2, tzinfo=timezone.utc), datetime(2026, 10, 3, tzinfo=timezone.utc))
    sources = ("jira", "jira", "slack", "slack")
    catalog = tuple(CatalogEntry(doc_id=doc_id, revision_id=hashlib.sha256(doc_id.encode()).hexdigest(),
        position=position, source_type=sources[position], text_length=100,
        timestamp=times[position], metadata_origin="synthetic") for position, doc_id in enumerate(doc_ids))
    snapshot = SnapshotRef(snapshot_id="tiny-dense", corpus_sha256="a" * 64,
        catalog_sha256=catalog_fingerprint(catalog, expected_count=len(doc_ids)), document_count=len(doc_ids))
    vectors = (
        np.array([[.8, .6, 0], [1, 0, 0], [.3, .4, .5]], dtype=dtype),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=dtype),
    )
    records = (((0, 0, 10), (1, 20, 30), (2, 0, 10)),
               ((0, 40, 50), (3, 0, 10), (1, 0, 10), (1, 0, 8)))
    manifest = {"version": 1, "complete": True, "documents": len(doc_ids), "shards": 2,
        "corpus_sha256": snapshot.corpus_sha256,
        "model": {"name": "fixture-model", "revision": "b" * 40, "normalize": True,
                  "fp16": False, "max_seq_length": 512, "query_prefix": ""}}
    _json(directory / "manifest.json", manifest)
    _json(directory / "doc_ids.json", doc_ids)
    for number, (rows, positions) in enumerate(zip(vectors, records)):
        np.save(directory / f"shard_{number:05d}.npy", rows, allow_pickle=False)
        _write_metadata(directory / f"shard_{number:05d}.parquet", positions)
    return SimpleNamespace(directory=directory, doc_ids=doc_ids, snapshot=snapshot,
        catalog=catalog, vectors=vectors, records=records, manifest=manifest, block_rows=block_rows)


def load(data, **changes):
    return ShardedDense.load(data.directory, snapshot=data.snapshot,
                            catalog=data.catalog, block_rows=data.block_rows, **changes)


def scope(data, allowed=None, *, snapshot=None):
    return LegacySetScope(binding=ScopeBinding(scope_id="test", user_id="tester",
        snapshot=snapshot or data.snapshot, policy_revision=1, session_version=1),
        allowed_doc_ids=frozenset(data.doc_ids if allowed is None else allowed))


def reference(data, vector, top_k, mask=None):
    rows = np.concatenate([np.array(part, dtype=np.float32, copy=True, order="C") for part in data.vectors])
    faiss.normalize_L2(rows)
    index = faiss.IndexFlatIP(rows.shape[1])
    index.add(rows)
    values, positions = index.search(np.asarray(vector, dtype=np.float32).reshape(1, -1), len(rows))
    metadata = [row for shard in data.records for row in shard]
    best = {}
    for value, position in zip(values[0], positions[0]):
        doc_index, start, end = metadata[int(position)]
        if mask is not None and not mask[doc_index]:
            continue
        doc_id = data.doc_ids[doc_index]
        old = best.get(doc_id)
        current = (float(value), start, end)
        if old is None or current[0] > old[0] or (current[0] == old[0] and current[1:] < old[1:]):
            best[doc_id] = current
    return [(doc_id, *score) for doc_id, score in sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("query", [[1, 0, 0], [2, 0, 0], [.2, -.3, .7], [0, 0, 0]])
def test_exact_rankings_match_faiss_with_crossblock_crossshard_document_maxima(tmp_path, dtype, query):
    data = artifact(tmp_path, dtype=dtype)
    backend = load(data)
    for mask in (None, [True, True, True, False], [False, True, False, False], [False] * 4):
        expected = reference(data, query, 10, mask)
        actual = backend.rank_vector(query, 10, mask)
        assert [row[0] for row in actual] == [row[0] for row in expected]
        np.testing.assert_allclose([row[1] for row in actual], [row[1] for row in expected], rtol=2e-6, atol=2e-6)


def test_best_passage_ties_and_document_ties_are_stable_and_candidates_have_no_bodies(tmp_path):
    data = artifact(tmp_path)
    backend = load(data)
    candidates = backend.candidates_vector([1, 0, 0], 10, scope=scope(data))
    assert [row.doc_id for row in candidates] == ["alpha", "secret", "zeta", "middle"]
    assert [(row.start_char, row.end_char) for row in candidates[:3]] == [(0, 8), (0, 10), (40, 50)]
    assert len({row.doc_id for row in candidates}) == len(candidates)
    for row in candidates:
        entry = data.catalog[data.doc_ids.index(row.doc_id)]
        row.validate_catalog(data.snapshot, entry)
        assert row.snapshot == data.snapshot and row.revision_id == entry.revision_id
        assert row.scoring_provenance == backend.scoring_provenance
        assert not {"text", "title", "content"} & row.model_dump().keys()
    assert backend.doc_ids == data.doc_ids and backend.dimension == 3
    assert backend.total_vectors == 7 and backend.block_rows == 2
    exposed = backend.manifest
    exposed["model"]["normalize"] = False
    assert backend.manifest["model"]["normalize"] is True


@pytest.mark.parametrize("seed", range(8))
def test_exact_kth_cutoff_pruning_matches_full_python_oracle_across_random_scores_and_scopes(seed):
    rng = np.random.default_rng(seed)
    for count in (0, 1, 2, 17, 257, 2049):
        doc_ids = tuple(f"doc-{value:06d}" for value in rng.permutation(count))
        starts = rng.integers(0, 100, size=count, dtype=np.int64)
        ends = starts + rng.integers(1, 100, size=count, dtype=np.int64)
        for distribution in ("continuous", "ties", "negative", "zero", "denied"):
            scores = rng.normal(size=count).astype(np.float32)
            if distribution == "ties":
                scores = rng.integers(-3, 4, size=count).astype(np.float32)
            elif distribution == "negative":
                scores = -np.abs(scores) - np.float32(.25)
            elif distribution == "zero":
                scores.fill(0)
            if distribution == "denied":
                scores.fill(-np.inf)
            else:
                scores[rng.random(count) < .3] = -np.inf
            original = scores.copy()
            permitted = [index for index, value in enumerate(scores) if np.isfinite(value)]
            key = lambda index: (-float(scores[index]), doc_ids[index], int(starts[index]), int(ends[index]))
            for top_k in sorted({1, 3, 50, max(1, count), count + 5}):
                actual = _top_candidates(scores, top_k)
                expected_top = sorted(permitted, key=key)[:top_k]
                assert heapq.nsmallest(top_k, actual, key=key) == expected_top
                if len(permitted) > top_k:
                    cutoff = sorted((float(scores[index]) for index in permitted), reverse=True)[top_k - 1]
                    assert actual.tolist() == [index for index in permitted if scores[index] >= cutoff]
                else:
                    assert actual.tolist() == permitted
                np.testing.assert_array_equal(scores, original)


def test_exact_cutoff_keeps_all_negative_ties_and_excludes_nonfinite_sentinels():
    scores = np.array([-5, -3, -3, -3, -4, -np.inf, np.nan, np.inf], dtype=np.float32)
    assert _top_candidates(scores, 2).tolist() == [1, 2, 3]
    assert _top_candidates(np.zeros(20, dtype=np.float32), 1).tolist() == list(range(20))


@pytest.mark.parametrize("query", [[1, 0, 0], [-1, 0, 0], [0, 0, 0]])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_pruned_public_candidates_keep_exact_scope_order_and_winning_spans(tmp_path, query, top_k):
    data = artifact(tmp_path)
    backend = load(data)
    for mask in ([True] * 4, [False, True, False, True], [False] * 4):
        allowed = [doc_id for doc_id, permit in zip(data.doc_ids, mask) if permit]
        expected = reference(data, query, top_k, mask)
        actual = backend.candidates_vector(query, top_k, scope=scope(data, allowed))
        assert [(item.doc_id, item.start_char, item.end_char) for item in actual] == [
            (doc_id, start, end) for doc_id, _score, start, end in expected]
        np.testing.assert_allclose([item.score for item in actual], [row[1] for row in expected], rtol=2e-6, atol=2e-6)


def test_public_heap_receives_every_exact_cutoff_tie_and_no_lower_score(tmp_path, monkeypatch):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    backend = load(data)
    original = module.heapq.nsmallest
    seen = []
    def watch(top_k, iterable, *, key):
        values = list(iterable)
        seen.append(values)
        return original(top_k, values, key=key)
    monkeypatch.setattr(module.heapq, "nsmallest", watch)
    hits = backend.candidates_vector([1, 0, 0], 1, scope=scope(data))
    assert seen == [[0, 1, 3]]  # zeta/alpha/secret tie at1; middle is strictly lower.
    assert [(item.doc_id, item.start_char, item.end_char) for item in hits] == [("alpha", 0, 8)]


def test_authorization_and_source_dates_filter_before_emitting_ranked_candidates(tmp_path):
    data = artifact(tmp_path)
    backend = load(data)
    allowed = scope(data, {"zeta", "alpha", "middle"})
    assert [row.doc_id for row in backend.candidates_vector([1, 0, 0], 10, scope=allowed)] == ["alpha", "zeta", "middle"]
    assert [row.doc_id for row in backend.candidates_vector([1, 0, 0], 10, scope=allowed,
        filters=ReadFilters(source_types=("jira",)))] == ["alpha", "zeta"]
    after = datetime(2026, 10, 2, tzinfo=timezone.utc)
    before = datetime(2026, 10, 2, tzinfo=timezone.utc)
    assert [row.doc_id for row in backend.candidates_vector([1, 0, 0], 10, scope=scope(data),
        filters=ReadFilters(after=after))] == ["secret", "middle"]
    assert [row.doc_id for row in backend.candidates_vector([1, 0, 0], 10, scope=scope(data),
        filters=ReadFilters(before=before))] == ["alpha", "middle"]
    assert backend.candidates_vector([1, 0, 0], 10, scope=allowed,
        filters=ReadFilters(source_types=("gmail",))) == ()


def test_scope_snapshot_and_malformed_scope_masks_fail_closed(tmp_path):
    data = artifact(tmp_path)
    backend = load(data)
    different = data.snapshot.model_copy(update={"catalog_sha256": "c" * 64})
    with pytest.raises(ValueError):
        backend.candidates_vector([1, 0, 0], 3, scope=scope(data, snapshot=different))

    class BrokenScope:
        binding = scope(data).binding
        def permits(self, _doc_id): return True
        def mask_for(self, _snapshot, _doc_ids): return [True]

    with pytest.raises(ValueError):
        backend.candidates_vector([1, 0, 0], 3, scope=BrokenScope())


def test_scope_mutation_during_candidate_permission_check_cannot_release_metadata(tmp_path):
    data = artifact(tmp_path)
    backend = load(data)

    class MutatingScope:
        binding = scope(data).binding
        permission_checks = 0
        def mask_for(self, _snapshot, doc_ids):
            return [True] * len(doc_ids)
        def permits(self, _doc_id):
            self.permission_checks += 1
            self.binding = self.binding.model_copy(update={"policy_revision": 2})
            return True

    changing = MutatingScope()
    with pytest.raises(StaleContext):
        backend.candidates_vector([1, 0, 0], 3, scope=changing)
    assert changing.permission_checks > 0


def test_empty_scope_skips_vector_conversion_mmaps_and_normalization(tmp_path, monkeypatch):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    backend = load(data)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Empty candidate scope must not access vectors")

    class NoVector:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Empty candidate scope must not convert the query")

    monkeypatch.setattr(module.np, "load", forbidden)
    monkeypatch.setattr(module, "_normalized_block", forbidden)
    assert backend.candidates_vector(NoVector(), 3, scope=scope(data, ())) == ()


@pytest.mark.parametrize("bad", [True, 1.0, np.int64(2), 0, -1, "2"])
def test_block_rows_is_a_strict_positive_python_integer(tmp_path, bad):
    data = artifact(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        ShardedDense.load(data.directory, snapshot=data.snapshot, catalog=data.catalog, block_rows=bad)


@pytest.mark.parametrize("bad", [True, 1.0, np.int64(2), 0, -1, "2"])
def test_top_k_is_strict_even_when_scope_is_empty(tmp_path, bad):
    data = artifact(tmp_path)
    backend = load(data)
    with pytest.raises((ValueError, TypeError)):
        backend.rank_vector([1, 0, 0], bad)
    with pytest.raises((ValueError, TypeError)):
        backend.candidates_vector([1, 0, 0], bad, scope=scope(data, ()))


@pytest.mark.parametrize("mask", [[1, 0, 1, 0], [True], [[True] * 4], ["true"] * 4])
def test_masks_cannot_coerce_numbers_strings_or_wrong_shapes_to_permission(tmp_path, mask):
    data = artifact(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        load(data).rank_vector([1, 0, 0], 3, mask)


@pytest.mark.parametrize("vector", [[], [1, 0], [1, 0, 0, 0], [[1], [0], [0]],
    [True, False, False], ["1", "0", "0"], [1 + 0j, 0, 0], [np.nan, 0, 0],
    [np.inf, 0, 0], [1e300, 0, 0], np.array([1, 0, 0], dtype=object), np.array(1)])
def test_invalid_query_shape_type_and_nonfinite_conversion_reject(tmp_path, vector):
    data = artifact(tmp_path)
    backend = load(data)
    with pytest.raises((ValueError, TypeError)):
        backend.rank_vector(vector, 3)
    with pytest.raises((ValueError, TypeError)):
        backend.rank_vector(vector, 3, [False] * 4)


def test_readonly_query_and_stored_vectors_are_not_mutated(tmp_path):
    data = artifact(tmp_path)
    backend = load(data)
    query = np.array([[2, 0, 0]], dtype=np.float32)
    query.flags.writeable = False
    before = [np.load(data.directory / f"shard_{i:05d}.npy").copy() for i in range(2)]
    backend.rank_vector(query, 4)
    np.testing.assert_array_equal(query, [[2, 0, 0]])
    for number, expected in enumerate(before):
        np.testing.assert_array_equal(np.load(data.directory / f"shard_{number:05d}.npy"), expected)


@pytest.mark.parametrize("change", [
    {"complete": False}, {"complete": 1}, {"version": 2}, {"version": True},
    {"documents": 3}, {"documents": True}, {"shards": 0}, {"shards": True}, {"shards": 10**12},
    {"corpus_sha256": "c" * 64}, {"model": None}, {"model": {"normalize": False}},
    {"model": {"normalize": 1}},
])
def test_invalid_or_incomplete_manifest_rejects(tmp_path, change):
    data = artifact(tmp_path)
    _json(data.directory / "manifest.json", {**data.manifest, **change})
    with pytest.raises((ValueError, TypeError, OSError)):
        load(data)


@pytest.mark.parametrize("kind", ["missing-vectors", "missing-metadata", "extra-vectors", "extra-metadata", "duplicate-doc", "blank-doc", "doc-order"])
def test_missing_extra_and_mismatched_artifact_identity_rejects(tmp_path, kind):
    data = artifact(tmp_path)
    if kind == "missing-vectors": (data.directory / "shard_00001.npy").unlink()
    elif kind == "missing-metadata": (data.directory / "shard_00001.parquet").unlink()
    elif kind == "extra-vectors": np.save(data.directory / "shard_00002.npy", np.ones((1, 3), dtype=np.float16))
    elif kind == "extra-metadata": _write_metadata(data.directory / "shard_00002.parquet", [(0, 0, 10)])
    elif kind == "duplicate-doc": _json(data.directory / "doc_ids.json", ["zeta", "alpha", "alpha", "secret"])
    elif kind == "blank-doc": _json(data.directory / "doc_ids.json", ["zeta", "alpha", "", "secret"])
    else: _json(data.directory / "doc_ids.json", list(reversed(data.doc_ids)))
    with pytest.raises((ValueError, TypeError, OSError)):
        load(data)


@pytest.mark.parametrize("kind", ["row-count", "missing-column", "doc-outside", "negative-doc", "fractional-doc", "null-doc",
                                    "negative-start", "empty-span", "outside-text"])
def test_metadata_row_identity_types_and_exact_spans_are_validated(tmp_path, kind):
    data = artifact(tmp_path)
    records = list(data.records[0])
    path = data.directory / "shard_00000.parquet"
    if kind == "missing-column":
        pq.write_table(pa.table({"doc_index": [0, 1, 2], "start_char": [0, 0, 0]}), path)
    else:
        if kind == "row-count": records.pop()
        elif kind == "doc-outside": records[0] = (4, 0, 10)
        elif kind == "negative-doc": records[0] = (-1, 0, 10)
        elif kind == "fractional-doc": records[0] = (0.5, 0, 10)
        elif kind == "null-doc": records[0] = (None, 0, 10)
        elif kind == "negative-start": records[0] = (0, -1, 10)
        elif kind == "empty-span": records[0] = (0, 10, 10)
        else: records[0] = (0, 0, 101)
        _write_metadata(path, records)
    with pytest.raises((ValueError, TypeError, OSError)):
        load(data)


@pytest.mark.parametrize("kind", ["float64", "integer", "non-native", "different-dimension", "mixed-dtype", "nan", "infinity", "not-matrix"])
def test_vector_artifacts_require_consistent_native_finite_float_matrices(tmp_path, kind):
    data = artifact(tmp_path)
    values = data.vectors[0].copy()
    if kind == "float64": values = values.astype(np.float64)
    elif kind == "integer": values = values.astype(np.int64)
    elif kind == "non-native": values = values.astype(np.dtype("f2").newbyteorder("S"))
    elif kind == "different-dimension": values = np.ones((3, 2), dtype=np.float16)
    elif kind == "mixed-dtype": values = values.astype(np.float32)
    elif kind == "nan": values[0, 0] = np.nan
    elif kind == "infinity": values[0, 0] = np.inf
    else: values = values[0]
    np.save(data.directory / "shard_00000.npy", values, allow_pickle=False)
    with pytest.raises((ValueError, TypeError, OSError)):
        load(data)


def test_catalog_is_single_pass_and_snapshot_fingerprint_and_order_are_verified(tmp_path):
    data = artifact(tmp_path)
    class Once:
        consumed = False
        def __iter__(self):
            assert not self.consumed
            self.consumed = True
            yield from data.catalog
    assert ShardedDense.load(data.directory, snapshot=data.snapshot, catalog=Once(), block_rows=2).doc_ids == data.doc_ids
    for bad_catalog in (data.catalog[:-1], tuple(reversed(data.catalog)),
        (data.catalog[0].model_copy(update={"text_length": 2}), *data.catalog[1:])):
        with pytest.raises((ValueError, TypeError)):
            ShardedDense.load(data.directory, snapshot=data.snapshot, catalog=bad_catalog, block_rows=2)


@pytest.mark.parametrize("file_cache_policy", ["discard", "retain"])
def test_normalized_copies_are_bounded_private_and_mmaps_do_not_escape_operations(tmp_path, monkeypatch, file_cache_policy):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    real_load, real_normalize = module.np.load, module._normalized_block
    mappings, blocks, dimensions = [], [], []
    maximum_open = 0

    def watched_load(*args, **kwargs):
        nonlocal maximum_open
        assert kwargs.get("mmap_mode") == "r"
        result = real_load(*args, **kwargs)
        assert isinstance(result, np.memmap)
        mappings.append(result._mmap)
        maximum_open = max(maximum_open, sum(not item.closed for item in mappings))
        return result

    def watched_normalize(stored, start, stop):
        result = real_normalize(stored, start, stop)
        assert 0 < stop - start <= 2
        assert result.dtype == np.float32 and result.flags.c_contiguous and result.flags.writeable
        assert result.flags.owndata and result.nbytes <= 2 * 3 * np.dtype(np.float32).itemsize
        blocks.append(weakref.ref(result))
        dimensions.append(result.shape)
        return result

    monkeypatch.setattr(module.np, "load", watched_load)
    monkeypatch.setattr(module, "_normalized_block", watched_normalize)
    backend = load(data, file_cache_policy=file_cache_policy)
    assert all(item.closed for item in mappings)
    backend.rank_vector([1, 0, 0], 4)
    backend.candidates_vector([1, 0, 0], 4, scope=scope(data, {"alpha"}))
    assert dimensions and maximum_open <= 1
    assert all(item.closed for item in mappings)
    assert all(block() is None for block in blocks)


@pytest.mark.parametrize("file_cache_policy", ["discard", "retain"])
def test_normalizer_failure_closes_mmaps_while_traceback_is_still_retained(tmp_path, monkeypatch, file_cache_policy):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    backend = load(data, file_cache_policy=file_cache_policy)
    real_load = module.np.load
    mappings = []

    def watched_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        mappings.append(result._mmap)
        return result

    def failing_normalizer(*_args):
        raise RuntimeError("injected normalization failure")

    monkeypatch.setattr(module.np, "load", watched_load)
    monkeypatch.setattr(module, "_normalized_block", failing_normalizer)
    with pytest.raises(RuntimeError, match="injected normalization failure") as retained:
        backend.rank_vector([1, 0, 0], 4)
    assert retained.value.__traceback__ is not None
    assert mappings and all(mapping.closed for mapping in mappings)


@pytest.mark.parametrize("bad", [None, True, False, 0, "", "Retain", "retain ", "auto", [], {}])
def test_file_cache_policy_is_explicit_and_never_coerces_invalid_values(tmp_path, bad):
    data = artifact(tmp_path)
    with pytest.raises(ValueError):
        load(data, file_cache_policy=bad)


def test_retaining_file_cache_changes_only_posix_hints_not_page_release_or_ranked_results(tmp_path, monkeypatch):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    native_load, native_release = module.np.load, module._release_pages
    mappings, hints, released = [], {"discard": [], "retain": []}, {"discard": [], "retain": []}
    mode = "discard"

    def watch_load(*args, **kwargs):
        result = native_load(*args, **kwargs)
        mappings.append(result._mmap)
        return result

    def watch_release(stored, start, stop):
        assert not stored._mmap.closed
        released[mode].append((Path(stored.filename).name, start, stop))
        return native_release(stored, start, stop)

    def watch_hint(descriptor, offset, length, advice):
        assert os.fstat(descriptor).st_size > 0
        hints[mode].append((offset, length, advice))

    monkeypatch.setattr(module.np, "load", watch_load)
    monkeypatch.setattr(module, "_release_pages", watch_release)
    monkeypatch.setattr(module.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(module.os, "posix_fadvise", watch_hint, raising=False)
    results = {}
    metadata = {}
    for mode in ("discard", "retain"):
        # Omission deliberately proves that the original default remains discard.
        backend = load(data) if mode == "discard" else load(data, file_cache_policy="retain")
        assert backend.file_cache_policy == mode
        metadata[mode] = (backend.snapshot, backend.catalog, backend.doc_ids, backend.manifest,
                          backend.scoring_provenance, backend.block_rows)
        results[mode] = (backend.rank_vector([1, 0, 0], 4),
            backend.candidates_vector([1, 0, 0], 4, scope=scope(data)),
            backend.candidates_vector([1, 0, 0], 4, scope=scope(data, {"alpha"}),
                filters=ReadFilters(source_types=("jira",), after=datetime(2026, 10, 1, tzinfo=timezone.utc))),
            backend.candidates_vector([1, 0, 0], 4, scope=scope(data, ())))
        assert all(mapping.closed for mapping in mappings)
    assert metadata["discard"] == metadata["retain"]
    assert results["discard"] == results["retain"]
    assert hints["discard"] and any(offset == length == 0 for offset, length, _ in hints["discard"])
    assert hints["retain"] == []
    assert released["discard"] and released["discard"] == released["retain"]


@pytest.mark.parametrize("file_cache_policy", ["discard", "retain"])
@pytest.mark.parametrize("ending", ["success", "early_close", "error"])
def test_cache_policy_never_retains_owned_descriptors_or_mappings(tmp_path, monkeypatch, file_cache_policy, ending):
    import src.retrieval.sharded_dense as module
    data = artifact(tmp_path)
    backend = load(data, file_cache_policy=file_cache_policy)
    native_open, native_close, native_load = module.os.open, module.os.close, module.np.load
    descriptors, active, mappings = [], set(), []

    def watch_open(path, *args, **kwargs):
        descriptor = native_open(path, *args, **kwargs)
        if str(path).endswith(".npy"):
            descriptors.append(descriptor)
            active.add(descriptor)
        return descriptor

    def watch_close(descriptor):
        result = native_close(descriptor)
        active.discard(descriptor)
        return result

    def watch_load(*args, **kwargs):
        result = native_load(*args, **kwargs)
        mappings.append(result._mmap)
        return result

    monkeypatch.setattr(module.os, "open", watch_open)
    monkeypatch.setattr(module.os, "close", watch_close)
    monkeypatch.setattr(module.np, "load", watch_load)
    if ending == "early_close":
        stream = backend._blocks(backend._shards[0])
        retained = next(stream)
        stream.close()
        assert retained[0]._mmap.closed
    elif ending == "error":
        def fail(*_args):
            raise RuntimeError("retained normalization traceback")
        monkeypatch.setattr(module, "_normalized_block", fail)
        with pytest.raises(RuntimeError, match="retained normalization") as retained_error:
            backend.rank_vector([1, 0, 0], 4)
        assert retained_error.value.__traceback__ is not None
    else:
        backend.rank_vector([1, 0, 0], 4)
    assert descriptors and not active
    assert mappings and all(mapping.closed for mapping in mappings)


def test_module_import_does_not_import_an_encoder_or_database():
    script = """
import builtins
real = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'sentence_transformers', 'sqlalchemy', 'psycopg'}:
        raise AssertionError('Unexpected heavy import: ' + name)
    return real(name, *args, **kwargs)
builtins.__import__ = checked
from src.retrieval.sharded_dense import ShardedDense
assert ShardedDense
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

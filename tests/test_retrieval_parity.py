"""Independent tiny-artifact parity and honest acceptance-report regressions.

The larger fixture has more than10,000 passage rows, not production corpus data.
FAISS scores are collected independently of the bounded scorer. Contract-store
fixtures establish exact evidence semantics; PostgreSQL/full-Docker acceptance
is reported separately by the integration run.
"""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.access.contracts import LegacySetScope, RequestContext, ResolvedIdentity
from src.contracts import ScopeBinding, SnapshotRef
from src.retrieval.sharded_dense import ShardedDense
from src.retrieval.store import StoreDenseRetriever
from src.retrieval.faiss_reference import (
    UnsupportedReferenceConfiguration, iter_scores, reference_configuration,
)
from src.storage.contracts import CatalogEntry, ReadFilters
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
from src.storage.testing import FixtureDocument, InMemoryDocumentStore
from scripts.compare_retrieval_backends import (
    RankHit, compare_hits, exhaustive_faiss as harness_reference, latency_gate,
    load_artifact, resource_gate, verify_source_evidence,
)
import scripts.compare_retrieval_backends as harness


NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
MODEL = {"name": "tiny-offline-parity-encoder", "revision": "b" * 40,
         "normalize": True, "fp16": False, "max_seq_length": 512, "query_prefix": ""}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parity_artifacts(tmp_path, *, large=False):
    """Cross-shard maxima, duplicate spans, exact ties, and a genuine near tie."""
    dense = tmp_path / "dense"
    dense.mkdir()
    doc_ids = ("zeta", "alpha", "beta", "secret", "late", "unclassified")
    titles = ("  Café 🧭  ", "Alpha", "Beta", "Executive", "Late", "Unclassified")
    contents = tuple((("naïve🙂 e\u0301\x00 and needle\n" if index == 0 else "naïve🙂 e\u0301 and needle\n") +
        "line\r\nspaces  ") * 30 + name for index, name in enumerate(doc_ids))
    texts = tuple(title + "\n" + content for title, content in zip(titles, contents))
    sources = ("jira", "jira", "slack", "gmail", "slack", "jira")
    timestamps = (None, NOW, NOW - timedelta(days=1), NOW, NOW + timedelta(days=1), None)
    rows = [dict(doc_id=name, source_type=source, title=title, content=content,
                 timestamp=timestamp, metadata_origin="synthetic")
            for name, source, title, content, timestamp in zip(doc_ids, sources, titles, contents, timestamps)]
    corpus = tmp_path / "documents.parquet"
    pq.write_table(pa.Table.from_pylist(rows), corpus)
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    catalog = tuple(CatalogEntry(doc_id=name, revision_id=canonical_revision_id(text), position=position,
        source_type=sources[position], text_length=len(text), timestamp=timestamps[position], metadata_origin="synthetic")
        for position, (name, text) in enumerate(zip(doc_ids, texts)))
    snapshot = SnapshotRef(snapshot_id="tiny-parity", corpus_sha256=corpus_hash,
        catalog_sha256=catalog_fingerprint(catalog, expected_count=len(catalog)), document_count=len(catalog))
    dimension, count = (64, 10016) if large else (4, 12)
    rng = np.random.default_rng(1729)
    vectors = rng.normal(size=(count, dimension)).astype(np.float32)
    query = rng.normal(size=dimension).astype(np.float32) if large else np.eye(1, dimension, dtype=np.float32)[0]
    query /= np.linalg.norm(query)
    orthogonal = np.eye(1, dimension, k=1, dtype=np.float32)[0]
    orthogonal -= float(orthogonal @ query) * query
    orthogonal /= np.linalg.norm(orthogonal)
    # A dense query and >10,000 rows exercise real reduction kernels. Filler
    # scores stay far from engineered maxima despite their rounding differences.
    projection = vectors @ query
    vectors -= (np.abs(projection) + 4)[:, None] * query
    metadata = np.column_stack((rng.integers(0, len(doc_ids), count),
        np.full(count, 10), np.full(count, 50))).astype(np.int64)
    metadata[4, 0] = 5
    boundary = count // 2
    engineered = ((0, 0, 0, 40, 0.), (1, 1, 0, 40, 0.), (2, 2, 20, 60, .004),
                  (3, 3, 0, 40, 0.), (boundary, 0, 60, 100, 0.),
                  (boundary + 1, 1, 60, 100, 0.), (count - 1, 4, 0, 40, 0.))
    for index, document, start, end, epsilon in engineered:
        vectors[index] = query + epsilon * orthogonal
        metadata[index] = document, start, end
    manifest = {"version": 1, "complete": True, "documents": len(doc_ids), "shards": 2,
        "corpus_sha256": corpus_hash, "model": dict(MODEL)}
    write_json(dense / "manifest.json", manifest)
    write_json(dense / "doc_ids.json", doc_ids)
    for shard, selection in enumerate((slice(0, boundary), slice(boundary, count))):
        np.save(dense / f"shard_{shard:05d}.npy", vectors[selection], allow_pickle=False)
        pq.write_table(pa.table({column: metadata[selection, index]
            for index, column in enumerate(("doc_index", "start_char", "end_char"))}),
            dense / f"shard_{shard:05d}.parquet")
    store = InMemoryDocumentStore(snapshot, (FixtureDocument(entry, text, title)
        for entry, text, title in zip(catalog, texts, titles)))
    catalog_file = tmp_path / "catalog.jsonl"
    catalog_file.write_text("".join(entry.model_dump_json() + "\n" for entry in catalog))
    return SimpleNamespace(dense=dense, corpus=corpus, catalog_file=catalog_file, snapshot=snapshot,
        catalog=catalog, store=store, texts=texts, doc_ids=doc_ids, vectors=vectors,
        metadata=metadata, query=query, manifest=manifest, count=count, dimension=dimension)


def context(data, allowed=None):
    binding = ScopeBinding(scope_id="parity-scope", user_id="parity-fixture", snapshot=data.snapshot,
        policy_revision=4, session_version=2)
    return RequestContext(ResolvedIdentity(user_id=binding.user_id, session_id="parity-session",
        session_version=2, expires_at=NOW + timedelta(hours=1)),
        LegacySetScope(binding=binding, allowed_doc_ids=frozenset(data.doc_ids if allowed is None else allowed)))


class Guard:
    def __init__(self):
        self.calls = 0

    def assert_current(self, context):
        self.calls += 1
        context.assert_bound()


def exhaustive_faiss(data, query, *, allowed=None, filters=ReadFilters(), top_k=6):
    """Independent all-passages max oracle, retaining both tie representatives.

    Full FAISS search prevents an early passage-fetch cutoff from masquerading
    as exhaustive document ranking. Compute canonical minimum spans from actual
    exact FAISS maxima; retain the first native FAISS representative separately.
    """
    index = faiss.IndexFlatIP(data.dimension)
    for shard in range(2):
        values = np.array(np.load(data.dense / f"shard_{shard:05d}.npy"), dtype=np.float32, order="C", copy=True)
        faiss.normalize_L2(values)
        index.add(values)
    scores, positions = index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), index.ntotal)
    allowed = set(data.doc_ids if allowed is None else allowed)
    best = {}
    for score, position in zip(scores[0], positions[0]):
        document, start, end = data.metadata[int(position)]
        name = data.doc_ids[int(document)]
        if name not in allowed or not filters.matches(data.catalog[int(document)]):
            continue
        if name not in best:
            best[name] = (float(score), int(start), int(end), 1, int(start), int(end))
        elif float(score) == best[name][0]:
            previous = best[name]
            minimum = min((int(start), int(end)), previous[1:3])
            best[name] = (previous[0], *minimum, previous[3] + 1, *previous[4:6])
    return [(name, *values) for name, values in sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]]


@pytest.mark.parametrize("large", [False, True])
def test_independent_all_passage_faiss_oracle_matches_bounded_document_maxima(tmp_path, large):
    data = parity_artifacts(tmp_path, large=large)
    if large:
        assert data.count >= 10000
    backend = ShardedDense.load(data.dense, snapshot=data.snapshot, catalog=data.catalog, block_rows=8192)
    cases = ((None, ReadFilters()), (("beta", "late"), ReadFilters()),
             ((), ReadFilters()), (None, ReadFilters(source_types=("slack",))),
             (None, ReadFilters(after=NOW, before=NOW)))
    for allowed, filters in cases:
        request = context(data, allowed)
        expected = exhaustive_faiss(data, data.query, allowed=allowed, filters=filters)
        actual = backend.candidates_vector(data.query, 6, scope=request.scope, filters=filters)
        assert [item.doc_id for item in actual] == [item[0] for item in expected]
        np.testing.assert_allclose([item.score for item in actual], [item[1] for item in expected], rtol=2e-6, atol=2e-6)
        assert len({item.doc_id for item in actual}) == len(actual)
        for item in actual:
            item.validate_catalog(data.snapshot, data.catalog[data.doc_ids.index(item.doc_id)])
    # The best passage for zeta lives in each shard. A bounded scan must not
    # lose the maximum or silently make the score proportional to passage count.
    full = backend.candidates_vector(data.query, 6, scope=context(data).scope)
    assert next(item.score for item in full if item.doc_id == "zeta") == pytest.approx(1.)


def test_harness_module_preflight_import_does_not_load_models_arrays_or_database():
    source = """
import importlib.abc
import sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','faiss','pyarrow','torch','transformers','sentence_transformers','sqlalchemy','pydantic','psutil','bm25s'}:
            raise RuntimeError('Heavy import during lightweight harness preflight: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
import scripts.compare_retrieval_backends
print('stdlib-preflight-ok')
"""
    completed = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "stdlib-preflight-ok"


@pytest.mark.parametrize("large", [False, True])
def test_harness_reference_matches_independent_full_faiss_scores_spans_and_tie_proof(tmp_path, monkeypatch, large):
    data = parity_artifacts(tmp_path, large=large)
    masks = {"all": [True] * len(data.doc_ids), "narrow": [name in {"alpha", "beta"} for name in data.doc_ids],
             "denied": [False] * len(data.doc_ids)}

    def forbidden(*_args, **_kwargs):
        pytest.fail("The reference oracle invoked the bounded ranking algorithm")

    monkeypatch.setattr(ShardedDense, "rank_vector", forbidden)
    monkeypatch.setattr(ShardedDense, "candidates_vector", forbidden)
    monkeypatch.setattr(ShardedDense, "_rank_passages", forbidden)
    vectors = np.stack((data.query, data.query * 1.7))
    result = harness_reference(data.dense, vectors, snapshot=data.snapshot, catalog=data.catalog,
                               masks=masks, top_k=6, block_rows=4096)
    assert (result.documents, result.passages, result.shards, result.dimension) == (6, data.count, 2, data.dimension)
    for name, mask in masks.items():
        for index, query in enumerate(vectors):
            expected = exhaustive_faiss(data, query, allowed=[doc for doc, permits in zip(data.doc_ids, mask) if permits])
            actual = result.rankings[name][index]
            assert [item.doc_id for item in actual] == [item[0] for item in expected]
            np.testing.assert_allclose([item.score for item in actual], [item[1] for item in expected], rtol=2e-6, atol=2e-6)
            assert [(item.start_char, item.end_char, item.tie_count) for item in actual] == [item[2:5] for item in expected]
            assert [(item.legacy_start_char, item.legacy_end_char) for item in actual] == [item[5:7] for item in expected]


def test_reference_preserves_single_query_faiss_kernel_for_a_many_query_batch(tmp_path, monkeypatch):
    data = parity_artifacts(tmp_path, large=True)
    searched = []
    original = faiss.IndexFlatIP.search

    def watch(index, vectors, *args, **kwargs):
        searched.append(np.asarray(vectors).shape[0])
        return original(index, vectors, *args, **kwargs)

    monkeypatch.setattr(faiss.IndexFlatIP, "search", watch)
    queries = np.stack([data.query * scale for scale in np.linspace(.5, 1.5, 21)])
    result = harness_reference(data.dense, queries, snapshot=data.snapshot, catalog=data.catalog,
                               masks={"all": [True] * 6}, top_k=3, block_rows=4096)
    assert len(result.rankings["all"]) == 21
    assert searched and set(searched) == {1}


def test_later_exact_maximum_with_earlier_start_keeps_its_actual_end_offset(tmp_path):
    data = parity_artifacts(tmp_path)
    boundary = data.count // 2
    data.metadata[0] = (0, 60, 80)
    data.metadata[boundary] = (0, 0, 100)
    for shard, selection in enumerate((slice(0, boundary), slice(boundary, data.count))):
        pq.write_table(pa.table({name: data.metadata[selection, index]
            for index, name in enumerate(("doc_index", "start_char", "end_char"))}),
            data.dense / f"shard_{shard:05d}.parquet")
    result = harness_reference(data.dense, np.stack((data.query,)), snapshot=data.snapshot,
                               catalog=data.catalog, masks={"all": [True] * 6}, top_k=6, block_rows=4)
    selected = next(hit for hit in result.rankings["all"][0] if hit.doc_id == "zeta")
    assert (selected.start_char, selected.end_char, selected.tie_count) == (0, 100, 2)


def test_score_tolerance_preserves_raw_delta_and_never_excuses_a_near_tie_rank_swap():
    reference = (RankHit("alpha", 1., 0, 20), RankHit("beta", .9999998, 0, 20))
    candidate = (RankHit("alpha", 1.0000001, 0, 20), RankHit("beta", .9999998, 0, 20))
    close = compare_hits(reference, candidate, atol=2e-6, rtol=2e-6)
    assert close["passed"] is True and close["strict_equal"] is False
    differences = [item for item in close["differences"] if item["kind"] == "score"]
    assert differences and differences[0]["delta"] == pytest.approx(1e-7)
    assert differences[0]["within_tolerance"] is True
    swapped = compare_hits(reference, tuple(reversed(candidate)), atol=2e-6, rtol=2e-6)
    assert swapped["passed"] is False and swapped["strict_equal"] is False
    assert swapped["differences"]


def test_primary_span_changes_fail_even_with_genuine_exact_maximum_tie_proof():
    canonical = RankHit("alpha", 1., 0, 20, tie_count=2, legacy_start_char=60, legacy_end_char=80)
    changed = RankHit("alpha", 1., 60, 80)
    report = compare_hits((canonical,), (changed,))
    assert report["passed"] is False and report["strict_equal"] is False
    assert any(item["kind"] == "span" and item["genuine_exact_tie"] is True for item in report["differences"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_cannot_produce_a_successful_comparison(value):
    with pytest.raises(ValueError):
        compare_hits((RankHit("alpha", value, 0, 20),), (RankHit("alpha", 1., 0, 20),))


@pytest.mark.parametrize("bad", ["dimension", "nan", "infinity", "bool", "complex", "empty", "rank_three"])
def test_reference_rejects_malformed_query_vectors_instead_of_reporting_parity(tmp_path, bad):
    data = parity_artifacts(tmp_path)
    vectors = {"dimension": np.zeros((1, data.dimension + 1)),
        "nan": np.full((1, data.dimension), np.nan), "infinity": np.full((1, data.dimension), np.inf),
        "bool": np.zeros((1, data.dimension), dtype=bool),
        "complex": np.zeros((1, data.dimension), dtype=np.complex64),
        "empty": np.empty((0, data.dimension)), "rank_three": np.zeros((1, 1, data.dimension))}[bad]
    with pytest.raises((ValueError, TypeError)):
        harness_reference(data.dense, vectors, snapshot=data.snapshot, catalog=data.catalog,
                          masks={"all": [True] * 6}, top_k=3)


@pytest.mark.parametrize("bad", [[True], [1] * 6, [True] * 7])
def test_reference_masks_must_be_exact_catalog_aligned_booleans(tmp_path, bad):
    data = parity_artifacts(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        harness_reference(data.dense, np.stack((data.query,)), snapshot=data.snapshot,
                          catalog=data.catalog, masks={"broken": bad}, top_k=3)


@pytest.mark.parametrize("field,value", [("top_k", 0), ("top_k", True), ("top_k", 1.5),
                                        ("block_rows", 0), ("block_rows", True)])
def test_reference_numeric_configuration_is_strict(tmp_path, field, value):
    data = parity_artifacts(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        harness_reference(data.dense, np.stack((data.query,)), snapshot=data.snapshot,
                          catalog=data.catalog, masks={"all": [True] * 6}, **{field: value})


@pytest.mark.parametrize("corruption", ["incomplete", "missing", "extra", "corpus", "catalog"])
def test_reference_rejects_incomplete_or_mismatching_artifact_identity(tmp_path, corruption):
    data = parity_artifacts(tmp_path)
    snapshot = data.snapshot
    if corruption == "incomplete":
        write_json(data.dense / "manifest.json", {**data.manifest, "complete": False})
    elif corruption == "missing":
        (data.dense / "shard_00001.parquet").unlink()
    elif corruption == "extra":
        np.save(data.dense / "shard_00099.npy", np.zeros((1, data.dimension)), allow_pickle=False)
    elif corruption == "corpus":
        snapshot = snapshot.model_copy(update={"corpus_sha256": "f" * 64})
    else:
        snapshot = snapshot.model_copy(update={"catalog_sha256": "e" * 64})
    with pytest.raises((ValueError, OSError)):
        harness_reference(data.dense, np.stack((data.query,)), snapshot=snapshot,
                          catalog=data.catalog, masks={"all": [True] * 6})


class Encoder:
    def __init__(self, query):
        self.identity, self.query, self.calls = dict(MODEL), query, 0

    def encode(self, texts, *, query):
        assert query is True
        self.calls += 1
        return np.stack([self.query for _ in texts])


def materialized_evidence(data, allowed=None, filters=ReadFilters()):
    index = ShardedDense.load(data.dense, snapshot=data.snapshot, catalog=data.catalog, block_rows=4)
    encoder, guard = Encoder(data.query), Guard()
    retriever = StoreDenseRetriever(index, encoder, data.store)
    request = context(data, allowed)
    evidence = retriever.search(request, "needle", guard=guard, top_k=6, filters=filters)
    return SimpleNamespace(evidence=evidence, retriever=retriever, context=request, guard=guard, encoder=encoder)


@pytest.mark.parametrize("allowed,filters", [(None, ReadFilters()), (("alpha", "beta"), ReadFilters()),
    ((), ReadFilters()), (None, ReadFilters(source_types=("slack",))),
    (None, ReadFilters(after=NOW, before=NOW))])
def test_actual_store_evidence_and_adjacent_windows_verify_exact_unicode_and_provenance(tmp_path, allowed, filters):
    data = parity_artifacts(tmp_path)
    result = materialized_evidence(data, allowed, filters)
    checked = verify_source_evidence(data.corpus, result.evidence, snapshot=data.snapshot, catalog=data.catalog)
    assert checked["checked"] == len(result.evidence)
    expected = exhaustive_faiss(data, data.query, allowed=allowed, filters=filters)
    assert [item.chunk.doc_id for item in result.evidence] == [item[0] for item in expected]
    for item in result.evidence:
        position = data.doc_ids.index(item.chunk.doc_id)
        assert item.chunk.text == data.texts[position][item.chunk.start_char:item.chunk.end_char]
        assert item.snapshot_id == data.snapshot.snapshot_id and item.revision_id == data.catalog[position].revision_id
    if not result.evidence:
        assert result.encoder.calls == 0
    anchors = tuple(result.evidence[:2])
    expanded = result.retriever.read_more(result.context, anchors, guard=result.guard, max_chars=80, filters=filters)
    assert len(expanded) <= len(anchors)
    assert verify_source_evidence(data.corpus, expanded, snapshot=data.snapshot, catalog=data.catalog)["checked"] == len(expanded)
    for item in expanded:
        parent = next(anchor for anchor in anchors if anchor.chunk.doc_id == item.chunk.doc_id)
        assert item.chunk.end_char > parent.chunk.end_char and len(item.chunk.text) <= 80


@pytest.mark.parametrize("corruption", ["text", "title", "revision", "snapshot", "offset",
                                        "source", "chunk_id", "origin", "timestamp"])
def test_evidence_verifier_rejects_plausible_but_noncanonical_exported_sources(tmp_path, corruption):
    data = parity_artifacts(tmp_path)
    item = materialized_evidence(data).evidence[0].model_dump(mode="json")
    if corruption == "text":
        item["chunk"]["text"] = "?" * len(item["chunk"]["text"])
    elif corruption == "title":
        item["chunk"]["title"] = "Invented source title"
    elif corruption == "revision":
        item["revision_id"] = "f" * 64
    elif corruption == "snapshot":
        item["snapshot_id"] = "different-snapshot"
    elif corruption == "source":
        item["chunk"]["source_type"] = "confluence"
    elif corruption == "chunk_id":
        item["chunk"]["chunk_id"] = "misleading-chunk"
    elif corruption == "origin":
        item["chunk"]["metadata_origin"] = "upstream"
    elif corruption == "timestamp":
        item["chunk"]["timestamp"] = "2020-01-01T00:00:00Z"
    else:
        item["chunk"]["start_char"] += 1
        item["chunk"]["end_char"] += 1
        key = f"{item['chunk']['doc_id']}:{item['chunk']['start_char']}:{item['chunk']['end_char']}"
        item["citation_id"] = item["chunk"]["chunk_id"] = key
    with pytest.raises(ValueError):
        verify_source_evidence(data.corpus, (item,), snapshot=data.snapshot, catalog=data.catalog)


@contextmanager
def native_threads(count):
    previous = faiss.omp_get_max_threads()
    try:
        if count is not None:
            faiss.omp_set_num_threads(count)
        yield faiss.omp_get_max_threads()
    finally:
        faiss.omp_set_num_threads(previous)


def full_native_scores(normalized_parts, queries):
    """The unchanged global IndexFlatIP baseline; no cvar overrides."""
    index = faiss.IndexFlatIP(normalized_parts[0].shape[1])
    for part in normalized_parts:
        index.add(part)
    total = sum(len(part) for part in normalized_parts)
    result = np.empty((len(queries), total), dtype=np.float32)
    for query_index, query in enumerate(queries):
        scores, positions = index.search(query.reshape(1, -1), total)
        result[query_index, positions[0]] = scores[0]
    return result


@pytest.mark.parametrize("dimension", [64, 768])
@pytest.mark.parametrize("total", [10016, 10017])
@pytest.mark.parametrize("threads", [None, 1, 2, 4])
@pytest.mark.parametrize("storage_dtype", [np.float16, np.float32])
def test_global_dispatch_helper_is_bit_exact_to_full_faiss_without_changing_baseline(dimension, total, threads, storage_dtype):
    rng = np.random.default_rng(413)
    stored = rng.normal(size=(total, dimension)).astype(storage_dtype)
    # The physical split differs from both the global thread partition and its
    # 1024-row tile boundary. Equal vectors straddle each of those boundaries.
    split = total // 3 + 17
    for target in (1023, 1024, total // 2 - 1, total // 2, split - 1, split, total - 1):
        stored[target] = stored[0]
    normalized_parts = []
    with native_threads(threads) as actual_threads:
        for part in (stored[:split], stored[split:]):
            normalized = np.array(part, dtype=np.float32, order="C", copy=True)
            faiss.normalize_L2(normalized)
            normalized_parts.append(normalized)
        query = rng.normal(size=dimension).astype(np.float32)
        query /= np.linalg.norm(query)
        queries = np.stack((query, query * 1.7, np.zeros(dimension, dtype=np.float32)))
        before = (faiss.omp_get_max_threads(), faiss.cvar.distance_compute_blas_threshold,
                  faiss.cvar.distance_compute_blas_query_bs, faiss.cvar.distance_compute_blas_database_bs,
                  faiss.cvar.distance_compute_min_k_reservoir)
        expected = full_native_scores(normalized_parts, queries)
        reads = []

        def reader(start, stop):
            reads.append((start, stop))
            if stop <= split:
                return normalized_parts[0][start:stop]
            if start >= split:
                return normalized_parts[1][start - split:stop - split]
            return np.concatenate((normalized_parts[0][start:], normalized_parts[1][:stop - split]))

        actual = np.empty_like(expected)
        cursor = 0
        for start, scores in iter_scores(queries, total_vectors=total, dimension=dimension,
                                          read_rows=reader, threads=actual_threads):
            assert start == cursor and scores.dtype == np.float32
            assert 0 < scores.shape[1] <= 1024
            actual[:, start:start + scores.shape[1]] = scores
            cursor += scores.shape[1]
        assert cursor == total and reads[0][0] == 0 and reads[-1][1] == total
        assert any(start < split < stop for start, stop in reads)
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert before == (faiss.omp_get_max_threads(), faiss.cvar.distance_compute_blas_threshold,
            faiss.cvar.distance_compute_blas_query_bs, faiss.cvar.distance_compute_blas_database_bs,
            faiss.cvar.distance_compute_min_k_reservoir)


def test_dispatch_reproduces_original_scaled_query_failure_under_unchanged_ambient_threads(tmp_path):
    data = parity_artifacts(tmp_path, large=True)
    parts = []
    for shard in range(2):
        part = np.array(np.load(data.dense / f"shard_{shard:05d}.npy"), dtype=np.float32, order="C", copy=True)
        faiss.normalize_L2(part)
        parts.append(part)
    normalized = np.concatenate(parts)
    queries = np.stack((data.query, data.query * 1.7))
    expected = full_native_scores(parts, queries)
    actual = np.empty_like(expected)
    threads = faiss.omp_get_max_threads()
    for start, scores in iter_scores(queries, total_vectors=data.count, dimension=data.dimension,
        read_rows=lambda start, stop: normalized[start:stop]):
        actual[:, start:start + scores.shape[1]] = scores
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert faiss.omp_get_max_threads() == threads


def test_dispatch_restores_cvars_before_yield_and_early_generator_close():
    query = np.ones((1, 64), dtype=np.float32)
    before = (faiss.omp_get_max_threads(), faiss.cvar.distance_compute_blas_threshold)
    stream = iter_scores(query, total_vectors=10016, dimension=64,
                         read_rows=lambda start, stop: np.zeros((stop - start, 64), np.float32))
    start, scores = next(stream)
    assert start == 0 and scores.shape[0] == 1
    assert before == (faiss.omp_get_max_threads(), faiss.cvar.distance_compute_blas_threshold)
    stream.close()
    assert before == (faiss.omp_get_max_threads(), faiss.cvar.distance_compute_blas_threshold)


def test_dispatch_restores_every_touched_native_setting_on_search_exception(monkeypatch):
    names = ("distance_compute_blas_threshold", "distance_compute_blas_query_bs",
             "distance_compute_blas_database_bs", "distance_compute_min_k_reservoir")
    before = tuple(getattr(faiss.cvar, name) for name in names)
    threads = faiss.omp_get_max_threads()

    def fail_search(*_args, **_kwargs):
        for name in names:
            setattr(faiss.cvar, name, getattr(faiss.cvar, name) + 1)
        faiss.omp_set_num_threads(threads + 1)
        raise RuntimeError("injected native search failure")

    monkeypatch.setattr(faiss.IndexFlatIP, "search", fail_search)
    with pytest.raises(RuntimeError, match="native search failure"):
        list(iter_scores(np.ones((1, 64), np.float32), total_vectors=10016, dimension=64,
             read_rows=lambda start, stop: np.zeros((stop - start, 64), np.float32)))
    assert tuple(getattr(faiss.cvar, name) for name in names) == before
    assert faiss.omp_get_max_threads() == threads


def test_reference_provenance_does_not_override_baseline_blas_or_openmp_settings():
    threads = faiss.omp_get_max_threads()
    description = reference_configuration(total_vectors=10016, dimension=768)
    assert description["faiss_version"] == "1.15.1" and description["faiss_threads"] == threads
    assert description["query_batch_size"] == 1 and description["blas_thread_override"] is False
    assert faiss.omp_get_max_threads() == threads


def test_unsupported_dispatch_configurations_fail_before_reading_rows(monkeypatch):
    def forbidden(*_args):
        pytest.fail("Unsupported native configuration read vector payloads")
    with pytest.raises(UnsupportedReferenceConfiguration):
        list(iter_scores(np.ones((1, 64)), total_vectors=10016, dimension=64,
                         read_rows=forbidden, threads=faiss.omp_get_max_threads() + 1))
    monkeypatch.setattr(faiss, "__version__", "different-version")
    with pytest.raises(UnsupportedReferenceConfiguration):
        list(iter_scores(np.ones((1, 64)), total_vectors=10016, dimension=64, read_rows=forbidden))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retained_artifact(path, manifest, records=()):
    """Build the public wire format independently, not via the producer helper."""
    path.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical({"case": record, "sha256": hashlib.sha256(canonical(record)).hexdigest()}) + b"\n"
                       for record in records)
    (path / "results.jsonl").write_bytes(payload)
    write_json(path / "manifest.json", {**manifest, "schema_version": "1", "complete": True,
        "records": len(records), "results_sha256": hashlib.sha256(payload).hexdigest()})


def measured_resources(scope, candidate):
    return {"complete": True, "scope": scope, "candidate_manifest_sha256": digest(candidate / "manifest.json"),
        "measurement_kind": "sampled_sequential_app_db_cgroup_1s", "sample_count": 5,
        "raw_sampler_sha256": "a" * 64, "app_supervisor_sha256": "b" * 64,
        "peak_aggregate_bytes": 6 * 1024**3, "min_vm_available_bytes": 512 * 1024**2,
        "oom_events_delta": 0, "guard_status": "not_triggered"}


def cli_artifacts(tmp_path):
    """Synthetic completed reports for CLI validation, not performance evidence.

    The query has no lexical matches, so independent dense-only RRF is a valid
    hybrid fixture. Exact excerpts still come from the real contract store.
    """
    source = tmp_path / "source"
    source.mkdir()
    data = parity_artifacts(source)
    questions = [{"question_id": name, "question": "quuxneedle",
        "question_sha256": hashlib.sha256(b"quuxneedle").hexdigest()} for name in harness.DEV_IDS]
    matrix = {"question_ids": list(harness.DEV_IDS), "narrow_doc_ids": ["alpha"],
        "sources_by_query": {name: ["jira"] for name in harness.DEV_IDS},
        "unknown_after": "2020-01-01T00:00:00Z", "candidate_k": 50, "rrf_k": 60,
        "bm25_weight": .7, "dense_weight": 1 - .7, "rank_prefixes": [5, 10, 20],
        "timed_top_k": 5, "warm_repeats": 3, "atol": 2e-6, "rtol": 2e-6, "reranker": None}
    shared = tmp_path / "shared"
    shared.mkdir()
    vectors = np.stack([data.query] * len(questions))
    np.save(shared / "query_vectors.npy", vectors, allow_pickle=False)
    hashes = harness._code()
    shared_manifest = {"stage": "encode", "snapshot": data.snapshot.model_dump(mode="json"),
        "questions": questions, "matrix": matrix, "matrix_sha256": hashlib.sha256(canonical(matrix)).hexdigest(),
        "vector_shape": list(vectors.shape), "vectors_sha256": digest(shared / "query_vectors.npy"),
        "model_identity": dict(MODEL), "source_hashes": hashes, "packages": {"fixture": "offline"},
        "artifact_hashes": {"corpus": digest(data.corpus)}}
    retained_artifact(shared, shared_manifest)
    shared_sha = digest(shared / "manifest.json")
    base_manifest = {"shared_manifest_sha256": shared_sha, "matrix_sha256": shared_manifest["matrix_sha256"],
        "vectors_sha256": shared_manifest["vectors_sha256"], "snapshot": shared_manifest["snapshot"],
        "coverage": {"documents": len(data.doc_ids), "passages": data.count, "shards": 2},
        "packages": {"fixture": "offline"}, "source_hashes": hashes}
    modes = (("full", None, ReadFilters()), ("full:source", None, ReadFilters(source_types=("jira",))),
        ("full:unknown_date", None, ReadFilters(after=datetime(2020, 1, 1, tzinfo=timezone.utc))),
        ("narrow", ("alpha",), ReadFilters()), ("empty", (), ReadFilters()))
    templates = {}
    for label, allowed, filters in modes:
        request = context(data, allowed)
        dense = exhaustive_faiss(data, data.query, allowed=allowed, filters=filters)
        dense_hits = [RankHit(row[0], row[1], row[2], row[3], row[4], row[5], row[6]) for row in dense]
        fused = [RankHit(hit.doc_id, matrix["dense_weight"] / (60 + rank), hit.start_char, hit.end_char)
                 for rank, hit in enumerate(dense_hits, 1)]
        evidence = []
        for hit in fused[:5]:
            entry = data.catalog[data.doc_ids.index(hit.doc_id)]
            value = data.store.read_passage(request, harness_candidate(data, hit, entry), guard=Guard(), filters=filters)
            evidence.append(value)
        expanded = data.store.read_more(request, tuple(evidence[:2]), guard=Guard(), max_chars=3000, filters=filters)
        legacy, primary = [], []
        for hit in dense_hits:
            if hit.tie_count > 1 and (hit.legacy_start_char, hit.legacy_end_char) != (hit.start_char, hit.end_char):
                entry = data.catalog[data.doc_ids.index(hit.doc_id)]
                for start, end, destination in ((hit.start_char, hit.end_char, primary),
                        (hit.legacy_start_char, hit.legacy_end_char, legacy)):
                    tied = RankHit(hit.doc_id, hit.score, start, end)
                    candidate = harness_candidate(data, tied, entry).model_copy(update={"retrieval_method": "dense"})
                    destination.append(data.store.read_passage(request, candidate, guard=Guard(), filters=filters))
        permitted = set(data.doc_ids if allowed is None else allowed)
        bits = bytes((sum(1 << position for position, name in enumerate(data.doc_ids) if name in permitted),))
        templates[label] = {"case": label, "filters": filters.model_dump(mode="json"),
            "scope_id": request.binding.scope_id, "allowed_documents": len(data.doc_ids if allowed is None else allowed),
            "policy_revision": request.binding.policy_revision, "permission_bitmap_sha256": hashlib.sha256(bits).hexdigest(),
            "dense_hits": [hit.as_dict() for hit in dense_hits], "hybrid_hits": [hit.as_dict() for hit in fused],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "read_more": [item.model_dump(mode="json") for item in expanded],
            "legacy_tie_evidence": [item.model_dump(mode="json") for item in legacy],
            "tie_primary_evidence": [item.model_dump(mode="json") for item in primary],
            "catalog": [entry.model_dump(mode="json") for entry in data.catalog]}
    reference_rows = []
    candidates = {scope: tmp_path / ("candidate-" + scope) for scope in ("full", "narrow", "empty")}
    candidate_rows = {scope: [] for scope in candidates}
    for question in questions:
        for label, template in templates.items():
            common = {**template, "question_id": question["question_id"], "question_sha256": question["question_sha256"]}
            reference_rows.append({**common, "iteration": 0})
            for iteration in range(4):
                candidate = {key: value for key, value in common.items()
                             if key not in {"legacy_tie_evidence", "tie_primary_evidence"}}
                candidate_rows[label.split(":")[0]].append({**candidate, "iteration": iteration,
                    "timings": {"total_seconds": 1., "encode_seconds": .1, "resolve_seconds": .1, "encode_calls": 1}})
    reference = tmp_path / "reference"
    retained_artifact(reference, {**base_manifest, "stage": "reference"}, reference_rows)
    resources = []
    for scope, path in candidates.items():
        candidate_rows[scope] = [{**row, "request_index": index}
                                 for index, row in enumerate(candidate_rows[scope])]
        retained_artifact(path, {**base_manifest, "stage": "candidate", "scope": scope, "startup_seconds": 1.},
                          candidate_rows[scope])
        resource = tmp_path / (scope + "-resource.json")
        write_json(resource, measured_resources(scope, path))
        resources.append(resource)
    return SimpleNamespace(data=data, shared=shared, reference=reference, candidates=candidates,
        resources=resources, reference_rows=reference_rows, candidate_rows=candidate_rows)


def harness_candidate(data, hit, entry):
    from src.storage.contracts import RankedCandidate
    return RankedCandidate(snapshot=data.snapshot, doc_id=hit.doc_id, revision_id=entry.revision_id,
        source_type=entry.source_type, start_char=hit.start_char, end_char=hit.end_char,
        score=hit.score, retrieval_method="hybrid", scoring_provenance="independent-parity-fixture-rrf")


def compare_cli(data, output, *, resources=None):
    arguments = ["compare", "--shared", str(data.shared), "--reference", str(data.reference),
                 "--corpus", str(data.data.corpus), "--output", str(output)]
    for candidate in data.candidates.values():
        arguments.extend(("--candidate", str(candidate)))
    for resource in data.resources if resources is None else resources:
        arguments.extend(("--resources", str(resource)))
    return harness.main(arguments)


def refresh_candidate(data, scope, *, manifest_changes=None, rows=None):
    path = data.candidates[scope]
    manifest, previous = load_artifact(path)
    retained_artifact(path, {**manifest, **(manifest_changes or {})}, previous if rows is None else rows)
    resource = next(path for path in data.resources if json.loads(path.read_text())["scope"] == scope)
    write_json(resource, measured_resources(scope, data.candidates[scope]))


def test_completed_cli_comparison_retains_all_cases_and_requires_measured_bound_resources(tmp_path):
    data = cli_artifacts(tmp_path)
    output = tmp_path / "comparison"
    assert compare_cli(data, output) == 0
    manifest, records = load_artifact(output)
    assert manifest["passed"] is True and manifest["matrix_complete"] is True
    assert manifest["resources_verified"] is True and len(records) == 10 * 5 * 4


@pytest.mark.parametrize("failure", ["missing_resource", "duplicate_resource", "wrong_candidate", "guard", "memory", "latency", "matrix", "rank"])
def test_cli_rejects_invalid_inputs_and_retains_measured_gate_failures(tmp_path, failure):
    data = cli_artifacts(tmp_path)
    resources = list(data.resources)
    if failure == "missing_resource":
        resources.pop()
    elif failure == "duplicate_resource":
        resources = [resources[0]] * 3
    elif failure in {"wrong_candidate", "guard", "memory"}:
        value = json.loads(resources[0].read_text())
        value[{"wrong_candidate": "candidate_manifest_sha256", "guard": "guard_status", "memory": "peak_aggregate_bytes"}[failure]] = {
            "wrong_candidate": "f" * 64, "guard": "rss_guard", "memory": 7 * 1024**3}[failure]
        write_json(resources[0], value)
    elif failure == "latency":
        refresh_candidate(data, "full", manifest_changes={"startup_seconds": 121.})
    else:
        rows = list(data.candidate_rows["full"])
        if failure == "matrix":
            rows.pop()
        else:
            rows[0] = {**rows[0], "dense_hits": list(reversed(rows[0]["dense_hits"]))}
        refresh_candidate(data, "full", rows=rows)
    output = tmp_path / "failed-comparison"
    if failure in {"duplicate_resource", "wrong_candidate"}:
        with pytest.raises(SystemExit) as error:
            compare_cli(data, output, resources=resources)
        assert error.value.code == 2 and not (output / "manifest.json").exists()
        return
    assert compare_cli(data, output, resources=resources) == 2
    manifest, _ = load_artifact(output)
    assert manifest["passed"] is False
    if failure == "latency":
        assert any(not gate["passed"] for gate in manifest["latency_gates"])
    elif failure == "matrix":
        assert manifest["matrix_complete"] is False


def test_existing_or_partial_output_directory_is_never_resumed_or_overwritten(tmp_path):
    data = cli_artifacts(tmp_path)
    output = tmp_path / "partial"
    output.mkdir()
    sentinel = output / "results.jsonl"
    sentinel.write_bytes(b"retained partial attempt\n")
    with pytest.raises(SystemExit) as error:
        compare_cli(data, output)
    assert error.value.code == 2 and sentinel.read_bytes() == b"retained partial attempt\n"


@pytest.mark.parametrize("corruption", ["outer_digest", "inner_digest", "count", "duplicate", "incomplete"])
def test_artifact_case_journal_rejects_corruption_or_duplicate_identities(tmp_path, corruption):
    path = tmp_path / "artifact"
    record = {"question_id": "q1", "case": "full", "iteration": 0,
              "dense_hits": [RankHit("alpha", 1., 0, 20).as_dict()], "hybrid_hits": []}
    retained_artifact(path, {"stage": "reference"}, (record,))
    manifest = json.loads((path / "manifest.json").read_text())
    if corruption == "outer_digest":
        with (path / "results.jsonl").open("ab") as handle:
            handle.write(b" ")
    elif corruption == "inner_digest":
        envelope = json.loads((path / "results.jsonl").read_text())
        envelope["case"]["dense_hits"][0]["score"] = .5
        payload = canonical(envelope) + b"\n"
        (path / "results.jsonl").write_bytes(payload)
        manifest["results_sha256"] = hashlib.sha256(payload).hexdigest()
    elif corruption == "duplicate":
        retained_artifact(path, manifest, (record, record))
        manifest = json.loads((path / "manifest.json").read_text())
    elif corruption == "count":
        manifest["records"] += 1
    else:
        manifest["complete"] = False
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_artifact(path)


@pytest.mark.parametrize("corruption", ["vectors", "nonfinite", "dtype", "shape", "question", "order", "extra_reference", "matrix"])
def test_frozen_input_artifact_rejects_changed_vectors_questions_and_configuration(tmp_path, corruption):
    data = cli_artifacts(tmp_path)
    path = data.shared
    manifest = json.loads((path / "manifest.json").read_text())
    vectors = np.load(path / "query_vectors.npy", allow_pickle=False)
    if corruption in {"vectors", "nonfinite", "dtype"}:
        vectors = vectors.copy()
        if corruption == "vectors":
            vectors[0, 0] += .25
        elif corruption == "nonfinite":
            vectors[0, 0] = np.nan
        else:
            vectors = vectors.astype(np.int32)
        np.save(path / "query_vectors.npy", vectors, allow_pickle=False)
        if corruption != "vectors":
            manifest["vectors_sha256"] = digest(path / "query_vectors.npy")
    elif corruption == "shape":
        manifest["vector_shape"][1] += 1
    elif corruption == "question":
        manifest["questions"][0]["question"] = "changed runtime question"
    elif corruption == "order":
        manifest["questions"] = list(reversed(manifest["questions"]))
    elif corruption == "extra_reference":
        manifest["questions"][0]["reference_answer"] = "GOLD_MARKER_NEVER_RUNTIME"
    else:
        manifest["matrix"]["dense_weight"] = .5
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_artifact(path)


@pytest.mark.parametrize("corruption", ["code", "package", "snapshot", "filters", "allowed_count", "epoch", "bitmap"])
def test_cli_rejects_cross_run_code_dataset_and_scope_configuration_mismatches(tmp_path, corruption):
    data = cli_artifacts(tmp_path)
    path = data.candidates["full"]
    manifest, rows = load_artifact(path)
    if corruption == "code":
        manifest["source_hashes"]["src/retrieval/faiss_reference.py"] = "f" * 64
    elif corruption == "package":
        manifest["packages"]["fixture"] = "different-native-build"
    elif corruption == "snapshot":
        manifest["snapshot"]["catalog_sha256"] = "f" * 64
    else:
        rows = list(rows)
        first = dict(rows[0])
        if corruption == "filters":
            first["filters"] = ReadFilters(source_types=("gmail",)).model_dump(mode="json")
        elif corruption == "allowed_count":
            first["allowed_documents"] += 1
        elif corruption == "epoch":
            first["policy_revision"] += 1
        else:
            first["permission_bitmap_sha256"] = "f" * 64
        rows[0] = first
    refresh_candidate(data, "full", manifest_changes=manifest, rows=rows)
    output = tmp_path / "mismatch"
    with pytest.raises(SystemExit) as error:
        compare_cli(data, output)
    assert error.value.code == 2 and not (output / "manifest.json").exists()


def test_cli_cannot_claim_legacy_tie_excerpts_source_checked_without_retained_proof(tmp_path):
    data = cli_artifacts(tmp_path)
    manifest, rows = load_artifact(data.reference)
    rows = list(rows)
    assert rows[0]["legacy_tie_evidence"] and rows[0]["tie_primary_evidence"]
    rows[0] = {**rows[0], "legacy_tie_evidence": []}
    retained_artifact(data.reference, manifest, rows)
    with pytest.raises(SystemExit) as error:
        compare_cli(data, tmp_path / "missing-proof")
    assert error.value.code == 2


def timing_record(iteration=0, seconds=1., *, request_index=0, question_id="q1", case="full"):
    return {"question_id": question_id, "case": case, "iteration": iteration, "request_index": request_index,
            "timings": {"total_seconds": seconds}}


def test_latency_limits_are_inclusive_and_report_every_failure():
    assert latency_gate(120, [timing_record(0, 30), timing_record(1, 10, request_index=1)])["passed"] is True
    result = latency_gate(120.1, [timing_record(0, 30.1), timing_record(1, 10.1, request_index=1)])
    assert result["passed"] is False
    assert {value["kind"] for value in result["violations"]} == {"startup", "first", "warm"}


@pytest.mark.parametrize("question,case", [("q2", "full"), ("q1", "full:source"), ("q1", "full:unknown_date")])
def test_only_process_first_request_gets_30_seconds_even_for_later_iteration_zero(question, case):
    records = [timing_record(0, 29., request_index=0),
               timing_record(0, 11., request_index=1, question_id=question, case=case)]
    result = latency_gate(1., records)
    assert result["passed"] is False
    assert result["violations"] == [{"kind": "warm", "question_id": question, "case": case,
        "iteration": 0, "request_index": 1, "seconds": 11., "limit": 10}]


@pytest.mark.parametrize("index", [None, True, -1, 1, .0])
def test_timing_request_sequence_requires_strict_contiguous_indices_starting_zero(index):
    with pytest.raises(ValueError):
        latency_gate(1., [timing_record(request_index=index)])


@pytest.mark.parametrize("index", [0, 2])
def test_timing_request_sequence_rejects_repeated_or_skipped_later_index(index):
    with pytest.raises(ValueError):
        latency_gate(1., [timing_record(), timing_record(request_index=index)])


def test_cli_new_question_iteration_zero_at_11_seconds_is_a_warm_failure(tmp_path):
    data = cli_artifacts(tmp_path)
    rows = list(data.candidate_rows["full"])
    target = next(index for index, row in enumerate(rows)
                  if row["question_id"] != rows[0]["question_id"] and row["iteration"] == 0)
    assert target > 0 and rows[target]["request_index"] == target
    rows[target] = {**rows[target], "timings": {**rows[target]["timings"], "total_seconds": 11.}}
    refresh_candidate(data, "full", rows=rows)
    output = tmp_path / "later-question-warm-failure"
    assert compare_cli(data, output) == 2
    manifest, _ = load_artifact(output)
    assert manifest["passed"] is False
    full = next(gate for gate in manifest["latency_gates"] if gate["scope"] == "full")
    assert full["passed"] is False
    assert full["violations"][0]["kind"] == "warm" and full["violations"][0]["request_index"] == target


@pytest.mark.parametrize("procfs_available", [True, False])
def test_cli_startup_boundary_uses_linux_process_age_with_driver_entry_fallback(tmp_path, monkeypatch, procfs_available):
    original_read = Path.read_text
    fields = ["S"] + ["0"] * 19
    fields[19] = "10000"  # Process start100s after boot at100 clock ticks/s.

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/stat":
            if not procfs_available:
                raise FileNotFoundError()
            return "123 (synthetic process) " + " ".join(fields)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(harness.time, "perf_counter", lambda: 1000.)
    monkeypatch.setattr(harness.time, "CLOCK_BOOTTIME", 98765, raising=False)
    monkeypatch.setattr(harness.time, "clock_gettime", lambda _clock: 107.)
    monkeypatch.setattr(harness.os, "sysconf", lambda _name: 100)
    observed = []
    monkeypatch.setattr(harness, "_compare", lambda args: observed.append(args.process_started) or True)
    arguments = ["compare", "--shared", "unused", "--reference", "unused", "--candidate", "unused",
                 "--corpus", "unused", "--output", str(tmp_path / "unused")]
    assert harness.main(arguments) == 0
    assert observed == [993. if procfs_available else 1000.]


@pytest.mark.parametrize("value", [True, None, -1., float("nan"), float("inf")])
def test_unknown_or_invalid_latency_measurements_never_pass(value):
    with pytest.raises(ValueError):
        latency_gate(value, [timing_record()])
    with pytest.raises(ValueError):
        latency_gate(1., [timing_record(seconds=value)])


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_latency_iteration_is_a_strict_nonnegative_integer(value):
    with pytest.raises(ValueError):
        latency_gate(1., [timing_record(iteration=value)])


@pytest.mark.parametrize("change", [{"peak_aggregate_bytes": 7 * 1024**3}, {"peak_aggregate_bytes": 0},
    {"min_vm_available_bytes": 0}, {"oom_events_delta": 1}, {"guard_status": "rss_guard"},
    {"complete": False}, {"sample_count": 0}, {"peak_aggregate_bytes": None}])
def test_resource_measurements_missing_failed_or_over_budget_remain_unaccepted(tmp_path, change):
    candidate = tmp_path / "candidate"
    retained_artifact(candidate, {"stage": "candidate"})
    measured = measured_resources("full", candidate)
    assert resource_gate(measured)["passed"] is True
    assert resource_gate({**measured, **change})["passed"] is False
    assert resource_gate({})["passed"] is False


@pytest.mark.parametrize("policy", ["discard", "retain"])
def test_cli_comparison_records_one_coherent_file_cache_policy_for_all_scope_processes(tmp_path, policy):
    data = cli_artifacts(tmp_path)
    for scope in data.candidates:
        refresh_candidate(data, scope, manifest_changes={"file_cache_policy": policy})
    output = tmp_path / "coherent-cache-policy"
    assert compare_cli(data, output) == 0
    manifest, _ = load_artifact(output)
    assert manifest["file_cache_policy"] == policy and manifest["passed"] is True


@pytest.mark.parametrize("policy", ["retain", "auto", None, True])
def test_cli_rejects_mixed_or_invalid_file_cache_policy_without_relaxing_gates(tmp_path, policy):
    data = cli_artifacts(tmp_path)
    # The two untouched older fixtures default to discard; retain is valid only
    # when every separately measured scope process explicitly uses that profile.
    refresh_candidate(data, "full", manifest_changes={"file_cache_policy": policy})
    output = tmp_path / "incoherent-cache-policy"
    with pytest.raises(SystemExit) as error:
        compare_cli(data, output)
    assert error.value.code == 2 and not (output / "manifest.json").exists()


def test_candidate_cli_cache_policy_default_and_explicit_values_are_reviewable(capsys):
    arguments = ["candidate", "--shared", "unused", "--credentials-file", "unused", "--scope", "full",
        "--corpus", "unused", "--dense-directory", "unused", "--bm25-directory", "unused", "--output", "unused"]
    assert harness.parser().parse_args(arguments).file_cache_policy == "discard"
    assert harness.parser().parse_args([*arguments, "--file-cache-policy", "retain"]).file_cache_policy == "retain"
    with pytest.raises(SystemExit) as error:
        harness.main(["candidate", "--help"])
    assert error.value.code == 0
    assert "--file-cache-policy {discard,retain}" in capsys.readouterr().out


def test_component_timings_are_inclusive_nested_observations_not_additive_costs(monkeypatch):
    ticks = iter((10., 11., 13., 15.))
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    observed = {}
    with harness._timed(observed, "outer"):
        with harness._timed(observed, "inner"):
            pass
    assert observed == {"inner_seconds": 2., "inner_calls": 1, "outer_seconds": 5., "outer_calls": 1}


@pytest.mark.parametrize("policy", ["discard", "retain"])
def test_candidate_cache_and_nested_timing_wrappers_keep_every_guard_and_exact_result(tmp_path, monkeypatch, policy):
    """Exercise real tiny adapters with model/DB doubles, not a performance run."""
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import EmbeddingModel
    data = cli_artifacts(tmp_path)
    shared, _ = load_artifact(data.shared)
    vectors = np.load(data.shared / "query_vectors.npy", allow_pickle=False)
    legacy = context(data.data, ("alpha",))

    class Scope:
        binding = legacy.binding
        allowed_count = 1
        bitmap_sha256 = hashlib.sha256(b"\x02").hexdigest()
        def permits(self, document):
            return legacy.scope.permits(document)
        def mask_for(self, snapshot, documents):
            return legacy.scope.mask_for(snapshot, documents)

    request = RequestContext(legacy.identity, Scope())

    class Resolver:
        calls = 0
        def resolve(self, identity):
            assert identity == request.identity
            return request
        def assert_current(self, context):
            self.calls += 1
            context.assert_bound()

    class DatabaseDouble:
        closed = False
        def close(self):
            self.closed = True

    class EmptyLexical:
        # The fixture's quuxneedle query genuinely has no lexical token match.
        doc_ids = data.data.doc_ids
        manifest = {"version": 1, "corpus_sha256": data.data.snapshot.corpus_sha256}
        def rank_candidates(self, query, *_args, **_kwargs):
            assert query == "quuxneedle"
            return ()

    resolver, database = Resolver(), DatabaseDouble()
    encoder = Encoder(data.data.query)
    monkeypatch.setattr(harness, "_shared", lambda _args: (shared, data.data.snapshot, vectors))
    monkeypatch.setattr(harness, "_runtime", lambda *_args, **_kwargs:
                        (database, request.identity, data.data.store, resolver))
    monkeypatch.setattr(CorpusBM25, "load", classmethod(lambda cls, *_args, **_kwargs: EmptyLexical()))
    monkeypatch.setattr(EmbeddingModel, "for_index", classmethod(lambda cls, *_args, **_kwargs: encoder))
    output = tmp_path / "instrumented-candidate"
    args = SimpleNamespace(output=output, shared=data.shared, scope="narrow", file_cache_policy=policy,
        dense_directory=data.data.dense, bm25_directory=tmp_path / "unused", block_rows=4)
    harness._candidate(args)
    manifest, records = load_artifact(output)
    assert database.closed and manifest["file_cache_policy"] == policy
    assert "inclusive" in manifest["component_timing_policy"] and "do not sum" in manifest["component_timing_policy"]
    for component in ("input_verification", "imports", "catalog", "dense_load", "bm25_load", "model_load", "adapters"):
        assert manifest["startup_components"][component + "_calls"] == 1
        assert manifest["startup_components"][component + "_seconds"] >= 0
    assert len(records) == 40 and encoder.calls == 40
    assert resolver.calls == sum(record["timings"]["guard_calls"] for record in records)
    for record in records:
        assert [item["chunk"]["doc_id"] for item in record["evidence"]] == ["alpha"]
        for component in ("rank_bm25", "rank_dense", "materialize", "read_more"):
            assert record["timings"][component + "_calls"] == 1
            assert record["timings"][component + "_seconds"] >= 0
        verify_source_evidence(data.data.corpus, (*record["evidence"], *record["read_more"]),
            snapshot=data.data.snapshot, catalog=data.data.catalog)

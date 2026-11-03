"""Tiny genuine artifact and scope checks for the observational MEM profiler.

No model downloads, full-corpus loads, paid calls or live process supervision.
The loader test uses the real BM25/FAISS loaders with a tiny fake query encoder.
"""

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import profile_retrieval as profile
from src.contracts import Chunk, Evidence, SearchRequest, UserContext


@pytest.fixture
def artifacts(tmp_path):
    corpus, bm25, dense = tmp_path / "documents.parquet", tmp_path / "bm25", tmp_path / "dense"
    bm25.mkdir()
    dense.mkdir()
    rows = {"doc_id": ["a", "b", "c"], "source_type": ["jira", "slack", "jira"],
            "title": ["A", "B", "C"], "content": ["canary rollback", "retention policy", "upload limit"]}
    table = pa.table(rows)
    pq.write_table(table, corpus)
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    model = {"name": "synthetic-encoder", "revision": "test-only", "fp16": False}
    (bm25 / "manifest.json").write_text(json.dumps({"version": 1,
        "corpus_sha256": corpus_hash, "stats": {"documents": 3}}))
    (dense / "manifest.json").write_text(json.dumps({"version": 1, "complete": True,
        "documents": 3, "corpus_sha256": corpus_hash, "shards": 2, "model": model}))
    (dense / "doc_ids.json").write_text(json.dumps(rows["doc_id"]))
    vectors = (np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float16),
               np.array([[.8, .6, 0], [0, 0, 1]], dtype=np.float16))
    for index, (values, documents) in enumerate(zip(vectors, ([0, 1], [0, 2]))):
        stem = f"shard_{index:05d}"
        np.save(dense / f"{stem}.npy", values)
        pq.write_table(pa.table({"doc_index": documents, "start_char": [0, 0],
            "end_char": [len(rows["title"][i] + "\n" + rows["content"][i]) for i in documents]}),
            dense / f"{stem}.parquet")
    return corpus, bm25, dense


def edit_manifest(directory, **changes):
    path = directory / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def test_inventory_reads_every_real_numpy_parquet_shard_and_verifies_payload_hashes(artifacts):
    corpus, bm25, dense = artifacts
    result = profile.inspect_artifacts(*artifacts, hash_payloads=True)
    assert (result["documents"], result["passages"], result["dimension"], result["shard_count"]) == (3, 4, 3, 2)
    assert [row["name"] for row in result["shards"]] == ["shard_00000", "shard_00001"]
    assert [row["stored_dtype"] for row in result["shards"]] == ["float16", "float16"]
    assert result["calculated_float32_vector_bytes"] == 4 * 3 * 4
    assert result["largest_calculated_conversion_bytes"] == 2 * 3 * 4
    assert result["payload_hashes_verified"] is True
    assert result["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    expected = {path for path in (corpus, *bm25.iterdir(), *dense.iterdir()) if path.is_file()}
    assert {Path(item["path"]) for item in result["files"]} == expected
    for item in result["files"]:
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert item["size_bytes"] == Path(item["path"]).stat().st_size


def test_inventory_header_only_inspection_does_not_claim_payload_verification(artifacts):
    result = profile.inspect_artifacts(*artifacts)
    assert result["payload_hashes_verified"] is False
    assert all(item["sha256"] is None for item in result["files"])


@pytest.mark.parametrize("change", ["missing_vector", "missing_metadata", "extra_vector", "extra_metadata"])
def test_inventory_rejects_missing_and_extra_shards(artifacts, change):
    _, _, dense = artifacts
    if change.startswith("missing"):
        (dense / ("shard_00001.npy" if change.endswith("vector") else "shard_00001.parquet")).unlink()
    elif change.endswith("vector"):
        np.save(dense / "shard_00002.npy", np.ones((1, 3), dtype=np.float16))
    else:
        pq.write_table(pa.table({"doc_index": [0]}), dense / "shard_00002.parquet")
    with pytest.raises(ValueError, match="Missing or unexpected dense shards"):
        profile.inspect_artifacts(*artifacts)


@pytest.mark.parametrize("change,message", [
    ("dimensions", "dimensions differ"), ("rows", "rows differ"),
    ("integer_vectors", "vector header"), ("one_dimensional", "vector header"),
    ("documents", "document counts differ"), ("hash_identity", "different corpus"),
    ("actual_hash", "Actual corpus hash"), ("incomplete", "Complete supported"),
])
def test_inventory_fails_closed_on_inconsistent_artifacts(artifacts, change, message):
    _, bm25, dense = artifacts
    if change == "dimensions":
        np.save(dense / "shard_00001.npy", np.ones((2, 4), dtype=np.float16))
    elif change == "rows":
        pq.write_table(pa.table({"doc_index": [0]}), dense / "shard_00001.parquet")
    elif change == "integer_vectors":
        np.save(dense / "shard_00001.npy", np.ones((2, 3), dtype=np.int32))
    elif change == "one_dimensional":
        np.save(dense / "shard_00001.npy", np.ones(3, dtype=np.float16))
    elif change == "documents":
        edit_manifest(dense, documents=2)
    elif change == "hash_identity":
        edit_manifest(dense, corpus_sha256="d" * 64)
    elif change == "actual_hash":
        edit_manifest(dense, corpus_sha256="d" * 64)
        edit_manifest(bm25, corpus_sha256="d" * 64)
    else:
        edit_manifest(dense, complete=False)
    with pytest.raises(ValueError, match=message):
        profile.inspect_artifacts(*artifacts, hash_payloads=True)


@pytest.mark.parametrize("shards", [True, 0, -1, "2"])
def test_inventory_requires_a_positive_integer_shard_count(artifacts, shards):
    edit_manifest(artifacts[2], shards=shards)
    with pytest.raises(ValueError, match="positive integer"):
        profile.inspect_artifacts(*artifacts)


def runtime_tree(root):
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "src/runtime.py").write_text("setting = 1\n")
    (root / "scripts/profile_retrieval.py").write_text("profile = 1\n")
    (root / "scripts/benchmark_web.py").write_text("workloads = ()\n")
    (root / "requirements.txt").write_text("numpy==2\n")
    return root


def fake_git(monkeypatch, *, commit="a" * 40, dirty=False):
    def output(command, **kwargs):
        assert command[0] == "git"
        return commit + "\n" if "rev-parse" in command else (" M src/runtime.py\n" if dirty else "")
    monkeypatch.setattr(profile.subprocess, "check_output", output)


def test_code_identity_is_stable_but_changes_with_actual_runtime_bytes(tmp_path, monkeypatch):
    root = runtime_tree(tmp_path)
    fake_git(monkeypatch, dirty=True)
    first = profile.code_identity(root, declared_commit="a" * 40)
    second = profile.code_identity(root, declared_commit="a" * 40)
    assert first == second
    assert first["commit_verification"] == "git" and first["dirty"] is True
    assert first["runtime_files"]["src/runtime.py"] == hashlib.sha256(b"setting = 1\n").hexdigest()
    (root / "notes.txt").write_text("not runtime input\n")
    assert profile.code_identity(root)["runtime_source_sha256"] == first["runtime_source_sha256"]
    (root / "src/runtime.py").write_text("setting = 2\n")
    changed = profile.code_identity(root)
    assert changed["commit"] == first["commit"]
    assert changed["runtime_source_sha256"] != first["runtime_source_sha256"]


def test_code_identity_rejects_a_stale_declared_commit(tmp_path, monkeypatch):
    fake_git(monkeypatch)
    with pytest.raises(ValueError, match="differs from the checkout"):
        profile.code_identity(runtime_tree(tmp_path), declared_commit="b" * 40)


def test_code_identity_labels_external_declaration_when_git_is_unavailable(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(profile.subprocess, "check_output", unavailable)
    result = profile.code_identity(runtime_tree(tmp_path), declared_commit="c" * 40)
    assert result["commit"] == "c" * 40
    assert result["commit_verification"] == "external_declaration" and result["dirty"] is None


@pytest.mark.parametrize("commit", ["short", "A" * 40, "a" * 39])
def test_code_identity_rejects_malformed_declared_commits(tmp_path, commit):
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        profile.code_identity(tmp_path, declared_commit=commit)


def hit(doc, *, source="jira", start=0, text="Allowed exact excerpt"):
    key = f"{doc}:{start}:{start + len(text)}"
    return Evidence(citation_id=key, chunk=Chunk(chunk_id=key, doc_id=doc,
        source_type=source, text=text, start_char=start, end_char=start + len(text)),
        score=1, retrieval_method="hybrid")


class ScopedRetriever:
    doc_ids = ("a", "b", "c")
    items = (hit("a"), hit("b", source="slack"), hit("c"))

    def __init__(self):
        self.searches, self.reads = [], []

    def search(self, request):
        self.searches.append(request)
        return tuple(item for item in self.items if request.user.permits(item.chunk.doc_id)
            and (not request.source_types or item.chunk.source_type in request.source_types))[:request.top_k]

    def read_more(self, request, anchors, *, max_chars):
        self.reads.append((request, anchors, max_chars))
        return tuple(hit(item.chunk.doc_id, source=item.chunk.source_type, start=100)
                     for item in anchors)


@pytest.fixture
def observation(monkeypatch):
    events = []
    monkeypatch.setattr(profile, "emit", lambda event, **fields: events.append({"event": event, **fields}))
    monkeypatch.setattr(profile, "sample_memory", lambda pid: {"rss_bytes": 0})
    return events


def test_query_workloads_exercises_broad_narrow_denied_filters_and_first_read_more(observation):
    from scripts.benchmark_web import WORKLOADS
    retriever = ScopedRetriever()
    profile.query_workloads(retriever, repeats=2, top_k=2, narrow_size=1)
    queries = [event for event in observation if event["event"] == "query_result"]
    assert len(queries) == 3 * 2 * len(WORKLOADS)
    assert {event["scope"] for event in queries} == {"broad", "narrow", "denied"}
    assert all(event["result_count"] == 0 for event in queries if event["scope"] == "denied")
    assert all({hit["doc_id"] for hit in event["hits"]} == {"a"}
               for event in queries if event["scope"] == "narrow")
    scopes = [event for event in observation if event["event"] == "scope"]
    assert [event["document_count"] for event in scopes] == [3, 1, 0]
    assert len({event["sha256"] for event in scopes}) == 3
    assert len(retriever.reads) == 2 * len(WORKLOADS)  # Once per nonempty scope/workload.
    for request, anchors, max_chars in retriever.reads:
        assert len(anchors) <= 2 and max_chars == 3000
        assert all(request.user.permits(item.chunk.doc_id) for item in anchors)
    filtered = [request for request in retriever.searches if request.source_types]
    assert len(filtered) == 2 * len(WORKLOADS)
    assert all(request.source_types == ("jira",) for request in filtered)
    assert all(event.get("cache_state") != "cold" for event in observation)


@pytest.mark.parametrize("scope", ["broad", "narrow", "denied"])
def test_query_workloads_stops_on_permission_leaks_in_each_scope(observation, scope):
    class Leaking(ScopedRetriever):
        def search(self, request):
            allowed = request.user.allowed_doc_ids
            target = "broad" if len(allowed) == 3 else "narrow" if allowed else "denied"
            return (hit("secret"),) if target == scope else super().search(request)
    with pytest.raises(ValueError, match="authorized"):
        profile.query_workloads(Leaking(), repeats=1, top_k=2, narrow_size=1)
    failed = [event for event in observation if event["event"] == "end" and not event["success"]]
    assert failed


def test_query_workloads_stops_when_search_exceeds_top_k(observation):
    class OverLimit(ScopedRetriever):
        def search(self, request):
            return self.items
    with pytest.raises(ValueError, match="bound"):
        profile.query_workloads(OverLimit(), repeats=1, top_k=1, narrow_size=1)


@pytest.mark.parametrize("failure", ["unauthorized", "too_many", "too_long", "unrelated"])
def test_query_workloads_stops_on_expansion_permission_anchor_or_size_failure(observation, failure):
    class BadRead(ScopedRetriever):
        def read_more(self, request, anchors, *, max_chars):
            if failure == "unauthorized":
                return (hit("secret"),)
            if failure == "too_many":
                return tuple(hit("a", start=i * 100) for i in range(3))
            if failure == "too_long":
                return (hit("a", text="x" * 3001),)
            return (hit("c"),)  # Authorized broadly, but not an anchor when top_k=1.
    with pytest.raises(ValueError):
        profile.query_workloads(BadRead(), repeats=1, top_k=1, narrow_size=1)


@pytest.mark.parametrize("failure", ["wrong_source", "too_many"])
def test_query_workloads_stops_on_source_filter_scope_and_result_bound_failure(observation, failure):
    class BadFilter(ScopedRetriever):
        def search(self, request):
            if request.source_types:
                return ((hit("b", source="slack"),) if failure == "wrong_source"
                        else (hit("a"), hit("c")))
            return super().search(request)
    with pytest.raises(ValueError):
        profile.query_workloads(BadFilter(), repeats=1, top_k=1, narrow_size=1)


@pytest.mark.parametrize("flag,value", [
    (flag, value) for flag in ("--max-rss-gib", "--max-cgroup-gib", "--timeout", "--sample-seconds")
    for value in ("nan", "inf", "-inf", "0")
] + [("--min-vm-available-mib", value) for value in ("nan", "inf", "-inf", "-1")]
  + [("--threads", "0"), ("--repeats", "0"), ("--top-k", "0"), ("--top-k", "101")])
def test_main_rejects_invalid_limits_before_creating_outputs_or_resolving_source(tmp_path, monkeypatch, flag, value):
    output = tmp_path / "must-not-exist"
    argv = ["profile_retrieval.py", "--mode=inventory", f"--corpus={tmp_path / 'none.parquet'}",
            f"--bm25-index={tmp_path / 'bm25'}", f"--dense-index={tmp_path / 'dense'}",
            f"--output={output}", f"{flag}={value}"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(profile, "code_identity", lambda **kwargs: pytest.fail("Invalid limits reached source resolution"))
    monkeypatch.setattr(profile, "supervise", lambda *args, **kwargs: pytest.fail("Invalid limits launched a worker"))
    with pytest.raises(SystemExit):
        profile.main()
    assert not output.exists()


@pytest.mark.parametrize("minimum_mib", [0, 256])
def test_main_resolves_relative_paths_and_passes_finite_vm_floor_to_supervisor(
        tmp_path, monkeypatch, minimum_mib):
    monkeypatch.chdir(tmp_path)
    argv = ["profile_retrieval.py", "--mode=inventory", "--corpus=documents.parquet",
            "--bm25-index=indexes/bm25", "--dense-index=indexes/dense", "--output=report",
            f"--min-vm-available-mib={minimum_mib}"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(profile, "code_identity", lambda **kwargs: {"commit": "a" * 40})
    calls = []
    def supervise(command, output, **kwargs):
        calls.append((command, output, kwargs))
        return {"status": "completed"}
    monkeypatch.setattr(profile, "supervise", supervise)
    profile.main()
    command, output, settings = calls[0]
    assert command[-1] == "--worker"
    assert output == tmp_path / "report"
    assert settings["min_vm_available_bytes"] == minimum_mib * 1024**2
    manifest = json.loads((output / "manifest.json").read_text())
    for key, relative in (("corpus", "documents.parquet"), ("bm25_index", "indexes/bm25"),
                          ("dense_index", "indexes/dense"), ("output", "report")):
        assert manifest["settings"][key] == str(tmp_path / relative)
    assert json.loads((output / "summary.json").read_text()) == {"status": "completed"}


@pytest.mark.parametrize("fail_after_load", [False, True])
def test_loader_instrumentation_preserves_tiny_real_loader_results_and_restores_patches(
        artifacts, observation, monkeypatch, fail_after_load):
    import bm25s
    import faiss
    from src.config import RetrievalSettings
    from src.retrieval import corpus_bm25, corpus_dense, factory

    corpus, bm25, dense = artifacts
    corpus_bm25.CorpusBM25.build(corpus).save(bm25)
    manifest = json.loads((dense / "manifest.json").read_text())
    class Encoder:
        identity = manifest["model"]
        def encode(self, texts, *, query=False):
            assert query is True
            return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(corpus_dense.BGEModel, "for_index", classmethod(lambda cls, *args, **kwargs: Encoder()))
    settings = RetrievalSettings(corpus=corpus, bm25_index=bm25, dense_index=dense,
                                 retrieval="hybrid", device="cpu")
    request = SearchRequest(query="canary retention", user=UserContext(
        user_id="fixture", allowed_doc_ids=frozenset({"a", "b", "c"})), top_k=3)
    reference = factory.load_retriever(settings).search(request)
    touched = [(pq, "read_table"), (corpus_bm25, "file_sha256"), (bm25s.BM25, "load"),
               (np, "array"), (np, "load"), (factory, "load_dense"),
               (corpus_dense, "file_sha256"), (corpus_dense.BGEModel, "for_index"),
               (faiss, "normalize_L2"), (faiss.IndexFlatIP, "add")]
    originals = [inspect.getattr_static(obj, name) for obj, name in touched]

    @contextmanager
    def expected_outcome():
        if fail_after_load:
            with pytest.raises(RuntimeError, match="injected"):
                yield
        else:
            yield

    with expected_outcome():
        with profile.instrument_loaders(settings) as observed_factory:
            observed = observed_factory.load_retriever(settings)
            assert observed.bm25.corpus is observed.dense.corpus
            assert observed.dense.index.ntotal == 4
            assert observed.search(request) == reference
            if fail_after_load:
                raise RuntimeError("injected after loading")
    for (obj, name), original in zip(touched, originals):
        assert inspect.getattr_static(obj, name) is original
    starts = {event["phase"] for event in observation if event["event"] == "start"}
    assert {"corpus.arrow", "corpus.hash_verification", "bm25.index_mmap", "model.load",
            "dense.load", "dense.convert.shard_00000", "dense.convert.shard_00001"} <= starts
    growth = [event["index_rows"] for event in observation if event["event"] == "index_growth"]
    assert growth == [2, 4]

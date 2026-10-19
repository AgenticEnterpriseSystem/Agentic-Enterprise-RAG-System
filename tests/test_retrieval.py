import bm25s
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import re
from datetime import datetime, timezone
from hashlib import md5
from src.contracts import Chunk, Evidence, SearchRequest, UserContext
from src.data.corpus import build_corpus, extract_document, question_coverage
from src.evaluation.run_retrieval import load_questions, run
from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25, tokenize
from src.retrieval.corpus_dense import CorpusDense, embed_corpus, windows
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridConfig, HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


# Corpus and BM25

DOCS = [
    "kappa elevation scale for data grids",
    "rollback the canary deployment safely",
    "canary canary rollback checklist for on call",
    "quarterly sales forecast and pipeline review",
    "grid density tokens and elevation review",
]


def write_doc(root, relative, doc_id, title, content, **extra):
    path = root / "generated_data" / "sources" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_doc_uuid": doc_id,
        "title_field_name": "title",
        "content_field_names": ["body"],
        "title": title,
        "body": content,
        **extra,
    }
    path.write_text(json.dumps(data))


def test_extract_document_matches_upstream_labels():
    single = {"title_field_name": "t", "content_field_names": ["c"], "t": "T", "c": "C"}
    assert extract_document(single) == ("T", "C")
    multi = {
        "title_field_name": "t",
        "content_field_names": ["a", "b"],
        "t": 1,
        "a": ["x", "y"],
        "b": "z",
    }
    assert extract_document(multi) == ("1", "a:\nx\ny\n\nb:\nz")
    with pytest.raises(ValueError):
        extract_document({"title_field_name": "t", "t": "T"})


def test_build_corpus_uses_canonical_file_for_reused_ids(tmp_path):
    repo = tmp_path / "repo"
    write_doc(repo, "slack/a.json", "d1", "A", "alpha")
    write_doc(repo, "jira/b.json", "d2", "B", "beta")
    write_doc(repo, "confluence/dup.json", "d2", "B2", "beta copy")
    index = {"d1": "slack/a.json", "d2": "jira/b.json"}
    (repo / "generated_data" / "uuid_index.json").write_text(json.dumps(index))
    output = tmp_path / "docs.parquet"
    report = build_corpus(repo, output, batch_size=1)
    rows = pq.read_table(output).to_pylist()
    assert [r["doc_id"] for r in rows] == ["d2", "d1"]  # Sorted source directories.
    assert rows[0]["source_type"] == "jira" and rows[0]["content"] == "beta"
    assert report["source_counts"] == {"jira": 1, "slack": 1}
    assert report["skipped"] == [
        {"path": "confluence/dup.json", "reason": "not canonical for d2"}
    ]
    coverage = question_coverage(
        [{"expected_doc_ids": ["d1", "missing"]}, {"expected_doc_ids": []}],
        report["doc_ids"],
    )
    assert coverage["referenced_documents_missing"] == ["missing"]
    assert coverage["questions_without_references"] == 1


def test_build_corpus_fails_when_indexed_document_is_missing(tmp_path):
    repo = tmp_path / "repo"
    write_doc(repo, "slack/a.json", "d1", "A", "alpha")
    index = {"d1": "slack/a.json", "d9": "slack/gone.json"}
    (repo / "generated_data" / "uuid_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="not exported"):
        build_corpus(repo, tmp_path / "docs.parquet")


def test_vectorized_index_matches_bm25s():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(200)]
    docs = [" ".join(rng.choice(words, rng.integers(1, 50))) for _ in range(300)]
    model, stats = build_bm25(docs, BM25Config())
    tokens = [tokenize(d) for d in docs]
    reference = bm25s.BM25(k1=1.5, b=0.75, delta=1, method="bm25+")
    reference.index(tokens, show_progress=False)
    assert stats["documents"] == 300
    for query in (["w1", "w2"], ["w7", "w7", "w99"], ["w150"]):
        ours = model.get_scores_from_ids([model.vocab_dict[t] for t in query])
        expected = reference.get_scores(query)
        # float32 rounding differs slightly; rankings must not.
        np.testing.assert_allclose(ours, expected, rtol=1e-6)
        np.testing.assert_array_equal(
            np.round(ours, 4).argsort(kind="stable"),
            np.round(expected, 4).argsort(kind="stable"),
        )


@pytest.fixture
def corpus_index(tmp_path):
    table = {
        "doc_id": [f"d{i}" for i in range(len(DOCS))],
        "source_type": ["linear", "slack", "confluence", "gmail", "linear"],
        "title": ["Grid", "", "Runbook", "Sales", "Tokens"],
        "content": DOCS,
        "path": [f"p{i}" for i in range(len(DOCS))],
    }
    import pyarrow as pa

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table(table), path)
    index = CorpusBM25.build(path)
    index.save(tmp_path / "index")
    return CorpusBM25.load(tmp_path / "index", corpus_path=path), path


def test_rank_skips_nonmatches_and_respects_masks(corpus_index):
    index, _ = corpus_index
    hits = index.rank("canary rollback", top_k=10)
    assert [d for d, _ in hits] == ["d2", "d1"]
    assert hits[0][1] > hits[1][1]
    assert index.rank("unknownword", top_k=5) == []
    mask = np.array([True, True, False, True, True])
    assert [d for d, _ in index.rank("canary rollback", 10, mask)] == ["d1"]
    assert len(index.rank("elevation grid review", top_k=1)) == 1


def test_search_enforces_permissions_and_sources(corpus_index):
    index, _ = corpus_index
    user = UserContext(user_id="u", allowed_doc_ids=frozenset({"d1", "d4"}))
    results = index.search(
        SearchRequest(query="canary elevation review", user=user, top_k=5)
    )
    assert {e.chunk.doc_id for e in results} == {"d1", "d4"}
    only = index.search(
        SearchRequest(query="canary elevation", user=user, source_types=("linear",))
    )
    assert [e.chunk.doc_id for e in only] == ["d4"]
    assert only[0].chunk.text == "Tokens\ngrid density tokens and elevation review"
    anonymous = UserContext(allowed_doc_ids=frozenset({"d1"}))
    assert index.search(SearchRequest(query="canary", user=anonymous)) == ()


def test_load_rejects_different_corpus(corpus_index, tmp_path):
    _, path = corpus_index
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Corpus differs"):
        CorpusBM25.load(tmp_path / "index", corpus_path=path)


def test_runner_records_rankings_and_errors(tmp_path):
    questions = tmp_path / "q.jsonl"
    rows = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "source_types": ["slack"],
            "question": "canary",
            "expected_doc_ids": ["d1"],
        },
        {
            "question_id": "q2",
            "question_type": "basic",
            "source_types": [],
            "question": "boom",
            "expected_doc_ids": ["d2"],
        },
    ]
    questions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"development": ["q1", "q2"]}}))

    class Fake:
        def rank(self, query, top_k):
            if query == "boom":
                raise RuntimeError("failed")
            return [("d1", 2.0), ("d3", 1.0)]

    selected = load_questions(questions, splits, "development")
    output, latencies = run(Fake(), selected, top_k=20)
    assert output[0]["retrieved_doc_ids"] == ["d1", "d3"] and output[0]["error"] is None
    assert output[1]["retrieved_doc_ids"] == [] and "failed" in output[1]["error"]
    assert len(latencies) == 2
    splits.write_text(json.dumps({"splits": {"development": ["q1", "q9"]}}))
    with pytest.raises(ValueError):
        load_questions(questions, splits, "development")


# Dense retrieval

class FakeModel:
    """Whitespace tokens and hashed bag-of-words vectors; no model download."""

    identity = {"name": "fake", "fp16": False}

    def __init__(self):
        self.encoded = 0

    def offsets(self, texts):
        return [[m.span() for m in re.finditer(r"\S+", t)] for t in texts]

    def encode(self, texts, query=False):
        self.encoded += len(texts)
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(md5(word.encode()).hexdigest(), 16) % 32] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def test_windows_cover_document_with_overlap():
    offsets = [(i * 2, i * 2 + 1) for i in range(7)]  # Seven one-character tokens.
    assert windows(offsets, window=3, overlap=1) == [(0, 5), (4, 9), (8, 13)]
    assert windows(offsets[:2], window=3, overlap=1) == [(0, 3)]
    assert windows([], window=3, overlap=1) == []
    with pytest.raises(ValueError):
        windows(offsets, window=3, overlap=3)


def test_dense_streaming_matches_reference_and_reuses_corpus(corpus, tmp_path):
    import faiss
    out = tmp_path / "dense"
    model = FakeModel()
    manifest = embed_corpus(corpus, out, model, shard_docs=2, window=4, overlap=1)
    names = [f"shard_{i:05d}" for i in range(manifest["shards"])]
    vectors = np.concatenate([np.load(out / f"{n}.npy") for n in names]).astype(np.float32)
    faiss.normalize_L2(vectors)
    metadata = [pq.read_table(out / f"{n}.parquet") for n in names]
    docs = np.concatenate([t.column("doc_index").to_numpy() for t in metadata])
    ids = json.loads((out / "doc_ids.json").read_text())
    reference = CorpusDense(vectors, docs, ids, model, manifest)
    shared = pq.read_table(corpus)
    loaded = CorpusDense.load(out, model, corpus_path=corpus, corpus=shared)
    assert loaded.corpus is shared
    assert loaded.index.ntotal == len(vectors)
    for query in ("canary rollback", "grid elevation", "quarterly forecast"):
        assert loaded.rank(query, 4) == reference.rank(query, 4)
    first = out / "shard_00000.npy"
    np.save(first, np.load(first)[:-1])
    with pytest.raises(ValueError, match="inconsistent"):
        CorpusDense.load(out, model)


@pytest.fixture
def corpus(tmp_path):
    table = {
        "doc_id": ["d0", "d1", "d2", "d3", "d4"],
        "source_type": ["slack"] * 5,
        "title": ["Deploy", "", "Sales", "Grid", ""],
        "content": [
            "canary rollback steps for the deployment",
            "lunch menu " * 6 + "canary rollback",
            "quarterly forecast review",
            "elevation tokens for grid density",
            "",
        ],
        "path": ["p"] * 5,
    }
    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table(table), path)
    return path


def test_embed_is_resumable_and_rank_uses_best_passage(corpus, tmp_path):
    out = tmp_path / "dense"
    model = FakeModel()
    manifest = embed_corpus(corpus, out, model, shard_docs=2, window=4, overlap=1)
    assert manifest["complete"] and manifest["shards"] == 3
    passages = pq.read_table(out / "shard_00000.parquet").to_pydict()
    # d1 is long enough for several windows; every window maps back to d1.
    assert passages["doc_index"].count(1) > 1
    first = model.encoded
    embed_corpus(corpus, out, model, shard_docs=2, window=4, overlap=1)
    assert model.encoded == first  # Completed shards are not recomputed.
    with pytest.raises(ValueError, match="different settings"):
        embed_corpus(corpus, out, model, shard_docs=2, window=8, overlap=1)

    index = CorpusDense.load(out, model, corpus_path=corpus)
    hits = index.rank("canary rollback", top_k=2)
    assert {d for d, _ in hits} == {"d0", "d1"}
    # d4 has no text, so it has no passages and can never be returned.
    assert {d for d, _ in index.rank("canary rollback", top_k=10)} == {
        "d0",
        "d1",
        "d2",
        "d3",
    }
    mask = np.array([False, True, True, True, True])
    assert "d0" not in [d for d, _ in index.rank("canary rollback", 2, mask)]
    scores = [s for _, s in index.rank("grid elevation", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_load_rejects_incomplete_or_mismatched_embeddings(corpus, tmp_path):
    out = tmp_path / "dense"
    embed_corpus(corpus, out, FakeModel(), shard_docs=5, window=4, overlap=1)
    other = FakeModel()
    other.identity = {"name": "other", "fp16": False}
    with pytest.raises(ValueError, match="Query model"):
        CorpusDense.load(out, other)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["complete"] = False
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        CorpusDense.load(out, FakeModel())


def test_bm25_cutoff_ties_are_independent_of_document_order():
    for doc_ids in (["z", "y", "x", "w", "v", "a"], ["v", "a", "w", "x", "y", "z"]):
        model, _ = build_bm25(["shared"] * len(doc_ids), BM25Config())
        index = CorpusBM25(model, doc_ids, BM25Config())
        assert [doc for doc, _ in index.rank("shared", 2)] == ["a", "v"]
        mask = np.array([doc != "a" for doc in doc_ids])
        assert [doc for doc, _ in index.rank("shared", 2, mask)] == ["v", "w"]


def test_passages_keep_exact_offsets_and_dense_representative():
    from src.contracts import Chunk, Evidence
    from src.retrieval.passages import bounded_passage
    from src.retrieval.fusion import reciprocal_rank_fusion
    text = "irrelevant filler " * 400 + " rollback canary recovery procedure " + "more filler " * 200
    chunk = Chunk(chunk_id="original", doc_id="d", source_type="slack", text=text,
                  start_char=17, end_char=17 + len(text))
    whole = Evidence(citation_id="original", chunk=chunk, score=2, retrieval_method="bm25")
    selected = bounded_passage(whole, "What is the canary rollback recovery procedure?", 500)
    assert "rollback canary recovery procedure" in selected.chunk.text
    lo, hi = selected.chunk.start_char - 17, selected.chunk.end_char - 17
    assert selected.chunk.text == text[lo:hi]
    assert selected.citation_id == f"d:{17 + lo}:{17 + hi}"
    assert selected.chunk.chunk_id == selected.citation_id
    assert len(selected.chunk.text) == 500
    dense = selected.model_copy(update={"retrieval_method": "dense"})
    fused = reciprocal_rank_fusion(((whole,), (dense,)))
    assert fused[0].chunk == dense.chunk
    assert fused[0].citation_id == dense.citation_id
    assert fused[0].score == pytest.approx(2 / 61)
    assert bounded_passage(dense, "other query", 500) is dense


def test_bm25_search_bounds_long_document_without_changing_rank(tmp_path):
    text = "filler " * 1200 + " rollback canary recovery procedure " + "filler " * 500
    path = tmp_path / "long.parquet"
    table = pa.table({"doc_id": ["d"], "title": [""], "content": [text], "source_type": ["slack"]})
    pq.write_table(table, path)
    index = CorpusBM25.build(path)
    index.corpus = table
    request = SearchRequest(query="canary rollback", user=UserContext(user_id="u", allowed_doc_ids={"d"}))
    result = index.search(request)[0]
    assert len(result.chunk.text) == 3000
    assert "rollback canary recovery" in result.chunk.text
    assert result.chunk.text == text[result.chunk.start_char:result.chunk.end_char]
    assert result.score == pytest.approx(index.rank(request.query, 1)[0][1])


def test_bm25_load_reuses_validated_corpus_table(corpus_index, tmp_path):
    existing, path = corpus_index
    shared = CorpusBM25.load(tmp_path / "index", corpus_path=path, corpus=existing.corpus)
    assert shared.corpus is existing.corpus
    with pytest.raises(ValueError, match="order"):
        CorpusBM25.load(tmp_path / "index", corpus_path=path, corpus=existing.corpus.take(pa.array([4, 3, 2, 1, 0])))


def test_reranker_receives_bounded_text_and_records_token_limit():
    from src.contracts import Chunk, Evidence
    from src.retrieval.reranker import CrossEncoderReranker
    text = "filler " * 2000 + " canary rollback procedure "
    chunk = Chunk(chunk_id="whole", doc_id="d", source_type="docs", text=text,
                  start_char=0, end_char=len(text))
    item = Evidence(citation_id="whole", chunk=chunk, score=1, retrieval_method="bm25")
    seen = []
    def score(pairs):
        seen.extend(pairs)
        return [1]
    reranker = CrossEncoderReranker("offline", "a" * 40, scorer=score, max_length=256)
    result = reranker.rerank("canary rollback", (item,), 1)[0]
    assert len(seen[0][1]) == 3000
    assert "canary rollback procedure" in seen[0][1]
    assert result.chunk.text == seen[0][1]
    assert result.citation_id == result.chunk.chunk_id
    assert reranker.identity["max_length"] == 256


# Hybrid fusion, reranking, and dense evidence



def evidence(doc, *, source="slack", timestamp=None, text="text"):
    chunk = Chunk(chunk_id=f"{doc}:0:{len(text)}", doc_id=doc, source_type=source,
                  text=text, start_char=0, end_char=len(text), timestamp=timestamp)
    return Evidence(citation_id=chunk.chunk_id, chunk=chunk, score=1, retrieval_method="bm25")


class Fixed:
    def __init__(self, *items):
        self.items = items
        self.requests = []
    def search(self, request):
        self.requests.append(request)
        return self.items


def request(**kwargs):
    return SearchRequest(query="question", user=UserContext(user_id="u", allowed_doc_ids={"a", "b", "c"}), **kwargs)


def test_rrf_deduplicates_documents_and_deterministic_ties():
    a, b = evidence("a"), evidence("b")
    duplicate = evidence("a", text="second passage")
    out = reciprocal_rank_fusion(((b, a, duplicate), (a, b)))
    assert [x.chunk.doc_id for x in out] == ["a", "b"]
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert out[0].chunk.text == "text"
    assert all(x.retrieval_method == "hybrid" for x in out)
    assert reciprocal_rank_fusion(((a,), (b,)), weights=(0, 1))[0].chunk.doc_id == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(((a,),), weights=(float("nan"),))


def test_hybrid_filters_before_reranker_and_preserves_scope():
    old = evidence("a", timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc))
    fresh = evidence("b", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    wrong = evidence("c", source="wiki", timestamp=fresh.chunk.timestamp)
    scored = []
    def scorer(pairs):
        scored.extend(pairs)
        return [5] * len(pairs)
    ranker = CrossEncoderReranker("offline", "a" * 40, scorer=scorer)
    left, right = Fixed(old, fresh, wrong), Fixed(fresh)
    hybrid = HybridRetriever(left, right, config=HybridConfig(candidate_k=30), reranker=ranker)
    req = request(top_k=1, source_types=("slack",), after=datetime(2024, 1, 1, tzinfo=timezone.utc))
    out = hybrid.search(req)
    assert [x.chunk.doc_id for x in out] == ["b"]
    assert scored == [("question", "text")]
    assert left.requests[0].user == req.user
    assert left.requests[0].top_k == 30
    assert hybrid.manifest["reranker"] == ranker.identity
    with pytest.raises(ValueError, match="authorized"):
        HybridRetriever(Fixed(evidence("secret")), right).search(req)


def test_unknown_date_filtered_and_empty_identity_skips_search():
    backend = Fixed(evidence("a"))
    hybrid = HybridRetriever(backend, backend)
    assert hybrid.search(request(after=datetime(2024, 1, 1, tzinfo=timezone.utc))) == ()
    backend.requests.clear()
    assert hybrid.search(SearchRequest(query="q", user=UserContext())) == ()
    assert backend.requests == []


def test_reranker_checks_scores_and_pin():
    with pytest.raises(ValueError, match="revision"):
        CrossEncoderReranker("model", "main")
    ranker = CrossEncoderReranker("offline", "a" * 40, scorer=lambda pairs: [0, 5])
    assert ranker.rerank("q", (evidence("a"), evidence("b")), 1)[0].chunk.doc_id == "b"
    for scores in ([], [float("inf")]):
        ranker.scorer = lambda pairs: scores
        with pytest.raises(ValueError):
            ranker.rerank("q", (evidence("a"),), 1)


class Model:
    def encode(self, texts, query=False):
        return np.array([[1, 0]], dtype=np.float32)


def dense_index(**kwargs):
    corpus = pa.table({"doc_id": ["a", "b"], "source_type": ["slack", "wiki"],
                       "title": ["Title", ""], "content": ["some exact passage", "private"]})
    options = {"corpus": corpus, "spans": np.array([[6, 10], [11, 24], [0, 7]])}
    options.update(kwargs)
    return CorpusDense(np.array([[.8, .2], [1., 0], [.99, .1]], dtype=np.float32),
                       np.array([0, 0, 1]), ["a", "b"], Model(), {}, **options)


def test_dense_returns_best_exact_passage_with_scope_and_sources():
    index = dense_index()
    req = SearchRequest(query="q", user=UserContext(user_id="u", allowed_doc_ids={"a"}), top_k=1)
    item = index.search(req)[0]
    assert item.chunk.text == "exact passage"
    assert (item.chunk.start_char, item.chunk.end_char) == (11, 24)
    assert item.citation_id == "a:11:24"
    assert item.retrieval_method == "dense"
    assert index.search(req.model_copy(update={"source_types": ("wiki",)})) == ()
    with pytest.raises(ValueError, match="offsets"):
        dense_index(spans=np.array([[0, 5], [0, 100], [0, 7]])).search(req)
    with pytest.raises(ValueError, match="order"):
        dense_index(corpus=pa.table({"doc_id": ["b", "a"]}))


def test_reranker_cannot_inject_or_change_candidates():
    class Inject:
        def rerank(self, query, items, top_k):
            return (evidence("b").model_copy(update={"retrieval_method": "hybrid"}),)
    with pytest.raises(ValueError, match="outside filtered"):
        HybridRetriever(Fixed(evidence("a")), Fixed(), reranker=Inject()).search(request())


def test_hybrid_requires_matching_index_documents():
    a, b = Fixed(), Fixed()
    a.doc_ids, b.doc_ids = ["a"], ["b"]
    with pytest.raises(ValueError, match="same documents"):
        HybridRetriever(a, b)


class RecordingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.passages = []

    def encode(self, texts, query=False):
        if not query:
            self.passages.extend(texts)
        return super().encode(texts, query)


def test_title_on_every_passage_keeps_exact_spans(corpus, tmp_path):
    from src.retrieval.corpus_bm25 import document_text
    from src.retrieval.corpus_dense import document_passages

    model = FakeModel()
    titled = list(document_passages(["Deploy"], ["a b c d e f g"], model.offsets, 4, 1, True))[0]
    # Title takes one token, so content windows hold three tokens each.
    assert [text for text, _, _ in titled] == ["Deploy\na b c", "Deploy\nc d e", "Deploy\ne f g"]
    full = document_text("Deploy", "a b c d e f g")
    assert [full[s:e] for _, s, e in titled] == ["a b c", "c d e", "e f g"]
    plain = list(document_passages(["Deploy"], ["a b c d e f g"], model.offsets, 4, 1))[0]
    assert sum("Deploy" in text for text, _, _ in plain) == 1
    # Untitled documents fall back to plain windows; title-only documents embed the title.
    assert list(document_passages(["", "Only"], ["x y", ""], model.offsets, 4, 1, True)) == [
        [("x y", 0, 3)],
        [("Only", 0, 4)],
    ]

    out = tmp_path / "titled"
    recorder = RecordingModel()
    manifest = embed_corpus(corpus, out, recorder, shard_docs=2, window=4, overlap=1, title_every=True)
    assert manifest["title_every_passage"] is True
    # d0 ("Deploy", 6 words) spans several windows; each carries the title.
    d0 = [p for p in recorder.passages if p.startswith("Deploy\n")]
    assert len(d0) > 1 and any("deployment" in p for p in d0)
    with pytest.raises(ValueError, match="different settings"):
        embed_corpus(corpus, out, recorder, shard_docs=2, window=4, overlap=1)

    table = pq.read_table(corpus)
    index = CorpusDense.load(out, FakeModel(), corpus_path=corpus, corpus=table)
    user = UserContext(user_id="u", allowed_doc_ids=frozenset({"d0", "d1", "d2", "d3"}))
    evidence = index.search(SearchRequest(query="deploy canary rollback", user=user, top_k=4))
    assert evidence and evidence[0].chunk.doc_id == "d0"
    for item in evidence:
        row = table.slice(index.position[item.chunk.doc_id], 1).to_pylist()[0]
        text = document_text(row["title"], row["content"])
        assert text[item.chunk.start_char:item.chunk.end_char] == item.chunk.text


def test_bge_model_choice_and_query_model_from_manifest(tmp_path, monkeypatch):
    import sys
    import types
    from src.config import EMBEDDING_MODELS, EMBEDDING_REVISION
    from src.retrieval.corpus_dense import BGEModel, method_label

    loaded = []

    class FakeST:
        def __init__(self, name, revision, **kwargs):
            loaded.append((name, revision))
            self.max_seq_length = 512

        def half(self):
            pass

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    assert method_label(EMBEDDING_MODELS["bge-small"]) == "bge-small"
    assert method_label(EMBEDDING_MODELS["bge-base"]) == "bge-base"
    assert BGEModel().identity["revision"] == EMBEDDING_REVISION
    with pytest.raises(ValueError, match="Unsupported"):
        BGEModel(name="other/model", revision="0" * 40)
    with pytest.raises(ValueError, match="commit hash"):
        BGEModel(name=EMBEDDING_MODELS["bge-base"], revision="main")

    base = BGEModel("cuda", name=EMBEDDING_MODELS["bge-base"], revision="a" * 40)
    (tmp_path / "manifest.json").write_text(json.dumps({"model": base.identity}))
    query = BGEModel.for_index(tmp_path)  # CPU device, but precision follows the index.
    assert query.identity == base.identity
    assert loaded[-1] == (EMBEDDING_MODELS["bge-base"], "a" * 40)

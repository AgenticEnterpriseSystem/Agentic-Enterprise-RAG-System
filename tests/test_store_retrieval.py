"""Store adapter behavior against real tiny scores, spans and legacy oracles.

Only accepted contracts and the eager contract fixture are used. No PostgreSQL,
canonical production corpus, downloaded model or runtime factory is involved.
"""

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, PolicyUnavailable,
    RequestContext, ResolvedIdentity, SessionExpired, SnapshotMismatch, StaleContext,
)
from src.contracts import Chunk, PassageEvidence, ScopeBinding, SearchRequest, SnapshotRef, UserContext
from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridConfig, HybridRetriever
from src.retrieval.passages import bounded_passage
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sharded_dense import ShardedDense
from src.retrieval.store import (
    StoreBM25Retriever, StoreDenseRetriever, StoreHybridRetriever, _StoreReader, reciprocal_rank_candidates,
)
from src.storage.contracts import CatalogEntry, RankedCandidate, ReadFilters
from src.storage.fingerprints import ValidatedCatalog, canonical_revision_id, catalog_fingerprint
from src.storage.testing import FixtureDocument, InMemoryDocumentStore


NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
MODEL = {"name": "tiny-test-encoder", "revision": "b" * 40, "normalize": True}


class Guard:
    def __init__(self):
        self.revoked = False
        self.calls = 0
        self.failure = None

    def assert_current(self, context):
        self.calls += 1
        if self.failure:
            raise self.failure("Policy/session is unavailable")
        if self.revoked or context.identity.expires_at <= NOW:
            raise SessionExpired("Session is no longer current")


class TrackingStore(InMemoryDocumentStore):
    def __init__(self, snapshot, documents):
        super().__init__(snapshot, documents)
        self.spans = []

    def read_passage(self, context, candidate, *, guard, filters=ReadFilters()):
        self.spans.append((candidate.doc_id, candidate.start_char, candidate.end_char))
        return super().read_passage(context, candidate, guard=guard, filters=filters)


def fixture_store(texts=None, *, sources=None, timestamps=None):
    texts = texts or {"alpha": "needle alpha needle", "beta": "needle beta", "secret": "needle executive secret"}
    sources = sources or {}
    timestamps = timestamps or {}
    entries = tuple(CatalogEntry(doc_id=doc_id, revision_id=canonical_revision_id(text),
        position=position, source_type=sources.get(doc_id, "jira"), text_length=len(text),
        timestamp=timestamps.get(doc_id)) for position, (doc_id, text) in enumerate(texts.items()))
    snapshot = SnapshotRef(snapshot_id="tiny-store", corpus_sha256="a" * 64,
        catalog_sha256=catalog_fingerprint(entries, expected_count=len(entries)), document_count=len(entries))
    return TrackingStore(snapshot, (FixtureDocument(entry, texts[entry.doc_id]) for entry in entries))


def entries(store):
    return tuple(entry for batch in store.iter_catalog() for entry in batch)


def context(store, allowed=None, *, snapshot=None, scope=None):
    snapshot = snapshot or store.snapshot()
    binding = ScopeBinding(scope_id="resolved", user_id="intern", snapshot=snapshot,
                           policy_revision=4, session_version=2)
    identity = ResolvedIdentity(user_id="intern", session_id="opaque-row-id", session_version=2,
                               expires_at=NOW + timedelta(hours=1))
    if scope is None:
        scope = LegacySetScope(binding=binding, allowed_doc_ids=frozenset(
            store._documents if allowed is None else allowed))
    return RequestContext(identity, scope)


def bm25_index(store, *, legacy=False):
    records = tuple(store._documents.values())
    config = BM25Config()
    model, _ = build_bm25((record.text for record in records), config)
    table = pa.table({"doc_id": [r.entry.doc_id for r in records],
                      "source_type": [r.entry.source_type for r in records],
                      "title": [r.title for r in records], "content": [r.text for r in records]}) if legacy else None
    return CorpusBM25(model, [r.entry.doc_id for r in records], config, corpus=table,
                      manifest={"version": 1, "corpus_sha256": store.snapshot().corpus_sha256})


def candidate(store, doc_id, *, method="dense", score=1., start=0, end=None):
    entry = store._documents[doc_id].entry
    return RankedCandidate(snapshot=store.snapshot(), doc_id=doc_id, revision_id=entry.revision_id,
        source_type=entry.source_type, start_char=start, end_char=entry.text_length if end is None else end,
        score=float(score), retrieval_method=method, scoring_provenance=method + "-fixture")


def passage(store, value):
    record = store._documents[value.doc_id]
    key = f"{value.doc_id}:{value.start_char}:{value.end_char}"
    return PassageEvidence(snapshot_id=store.snapshot().snapshot_id, revision_id=value.revision_id,
        citation_id=key, score=value.score, retrieval_method=value.retrieval_method,
        chunk=Chunk(chunk_id=key, doc_id=value.doc_id, source_type=value.source_type,
            title=record.title, timestamp=record.entry.timestamp, metadata_origin=record.entry.metadata_origin,
            start_char=value.start_char, end_char=value.end_char,
            text=record.text[value.start_char:value.end_char]))


class Encoder:
    def __init__(self, callback=None):
        self.identity = dict(MODEL)
        self.calls = []
        self.callback = callback

    def encode(self, texts, *, query):
        self.calls.append((texts, query))
        if self.callback:
            self.callback()
        return np.array([[1., 0.]], dtype=np.float32)


class ScriptedDense:
    def __init__(self, store, rows=None):
        self.snapshot = store.snapshot()
        self.catalog = entries(store)
        self.doc_ids = tuple(entry.doc_id for entry in self.catalog)
        self.manifest = {"model": dict(MODEL)}
        self.rows = tuple(rows) if rows is not None else tuple(candidate(store, doc) for doc in self.doc_ids)
        self.calls = []

    def candidates_vector(self, vector, top_k, *, scope, filters):
        self.calls.append((np.asarray(vector).copy(), top_k, scope, filters))
        catalog = {entry.doc_id: entry for entry in self.catalog}
        return tuple(value for value in self.rows if scope.permits(value.doc_id)
                     and filters.matches(catalog[value.doc_id]))[:top_k]


def hybrid(store, *, rows=None, config=None, reranker=None):
    bm25 = StoreBM25Retriever(bm25_index(store), store)
    index, encoder = ScriptedDense(store, rows), Encoder()
    dense = StoreDenseRetriever(index, encoder, store)
    return StoreHybridRetriever(bm25, dense, config=config, reranker=reranker), index, encoder


def genuine_dense(tmp_path, store, rows):
    directory = tmp_path / "dense"
    directory.mkdir()
    catalog = entries(store)
    positions = {entry.doc_id: entry.position for entry in catalog}
    values = np.array([vector for _doc, _start, _end, vector in rows], dtype=np.float16)
    np.save(directory / "shard_00000.npy", values, allow_pickle=False)
    pq.write_table(pa.table({"doc_index": [positions[doc] for doc, *_ in rows],
                            "start_char": [start for _, start, _, _ in rows],
                            "end_char": [end for _, _, end, _ in rows]}), directory / "shard_00000.parquet")
    manifest = {"version": 1, "complete": True, "documents": len(catalog), "shards": 1,
                "corpus_sha256": store.snapshot().corpus_sha256, "model": dict(MODEL)}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "doc_ids.json").write_text(json.dumps([entry.doc_id for entry in catalog]))
    return ShardedDense.load(directory, snapshot=store.snapshot(), catalog=catalog, block_rows=2)


@pytest.mark.parametrize("weights", [(1., 1.), (2., .25), (0., 1.), (1., 0.), (0., 0.)])
@pytest.mark.parametrize("k", [0., 60.])
def test_metadata_fusion_matches_legacy_weights_duplicate_rank_and_dense_preference(weights, k):
    store = fixture_store()
    lexical = tuple(candidate(store, doc, method="bm25", score=4.) for doc in ("beta", "beta", "alpha"))
    dense = tuple(candidate(store, doc, score=.7, start=1) for doc in ("alpha", "secret", "secret", "beta"))
    rankings = (lexical, dense)
    old = reciprocal_rank_fusion(tuple(tuple(passage(store, item) for item in ranking) for ranking in rankings),
                                 k=k, weights=weights)
    new = reciprocal_rank_candidates(rankings, k=k, weights=weights)
    assert [(item.doc_id, item.score, item.start_char, item.end_char, item.retrieval_method) for item in new] == [
        (item.chunk.doc_id, item.score, item.chunk.start_char, item.chunk.end_char, item.retrieval_method) for item in old]
    assert all(item.scoring_provenance.startswith("rrf-v1:") for item in new)
    assert store.body_reads == []


def test_metadata_fusion_ties_use_doc_id_and_reject_cross_snapshot_or_revision():
    store = fixture_store()
    first, second = candidate(store, "beta"), candidate(store, "alpha")
    result = reciprocal_rank_candidates(((first,), (second,)), limit=1)
    assert [item.doc_id for item in result] == ["alpha"]
    for changed in (first.model_copy(update={"revision_id": "other"}), first.model_copy(update={
            "snapshot": first.snapshot.model_copy(update={"snapshot_id": "other"})})):
        with pytest.raises(EvidenceValidationError):
            reciprocal_rank_candidates(((first,), (changed,)))
    changed = second.model_copy(update={"snapshot": second.snapshot.model_copy(update={"snapshot_id": "other"})})
    with pytest.raises(SnapshotMismatch):
        reciprocal_rank_candidates(((first,), (changed,)))
    with pytest.raises(ValueError, match="finite"):
        reciprocal_rank_candidates(((first,), (first,), (first,)), k=0, weights=(1e308,) * 3)


@pytest.mark.parametrize("changes", [{"weights": (float("nan"),)}, {"weights": (-1.,)},
    {"k": float("inf")}, {"k": -1}, {"limit": 0}, {"limit": True}])
def test_metadata_fusion_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):
        reciprocal_rank_candidates(((),), **changes)


@pytest.mark.parametrize("query", ["needle trail", "the and to", "Needle needle TRAIL", "不存在 needle"])
@pytest.mark.parametrize("length", [3000, 3001, 6000, 7601])
def test_bm25_streamed_windows_match_old_passage_selector_including_final_start(query, length):
    text = ("plain " * (length // 6 + 1))[:length - 15] + " needle trail  "
    text = text[:length]
    store = fixture_store({"long": text, "other": "completely different words"})
    index = bm25_index(store)
    retriever = StoreBM25Retriever(index, store)
    ctx, guard = context(store), Guard()
    # The pure selector is the oracle even when the query has only stopwords.
    selected = candidate(store, "long", method="bm25", score=3.)
    actual = retriever._materialize(ctx, selected, query, guard, ReadFilters())
    expected = bounded_passage(passage(store, selected), query)
    assert actual == expected
    assert all(end - start <= 3000 for _, start, end in store.spans)
    assert {doc for doc, _, _ in store.spans} == {"long"}
    if length > 3000:
        assert ("long", length - 3000, length) in store.spans
    assert all(read == "long" for read in store.body_reads)
    assert index.corpus is None


def test_bm25_real_scores_and_short_citations_match_legacy_without_arrow_body_table():
    store = fixture_store()
    legacy = bm25_index(store, legacy=True)
    retriever = StoreBM25Retriever(bm25_index(store), store)
    ctx = context(store, {"alpha", "beta"})
    expected = legacy.search(SearchRequest(query="needle needle", top_k=2,
        user=UserContext(user_id="intern", allowed_doc_ids={"alpha", "beta"})))
    actual = retriever.search(ctx, "needle needle", guard=Guard(), top_k=2)
    assert [(item.chunk, item.citation_id, item.score, item.retrieval_method) for item in actual] == [
        (item.chunk, item.citation_id, item.score, item.retrieval_method) for item in expected]
    assert "secret" not in store.body_reads
    assert all(item.snapshot_id == store.snapshot().snapshot_id for item in actual)


@pytest.mark.parametrize("text,query", [
    ("needle " * 500 + " " * 4000 + "needle trail", "needle trail"),
    ("needle " + " " * 7000 + "needle " * 12, "needle"),
    ("🙂 café " * 1000 + "終点 needle trail", "café needle trail"),
    ("🙂\x00 café " * 1000 + "終点 needle trail", "café needle trail"),
])
def test_selector_preserves_distinct_term_priority_then_frequency_with_unicode(text, query):
    store = fixture_store({"long": text, "other": "unrelated"})
    retriever = StoreBM25Retriever(bm25_index(store), store)
    selected = candidate(store, "long", method="bm25")
    expected = bounded_passage(passage(store, selected), query)
    actual = retriever._materialize(context(store), selected, query, Guard(), ReadFilters())
    assert actual == expected
    assert actual.chunk.start_char > 0
    assert actual.chunk.text == text[actual.chunk.start_char:actual.chunk.end_char]


def test_source_and_dates_narrow_before_bm25_ranking_and_can_skip_dense_encoding():
    store = fixture_store({"secret": "needle " * 20, "wrong-source": "needle " * 10,
                           "unknown-date": "needle " * 8, "allowed": "needle item"},
        sources={"wrong-source": "slack"}, timestamps={"secret": NOW, "wrong-source": NOW, "allowed": NOW})
    index = bm25_index(store)
    original = index.rank
    observed = []

    def rank(query, top_k, mask):
        observed.append(mask.copy())
        return original(query, top_k, mask)

    index.rank = rank
    retriever = StoreBM25Retriever(index, store)
    ctx = context(store, {"wrong-source", "unknown-date", "allowed"})
    filters = ReadFilters(source_types=("jira",), after=NOW, before=NOW)
    actual = retriever.search(ctx, "needle", guard=Guard(), top_k=1, filters=filters)
    assert [item.chunk.doc_id for item in actual] == ["allowed"]
    assert observed[0].tolist() == [False, False, False, True]
    assert store.body_reads == ["allowed"]
    dense_index, encoder = ScriptedDense(store), Encoder()
    dense = StoreDenseRetriever(dense_index, encoder, store)
    assert dense.search(ctx, "needle", guard=Guard(), filters=ReadFilters(source_types=("missing",))) == ()
    assert encoder.calls == dense_index.calls == []


def test_genuine_shards_dense_candidate_span_and_bounded_standalone_and_hybrid_outputs(tmp_path):
    text = "prefix " * 800 + "needle trail " * 170 + " suffix" * 100
    store = fixture_store({"long": text, "other": "needle other"})
    span = candidate(store, "long", start=700, end=len(text) - 100)
    backend = genuine_dense(tmp_path, store, [("long", span.start_char, span.end_char, [1., 0.]),
                                             ("other", 0, 12, [.4, .6])])
    encoder = Encoder()
    dense = StoreDenseRetriever(backend, encoder, store)
    ctx, guard = context(store), Guard()
    candidates = dense.candidates(ctx, "needle trail", guard=guard, top_k=1)
    assert len(candidates) == 1 and candidates[0].start_char == span.start_char
    assert candidates[0].end_char == span.end_char and span.end_char - span.start_char > 3000
    assert store.body_reads == []
    expected = bounded_passage(passage(store, candidates[0]), "needle trail")
    standalone = dense.search(ctx, "needle trail", guard=guard, top_k=1)
    assert standalone == (expected,)
    assert all(end - start <= 3000 for _, start, end in store.spans)
    store.body_reads.clear()
    store.spans.clear()
    bm25 = StoreBM25Retriever(bm25_index(store), store, catalog=backend.catalog)
    combined = StoreHybridRetriever(bm25, dense, config=HybridConfig(candidate_k=2, rerank_k=2))
    result = combined.search(ctx, "needle trail", guard=guard, top_k=1)
    assert len(result[0].chunk.text) <= 3000
    assert result[0].chunk == expected.chunk
    assert result[0].retrieval_method == "hybrid"
    assert set(store.body_reads) == {"long"}
    assert all(bm25.catalog[i] is backend.catalog[i] for i in range(len(backend.catalog)))


def test_dense_store_adapter_passes_source_date_and_scope_restrictions_before_top_k(tmp_path):
    store = fixture_store({"secret": "needle executive", "wrong-source": "needle slack",
                           "unknown-date": "needle unknown", "allowed": "needle allowed"},
        sources={"wrong-source": "slack"}, timestamps={"secret": NOW, "wrong-source": NOW, "allowed": NOW})
    vectors = ([1., 0.], [.9, .1], [.8, .2], [.5, .5])
    rows = [(entry.doc_id, 0, entry.text_length, vector) for entry, vector in zip(entries(store), vectors)]
    backend = genuine_dense(tmp_path, store, rows)
    encoder = Encoder()
    dense = StoreDenseRetriever(backend, encoder, store)
    result = dense.search(context(store, {"wrong-source", "unknown-date", "allowed"}), "needle",
        guard=Guard(), top_k=1, filters=ReadFilters(source_types=("jira",), after=NOW, before=NOW))
    assert [item.chunk.doc_id for item in result] == ["allowed"]
    assert encoder.calls == [(["needle"], True)]
    assert store.body_reads == ["allowed"]


@pytest.mark.parametrize("with_reranker", [False, True])
def test_hybrid_matches_legacy_final_evidence_but_fetches_only_selected_documents(with_reranker):
    store = fixture_store({"alpha": "needle " * 1000 + "tail", "beta": "needle beta", "secret": "needle secret"})
    rows = (candidate(store, "alpha", start=300, end=6500, score=.9),
            candidate(store, "beta", score=.7), candidate(store, "secret", score=.6))
    config = HybridConfig(candidate_k=3, rerank_k=2, bm25_weight=2., dense_weight=.25)

    class LegacyDense:
        doc_ids = list(store._documents)
        def search(self, request):
            return tuple(passage(store, value) for value in rows if request.user.permits(value.doc_id))[:request.top_k]

    def reranker():
        return CrossEncoderReranker("tiny", "b" * 40,
            scorer=lambda pairs: [float("beta" in text) for _, text in pairs]) if with_reranker else None

    legacy = HybridRetriever(bm25_index(store, legacy=True), LegacyDense(), config=config, reranker=reranker())
    expected = legacy.search(SearchRequest(query="needle tail", top_k=1,
        user=UserContext(user_id="intern", allowed_doc_ids=frozenset(store._documents))))
    combined, _, _ = hybrid(store, rows=rows, config=config, reranker=reranker())
    actual = combined.search(context(store), "needle tail", guard=Guard(), top_k=1)
    assert [(item.citation_id, item.chunk, item.score) for item in actual] == [
        (item.citation_id, item.chunk, item.score) for item in expected]
    assert set(store.body_reads) == ({"alpha", "beta"} if with_reranker else {"alpha"})
    assert all(end - start <= 3000 for _, start, end in store.spans)


@pytest.mark.parametrize("kind", ["bm25", "dense", "hybrid"])
def test_denied_scope_avoids_scoring_encoding_and_all_body_reads(kind):
    store = fixture_store()
    combined, index, encoder = hybrid(store)
    bm25 = combined.bm25
    bm25.index.rank_candidates = lambda *_args, **_kwargs: pytest.fail("Denied scope must not score BM25")
    retriever = {"bm25": bm25, "dense": combined.dense, "hybrid": combined}[kind]
    assert retriever.search(context(store, ()), "needle", guard=Guard()) == ()
    assert store.body_reads == store.spans == encoder.calls == index.calls == []


@pytest.mark.parametrize("error", [SessionExpired, StaleContext, PolicyUnavailable])
def test_freshness_failure_prevents_even_empty_search_and_expansion(error):
    store = fixture_store()
    combined, index, encoder = hybrid(store)
    guard = Guard()
    guard.failure = error
    for operation in (lambda: combined.search(context(store, ()), "needle", guard=guard),
                      lambda: combined.read_more(context(store), (), guard=guard)):
        with pytest.raises(error):
            operation()
    assert store.body_reads == encoder.calls == index.calls == []


def test_revocation_during_encode_stops_scan_and_body_access():
    store = fixture_store()
    index, guard = ScriptedDense(store), Guard()
    encoder = Encoder(lambda: setattr(guard, "revoked", True))
    retriever = StoreDenseRetriever(index, encoder, store)
    with pytest.raises(SessionExpired):
        retriever.search(context(store), "needle", guard=guard)
    assert len(encoder.calls) == 1 and index.calls == store.body_reads == []


def test_revocation_after_bm25_metadata_ranking_prevents_all_body_reads():
    store = fixture_store()
    index, guard = bm25_index(store), Guard()
    original = index.rank_candidates

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        guard.revoked = True
        return result

    index.rank_candidates = changed
    with pytest.raises(SessionExpired):
        StoreBM25Retriever(index, store).search(context(store), "needle", guard=guard)
    assert store.body_reads == store.spans == []


def test_revocation_after_first_window_prevents_more_windows_or_evidence_release():
    store = fixture_store({"long": "needle " * 1500, "other": "other"})
    original = store.read_passage
    guard = Guard()

    def revoked(*args, **kwargs):
        value = original(*args, **kwargs)
        guard.revoked = True
        return value

    store.read_passage = revoked
    with pytest.raises(SessionExpired):
        StoreBM25Retriever(bm25_index(store), store).search(context(store), "needle", guard=guard)
    assert len(store.spans) == len(store.body_reads) == 1


@pytest.mark.parametrize("corruption", ["denied", "source", "revision", "snapshot", "bounds"])
def test_all_ranked_candidates_are_checked_before_first_body_read(corruption):
    store = fixture_store(sources={"secret": "slack"})
    index = bm25_index(store)
    valid = candidate(store, "alpha", method="bm25")
    invalid = candidate(store, "secret", method="bm25")
    ctx = context(store, {"alpha", "beta"} if corruption == "denied" else None)
    filters = ReadFilters(source_types=("jira",)) if corruption == "source" else ReadFilters()
    if corruption == "revision":
        invalid = invalid.model_copy(update={"revision_id": "forged"})
    elif corruption == "snapshot":
        invalid = invalid.model_copy(update={"snapshot": invalid.snapshot.model_copy(update={"snapshot_id": "other"})})
    elif corruption == "bounds":
        invalid = invalid.model_copy(update={"end_char": 99999})
    index.rank_candidates = lambda *_args, **_kwargs: (valid, invalid)
    with pytest.raises((AccessDenied, EvidenceValidationError, SnapshotMismatch)):
        StoreBM25Retriever(index, store).search(ctx, "needle", guard=Guard(), filters=filters)
    assert store.body_reads == store.spans == []


@pytest.mark.parametrize("corruption", ["revision", "snapshot", "score", "method", "span", "legacy"])
def test_store_output_must_match_the_exact_authorized_ranked_span(corruption):
    store = fixture_store()
    original = store.read_passage

    def corrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        if corruption == "revision":
            return value.model_copy(update={"revision_id": "other"})
        if corruption == "snapshot":
            return value.model_copy(update={"snapshot_id": "other"})
        if corruption == "score":
            return value.model_copy(update={"score": value.score + 1})
        if corruption == "method":
            return value.model_copy(update={"retrieval_method": "dense"})
        if corruption == "span":
            return passage(store, candidate(store, value.chunk.doc_id, method="bm25", start=1))
        from src.contracts import Evidence
        return Evidence(citation_id=value.citation_id, chunk=value.chunk,
                        score=value.score, retrieval_method=value.retrieval_method)

    store.read_passage = corrupted
    with pytest.raises((EvidenceValidationError, SnapshotMismatch)):
        StoreBM25Retriever(bm25_index(store), store).search(context(store), "needle", guard=Guard(), top_k=1)
    assert len(store.body_reads) == 1


def test_read_more_preserves_matched_document_score_method_and_legacy_unicode_offsets():
    store = fixture_store({"long": "🙂 needle " * 1000, "other": "needle other"})
    retriever = StoreBM25Retriever(bm25_index(store), store)
    ctx = context(store)
    anchor = retriever.search(ctx, "needle", guard=Guard(), top_k=1)[0]
    store.body_reads.clear()
    expanded = retriever.read_more(ctx, (anchor,), guard=Guard(), max_chars=1000)
    old = bm25_index(store, legacy=True).read_more(SearchRequest(query="needle", top_k=1,
        user=UserContext(user_id="intern", allowed_doc_ids=frozenset(store._documents))), (anchor,), max_chars=1000)
    assert [(value.citation_id, value.chunk, value.score, value.retrieval_method) for value in expanded] == [
        (value.citation_id, value.chunk, value.score, value.retrieval_method) for value in old]
    assert set(store.body_reads) == {anchor.chunk.doc_id}
    assert expanded[0].chunk.text == store._documents[anchor.chunk.doc_id].text[
        expanded[0].chunk.start_char:expanded[0].chunk.end_char]


@pytest.mark.parametrize("bad", ["denied", "revision", "citation", "snapshot"])
def test_expansion_checks_all_anchors_before_any_body_access(bad):
    store = fixture_store()
    retriever = StoreBM25Retriever(bm25_index(store), store)
    first = passage(store, candidate(store, "alpha"))
    second = passage(store, candidate(store, "secret"))
    if bad == "revision":
        second = second.model_copy(update={"revision_id": "wrong"})
    elif bad == "citation":
        second = second.model_copy(update={"citation_id": "forged"})
    elif bad == "snapshot":
        second = second.model_copy(update={"snapshot_id": "other"})
    ctx = context(store, {"alpha"} if bad == "denied" else None)
    with pytest.raises((AccessDenied, EvidenceValidationError, SnapshotMismatch)):
        retriever.read_more(ctx, (first, second), guard=Guard())
    assert store.body_reads == []


def test_excluded_expansion_anchor_reads_no_body_and_expansion_cannot_add_a_document():
    store = fixture_store(sources={"alpha": "jira", "secret": "slack"})
    retriever = StoreBM25Retriever(bm25_index(store), store)
    anchor = passage(store, candidate(store, "alpha", end=5))
    ctx = context(store)
    assert retriever.read_more(ctx, (anchor,), guard=Guard(), filters=ReadFilters(source_types=("slack",))) == ()
    assert store.body_reads == []
    store.read_more = lambda *_args, **_kwargs: (passage(store, candidate(store, "secret")),)
    with pytest.raises(EvidenceValidationError):
        retriever.read_more(ctx, (anchor,), guard=Guard())


@pytest.mark.parametrize("corruption", ["new-document", "changed-text", "changed-revision", "duplicate", "too-many"])
def test_reranker_cannot_replace_evidence_or_exceed_candidate_bounds(corruption):
    store = fixture_store()

    class BadReranker:
        def rerank(self, query, values, top_k):
            value = values[0]
            if corruption == "new-document":
                return (passage(store, candidate(store, "secret", method="hybrid")),)
            if corruption == "changed-text":
                return (value.model_copy(update={"chunk": value.chunk.model_copy(update={"text": "x" * len(value.chunk.text)})}),)
            if corruption == "changed-revision":
                return (value.model_copy(update={"revision_id": "forged"}),)
            if corruption == "duplicate":
                return (value, value)
            return tuple(values)

    combined, _, _ = hybrid(store, config=HybridConfig(candidate_k=3, rerank_k=2), reranker=BadReranker())
    with pytest.raises(EvidenceValidationError):
        combined.search(context(store), "needle", guard=Guard(), top_k=1)


def test_reranker_revocation_cannot_release_previously_read_passages():
    store, guard = fixture_store(), Guard()

    class RevokedReranker:
        def rerank(self, query, values, top_k):
            guard.revoked = True
            return values[:top_k]

    combined, _, _ = hybrid(store, reranker=RevokedReranker())
    with pytest.raises(SessionExpired):
        combined.search(context(store), "needle", guard=guard, top_k=1)
    assert store.body_reads


@pytest.mark.parametrize("mask", [[True], [1, 1, 1], [[True, True, True]], ["yes"] * 3])
def test_malformed_permission_mask_fails_before_encoding_or_body_access(mask):
    store = fixture_store()
    ctx = context(store)

    class BadScope:
        binding = ctx.binding
        def permits(self, _doc): return True
        def mask_for(self, _snapshot, _ids): return mask

    combined, index, encoder = hybrid(store)
    with pytest.raises(EvidenceValidationError):
        combined.search(context(store, scope=BadScope()), "needle", guard=Guard())
    assert store.body_reads == index.calls == encoder.calls == []


def test_binding_mutation_while_resolving_mask_cannot_encode_or_release_results():
    store = fixture_store()
    original = context(store)

    class ChangingScope:
        binding = original.binding
        def permits(self, _doc): return True
        def mask_for(self, _snapshot, ids):
            self.binding = self.binding.model_copy(update={"policy_revision": 5})
            return (True,) * len(ids)

    combined, index, encoder = hybrid(store)
    with pytest.raises(StaleContext):
        combined.search(context(store, scope=ChangingScope()), "needle", guard=Guard())
    assert store.body_reads == index.calls == encoder.calls == []


def test_bitmap_style_scope_never_needs_legacy_allowed_id_attribute():
    store = fixture_store()
    original = context(store)

    class Scope:
        binding = original.binding
        @property
        def allowed_doc_ids(self): pytest.fail("Store retrieval must not copy scope into legacy sets")
        def permits(self, doc): return doc == "beta"
        def mask_for(self, snapshot, ids):
            assert snapshot == self.binding.snapshot
            return np.fromiter((self.permits(doc) for doc in ids), dtype=bool, count=len(ids))

    combined, _, _ = hybrid(store)
    result = combined.search(context(store, scope=Scope()), "needle", guard=Guard())
    assert [item.chunk.doc_id for item in result] == ["beta"]
    assert store.body_reads == ["beta"]


def test_cross_snapshot_request_and_changed_model_or_index_fail_before_expensive_work():
    store = fixture_store()
    combined, index, encoder = hybrid(store)
    other = store.snapshot().model_copy(update={"catalog_sha256": "c" * 64})
    with pytest.raises(SnapshotMismatch):
        combined.search(context(store, snapshot=other), "needle", guard=Guard())
    encoder.identity["revision"] = "d" * 40
    with pytest.raises(EvidenceValidationError):
        combined.dense.search(context(store), "needle", guard=Guard())
    combined.bm25.index.manifest["corpus_sha256"] = "d" * 64
    with pytest.raises(SnapshotMismatch):
        combined.bm25.search(context(store), "needle", guard=Guard())
    assert store.body_reads == encoder.calls == index.calls == []


def test_catalogue_hash_order_and_source_enumeration_are_metadata_only():
    store = fixture_store(sources={"alpha": "slack"})
    index = bm25_index(store)
    retriever = StoreBM25Retriever(index, store)
    assert retriever.source_types() == ("jira", "slack")
    assert store.body_reads == []
    reversed_index = bm25_index(store)
    reversed_index.doc_ids.reverse()
    with pytest.raises(SnapshotMismatch):
        StoreBM25Retriever(reversed_index, store)
    altered = list(entries(store))
    altered[0] = altered[0].model_copy(update={"source_type": "forged"})
    with pytest.raises(SnapshotMismatch):
        StoreBM25Retriever(index, store, catalog=altered)
    assert store.body_reads == []


def proof_hybrid(store, *, handle=None):
    lexical = bm25_index(store)
    index, encoder = ScriptedDense(store), Encoder()
    if handle is not None:
        index.catalog_handle, index.catalog, index.doc_ids = handle, handle.entries, handle.doc_ids
    bm25 = StoreBM25Retriever(lexical, store, catalog=handle)
    dense = StoreDenseRetriever(index, encoder, store)
    return StoreHybridRetriever(bm25, dense, config=HybridConfig(candidate_k=3, rerank_k=3)), index, encoder


def test_reader_proof_path_shares_exact_metadata_and_never_revalidates_or_enumerates_again(monkeypatch):
    import src.retrieval.store as module
    store = fixture_store()
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    lexical, index, encoder = bm25_index(store), ScriptedDense(store), Encoder()
    index.catalog_handle, index.catalog, index.doc_ids = handle, handle.entries, handle.doc_ids
    def unexpected(*_args, **_kwargs):
        pytest.fail("Issued metadata was revalidated, fingerprinted or enumerated again")
    with monkeypatch.context() as patch:
        patch.setattr(module, "catalog_fingerprint", unexpected)
        patch.setattr(CatalogEntry, "model_validate", classmethod(unexpected))
        patch.setattr(store, "iter_catalog", unexpected)
        bm25 = StoreBM25Retriever(lexical, store, catalog=handle)
        dense = StoreDenseRetriever(index, encoder, store)
        combined = StoreHybridRetriever(bm25, dense)
    assert bm25.catalog_handle is dense.catalog_handle is handle
    assert bm25.catalog is dense.catalog is handle.entries
    assert bm25.doc_ids is dense.doc_ids is handle.doc_ids
    assert bm25._positions is dense._positions is handle.positions
    assert combined.doc_ids is handle.doc_ids and type(bm25.catalog) is tuple
    assert store.body_reads == store.spans == encoder.calls == index.calls == []
    assert bm25.source_types() == ("jira",)


@pytest.mark.parametrize("allowed,filters,encoding_calls", [
    (("alpha", "beta"), ReadFilters(), 1),
    ((), ReadFilters(), 0),
    (("alpha", "beta"), ReadFilters(source_types=("jira",)), 1),
    (("alpha", "beta"), ReadFilters(source_types=("unknown",)), 0),
    (("alpha", "beta"), ReadFilters(after=NOW, before=NOW), 1),
    (("alpha", "beta"), ReadFilters(after=NOW + timedelta(days=2)), 0),
])
def test_shared_proof_preserves_exact_hybrid_windows_filters_guards_and_model_calls(allowed, filters, encoding_calls):
    store = fixture_store({"alpha": "🙂\x00 café needle trail " * 200, "beta": "needle beta", "secret": "needle executive"},
        sources={"alpha": "jira", "beta": "slack"}, timestamps={"alpha": NOW, "beta": NOW + timedelta(days=1)})
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    ordinary, ordinary_index, ordinary_encoder = proof_hybrid(store)
    shared, shared_index, shared_encoder = proof_hybrid(store, handle=handle)
    assert ordinary.bm25.catalog_handle is ordinary.dense.catalog_handle is None
    request = context(store, allowed)
    results, expansions, counts = [], [], []
    for reader, index, encoder in ((ordinary, ordinary_index, ordinary_encoder), (shared, shared_index, shared_encoder)):
        store.body_reads.clear()
        store.spans.clear()
        guard = Guard()
        result = reader.search(request, "needle trail", guard=guard, top_k=2, filters=filters)
        expansion = reader.read_more(request, result[:2], guard=guard, max_chars=1000, filters=filters)
        assert len(encoder.calls) == encoding_calls
        assert len(index.calls) == encoding_calls
        assert set(store.body_reads) <= set(allowed)
        assert all(end-start <= 3000 for _doc, start, end in store.spans)
        for passage_value in (*result, *expansion):
            record = store._documents[passage_value.chunk.doc_id]
            assert passage_value.snapshot_id == store.snapshot().snapshot_id
            assert passage_value.revision_id == record.entry.revision_id
            assert passage_value.chunk.text == record.text[passage_value.chunk.start_char:passage_value.chunk.end_char]
        results.append(result)
        expansions.append(expansion)
        counts.append(guard.calls)
    assert results[0] == results[1] and expansions[0] == expansions[1]
    assert counts[0] == counts[1]


@pytest.mark.parametrize('allowed', [None, (), ('empty', 'current'), ('future', 'unknown')])
@pytest.mark.parametrize('filters', [ReadFilters(), ReadFilters(source_types=('jira',)),
    ReadFilters(source_types=('slack',)), ReadFilters(after=NOW), ReadFilters(before=NOW),
    ReadFilters(after=NOW, before=NOW), ReadFilters(source_types=('jira',), after=NOW)])
def test_issued_catalogue_masks_equal_independent_old_loop_for_all_scope_and_date_cases(allowed, filters):
    store = fixture_store({'empty': '', 'current': 'needle current', 'future': 'needle future', 'unknown': 'needle unknown'},
        sources={'future': 'slack'}, timestamps={'empty': NOW, 'current': NOW, 'future': NOW + timedelta(days=1)})
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    reader, request, guard = _StoreReader(store, handle), context(store, allowed), Guard()
    # Independent former algorithm: authority from permits, then the original
    # per-entry source/date/unknown-time predicate and empty-body exclusion.
    expected = np.array([request.scope.permits(entry.doc_id) for entry in handle.entries], dtype=bool)
    for position in np.flatnonzero(expected):
        entry = handle.entries[position]
        if not filters.matches(entry) or not entry.text_length:
            expected[position] = False
    received = reader._mask(request, guard, filters)
    np.testing.assert_array_equal(received, expected)
    assert received.dtype == np.dtype(bool) and received.shape == (4,)
    assert guard.calls == 2 and store.body_reads == []


def test_empty_filter_fast_path_uses_one_immutable_metadata_mask_and_fresh_isolated_authority(monkeypatch):
    store = fixture_store({'empty': '', 'alpha': 'needle alpha', 'beta': 'needle beta'})
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    reader = _StoreReader(store, handle)
    original = context(store)
    live = np.array([True, True, False], dtype=bool)

    class Scope:
        binding = original.binding
        calls = 0
        def permits(self, doc): return bool(live[handle.positions[doc]])
        def mask_for(self, _snapshot, _ids):
            self.calls += 1
            return live

    scope = Scope()
    request = context(store, scope=scope)
    cached = reader._nonempty_text_mask
    assert cached is not None and not cached.flags.writeable
    with pytest.raises(ValueError):
        cached.setflags(write=True)
    monkeypatch.setattr(ReadFilters, 'matches', lambda *_: pytest.fail('No-filter issued path walked per-entry predicates'))
    first = reader._mask(request, Guard(), ReadFilters())
    np.testing.assert_array_equal(first, [False, True, False])
    assert not np.shares_memory(first, live) and not np.shares_memory(first, cached)
    first[:] = False
    np.testing.assert_array_equal(live, [True, True, False])
    live[:] = [False, False, True]
    second = reader._mask(request, Guard(), ReadFilters())
    np.testing.assert_array_equal(second, [False, False, True])
    assert scope.calls == 2 and reader._nonempty_text_mask is cached
    np.testing.assert_array_equal(cached, [False, True, True])
    assert store.body_reads == []


def test_generic_catalogue_and_custom_empty_filter_keep_original_predicate_dispatch(monkeypatch):
    store = fixture_store()
    ordinary = _StoreReader(store)
    calls = []
    monkeypatch.setattr(ReadFilters, 'matches', lambda _filter, entry: calls.append(entry.doc_id) or False)
    assert not ordinary._mask(context(store), Guard(), ReadFilters()).any()
    assert calls == list(ordinary.doc_ids) and ordinary._nonempty_text_mask is None

    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    issued = _StoreReader(store, handle)
    custom_calls = []

    class CustomFilters(ReadFilters):
        def matches(self, entry):
            custom_calls.append(entry.doc_id)
            return False

    assert not issued._mask(context(store), Guard(), CustomFilters()).any()
    assert custom_calls == list(issued.doc_ids)


def test_issued_empty_catalogue_preserves_zero_length_boolean_mask_and_guards():
    snapshot = SnapshotRef(snapshot_id='empty-catalogue', corpus_sha256='a' * 64,
        catalog_sha256=catalog_fingerprint((), expected_count=0), document_count=0)
    store = TrackingStore(snapshot, ())
    handle = ValidatedCatalog.build(snapshot, (), issuer=store)
    reader, guard = _StoreReader(store, handle), Guard()
    result = reader._mask(context(store), guard, ReadFilters())
    assert result.shape == (0,) and result.dtype == np.dtype(bool)
    assert not reader._nonempty_text_mask.flags.writeable
    assert guard.calls == 2 and store.body_reads == []


def test_issued_scope_with_only_empty_text_denies_before_encoder_ranking_or_body():
    store = fixture_store({'empty': '', 'other': 'needle body'})
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    index, encoder = ScriptedDense(store, rows=()), Encoder()
    index.catalog_handle, index.catalog, index.doc_ids = handle, handle.entries, handle.doc_ids
    dense = StoreDenseRetriever(index, encoder, store)
    lexical = StoreBM25Retriever(bm25_index(store), store, catalog=handle)
    request = context(store, ('empty',))
    assert dense.search(request, 'needle', guard=Guard()) == ()
    assert lexical.search(request, 'needle', guard=Guard()) == ()
    assert encoder.calls == index.calls == store.body_reads == store.spans == []


@pytest.mark.parametrize('mask', [[True], [1, 1, 1], [[True, True, True]], ['yes'] * 3])
def test_issued_fast_path_still_rejects_malformed_scope_masks_before_expensive_work(mask):
    store = fixture_store()
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    original = context(store)

    class BadScope:
        binding = original.binding
        def permits(self, _doc): return True
        def mask_for(self, _snapshot, _ids): return mask

    combined, index, encoder = proof_hybrid(store, handle=handle)
    with pytest.raises(EvidenceValidationError):
        combined.search(context(store, scope=BadScope()), 'needle', guard=Guard())
    assert store.body_reads == index.calls == encoder.calls == []


@pytest.mark.parametrize('change', ['revoked', 'binding'])
def test_issued_fast_path_keeps_final_freshness_and_binding_check_before_expensive_work(change):
    store = fixture_store()
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    original, guard = context(store), Guard()

    class ChangingScope:
        binding = original.binding
        def permits(self, _doc): return True
        def mask_for(self, _snapshot, ids):
            if change == 'revoked': guard.revoked = True
            else: self.binding = self.binding.model_copy(update={'policy_revision': 5})
            return (True,) * len(ids)

    combined, index, encoder = proof_hybrid(store, handle=handle)
    with pytest.raises(SessionExpired if change == 'revoked' else StaleContext):
        combined.search(context(store, scope=ChangingScope()), 'needle', guard=guard)
    assert store.body_reads == index.calls == encoder.calls == []


@pytest.mark.parametrize("kind", ["unissued", "snapshot", "copied-fields"])
def test_readers_reject_invalid_dense_handle_without_falling_back_to_valid_catalog_tuple(kind):
    store = fixture_store()
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    invalid = object.__new__(ValidatedCatalog)
    failure = EvidenceValidationError
    if kind == "snapshot":
        other = store.snapshot().model_copy(update={"snapshot_id": "other-proof-snapshot"})
        invalid = ValidatedCatalog.build(other, entries(store), issuer=store)
        failure = SnapshotMismatch
    elif kind == "copied-fields":
        for slot in ValidatedCatalog.__slots__:
            if slot != "__weakref__":
                object.__setattr__(invalid, slot, getattr(handle, slot))
    lexical, index, encoder = bm25_index(store), ScriptedDense(store), Encoder()
    index.catalog_handle = invalid
    with pytest.raises(failure):
        StoreBM25Retriever(lexical, store, catalog=invalid)
    with pytest.raises(failure):
        StoreDenseRetriever(index, encoder, store)
    assert store.body_reads == store.spans == encoder.calls == index.calls == []


@pytest.mark.parametrize("bad", [True, -1])
def test_ordinary_reader_model_copies_still_receive_full_validation(bad):
    store = fixture_store()
    rows = entries(store)
    invalid = (rows[0].model_copy(update={"text_length": bad}), *rows[1:])
    index, encoder = ScriptedDense(store), Encoder()
    index.catalog = invalid
    with pytest.raises(ValueError):
        StoreBM25Retriever(bm25_index(store), store, catalog=invalid)
    with pytest.raises(ValueError):
        StoreDenseRetriever(index, encoder, store)
    assert store.body_reads == encoder.calls == index.calls == []


@pytest.mark.parametrize("failure", [SessionExpired, StaleContext, PolicyUnavailable])
def test_metadata_proof_never_bypasses_a_live_guard_failure_before_model_or_body(failure):
    store = fixture_store()
    handle = ValidatedCatalog.build(store.snapshot(), entries(store), issuer=store)
    combined, index, encoder = proof_hybrid(store, handle=handle)
    guard = Guard()
    guard.failure = failure
    with pytest.raises(failure):
        combined.search(context(store), "needle", guard=guard)
    assert store.body_reads == store.spans == encoder.calls == index.calls == []


@pytest.mark.parametrize("bad", [True, 0, 101, -1, 1.2, "1"])
def test_invalid_top_k_is_rejected_before_work(bad):
    store = fixture_store()
    combined, index, encoder = hybrid(store)
    with pytest.raises(ValueError):
        combined.search(context(store), "needle", guard=Guard(), top_k=bad)
    assert store.body_reads == encoder.calls == index.calls == []


@pytest.mark.parametrize("kind", ["bm25", "dense", "hybrid"])
def test_oversized_query_is_rejected_before_scoring_encoding_or_body_access(kind):
    store = fixture_store()
    combined, index, encoder = hybrid(store)
    combined.bm25.index.rank_candidates = lambda *_args, **_kwargs: pytest.fail("Oversized query must not score")
    retriever = {"bm25": combined.bm25, "dense": combined.dense, "hybrid": combined}[kind]
    with pytest.raises(ValueError, match="8000"):
        retriever.search(context(store), "q" * 8001, guard=Guard())
    assert store.body_reads == store.spans == encoder.calls == index.calls == []

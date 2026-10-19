import pytest
from math import log2
from src.evaluation.retrieval_metrics import evaluate_retrieval, ndcg_at_k, recall_at_k
from collections import defaultdict
from src.evaluation.splits import build_splits


# Retrieval metrics

def test_hand_calculated_ranking_and_deduplication():
    # Unique ranking is a, irrelevant, b. Both a and b are relevant.
    hits = ["a", "a", "x", "b"]
    assert recall_at_k(hits, ["a", "b"], 2) == 0.5
    assert recall_at_k(hits, ["a", "b"], 3) == 1
    assert ndcg_at_k(hits, ["a", "b"], 3) == pytest.approx(1.5 / (1 + 1 / log2(3)))
    assert ndcg_at_k(["b", "a"], ["a", "b"], 2) == 1
    assert recall_at_k(["a"], ["a", "a"], 1) == 1


def test_empty_predictions_and_unlabeled():
    assert recall_at_k([], ["a"], 10) == 0
    assert ndcg_at_k([], ["a"], 10) == 0
    assert recall_at_k(["a"], [], 10) is None
    assert ndcg_at_k([], [], 10) is None


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([], [], k)


def test_aggregation_includes_errors_and_reports_exclusions():
    rows = [
        {
            "question_id": "1",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
        },
        {
            "question_id": "2",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
            "error": "timeout",
        },
        {
            "question_id": "3",
            "question_type": "unknown",
            "retrieved_doc_ids": [],
            "relevant_doc_ids": [],
        },
    ]
    report = evaluate_retrieval(rows, ks=(10,))
    assert report["overall"]["recall@10"] == {"mean": 0.5, "scored": 2, "excluded": 1}
    assert report["overall"]["errors"] == 1
    assert report["slices"]["unknown"]["recall@10"]["mean"] is None
    with pytest.raises(ValueError):
        evaluate_retrieval(rows + rows)


# Question splits

def questions():
    return [
        {
            "question_id": f"q{i:03}",
            "question_type": f"type{i % 3}",
            "question": f"Question {i}",
            "expected_doc_ids": [f"d{i // 2}"],
        }
        for i in range(60)
    ]


def test_split_reproducible_disjoint_complete_and_grouped():
    rows = questions()
    rows[4]["expected_doc_ids"] = ["d1", "d2"]  # Transitive overlap.
    result = build_splits(rows)
    assert result == build_splits(list(reversed(rows)))
    splits = result["splits"]
    all_ids = [q for ids in splits.values() for q in ids]
    assert len(all_ids) == len(set(all_ids)) == len(rows)
    assignments = {q: name for name, ids in splits.items() for q in ids}
    doc_splits = defaultdict(set)
    for q in rows:
        for doc in q["expected_doc_ids"]:
            doc_splits[doc].add(assignments[q["question_id"]])
    assert all(len(names) == 1 for names in doc_splits.values())
    assert set(result["golden_development_ids"]) <= set(splits["development"])
    assert all(splits.values())


def test_duplicate_text_is_grouped_without_reference_docs():
    rows = questions()
    rows[0].update(question="same question", expected_doc_ids=[])
    rows[20].update(question=" SAME  question ", expected_doc_ids=[])
    result = build_splits(rows)
    assert any({"q000", "q020"} <= set(ids) for ids in result["splits"].values())


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        build_splits([questions()[0], questions()[0]])


def test_committed_split_manifest_is_consistent():
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "evaluation"
    manifest = json.loads((base / "splits.json").read_text())
    parts = manifest["splits"]
    ids = [qid for values in parts.values() for qid in values]
    assert len(ids) == len(set(ids)) == 500
    assert [len(parts[k]) for k in ("development", "calibration", "final")] == [
        300,
        100,
        100,
    ]
    assert len(manifest["golden_development_ids"]) == 60
    assert set(manifest["golden_development_ids"]) <= set(parts["development"])


# Hybrid variants over shared searches

def test_hybrid_variants_share_searches_and_match_live_hybrid(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.contracts import SearchRequest, UserContext
    from src.evaluation.run_hybrid import run_single, run_variant, search_once
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import CorpusDense, embed_corpus
    from src.retrieval.hybrid import HybridConfig, HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from tests.test_retrieval import FakeModel

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({
        "doc_id": [f"d{i}" for i in range(5)],
        "source_type": ["slack", "gmail", "linear", "slack", "jira"],
        "title": ["Deploy", "", "Sales", "Grid", "Canary"],
        "content": ["canary rollback steps for the deployment", "lunch menu canary rollback",
                    "quarterly forecast review", "elevation tokens for grid density",
                    "canary release checklist and rollback owner"],
        "path": ["p"] * 5,
    }), path)
    CorpusBM25.build(path).save(tmp_path / "bm25")
    bm25 = CorpusBM25.load(tmp_path / "bm25", corpus_path=path)
    embed_corpus(path, tmp_path / "dense", FakeModel(), shard_docs=2, window=4, overlap=1)
    dense = CorpusDense.load(tmp_path / "dense", FakeModel(), corpus_path=path, corpus=bm25.corpus)

    class Counting:
        def __init__(self, retriever):
            self.retriever, self.calls = retriever, 0
            self.doc_ids, self.manifest = retriever.doc_ids, {}

        def search(self, request):
            self.calls += 1
            return self.retriever.search(request)

    questions = [{"question_id": f"q{i}", "question_type": "basic", "source_types": ["slack"],
                  "question": text, "expected_doc_ids": ["d0"]}
                 for i, text in enumerate(["canary rollback", "grid density", "canary rollback"])]
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    counted = (Counting(bm25), Counting(dense))
    caches, seconds, errors = search_once(counted, questions, user, depth=5)
    assert [r.calls for r in counted] == [2, 2] and not errors  # Repeated question searched once.

    live = HybridRetriever(bm25, dense, config=HybridConfig(candidate_k=5, rerank_k=5))
    rows, latencies, manifest = run_variant(counted, caches, seconds, errors, questions, user,
                                            bm25_weight=0.5, top_k=5, candidate_k=5)
    assert manifest["method"] == "hybrid-rrf" and len(latencies) == 3
    for q, row in zip(questions, rows):
        expected = live.search(SearchRequest(query=q["question"], user=user, top_k=5))
        assert row["retrieved_doc_ids"] == [e.chunk.doc_id for e in expected]
    assert [r.calls for r in counted] == [2, 2]  # Variants replay the saved results.

    keyword_only, _, _ = run_variant(counted, caches, seconds, errors, questions, user,
                                     bm25_weight=1.0, top_k=5, candidate_k=5)
    bm25_rows, _ = run_single(caches[0], seconds, errors, questions, 5)
    assert [r["retrieved_doc_ids"] for r in keyword_only] == [r["retrieved_doc_ids"] for r in bm25_rows]

    # The reranker reorders the fused candidates; here it prefers the longest passage.
    reranker = CrossEncoderReranker("offline-test", "a" * 40,
                                    scorer=lambda pairs: [len(text) for _, text in pairs])
    reranked, _, manifest = run_variant(counted, caches, seconds, errors, questions[:1], user,
                                        bm25_weight=0.5, reranker=reranker, top_k=5, candidate_k=5)
    assert set(reranked[0]["retrieved_doc_ids"]) == set(rows[0]["retrieved_doc_ids"])
    assert reranked[0]["scores"] == sorted(reranked[0]["scores"], reverse=True)
    assert reranked[0]["scores"] != rows[0]["scores"]
    assert manifest["reranker"]["name"] == "offline-test"

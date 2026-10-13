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

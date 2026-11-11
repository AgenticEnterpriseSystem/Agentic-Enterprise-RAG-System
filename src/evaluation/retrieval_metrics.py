"""Document-level binary relevance metrics; unlabeled questions return None."""

from collections.abc import Iterable
from math import log2
from statistics import mean


def ranked_documents(doc_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def recall_at_k(
    retrieved_doc_ids: Iterable[str], relevant_doc_ids: Iterable[str], k=10
):
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    retrieved = set(ranked_documents(retrieved_doc_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def ndcg_at_k(retrieved_doc_ids: Iterable[str], relevant_doc_ids: Iterable[str], k=10):
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    dcg = sum(
        1 / log2(rank + 2)
        for rank, doc in enumerate(ranked_documents(retrieved_doc_ids)[:k])
        if doc in relevant
    )
    ideal = sum(1 / log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal


def evaluate_retrieval(rows: Iterable[dict], ks=(5, 10, 20)) -> dict:
    """Rows: question_id, relevant_doc_ids, retrieved_doc_ids, question_type, error.

    Pass doc IDs in chunk-hit order; repeated chunks collapse to the first hit.
    Error rows receive zero for eligible metrics; missing relevance stays N/A.
    """
    rows = list(rows)
    ks = tuple(ks)
    if not ks or len(ks) != len(set(ks)):
        raise ValueError("Specify distinct k values")
    for k in ks:
        _validate_k(k)
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    results = []
    for row in rows:
        predictions = [] if row.get("error") else row["retrieved_doc_ids"]
        result = {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "error": bool(row.get("error")),
        }
        for k in ks:
            result[f"recall@{k}"] = recall_at_k(predictions, row["relevant_doc_ids"], k)
            result[f"ndcg@{k}"] = ndcg_at_k(predictions, row["relevant_doc_ids"], k)
        results.append(result)

    def summarize(group):
        values = {"questions": len(group), "errors": sum(r["error"] for r in group)}
        for key in [f"{metric}@{k}" for k in ks for metric in ("recall", "ndcg")]:
            scores = [r[key] for r in group if r[key] is not None]
            values[key] = {
                "mean": mean(scores) if scores else None,
                "scored": len(scores),
                "excluded": len(group) - len(scores),
            }
        return values

    return {
        "overall": summarize(results),
        "slices": {
            category: summarize([r for r in results if r["question_type"] == category])
            for category in sorted({r["question_type"] for r in results})
        },
        "per_question": results,
    }


def paired_bootstrap(rows: list[dict], baseline: list[dict], metric=recall_at_k, k=10,
                     iterations=10000, seed=5980) -> dict:
    """Mean change versus `baseline` on the same questions, with a 95% bootstrap interval.

    Questions are resampled with replacement; unlabeled questions are skipped. Also counts
    questions that improved or got worse.
    """
    import random

    by_id = {row["question_id"]: row for row in baseline}
    changes = []
    for row in rows:
        before = by_id.get(row["question_id"])
        if before is None:
            raise ValueError("Rows and baseline must cover the same questions")
        new = metric(row["retrieved_doc_ids"], row["relevant_doc_ids"], k)
        old = metric(before["retrieved_doc_ids"], before["relevant_doc_ids"], k)
        if new is not None and old is not None:
            changes.append(new - old)
    if not changes:
        raise ValueError("No labeled questions to compare")
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(changes, k=len(changes))) for _ in range(iterations))
    return {"change": mean(changes), "low": means[int(0.025 * iterations)],
            "high": means[int(0.975 * iterations) - 1], "questions": len(changes),
            "better": sum(c > 0 for c in changes), "worse": sum(c < 0 for c in changes)}


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL predictions with reference document IDs",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    print(json.dumps(evaluate_retrieval(rows), indent=2))


if __name__ == "__main__":
    main()

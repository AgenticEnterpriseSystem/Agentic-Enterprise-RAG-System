"""Deterministic document-level reciprocal rank fusion."""
from collections.abc import Sequence
import math
from src.contracts import Evidence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Evidence]], *, k: float = 60,
                           weights: Sequence[float] | None = None,
                           limit: int | None = None) -> tuple[Evidence, ...]:
    """One vote per document per ranking; prefer its dense passage over whole-document evidence."""
    weights = tuple(weights) if weights is not None else (1.0,) * len(rankings)
    if not math.isfinite(k) or k < 0 or len(weights) != len(rankings):
        raise ValueError("Invalid RRF constant or weights")
    if any(not math.isfinite(w) or w < 0 for w in weights) or (limit is not None and limit < 1):
        raise ValueError("Weights must be finite and nonnegative; limit must be positive")
    scores, representatives = {}, {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        rank = 0
        for item in ranking:
            doc = item.chunk.doc_id
            if doc in seen:
                continue
            seen.add(doc)
            rank += 1
            if weight == 0:
                continue
            previous = representatives.get(doc)
            if previous is None or (item.retrieval_method == "dense" and previous.retrieval_method != "dense"):
                representatives[doc] = item
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
    docs = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return tuple(representatives[doc].model_copy(update={"score": scores[doc], "retrieval_method": "hybrid"})
                 for doc in docs[:limit])

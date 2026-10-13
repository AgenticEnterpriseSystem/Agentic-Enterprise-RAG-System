"""Optional cross-encoder inference with explicit immutable model provenance."""
import math
import re
from collections.abc import Callable, Sequence
from src.contracts import Evidence
from src.retrieval.passages import bounded_passage


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str, *, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512,
                 scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None):
        if not model_name.strip() or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Provide a model name and immutable 40-character commit revision")
        if max_length < 2:
            raise ValueError("max_length must fit query and passage tokens")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.identity = {"name": model_name, "revision": revision, "max_length": max_length}
        self.device, self.batch_size = device, batch_size
        self.scorer = scorer
        self.model = None

    def rerank(self, query: str, evidence: Sequence[Evidence], top_k: int) -> tuple[Evidence, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not evidence:
            return ()
        evidence = tuple(bounded_passage(item, query) for item in evidence)
        pairs = [(query, item.chunk.text) for item in evidence]
        if self.scorer is not None:
            scores = list(self.scorer(pairs))
        else:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.identity["name"], revision=self.identity["revision"],
                                          device=self.device, trust_remote_code=False,
                                          max_length=self.identity["max_length"])
            scores = list(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))
        if len(scores) != len(evidence) or any(not math.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker must return one finite scalar per candidate")
        ranked = sorted(zip(evidence, scores), key=lambda pair: (-float(pair[1]), pair[0].chunk.doc_id))
        return tuple(item.model_copy(update={"score": float(score), "retrieval_method": "hybrid"})
                     for item, score in ranked[:top_k])

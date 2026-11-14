"""Optional cross-encoder inference with explicit immutable model provenance.

By default each candidate is scored on one query-relevant slice (`bounded_passage`).
With `window_chars` set, each candidate is scored on several windows across its whole
document, each prefixed with the title, and keeps its best window's score (MaxP). This
stops long documents losing out when the single slice misses the answer.
"""
import math
import re
from collections.abc import Callable, Sequence
from src.contracts import Evidence
from src.retrieval.passages import _STOPWORDS, bounded_passage


def document_windows(text: str, query: str, window_chars: int, stride: int,
                     max_windows: int) -> list[tuple[int, int]]:
    """Character spans covering `text`; at most `max_windows`, in document order.

    The first window (the document's lead) is always kept. When a document needs more
    windows than allowed, the rest are those covering the most distinct query terms.
    """
    if window_chars < 1 or stride < 1 or max_windows < 1:
        raise ValueError("Window size, stride and window count must be positive")
    if len(text) <= window_chars:
        return [(0, len(text))]
    starts = list(range(0, len(text) - window_chars + 1, stride))
    if starts[-1] != len(text) - window_chars:
        starts.append(len(text) - window_chars)
    if len(starts) > max_windows:
        terms = set(re.findall(r"\w+", query.lower())) - _STOPWORDS

        def coverage(start):
            words = re.findall(r"\w+", text[start:start + window_chars].lower())
            matches = [w for w in words if w in terms]
            return (len(set(matches)), len(matches), -start)

        starts = [starts[0]] + sorted(sorted(starts[1:], key=coverage, reverse=True)[:max_windows - 1])
    return [(s, s + window_chars) for s in starts]


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str, *, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512, fp16: bool = False,
                 scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
                 window_chars: int | None = None, window_stride: int | None = None,
                 max_windows: int = 8, document_text: Callable[[str], str | None] | None = None,
                 document_header: Callable[[str], str] | None = None):
        if not model_name.strip() or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Provide a model name and immutable 40-character commit revision")
        if max_length < 2:
            raise ValueError("max_length must fit query and passage tokens")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if window_chars is not None and (window_chars < 1 or max_windows < 1
                                         or (window_stride is not None and window_stride < 1)):
            raise ValueError("Window size, stride and window count must be positive")
        self.identity = {"name": model_name, "revision": revision, "max_length": max_length,
                         "fp16": fp16}
        self.window_chars = window_chars
        self.window_stride = window_stride or (window_chars // 2 if window_chars else None)
        self.max_windows = max_windows
        self.document_text = document_text
        self.document_header = document_header
        if window_chars is not None:
            self.identity["windows"] = {"chars": window_chars, "stride": self.window_stride,
                                        "max": max_windows, "aggregate": "max",
                                        "full_document": document_text is not None}
            if document_header is not None:
                self.identity["windows"]["header"] = "path-v1"
        self.device, self.batch_size = device, batch_size
        self.scorer = scorer
        self.model = None

    def _pairs(self, query: str, evidence: Sequence[Evidence]):
        """(query, text) pairs plus the candidate index each pair belongs to."""
        if self.window_chars is None:
            return [(query, item.chunk.text) for item in evidence], list(range(len(evidence)))
        pairs, owners = [], []
        for index, item in enumerate(evidence):
            text = (self.document_text(item.chunk.doc_id) if self.document_text else None) \
                or item.chunk.text
            title = (item.chunk.title or "").strip()
            for start, end in document_windows(text, query, self.window_chars,
                                               self.window_stride, self.max_windows):
                window = text[start:end]
                if self.document_header is not None:
                    # Every window starts with the path header and the title.
                    body = window[len(title):].lstrip("\n") if title and window.startswith(title) else window
                    window = "\n".join(p for p in (self.document_header(item.chunk.doc_id), title, body) if p)
                # Later windows keep the document's name, like the dense index's passages.
                elif title and not window.startswith(title):
                    window = f"{title}\n{window}"
                pairs.append((query, window))
                owners.append(index)
        return pairs, owners

    def load(self):
        """Load the model now instead of on the first query (no-op with a test scorer)."""
        if self.scorer is None and self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.identity["name"], revision=self.identity["revision"],
                                      device=self.device, trust_remote_code=False,
                                      max_length=self.identity["max_length"])
            if self.identity["fp16"]:
                self.model.half()  # Half precision for GPU inference only.
        return self

    def _score(self, pairs):
        if self.scorer is not None:
            return list(self.scorer(pairs))
        self.load()
        return list(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))

    def rerank(self, query: str, evidence: Sequence[Evidence], top_k: int) -> tuple[Evidence, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not evidence:
            return ()
        evidence = tuple(bounded_passage(item, query) for item in evidence)
        pairs, owners = self._pairs(query, evidence)
        # Windows scored per candidate, for cost estimates in evaluation runs.
        self.last_window_counts = {}
        for owner in owners:
            key = evidence[owner].citation_id
            self.last_window_counts[key] = self.last_window_counts.get(key, 0) + 1
        scores = self._score(pairs)
        if len(scores) != len(pairs) or any(not math.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker must return one finite scalar per candidate")
        best = [-math.inf] * len(evidence)
        for owner, score in zip(owners, scores):
            best[owner] = max(best[owner], float(score))
        ranked = sorted(zip(evidence, best), key=lambda pair: (-pair[1], pair[0].chunk.doc_id))
        # The returned evidence is the candidate's original passage; only its order changes.
        return tuple(item.model_copy(update={"score": score, "retrieval_method": "hybrid"})
                     for item, score in ranked[:top_k])

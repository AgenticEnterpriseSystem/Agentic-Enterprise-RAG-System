"""Scoped BM25/dense fusion with optional reranking."""
from pydantic import Field
from src.contracts import Contract, Retriever, SearchRequest, validate_evidence
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.passages import bounded_passage


class HybridConfig(Contract):
    candidate_k: int = Field(default=50, ge=1, le=100)
    rerank_k: int = Field(default=50, ge=1, le=200)
    rrf_k: float = Field(default=60, ge=0)
    bm25_weight: float = Field(default=1, ge=0)
    dense_weight: float = Field(default=1, ge=0)
    # Share of a reranker in the final order: 1 replaces the fused order with the reranker's;
    # lower values blend the two ranks with RRF, keeping BM25+'s exact-match wins.
    rerank_weight: float = Field(default=1, ge=0, le=1)


def matches_filters(item, request):
    chunk = item.chunk
    return (request.user.permits(chunk.doc_id)
            and (not request.source_types or chunk.source_type in request.source_types)
            and (request.after is None or (chunk.timestamp is not None and chunk.timestamp >= request.after))
            and (request.before is None or (chunk.timestamp is not None and chunk.timestamp <= request.before)))


class HybridRetriever:
    def __init__(self, bm25: Retriever, dense: Retriever, *, config: HybridConfig | None = None, reranker=None):
        self.bm25, self.dense = bm25, dense
        self.config = config or HybridConfig()
        self.reranker = reranker
        if hasattr(bm25, "doc_ids") and hasattr(dense, "doc_ids"):
            if set(bm25.doc_ids) != set(dense.doc_ids):
                raise ValueError("Hybrid indexes must cover the same documents")
            self.doc_ids = list(bm25.doc_ids)
        self.manifest = {"method": "hybrid-rrf", "config": self.config.model_dump(),
                         "bm25": getattr(bm25, "manifest", {}),
                         "dense": getattr(dense, "manifest", {}),
                         "reranker": getattr(reranker, "identity", None)}

    def search(self, request: SearchRequest):
        if request.user.user_id is None or not request.user.allowed_doc_ids:
            return ()
        candidate_request = request.model_copy(update={"top_k": max(request.top_k, self.config.candidate_k)})
        rankings = []
        for retriever in (self.bm25, self.dense):
            items = tuple(retriever.search(candidate_request))
            # Fail closed on scope violations; filter stale/wrong-source results before scoring.
            validate_evidence(request.user, items)
            rankings.append(tuple(item for item in items if matches_filters(item, request)))
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k,
                    weights=(self.config.bm25_weight, self.config.dense_weight),
                    limit=max(request.top_k, self.config.rerank_k))
        fused = tuple(bounded_passage(item, request.query) for item in fused)
        validate_evidence(request.user, fused)
        if self.reranker is None:
            result = fused[:request.top_k]
        elif self.config.rerank_weight >= 1:
            result = self.reranker.rerank(request.query, fused, request.top_k)
        else:
            result = self.blend(fused, self.reranker.rerank(request.query, fused, len(fused)),
                                request.top_k)
        result = tuple(result)
        candidates = {item.citation_id: item for item in fused}
        if len(result) > request.top_k or any(
            item.citation_id not in candidates
            or item.chunk != candidates[item.citation_id].chunk
            or item.retrieval_method != "hybrid"
            or not matches_filters(item, request)
            for item in result
        ):
            raise ValueError("Reranker returned evidence outside filtered candidates")
        validate_evidence(request.user, result)
        return result

    def blend(self, fused, reranked, top_k):
        """Weighted RRF of the fused rank and the reranker's rank (ties keep fused order)."""
        weight, k = self.config.rerank_weight, self.config.rrf_k
        position = {item.citation_id: rank for rank, item in enumerate(reranked, 1)}
        scored = []
        for rank, item in enumerate(fused, 1):
            score = (1 - weight) / (k + rank)
            if item.citation_id in position:
                score += weight / (k + position[item.citation_id])
            scored.append((score, rank, item))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return tuple(item.model_copy(update={"score": score}) for score, _, item in scored[:top_k])

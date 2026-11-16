"""Scoped BM25/dense fusion with optional reranking."""
from pydantic import Field
from src.contracts import Contract, Retriever, SearchRequest, validate_evidence
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.passages import bounded_passage
from src.observability import emit, span


class HybridConfig(Contract):
    candidate_k: int = Field(default=50, ge=1, le=100)
    rerank_k: int = Field(default=50, ge=1, le=200)
    rrf_k: float = Field(default=60, ge=0)
    bm25_weight: float = Field(default=1, ge=0)
    dense_weight: float = Field(default=1, ge=0)


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
        bm25_ids, dense_ids = getattr(bm25, 'doc_ids', None), getattr(dense, 'doc_ids', None)
        if bm25_ids is not None and dense_ids is not None:
            if set(bm25_ids) != set(dense_ids):
                raise ValueError("Hybrid indexes must cover the same documents")
            self.doc_ids = list(bm25_ids)
        self.manifest = {"method": "hybrid-rrf", "config": self.config.model_dump(),
                         "bm25": getattr(bm25, "manifest", {}),
                         "dense": getattr(dense, "manifest", {}),
                         "reranker": getattr(reranker, "identity", None)}

    def read_more(self, request: SearchRequest, anchors, *, max_chars=3000):
        validate_evidence(request.user, anchors)
        operation = getattr(self.bm25, 'read_more', None)
        if operation is None:
            return ()
        result = tuple(operation(request, anchors, max_chars=max_chars))
        matched_ids = {e.chunk.doc_id for e in anchors}
        if any(e.chunk.doc_id not in matched_ids or not matches_filters(e, request) for e in result):
            raise ValueError("Expansion returned evidence outside the matched document scope")
        validate_evidence(request.user, result)
        return result

    def search(self, request: SearchRequest):
        with span('hybrid') as observed:
            if request.user.user_id is None or not request.user.allowed_doc_ids:
                observed.set(returned_count=0)
                return ()
            candidate_request = request.model_copy(update={"top_k": max(request.top_k, self.config.candidate_k)})
            rankings = []
            for stage, retriever in (('bm25', self.bm25), ('dense', self.dense)):
                with span(stage) as branch:
                    items = tuple(retriever.search(candidate_request))
                    # Fail closed on scope violations; filter stale/wrong-source results before scoring.
                    validate_evidence(request.user, items)
                    ranking = tuple(item for item in items if matches_filters(item, request))
                    rankings.append(ranking)
                    branch.set(candidate_count=len(ranking),
                               unique_document_count=len({item.chunk.doc_id for item in ranking}))
            observed.set(candidate_count=sum(map(len, rankings)), bm25_candidate_count=len(rankings[0]),
                         dense_candidate_count=len(rankings[1]),
                         unique_document_count=len({item.chunk.doc_id for ranking in rankings for item in ranking}))
            with span('fusion') as fusion:
                fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k,
                            weights=(self.config.bm25_weight, self.config.dense_weight),
                            limit=max(request.top_k, self.config.rerank_k))
                fusion.set(fused_count=len(fused))
            with span('materialize') as materialized:
                fused = tuple(bounded_passage(item, request.query) for item in fused)
                validate_evidence(request.user, fused)
                materialized.set(returned_count=len(fused))
            if self.reranker is not None:
                with span('rerank', reranker_enabled=True) as reranked:
                    result = tuple(self.reranker.rerank(request.query, fused, request.top_k))
                    reranked.set(candidate_count=len(fused), returned_count=len(result))
            else:
                emit('rerank', 'skipped', status='skipped', reason='disabled', reranker_enabled=False)
                result = tuple(fused[:request.top_k])
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
            observed.set(fused_count=len(fused), returned_count=len(result), reranker_enabled=self.reranker is not None)
            return result

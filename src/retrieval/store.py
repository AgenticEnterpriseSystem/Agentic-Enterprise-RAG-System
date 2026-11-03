"""Store-backed retrieval with metadata fusion and bounded authorized reads.

No identity, policy, database or runtime factory is configured here. Callers
provide a server-resolved RequestContext and a FreshnessGuard for each operation.
The legacy corpus retrievers remain available as the comparison implementation.

Catalogue metadata is O(documents); bodies are read only for selected documents.
Preserving the old passage heuristic can require many overlapping 3000-character
reads for one long selected span. Only the current and best windows are retained;
bounded reads do not promise constant I/O or a full-corpus latency target.
"""

from collections.abc import Iterable, Sequence
from hashlib import sha256
import json
import math
import re

import numpy as np

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, FreshnessGuard, RequestContext,
    SnapshotMismatch, validate_grounded_evidence,
)
from src.contracts import PassageEvidence, SnapshotRef
from src.retrieval.hybrid import HybridConfig
from src.retrieval.passages import PASSAGE_CHARS, _STOPWORDS
from src.storage.contracts import (
    CatalogEntry, DocumentStore, RankedCandidate, ReadFilters,
    validate_read_more_bounds,
)
from src.storage.fingerprints import catalog_fingerprint


def _query(query, top_k):
    if not isinstance(query, str) or not query.strip() or len(query) > 8000:
        raise ValueError("Search requires a nonempty query of at most 8000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("Search top_k must be an integer between 1 and 100")


def _filters(filters):
    if not isinstance(filters, ReadFilters):
        raise EvidenceValidationError("Search restrictions require ReadFilters")
    try:
        return ReadFilters.model_validate(filters.model_dump(warnings=False))
    except ValueError:
        raise EvidenceValidationError("Invalid search restrictions") from None


def _candidate(value):
    if not isinstance(value, RankedCandidate):
        raise EvidenceValidationError("Ranking returned an unsupported candidate")
    try:
        return RankedCandidate.model_validate(value.model_dump(warnings=False))
    except ValueError:
        raise EvidenceValidationError("Ranking returned an invalid candidate") from None


def reciprocal_rank_candidates(
    rankings: Sequence[Sequence[RankedCandidate]], *, k: float = 60,
    weights: Sequence[float] | None = None, limit: int | None = None,
) -> tuple[RankedCandidate, ...]:
    """The legacy RRF arithmetic applied to metadata before any body access.

    Duplicate documents consume one rank per input ranking. Dense spans replace
    lexical representatives. Zero-weight rankings confer no vote or preferred
    span; score ties use lexical document IDs. This is a trusted ranking helper,
    not an authorization boundary.
    """
    weights = tuple(weights) if weights is not None else (1.0,) * len(rankings)
    if (not math.isfinite(k) or k < 0 or len(weights) != len(rankings)
            or any(not math.isfinite(weight) or weight < 0 for weight in weights)
            or (limit is not None and (type(limit) is not int or limit < 1))):
        raise ValueError("Invalid RRF constant, weights or limit")
    checked = tuple(tuple(_candidate(item) for item in ranking) for ranking in rankings)
    provenance = "rrf-v1:" + sha256(json.dumps({
        "k": k, "weights": weights,
        "scorers": [[item.scoring_provenance for item in ranking] for ranking in checked],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    scores, representatives, identities = {}, {}, {}
    snapshot = next((item.snapshot for ranking in checked for item in ranking), None)
    for ranking, weight in zip(checked, weights):
        seen = set()
        rank = 0
        for item in ranking:
            doc = item.doc_id
            identity = (item.snapshot, item.revision_id, item.source_type)
            if doc in identities and identities[doc] != identity:
                raise EvidenceValidationError("Fusion candidates disagree on document identity")
            identities[doc] = identity
            if item.snapshot != snapshot:
                raise SnapshotMismatch("Fusion candidates belong to different snapshots")
            if doc in seen:
                continue
            seen.add(doc)
            rank += 1
            if weight == 0:
                continue
            previous = representatives.get(doc)
            if previous is None or (item.retrieval_method == "dense"
                                    and previous.retrieval_method != "dense"):
                representatives[doc] = item
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
            if not math.isfinite(scores[doc]):
                raise ValueError("Fused scores must be finite")
    docs = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return tuple(representatives[doc].model_copy(update={
        "score": scores[doc], "retrieval_method": "hybrid",
        "scoring_provenance": provenance,
    }) for doc in docs[:limit])


class _StoreReader:
    def __init__(self, store: DocumentStore, catalog: Iterable[CatalogEntry] | None = None):
        self.store = store
        snapshot = store.snapshot()
        if not isinstance(snapshot, SnapshotRef):
            raise EvidenceValidationError("Document store requires a snapshot")
        self.snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        if catalog is None:
            catalog = (entry for batch in store.iter_catalog() for entry in batch)
        entries = []

        def checked():
            for supplied in catalog:
                if not isinstance(supplied, CatalogEntry):
                    raise EvidenceValidationError("Document store returned invalid catalogue metadata")
                # Validate untrusted model_copy updates, retain the frozen original
                # so accepted dense catalogue objects can be shared without copies.
                validated = CatalogEntry.model_validate(supplied.model_dump(warnings=False))
                retained = supplied if supplied == validated else validated
                entries.append(retained)
                yield retained

        digest = catalog_fingerprint(checked(), expected_count=self.snapshot.document_count)
        if digest != self.snapshot.catalog_sha256:
            raise SnapshotMismatch("Document catalogue differs from the snapshot")
        self.catalog = tuple(entries)
        self.doc_ids = tuple(entry.doc_id for entry in entries)
        self._positions = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        if len(self._positions) != len(self.doc_ids):
            raise EvidenceValidationError("Document catalogue contains duplicate IDs")

    def source_types(self) -> tuple[str, ...]:
        """Internal platform metadata; never an authorized public source list."""
        return tuple(sorted({entry.source_type for entry in self.catalog}))

    def _current(self, context, guard):
        if not isinstance(context, RequestContext):
            raise EvidenceValidationError("A server-resolved request context is required")
        guard.assert_current(context)
        context.assert_bound()
        if context.binding.snapshot != self.snapshot or self.store.snapshot() != self.snapshot:
            raise SnapshotMismatch("Retrieval context or store belongs to another snapshot")

    def _mask(self, context, guard, filters):
        self._current(context, guard)
        supplied = np.asarray(context.scope.mask_for(self.snapshot, self.doc_ids))
        if not self.doc_ids and supplied.shape == (0,):
            supplied = supplied.astype(bool)
        if supplied.dtype != np.dtype(bool) or supplied.shape != (len(self.doc_ids),):
            raise EvidenceValidationError("Authorization scope must return one boolean per document")
        mask = supplied.copy()
        for position in np.flatnonzero(mask):
            if not filters.matches(self.catalog[position]) or not self.catalog[position].text_length:
                mask[position] = False
        self._current(context, guard)
        return mask

    def _checked_candidate(self, context, candidate, filters):
        candidate = _candidate(candidate)
        position = self._positions.get(candidate.doc_id)
        if position is None:
            raise EvidenceValidationError("Ranking returned an unknown document")
        entry = self.catalog[position]
        if candidate.snapshot != self.snapshot:
            raise SnapshotMismatch("Ranking returned another snapshot")
        try:
            candidate.validate_catalog(self.snapshot, entry)
        except ValueError:
            raise EvidenceValidationError("Ranking differs from the canonical catalogue") from None
        permitted = context.scope.permits(candidate.doc_id)
        if type(permitted) not in (bool, np.bool_) or not permitted:
            raise AccessDenied("Ranking returned a document outside the authorized scope")
        if not filters.matches(entry):
            raise AccessDenied("Ranking returned a document outside source/date restrictions")
        context.assert_bound()
        return candidate

    def _checked_passage(self, context, value, filters, *, candidate=None):
        if not isinstance(value, PassageEvidence):
            raise EvidenceValidationError("Document store must return snapshot-bound passage evidence")
        try:
            value = PassageEvidence.model_validate(value.model_dump(warnings=False))
        except ValueError:
            raise EvidenceValidationError("Document store returned invalid passage evidence") from None
        chunk = value.chunk
        position = self._positions.get(chunk.doc_id)
        if position is None:
            raise EvidenceValidationError("Document store returned an unknown document")
        entry = self.catalog[position]
        if value.snapshot_id != self.snapshot.snapshot_id:
            raise SnapshotMismatch("Document store returned another snapshot")
        if (value.revision_id != entry.revision_id or chunk.source_type != entry.source_type
                or chunk.metadata_origin != entry.metadata_origin or chunk.timestamp != entry.timestamp
                or chunk.end_char > entry.text_length
                or value.citation_id != f"{chunk.doc_id}:{chunk.start_char}:{chunk.end_char}"
                or chunk.chunk_id != value.citation_id):
            raise EvidenceValidationError("Document store returned inconsistent passage metadata")
        permitted = context.scope.permits(chunk.doc_id)
        if type(permitted) not in (bool, np.bool_) or not permitted or not filters.matches(entry):
            raise AccessDenied("Document store returned a passage outside the requested scope")
        if candidate is not None and (
            chunk.doc_id != candidate.doc_id or chunk.start_char != candidate.start_char
            or chunk.end_char != candidate.end_char or value.score != candidate.score
            or value.retrieval_method != candidate.retrieval_method
        ):
            raise EvidenceValidationError("Document store returned a different ranked span")
        context.assert_bound()
        return value

    def _read(self, context, candidate, guard, filters):
        self._current(context, guard)
        candidate = self._checked_candidate(context, candidate, filters)
        value = self.store.read_passage(context, candidate, guard=guard, filters=filters)
        self._current(context, guard)
        return self._checked_passage(context, value, filters, candidate=candidate)

    def _materialize(self, context, candidate, query, guard, filters):
        """Stream the same windows/scoring used by legacy bounded_passage."""
        candidate = self._checked_candidate(context, candidate, filters)
        length = candidate.end_char - candidate.start_char
        if length <= PASSAGE_CHARS:
            return self._read(context, candidate, guard, filters)
        terms = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
        step = max(1, PASSAGE_CHARS // 2)
        final_start = length - PASSAGE_CHARS
        best, best_score = None, None
        for start in range(0, final_start + step, step):
            start = min(start, final_start)
            absolute = candidate.start_char + start
            window = candidate.model_copy(update={
                "start_char": absolute, "end_char": absolute + PASSAGE_CHARS,
            })
            value = self._read(context, window, guard, filters)
            matches = [word for word in re.findall(r"\w+", value.chunk.text.lower()) if word in terms]
            score = (len(set(matches)), len(matches), -start)
            if best_score is None or score > best_score:
                best, best_score = value, score
            if start == final_start:
                break
        return best

    def read_more(self, context: RequestContext, anchors: tuple[PassageEvidence, ...], *,
                  guard: FreshnessGuard, max_chars: int = 3000,
                  filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        self._current(context, guard)
        filters = _filters(filters)
        anchors = tuple(anchors)
        validate_read_more_bounds(len(anchors), max_chars)
        validate_grounded_evidence(context, anchors, guard)
        # Validate all metadata, even excluded source/date anchors, before the
        # store can read the body of an earlier allowed anchor.
        for anchor in anchors:
            self._checked_passage(context, anchor, ReadFilters())
        self._current(context, guard)
        result = tuple(self.store.read_more(context, anchors, guard=guard,
                                            max_chars=max_chars, filters=filters))
        self._current(context, guard)
        result = tuple(self._checked_passage(context, item, filters) for item in result)
        parents = {item.chunk.doc_id: item for item in reversed(anchors)}
        seen = set()
        for item in result:
            chunk = item.chunk
            parent = parents.get(chunk.doc_id)
            if (parent is None or chunk.doc_id in seen or len(chunk.text) > max_chars
                    or item.score != parent.score or item.retrieval_method != parent.retrieval_method
                    or (chunk.start_char >= parent.chunk.start_char
                        and chunk.end_char <= parent.chunk.end_char)):
                raise EvidenceValidationError("Expansion left its matched document or made no progress")
            seen.add(chunk.doc_id)
        validate_grounded_evidence(context, result, guard)
        self._current(context, guard)
        return result


class StoreBM25Retriever(_StoreReader):
    """Existing mmap BM25 scores, with all canonical bodies supplied lazily."""

    def __init__(self, index, store: DocumentStore, *, catalog: Iterable[CatalogEntry] | None = None):
        super().__init__(store, catalog)
        self.index = index
        if tuple(index.doc_ids) != self.doc_ids:
            raise SnapshotMismatch("BM25 document order differs from the store")
        if index.manifest.get("corpus_sha256") != self.snapshot.corpus_sha256:
            raise SnapshotMismatch("BM25 corpus differs from the store")
        self._manifest = json.dumps(index.manifest, sort_keys=True, allow_nan=False)

    def candidates(self, context: RequestContext, query: str, *, guard: FreshnessGuard,
                   top_k: int = 10, filters: ReadFilters = ReadFilters()) -> tuple[RankedCandidate, ...]:
        self._current(context, guard)
        _query(query, top_k)
        filters = _filters(filters)
        if (tuple(self.index.doc_ids) != self.doc_ids
                or json.dumps(self.index.manifest, sort_keys=True, allow_nan=False) != self._manifest):
            raise SnapshotMismatch("BM25 index identity changed after adapter construction")
        mask = self._mask(context, guard, filters)
        if not mask.any():
            self._current(context, guard)
            return ()
        ranked = self.index.rank_candidates(query, top_k, snapshot=self.snapshot,
                                            catalog=self.catalog, mask=mask)
        self._current(context, guard)
        result = tuple(self._checked_candidate(context, value, filters) for value in ranked)
        if len(result) > top_k or any(value.retrieval_method != "bm25" for value in result):
            raise EvidenceValidationError("BM25 returned invalid candidate bounds or method")
        self._current(context, guard)
        return result

    def search(self, context: RequestContext, query: str, *, guard: FreshnessGuard,
               top_k: int = 10, filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        filters = _filters(filters)
        candidates = self.candidates(context, query, guard=guard, top_k=top_k, filters=filters)
        result = tuple(self._materialize(context, value, query, guard, filters) for value in candidates)
        validate_grounded_evidence(context, result, guard)
        self._current(context, guard)
        return result


class StoreDenseRetriever(_StoreReader):
    """Accepted dense metadata search with a caller-supplied encoder.

    candidates preserves the complete winning embedding span. search returns
    at most 3000 characters from it, unlike the legacy standalone dense search
    which emitted the complete span. The selector matches legacy hybrid output.
    """

    def __init__(self, index, encoder, store: DocumentStore):
        super().__init__(store, index.catalog)
        if index.snapshot != self.snapshot or tuple(index.doc_ids) != self.doc_ids:
            raise SnapshotMismatch("Dense index differs from the store")
        self.index, self.encoder = index, encoder
        self._model_identity = json.loads(json.dumps(index.manifest["model"], allow_nan=False))
        self._check_model()

    def _check_model(self):
        if self.encoder.identity != self._model_identity:
            raise EvidenceValidationError("Query encoder settings differ from dense index settings")

    def candidates(self, context: RequestContext, query: str, *, guard: FreshnessGuard,
                   top_k: int = 10, filters: ReadFilters = ReadFilters()) -> tuple[RankedCandidate, ...]:
        self._current(context, guard)
        _query(query, top_k)
        filters = _filters(filters)
        if not self._mask(context, guard, filters).any():
            self._current(context, guard)
            return ()
        # Check immediately before encoding as well as after potentially slow
        # model work; no stale session may proceed to scanning or text access.
        self._current(context, guard)
        self._check_model()
        encoded = self.encoder.encode([query], query=True)
        self._current(context, guard)
        self._check_model()
        if len(encoded) != 1:
            raise EvidenceValidationError("Dense encoder must return one query vector")
        ranked = self.index.candidates_vector(encoded[0], top_k, scope=context.scope, filters=filters)
        self._current(context, guard)
        result = tuple(self._checked_candidate(context, value, filters) for value in ranked)
        if len(result) > top_k or any(value.retrieval_method != "dense" for value in result):
            raise EvidenceValidationError("Dense returned invalid candidate bounds or method")
        self._current(context, guard)
        return result

    def search(self, context: RequestContext, query: str, *, guard: FreshnessGuard,
               top_k: int = 10, filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        filters = _filters(filters)
        candidates = self.candidates(context, query, guard=guard, top_k=top_k, filters=filters)
        result = tuple(self._materialize(context, value, query, guard, filters) for value in candidates)
        validate_grounded_evidence(context, result, guard)
        self._current(context, guard)
        return result


class StoreHybridRetriever:
    """Fuse metadata first; read only final winners or reranker candidates."""

    def __init__(self, bm25: StoreBM25Retriever, dense: StoreDenseRetriever, *,
                 config: HybridConfig | None = None, reranker=None):
        self.bm25, self.dense = bm25, dense
        if bm25.snapshot != dense.snapshot or bm25.doc_ids != dense.doc_ids:
            raise SnapshotMismatch("Hybrid retrievers must cover the same store snapshot")
        self.snapshot, self.doc_ids = bm25.snapshot, bm25.doc_ids
        self.config = HybridConfig.model_validate((config or HybridConfig()).model_dump(warnings=False))
        self.reranker = reranker

    def source_types(self) -> tuple[str, ...]:
        return self.bm25.source_types()

    def read_more(self, context, anchors, *, guard, max_chars=3000, filters=ReadFilters()):
        return self.bm25.read_more(context, anchors, guard=guard, max_chars=max_chars, filters=filters)

    def search(self, context: RequestContext, query: str, *, guard: FreshnessGuard,
               top_k: int = 10, filters: ReadFilters = ReadFilters()) -> tuple[PassageEvidence, ...]:
        self.bm25._current(context, guard)
        self.dense._current(context, guard)
        _query(query, top_k)
        filters = _filters(filters)
        config = self.config
        candidate_k = max(top_k, config.candidate_k)
        rankings = tuple(retriever.candidates(context, query, guard=guard,
                    top_k=candidate_k, filters=filters) for retriever in (self.bm25, self.dense))
        fused = reciprocal_rank_candidates(rankings, k=config.rrf_k,
                    weights=(config.bm25_weight, config.dense_weight),
                    limit=max(top_k, config.rerank_k) if self.reranker is not None else top_k)
        # Validate every selected metadata item before reading the first body.
        fused = tuple(self.bm25._checked_candidate(context, value, filters) for value in fused)
        self.bm25._current(context, guard)
        passages = tuple(self.bm25._materialize(context, value, query, guard, filters) for value in fused)
        validate_grounded_evidence(context, passages, guard)
        result = tuple(self.reranker.rerank(query, passages, top_k)) if self.reranker is not None else passages
        self.bm25._current(context, guard)
        result = tuple(self.bm25._checked_passage(context, value, filters) for value in result)
        by_id = {value.citation_id: value for value in passages}
        if len(result) > top_k:
            raise EvidenceValidationError("Reranker exceeded the requested result limit")
        for value in result:
            candidate = by_id.get(value.citation_id)
            if (candidate is None or value.chunk != candidate.chunk
                    or value.snapshot_id != candidate.snapshot_id or value.revision_id != candidate.revision_id
                    or value.retrieval_method != "hybrid"):
                raise EvidenceValidationError("Reranker returned evidence outside its candidates")
        validate_grounded_evidence(context, result, guard)
        self.bm25._current(context, guard)
        self.dense._current(context, guard)
        return result

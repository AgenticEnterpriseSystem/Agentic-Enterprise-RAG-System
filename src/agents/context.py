"""Budgeted evidence selection across initial and follow-up searches."""
import math
import re
from collections import Counter

from src.contracts import AggregateEvidence, ContextSelectionStats, Evidence, StructuredRecord, validate_evidence
from src.generation.generator import evidence_chars, evidence_fingerprint, select_context
from src.retrieval.passages import bounded_passage, slice_passage

_STOP = frozenset('a an and are as at be by do does for from how i in is it of on or our should that the this to was we what when where which who why with'.split())


def terms(text):
    return set(re.findall(r'\w+', text.lower())) - _STOP


class ContextBudgetError(ValueError):
    """Previously supported evidence cannot fit without silently dropping facts."""

    def __init__(self, message, *, reason="retained_char_budget"):
        super().__init__(message)
        self.reason = reason


def evidence_document(item):
    if isinstance(item, Evidence):
        return item.chunk.doc_id
    if isinstance(item, StructuredRecord):
        return item.doc_id
    return None  # An aggregate remains atomic; its relation cardinality is not a cap input.


def document_ids(items):
    return tuple(dict.fromkeys(doc for item in items if (doc := evidence_document(item)) is not None))


def _contributes(parent, selected):
    if not isinstance(parent, Evidence):
        return parent == selected
    if not isinstance(selected, Evidence):
        return False
    a, b = parent.chunk, selected.chunk
    left, right = max(a.start_char, b.start_char), min(a.end_char, b.end_char)
    return (a.doc_id == b.doc_id and a.source_type == b.source_type and a.title == b.title
        and a.timestamp == b.timestamp and a.metadata_origin == b.metadata_origin
        and getattr(parent, 'snapshot_id', None) == getattr(selected, 'snapshot_id', None)
        and getattr(parent, 'revision_id', None) == getattr(selected, 'revision_id', None)
        and left < right and a.text[left-a.start_char:right-a.start_char] == b.text[left-b.start_char:right-b.start_char])


def selection_stats(request, selected, limit, document_limit, round_number, *, capped_docs=(), failure_reason=None):
    passages = tuple(item for item in request.evidence if isinstance(item, Evidence))
    used = tuple(item for item in selected if isinstance(item, Evidence))
    omitted = tuple(item for item in request.evidence if not any(_contributes(item, value) for value in selected))
    cropped = sum(not any(item == parent for parent in passages) for item in used)
    reasons = {}
    if failure_reason:
        reasons[failure_reason] = 1
    else:
        by_cap = sum(evidence_document(item) in capped_docs for item in omitted)
        if by_cap: reasons['document_limit'] = by_cap
        if len(omitted) > by_cap: reasons['char_budget'] = len(omitted) - by_cap
        if cropped: reasons['cropped'] = cropped
    return ContextSelectionStats(round_number=round_number, document_limit=document_limit,
        pool_documents=len(document_ids(request.evidence)), pool_passages=len(passages),
        selected_documents=len(document_ids(selected)), selected_passages=len(used),
        selected_native_items=sum(isinstance(item, (AggregateEvidence, StructuredRecord)) for item in selected),
        selected_chars=sum(evidence_chars(item) for item in selected), char_budget=limit,
        excluded_documents=len(set(document_ids(request.evidence)) - set(document_ids(selected))),
        excluded_passages=sum(not any(_contributes(item, value) for value in selected) for item in passages),
        cropped_passages=cropped, exclusion_reasons=reasons, evidence_sha256=evidence_fingerprint(selected))


def select_shared_context(request, limit, initial_ids, retained=(), *, document_limit, round_number=1):
    """Apply a global document ceiling after existing relevance/diversity ranking.

    This is a ceiling, never a quota: documents not selected by the current
    character-budgeted ranking are not padded into context. Exact supported
    passages/native items take precedence and are never silently discarded.
    """
    if type(document_limit) is not int or not 1 <= document_limit <= 20:
        raise ValueError('Document limit must be an integer from1 through20')
    retained_docs = document_ids(retained)
    if len(retained_docs) > document_limit:
        raise ContextBudgetError('Supported documents exceed the configured document limit', reason='retained_document_limit')
    ranked = tuple(select_agent_context(request, limit, initial_ids, retained))
    ranked_docs = tuple(dict.fromkeys((*retained_docs, *document_ids(ranked))))
    chosen_docs = list(retained_docs)
    original_docs = frozenset(evidence_document(item) for item in request.evidence if item.citation_id in initial_ids)
    original = tuple(doc for doc in ranked_docs if doc in original_docs)
    followup = tuple(doc for doc in ranked_docs if doc not in original_docs)
    if len(ranked_docs) > document_limit and original and followup:
        # The ranked output lists the initial group first. Applying a simple
        # prefix cap there would erase useful follow-up evidence even though
        # the character selector explicitly reserves room for both groups.
        for group, quota in ((original, document_limit // 2), (followup, document_limit - document_limit // 2)):
            available = max(0, quota - sum(doc in group for doc in chosen_docs))
            for doc in group:
                if available == 0 or len(chosen_docs) == document_limit: break
                if doc not in chosen_docs: chosen_docs.append(doc); available -= 1
    for doc in ranked_docs:
        if len(chosen_docs) == document_limit: break
        if doc not in chosen_docs: chosen_docs.append(doc)
    allowed = frozenset(chosen_docs)
    capped = frozenset(ranked_docs) - allowed
    if capped:
        pool = tuple(item for item in request.evidence if evidence_document(item) is None or evidence_document(item) in allowed)
        reduced = request.model_copy(update={'evidence': pool})
        selected = tuple(select_agent_context(reduced, limit, initial_ids, retained))
    else:
        selected = ranked
    if len(document_ids(selected)) > document_limit or any(item not in selected for item in retained):
        raise ContextBudgetError('Supported context cannot satisfy the configured document limit', reason='retained_document_limit')
    return selected, selection_stats(request, selected, limit, document_limit, round_number, capped_docs=capped)


def select_agent_context(request, limit, initial_ids, retained=()):
    """Keep relevant evidence from both rounds without consulting labels or answers.

    Initial-only requests preserve retrieval ranking. With useful follow-up hits,
    reserve up to half the character allowance for each group, rank passages by
    question coverage, and reward terms not already covered. Real token counting
    remains the provider's responsibility.
    """
    if any(isinstance(item, (AggregateEvidence, StructuredRecord)) for item in (*request.evidence, *retained)):
        validate_evidence(request.user, request.evidence)
        validate_evidence(request.user, tuple(retained))
        required_native = [item for item in retained if not isinstance(item, Evidence)]
        if any(item not in request.evidence for item in required_native):
            raise ValueError("Retained native evidence is outside the collected evidence")
        if sum(evidence_chars(item) for item in retained) > limit:
            raise ContextBudgetError("Supported evidence exceeds the remaining context allowance")
        chosen = list(required_native)
        remaining = limit - sum(evidence_chars(item) for item in required_native)
        passage_retained = tuple(item for item in retained if isinstance(item, Evidence))
        reserved = sum(evidence_chars(item) for item in passage_retained)
        for item in request.evidence:
            if isinstance(item, Evidence) or item in chosen:
                continue
            size = evidence_chars(item)
            if size <= remaining - reserved:
                chosen.append(item)
                remaining -= size
        passages = tuple(item for item in request.evidence if isinstance(item, Evidence))
        if passages:
            chosen.extend(select_agent_context(request.model_copy(update={"evidence": passages}),
                remaining, initial_ids, passage_retained))
        return chosen
    if retained:
        # Retain exact passages cited by the previous assessment, not its prose.
        # Remove overlapping text from candidates so it does not consume room twice.
        if sum(len(e.chunk.text) for e in retained) > limit:
            raise ContextBudgetError("Supported passages exceed the remaining context allowance")
        validate_evidence(request.user, tuple(retained))
        for keep in retained:
            if not any(e.chunk.doc_id == keep.chunk.doc_id
                       and e.chunk.source_type == keep.chunk.source_type
                       and e.chunk.start_char <= keep.chunk.start_char
                       and e.chunk.end_char >= keep.chunk.end_char
                       and e.chunk.text[keep.chunk.start_char-e.chunk.start_char:
                                        keep.chunk.end_char-e.chunk.start_char] == keep.chunk.text
                       for e in request.evidence):
                raise ValueError('Retained passage is outside the collected evidence')
        remaining = []
        consumed = [e.chunk for e in retained]
        original_ids = set(initial_ids)
        rewritten_ids = set()
        for item in request.evidence:
            spans = [(item.chunk.start_char, item.chunk.end_char)]
            for keep in consumed:
                if keep.doc_id != item.chunk.doc_id:
                    continue
                if item.chunk.source_type != keep.source_type:
                    raise ValueError("Retained passage source differs from the evidence pool")
                spans = [(left, right) for start, end in spans
                         for left, right in ((start, min(end, keep.start_char)),
                                             (max(start, keep.end_char), end)) if left < right]
            for start, end in spans:
                part = slice_passage(item, start, end)
                remaining.append(part)
                consumed.append(part.chunk)
                if item.citation_id in original_ids:
                    rewritten_ids.add(part.citation_id)
        allowance = limit - sum(len(e.chunk.text) for e in retained)
        rest = select_agent_context(request.model_copy(update={"evidence": tuple(remaining)}),
                                    allowance, rewritten_ids) if allowance else []
        return list(retained) + rest
    initial_ids = set(initial_ids)
    original = [e for e in request.evidence if e.citation_id in initial_ids]
    followup = [e for e in request.evidence if e.citation_id not in initial_ids]
    query = terms(request.question)
    candidates = [(e, terms(e.chunk.title + ' ' + e.chunk.text)) for e in request.evidence]
    frequency = Counter(t for _, words in candidates for t in words & query)
    anchors = {t.lower() for t in re.findall(r'\b[A-Z][A-Z0-9_]{1,}\b', request.question)}
    weights = {t: (1 + math.log((len(candidates) + 1) / (frequency[t] + 1)))
               * (3 if t in anchors else 1) for t in query}
    words_by_id = {e.citation_id: words & query for e, words in candidates}
    if not original or not followup or not any(words_by_id[e.citation_id] for e in followup) or limit < 128:
        return select_context(request, limit)

    selected, covered, selected_docs = [], set(), set()

    def take(pool, budget, keep_unmatched=False):
        remaining = budget
        while pool and remaining >= 32:
            def score(item):
                overlap = words_by_id[item.citation_id]
                title_terms = terms(item.chunk.title)
                return (sum(weights[t] * (1 if t not in covered else .3)
                            * (2 if t in title_terms else 1) for t in overlap)
                        * (0.5 if item.chunk.doc_id in selected_docs else 1))
            best = max(pool, key=score)
            pool.remove(best)
            if not words_by_id[best.citation_id] and not keep_unmatched:
                break
            # Shorter passages admit several sources rather than one long hit.
            cap = min(1500, remaining)
            if len(best.chunk.text) > cap and cap >= 128:
                # Preserve the retrieved passage's lead as well as a query window:
                # a lead can state scope/priority that a keyword-dense tail omits.
                lead = bounded_passage(best, "", cap // 2)
                window = bounded_passage(best, request.question, cap - len(lead.chunk.text))
                if window.chunk.start_char < lead.chunk.end_char:
                    pieces = [bounded_passage(best, "", cap)]
                else:
                    pieces = [lead, window]
            else:
                pieces = [bounded_passage(best, request.question, cap)]
            selected.extend(pieces)
            for item in pieces:
                covered.update(terms(item.chunk.text) & query)
                remaining -= len(item.chunk.text)
            selected_docs.add(best.chunk.doc_id)
        return remaining

    original_budget = limit // 2
    unused = take(original, original_budget, keep_unmatched=True)
    unused = take(followup, limit - original_budget + unused)
    take(original + followup, unused)
    return selected

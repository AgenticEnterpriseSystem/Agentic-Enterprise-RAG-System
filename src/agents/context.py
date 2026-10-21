"""Budgeted evidence selection across initial and follow-up searches."""
import math
import re
from collections import Counter

from src.generation.generator import select_context
from src.retrieval.passages import bounded_passage

_STOP = frozenset('a an and are as at be by do does for from how i in is it of on or our should that the this to was we what when where which who why with'.split())


def terms(text):
    return set(re.findall(r'\w+', text.lower())) - _STOP


def select_agent_context(request, limit, initial_ids):
    """Keep relevant evidence from both rounds without consulting labels or answers.

    Initial-only requests preserve retrieval ranking. With useful follow-up hits,
    reserve up to half the character allowance for each group, rank passages by
    question coverage, and reward terms not already covered. Real token counting
    remains the provider's responsibility.
    """
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

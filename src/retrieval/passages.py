"""Bound exact evidence slices without changing the indexed documents.

Character windows are a cheap lexical heuristic, not a claim of token-optimal
chunking. Dense evidence already carries its retrieval-selected passage.
"""
import re

from src.contracts import Evidence

PASSAGE_CHARS = 3000
_STOPWORDS = frozenset("a an and are as at be by do does for from how i in is it of on or that the this to was what when where which who why with".split())


def bounded_passage(item: Evidence, query: str, max_chars: int = PASSAGE_CHARS) -> Evidence:
    """Return a query-relevant exact slice, preserving short evidence verbatim.

    Windows overlap by half their length. Distinct non-stopword query matches
    take priority, then match count, then earliest offset. Citation and chunk
    IDs of new slices describe the original document coordinates.
    """
    if max_chars < 1:
        raise ValueError("Passage length must be positive")
    text = item.chunk.text
    if len(text) <= max_chars:
        return item
    terms = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    step = max(1, max_chars // 2)
    starts = list(range(0, len(text) - max_chars + 1, step))
    final_start = len(text) - max_chars
    if starts[-1] != final_start:
        starts.append(final_start)

    def score(start):
        words = re.findall(r"\w+", text[start:start + max_chars].lower())
        matches = [word for word in words if word in terms]
        return (len(set(matches)), len(matches), -start)

    start = max(starts, key=score)
    end = start + max_chars
    absolute_start = item.chunk.start_char + start
    absolute_end = item.chunk.start_char + end
    identifier = f"{item.chunk.doc_id}:{absolute_start}:{absolute_end}"
    chunk = item.chunk.model_copy(update={
        "chunk_id": identifier, "text": text[start:end],
        "start_char": absolute_start, "end_char": absolute_end,
    })
    return item.model_copy(update={"citation_id": identifier, "chunk": chunk})

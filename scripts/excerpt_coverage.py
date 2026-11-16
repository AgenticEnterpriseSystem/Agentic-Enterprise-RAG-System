"""Offline check: do the excerpts the agent shows contain the facts of the reference answer?

For each question, every reference document is cut into the excerpt the app would show for the
question (`bounded_passage`: a query-relevant slice of at most N characters), at several sizes.
The script then checks which of the gold answer's specific facts (numbers, dates, IDs, code
names) appear in those excerpts, compared with the whole documents. No searches and no model
calls: it needs only the corpus and question files, and takes about a minute.

A fact found in the whole document but missing from the excerpt is one the critic could not see.
Facts are matched as exact tokens, so this is a lower bound on what the excerpt conveys; the
answer may also be paraphrased. Retrieval is not tested here: every reference document is
assumed to have been found.

    python scripts/excerpt_coverage.py
    python scripts/excerpt_coverage.py --results .cache/agent-timing/<run>/results.json
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SIZES = (1500, 3000, 4500, 6000, 9000, 12000)
# A token with a digit, an underscore, an inner dot or hyphen, or two or more capitals:
# dates, amounts, percentages, versions, ticket keys, config names, acronyms and code names.
_TOKEN = re.compile(r"[A-Za-z0-9][\w.\-/:%]*[\w%]|\d")
_SPECIFIC = re.compile(r"\d|_|\w[.\-]\w|[A-Z].*[A-Z]")
_STOP = frozenset("""a an and are as at be by can do does for from has have how i if in into is it its
of on or our should so than that the their then there these they this to was we were what when
where which who why will with would also not no yes""".split())


def key_facts(answer: str) -> set[str]:
    """Specific tokens of a reference answer, lower-cased, without trailing punctuation."""
    facts = set()
    for token in _TOKEN.findall(answer):
        token = token.strip(".-/:")
        if len(token) >= 2 and _SPECIFIC.search(token):
            facts.add(token.lower())
    return facts


def content_words(answer: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", answer.lower()) if w not in _STOP}


def present(facts: set[str], text: str) -> set[str]:
    text = text.lower()
    return {f for f in facts if re.search(r"(?<!\w)" + re.escape(f) + r"(?!\w)", text)}


def excerpt(doc: dict, question: str, size: int) -> str:
    from src.contracts import Chunk, Evidence
    from src.retrieval.passages import bounded_passage

    text = doc["text"]
    chunk = Chunk(chunk_id=f"{doc['doc_id']}:0:{len(text)}", doc_id=doc["doc_id"],
                  source_type=doc["source_type"], title=doc["title"], text=text,
                  start_char=0, end_char=len(text))
    item = Evidence(citation_id=chunk.chunk_id, chunk=chunk, score=1, retrieval_method="bm25")
    return bounded_passage(item, question, size).chunk.text


def load_documents(corpus: Path, doc_ids: set[str]) -> dict[str, dict]:
    import pyarrow.parquet as pq
    from src.retrieval.corpus_bm25 import document_text

    table = pq.read_table(corpus, columns=["doc_id", "source_type", "title", "content"],
                          filters=[("doc_id", "in", sorted(doc_ids))])
    return {row["doc_id"]: {**row, "text": document_text(row["title"], row["content"])}
            for row in table.to_pylist()}


def measure(question: dict, documents: dict[str, dict], sizes) -> dict | None:
    """Fact coverage of one question's reference documents, whole and as excerpts."""
    docs = [documents[d] for d in question["expected_doc_ids"] if d in documents]
    if not docs or not question.get("gold_answer"):
        return None
    facts = key_facts(question["gold_answer"])
    whole_text = "\n".join(d["text"] for d in docs)
    in_docs = present(facts, whole_text)
    words = content_words(question["gold_answer"])
    words_in_docs = present(words, whole_text)
    row = {"question_id": question["question_id"], "question_type": question["question_type"],
           "references": len(docs), "facts": len(facts), "facts_in_documents": len(in_docs),
           "document_chars": [len(d["text"]) for d in docs], "sizes": {}}
    for size in sizes:
        text = "\n".join(excerpt(d, question["question"], size) for d in docs)
        seen = present(in_docs, text)
        row["sizes"][size] = {
            "facts_seen": len(seen), "missing": sorted(in_docs - seen),
            "words_seen": len(present(words_in_docs, text)), "words_in_documents": len(words_in_docs)}
    return row


def report(rows, sizes, label="All questions"):
    scored = [r for r in rows if r["facts_in_documents"]]
    print(f"\n{label}: {len(rows)} questions with reference documents and a gold answer; "
          f"{len(scored)} have answer facts (numbers, dates, IDs, names) in their documents.")
    if not scored:
        return
    print(f"  {'excerpt size':>14} {'facts seen':>11} {'all facts seen':>15} {'answer words seen':>18}")
    for size in sizes:
        share = statistics.mean(r["sizes"][size]["facts_seen"] / r["facts_in_documents"] for r in scored)
        complete = sum(not r["sizes"][size]["missing"] for r in scored)
        words = [r["sizes"][size]["words_seen"] / r["sizes"][size]["words_in_documents"]
                 for r in rows if r["sizes"][size]["words_in_documents"]]
        print(f"  {size:>14,} {share:>10.0%} {complete:>7} ({complete / len(scored):>4.0%}) "
              f"{statistics.mean(words) if words else 0:>17.0%}")
    print(f"  {'whole document':>14} {1:>10.0%} {len(scored):>7} (100%)")


def main():
    from run_web import build_environment

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / ".cache/corpus/questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation/splits.json")
    parser.add_argument("--split", default="development", choices=["development", "calibration", "final"])
    parser.add_argument("--corpus", type=Path, help="documents.parquet (default: the web app's RAG_CORPUS)")
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)), help="Excerpt sizes in characters")
    parser.add_argument("--results", type=Path,
                        help="A time_agent results.json: also report its unanswered questions one by one")
    parser.add_argument("--output", type=Path, help="Write per-question rows as JSON here")
    args = parser.parse_args()
    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
    corpus = args.corpus or Path(build_environment(ROOT, os.environ, "hybrid")["RAG_CORPUS"])

    from src.evaluation.run_retrieval import load_questions

    questions = load_questions(args.questions, args.splits, args.split)
    if questions and "gold_answer" not in questions[0]:
        parser.error("The question file has no gold_answer field")
    wanted = {d for q in questions for d in q["expected_doc_ids"]}
    print(f"Reading {len(wanted)} reference documents from {corpus}...", flush=True)
    documents = load_documents(corpus, wanted)
    rows = [r for r in (measure(q, documents, sizes) for q in questions) if r]
    lengths = [n for r in rows for n in r["document_chars"]]
    print(f"Reference documents: median {statistics.median(lengths):,.0f} characters; "
          f"{sum(n <= 3000 for n in lengths) / len(lengths):.0%} fit in one 3,000-character excerpt.")
    report(rows, sizes)

    print("\nBy question type (share of questions whose excerpt shows every answer fact the documents contain):")
    print(f"  {'question type':26} {'n':>3} " + " ".join(f"{s:>7,}" for s in sizes))
    by_type = defaultdict(list)
    for r in rows:
        if r["facts_in_documents"]:
            by_type[r["question_type"]].append(r)
    for qtype, group in sorted(by_type.items()):
        print(f"  {qtype:26} {len(group):>3} " + " ".join(
            f"{sum(not r['sizes'][s]['missing'] for r in group) / len(group):>7.0%}" for s in sizes))

    if args.results:
        runs = json.loads(args.results.read_text())["results"]
        records = [r for mode in runs.values() for arm in mode.values() for r in arm
                   if r.get("diagnosis") not in (None, "answered", "no_answer_expected")]
        by_id = {r["question_id"]: r for r in rows}
        print(f"\nUnanswered questions in {args.results.name} (facts seen in the excerpt / facts in the documents):")
        print(f"  {'question':10} {'type':25} {'diagnosis':20} " + " ".join(f"{s:>7,}" for s in sizes)
              + "  missing at 3,000")
        for record in records:
            r = by_id.get(record["question_id"])
            if r is None or not r["facts_in_documents"]:
                print(f"  {record['question_id']:10} {record['question_type']:25} {record['diagnosis']:20} "
                      "(no answer facts found in its documents)")
                continue
            cells = " ".join(f"{r['sizes'][s]['facts_seen']:>3}/{r['facts_in_documents']:<3}" for s in sizes)
            missing = r["sizes"].get(3000, {}).get("missing", [])
            print(f"  {r['question_id']:10} {r['question_type']:25} {record['diagnosis']:20} {cells}  "
                  f"{', '.join(missing[:5])}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"\nPer-question rows: {args.output}")


if __name__ == "__main__":
    main()

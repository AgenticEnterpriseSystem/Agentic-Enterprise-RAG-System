"""Build a fixed corpus sample for screening embedding models cheaply.

The sample keeps every reference document for one question split, the documents
that BM25+ and BGE-base ranked highest for those questions on the full corpus
(the look-alikes that compete with the answers), and seeded random documents.
Scores on a sample are higher than on the full corpus, so compare models with
each other on the same sample, not with full-corpus numbers.
"""

import argparse
import json
import random
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import CORPUS_DIR, ROOT
from src.evaluation.run_retrieval import load_questions

RESULTS = ROOT / "evaluation" / "results"
DEFAULT_RANKINGS = ("bm25_full_development", "dense_base_titled_full_development")


def sample_ids(corpus_ids, questions, rankings, *, top_k=20, random_docs=10000, seed=5980):
    """Reference, look-alike and random document IDs, plus counts for the manifest."""
    known = set(corpus_ids)
    reference = {d for q in questions for d in q["expected_doc_ids"]}
    if not reference <= known:
        raise ValueError("Reference documents are missing from the corpus")
    wanted = {q["question_id"] for q in questions}
    lookalikes = set()
    for rows in rankings:
        for row in rows:
            if row["question_id"] in wanted:
                lookalikes.update(row["retrieved_doc_ids"][:top_k])
    lookalikes -= reference
    rest = [d for d in corpus_ids if d not in reference and d not in lookalikes]
    if random_docs > len(rest):
        raise ValueError("Not enough remaining documents for the random part")
    extra = set(random.Random(seed).sample(rest, random_docs))
    counts = {"reference": len(reference), "lookalikes": len(lookalikes), "random": len(extra)}
    return reference | lookalikes | extra, counts


def build_sample(corpus_path, questions, ranking_dirs, output, *, split, top_k=20,
                 random_docs=10000, seed=5980):
    """Write documents.parquet (corpus order) and manifest.json under `output`."""
    rankings = [[json.loads(line) for line in (Path(d) / "predictions.jsonl").open()]
                for d in ranking_dirs]
    ids = pq.read_table(corpus_path, columns=["doc_id"]).column("doc_id").to_pylist()
    chosen, counts = sample_ids(ids, questions, rankings, top_k=top_k,
                                random_docs=random_docs, seed=seed)
    table = pq.read_table(corpus_path)
    table = table.filter(pc.is_in(table.column("doc_id"), value_set=pa.array(sorted(chosen))))
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output / "documents.parquet")
    kept = table.column("doc_id").to_pylist()
    manifest = {
        "split": split,
        "questions": len(questions),
        "rankings": [Path(d).name for d in ranking_dirs],
        "top_k": top_k,
        "seed": seed,
        "documents": len(kept),
        **counts,
        "doc_ids_sha256": sha256("\n".join(sorted(kept)).encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--questions", type=Path, default=CORPUS_DIR / "questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation" / "splits.json")
    parser.add_argument("--split", default="development",
                        help="Questions whose reference documents are kept (development only)")
    parser.add_argument("--rankings", type=Path, nargs="+",
                        default=[RESULTS / name for name in DEFAULT_RANKINGS],
                        help="Result folders whose top documents are kept as look-alikes")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-docs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=5980)
    parser.add_argument("--output", type=Path, default=CORPUS_DIR / "sample")
    args = parser.parse_args()
    if args.split != "development":
        parser.error("Build screening samples from development questions only")
    questions = load_questions(args.questions, args.splits, args.split)
    manifest = build_sample(args.corpus, questions, args.rankings, args.output, split=args.split,
                            top_k=args.top_k, random_docs=args.random_docs, seed=args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

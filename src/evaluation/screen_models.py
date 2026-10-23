"""Compare dense models on a corpus sample: alone, and merged with BM25+.

For each dense index (built on the same sample), reports Recall@10 and nDCG@10,
how many questions BM25+ misses that the model finds, and the hybrid score with
BM25+ (RRF via HybridRetriever). BM25+ is searched once and shared.
"""

import argparse
import json
import time
from pathlib import Path

from src.config import CORPUS_DIR, ROOT, RetrievalSettings
from src.retrieval.factory import load_dense
from src.contracts import UserContext
from src.evaluation.retrieval_metrics import evaluate_retrieval, recall_at_k
from src.evaluation.run_hybrid import compare, run_single, run_variant, search_once
from src.evaluation.run_retrieval import KS, build_summary, load_questions, write_run


def misses_found(rows, bm25_rows):
    """(questions BM25+ finds nothing for in its top 10, how many of those `rows` finds)."""
    missed = found = 0
    for row, base in zip(rows, bm25_rows):
        rel = row["relevant_doc_ids"]
        if rel and recall_at_k(base["retrieved_doc_ids"], rel, 10) == 0:
            missed += 1
            found += recall_at_k(row["retrieved_doc_ids"], rel, 10) > 0
    return missed, found


def parse_dense(text):
    label, sep, path = text.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError("Use LABEL=PATH, e.g. bge-base=/content/sample/bge-base")
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=CORPUS_DIR / "sample",
                        help="Folder with the sample documents.parquet and manifest.json")
    parser.add_argument("--bm25-index", type=Path, help="Default: <sample>/bm25")
    parser.add_argument("--dense", type=parse_dense, nargs="+", required=True,
                        help="LABEL=PATH for each dense index built on the sample")
    parser.add_argument("--questions", type=Path, default=CORPUS_DIR / "questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation" / "splits.json")
    parser.add_argument("--device", default=RetrievalSettings().device)
    parser.add_argument("--bm25-weight", type=float, default=RetrievalSettings().bm25_weight,
                        help="BM25 RRF weight for the hybrid (the full-corpus development choice)")
    parser.add_argument("--candidate-k", type=int, default=RetrievalSettings().candidate_k)
    parser.add_argument('--rerank-k', type=int, default=RetrievalSettings().rerank_k)
    parser.add_argument('--rrf-k', type=float, default=RetrievalSettings().rrf_k)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "evaluation" / "results" / "screening")
    args = parser.parse_args()

    from src.retrieval.corpus_bm25 import CorpusBM25

    sample_manifest = json.loads((args.sample / "manifest.json").read_text())
    split = sample_manifest["split"]
    corpus = args.sample / "documents.parquet"
    settings = RetrievalSettings(corpus=corpus, device=args.device, fp16=args.fp16,
        bm25_weight=args.bm25_weight, candidate_k=args.candidate_k,
        rerank_k=args.rerank_k, rrf_k=args.rrf_k)
    questions = load_questions(args.questions, args.splits, split)
    started = time.perf_counter()
    bm25 = CorpusBM25.load(args.bm25_index or args.sample / "bm25", corpus_path=corpus)
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    (bm25_cache,), bm25_seconds, bm25_errors = search_once([bm25], questions, user, max(max(KS), settings.candidate_k))
    bm25_rows, bm25_latency = run_single(bm25_cache, bm25_seconds, bm25_errors, questions, max(KS))
    extra = {"sample": sample_manifest}

    def save(name, rows, latencies, index, load_seconds):
        summary = build_summary(rows, latencies, method=name, split=split, top_k=max(KS),
                                index=index, load_seconds=load_seconds, **extra)
        write_run(args.output_dir / f"{name}_sample_{split}", rows, summary)

    save("bm25", bm25_rows, bm25_latency, bm25.manifest, time.perf_counter() - started)
    table = [("BM25+ alone", bm25_rows, None)]
    for label, path in args.dense:
        began = time.perf_counter()
        dense = load_dense(settings.model_copy(update={'dense_index': path}),
                           corpus_path=corpus, corpus=bm25.corpus, bm25_manifest=bm25.manifest)
        load_seconds = time.perf_counter() - began
        (cache,), seconds, errors = search_once([dense], questions, user, max(max(KS), settings.candidate_k))
        rows, latency = run_single(cache, seconds, errors, questions, max(KS))
        save(f"dense_{label}", rows, latency, dense.manifest, load_seconds)
        both_seconds = {q: seconds[q] + bm25_seconds[q] for q in seconds}
        hybrid, hybrid_latency, manifest = run_variant(
            (bm25, dense), (bm25_cache, cache), both_seconds, {**bm25_errors, **errors},
            questions, user, bm25_weight=args.bm25_weight, top_k=max(KS),
            candidate_k=settings.candidate_k, rerank_k=settings.rerank_k, rrf_k=settings.rrf_k)
        save(f"hybrid_{label}_w{args.bm25_weight:g}", hybrid, hybrid_latency, manifest, load_seconds)
        table += [(f"{label} alone", rows, None), (f"BM25+ + {label} (hybrid)", hybrid, label)]
        print(f"{label}: loaded in {load_seconds:.0f}s, searched in {sum(seconds.values()):.0f}s",
              flush=True)

    print(f"\nSample: {sample_manifest['documents']:,} documents; {split} questions "
          f"(scores are higher than on the full corpus; compare rows with each other)")
    print(f"{'run':40} {'R@10':>6} {'nDCG@10':>8} {'BM25+ misses found':>19} {'vs BM25+ (better/worse)':>24}")
    for name, rows, _ in table:
        o = evaluate_retrieval(rows, KS)["overall"]
        missed, found = misses_found(rows, bm25_rows)
        better, worse = compare(rows, bm25_rows)
        print(f"{name:40} {o['recall@10']['mean']:>6.3f} {o['ndcg@10']['mean']:>8.3f} "
              f"{f'{found} of {missed}':>19} {f'{better} / {worse}':>24}")


if __name__ == "__main__":
    main()

"""Score BM25+ / BGE hybrid variants (weighted RRF, optional cross-encoder) on one split.

Each question is searched once by each retriever (top --candidate-k documents);
every variant then runs HybridRetriever over those same saved results, so the
variants differ only in fusion weights and reranking. Run it on a GPU runtime.
"""

import argparse
import json
import time
from pathlib import Path

from src.config import CORPUS_DIR, ROOT
from src.contracts import SearchRequest, UserContext
from src.evaluation.run_retrieval import KS, build_summary, load_questions, write_run
from src.retrieval.hybrid import HybridConfig, HybridRetriever

RERANKERS = {
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


class CachedSearch:
    """Replays one retriever's saved results so every variant sees the same candidates."""

    def __init__(self, retriever, results):
        self.doc_ids = retriever.doc_ids
        self.manifest = getattr(retriever, "manifest", {})
        self.results = results

    def search(self, request):
        return self.results[request.query][: request.top_k]


def search_once(retrievers, questions, user, depth):
    """Search each distinct question once per retriever.

    Returns one {question: evidence} cache per retriever, search seconds per
    question, and errors per question (failed questions get empty results).
    """
    caches = [{} for _ in retrievers]
    seconds, errors = {}, {}
    for q in questions:
        text = q["question"]
        if text in seconds:
            continue
        started = time.perf_counter()
        try:
            hits = [tuple(r.search(SearchRequest(query=text, user=user, top_k=depth)))
                    for r in retrievers]
        except Exception as exc:  # Count failures as errors instead of dropping them.
            hits, errors[text] = [()] * len(retrievers), f"{type(exc).__name__}: {exc}"
        for cache, found in zip(caches, hits):
            cache[text] = found
        seconds[text] = time.perf_counter() - started
    return caches, seconds, errors


def result_row(q, hits, error):
    return {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "source_types": q["source_types"],
        "relevant_doc_ids": q["expected_doc_ids"],
        "retrieved_doc_ids": [e.chunk.doc_id for e in hits],
        "scores": [round(e.score, 4) for e in hits],
        "error": error,
    }


def run_single(cache, seconds, errors, questions, top_k):
    """Rows for one retriever alone, from its saved results."""
    rows = [result_row(q, cache[q["question"]][:top_k], errors.get(q["question"]))
            for q in questions]
    return rows, [seconds[q["question"]] for q in questions]


def run_variant(retrievers, caches, seconds, errors, questions, user, *, bm25_weight,
                reranker=None, top_k=max(KS), candidate_k=50):
    """Rows, latencies and manifest for one fusion/rerank setting."""
    config = HybridConfig(candidate_k=candidate_k, rerank_k=candidate_k,
                          bm25_weight=bm25_weight, dense_weight=1 - bm25_weight)
    hybrid = HybridRetriever(*(CachedSearch(r, c) for r, c in zip(retrievers, caches)),
                             config=config, reranker=reranker)
    rows, latencies = [], []
    for q in questions:
        started, hits, error = time.perf_counter(), (), errors.get(q["question"])
        if error is None:
            try:
                hits = hybrid.search(SearchRequest(query=q["question"], user=user, top_k=top_k))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        # Latency includes this question's share of the shared searches.
        latencies.append(seconds[q["question"]] + time.perf_counter() - started)
        rows.append(result_row(q, hits, error))
    return rows, latencies, hybrid.manifest


def reranker_identity(spec):
    """"label" or "label@revision" -> (label, model name, exact revision)."""
    label, _, revision = spec.partition("@")
    if label not in RERANKERS:
        raise ValueError(f"Unknown reranker {label!r}; choose from {sorted(RERANKERS)}")
    if not revision:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(RERANKERS[label]).sha
    return label, RERANKERS[label], revision


def recall10(rows):
    from src.evaluation.retrieval_metrics import evaluate_retrieval

    return evaluate_retrieval(rows, KS)["overall"]["recall@10"]["mean"]


def compare(rows, baseline):
    """Questions whose Recall@10 rose / fell versus the baseline rows."""
    from src.evaluation.retrieval_metrics import recall_at_k

    better = worse = 0
    for a, b in zip(rows, baseline):
        if a["relevant_doc_ids"]:
            change = (recall_at_k(a["retrieved_doc_ids"], a["relevant_doc_ids"], 10)
                      - recall_at_k(b["retrieved_doc_ids"], b["relevant_doc_ids"], 10))
            better, worse = better + (change > 0), worse + (change < 0)
    return better, worse


def parse_weights(text):
    weights = [float(w) for w in text.split(",") if w.strip()]
    if not weights or any(not 0 <= w <= 1 for w in weights):
        raise argparse.ArgumentTypeError("BM25 weights must be between 0 and 1")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-index", type=Path, default=CORPUS_DIR / "indexes" / "bm25")
    parser.add_argument("--dense-index", type=Path, default=CORPUS_DIR / "indexes" / "dense")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--questions", type=Path, default=CORPUS_DIR / "questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation" / "splits.json")
    parser.add_argument("--split", choices=["development", "calibration", "final"],
                        default="development")
    parser.add_argument("--device", default="cuda", help="Query encoder and reranker device")
    parser.add_argument("--bm25-weights", type=parse_weights, default=[0.5, 0.7],
                        help="BM25 RRF weights to score without reranking (BGE gets 1 - w)")
    parser.add_argument("--rerankers", default="bge-reranker-base,bge-reranker-v2-m3",
                        help="Comma-separated rerankers (label or label@revision); empty for none")
    parser.add_argument("--rerank-bm25-weight", default="best",
                        help='Fusion weight before reranking: a number, or "best" (highest '
                             "Recall@10 among --bm25-weights on this split; use on development only)")
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--write-baselines", action="store_true",
                        help="Also save BM25+ alone and BGE alone (for a new split)")
    args = parser.parse_args()
    if args.top_k < max(KS):
        parser.error(f"--top-k must be at least {max(KS)} to score Recall@{max(KS)}")
    if args.rerank_bm25_weight != "best":
        args.rerank_bm25_weight = parse_weights(args.rerank_bm25_weight)[0]
    rerankers = [reranker_identity(s.strip()) for s in args.rerankers.split(",") if s.strip()]

    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import BGEModel, CorpusDense, method_label
    from src.retrieval.reranker import CrossEncoderReranker

    questions = load_questions(args.questions, args.splits, args.split)
    started = time.perf_counter()
    bm25 = CorpusBM25.load(args.bm25_index, corpus_path=args.corpus)
    dense = CorpusDense.load(args.dense_index, BGEModel.for_index(args.dense_index, args.device),
                             corpus_path=args.corpus, corpus=bm25.corpus)
    load_seconds = time.perf_counter() - started
    dense_label = method_label(dense.manifest["model"]["name"])
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    print(f"Indexes loaded in {load_seconds:.0f}s; searching {len(questions)} questions...", flush=True)
    retrievers = (bm25, dense)
    caches, seconds, errors = search_once(retrievers, questions, user, args.candidate_k)
    print(f"Searched in {sum(seconds.values()):.0f}s ({len(errors)} errors)", flush=True)

    def save(name, rows, latencies, index, **extra):
        summary = build_summary(rows, latencies, method=name.rsplit("_full_", 1)[0],
                                split=args.split, top_k=args.top_k, index=index,
                                load_seconds=load_seconds, **extra)
        write_run(args.output_dir / name, rows, summary)
        return summary

    results = {}
    bm25_rows, bm25_latency = run_single(caches[0], seconds, errors, questions, args.top_k)
    results["BM25+ alone"] = bm25_rows
    if args.write_baselines:
        save(f"bm25_full_{args.split}", bm25_rows, bm25_latency, bm25.manifest)
        dense_rows, dense_latency = run_single(caches[1], seconds, errors, questions, args.top_k)
        save(f"dense_{dense_label}_full_{args.split}", dense_rows, dense_latency, dense.manifest)
        results[f"{dense_label} alone"] = dense_rows

    fusion = {}
    for weight in args.bm25_weights:
        name = f"hybrid_{dense_label}_w{weight:g}_full_{args.split}"
        rows, latencies, manifest = run_variant(retrievers, caches, seconds, errors, questions,
                                                user, bm25_weight=weight, top_k=args.top_k,
                                                candidate_k=args.candidate_k)
        save(name, rows, latencies, manifest)
        results[name], fusion[weight] = rows, recall10(rows)

    weight, choice = args.rerank_bm25_weight, "given"
    if weight == "best" and rerankers:
        if args.split != "development":
            parser.error('--rerank-bm25-weight "best" tunes on the split; pass a number here')
        weight, choice = max(fusion, key=fusion.get), "best development Recall@10"
    for label, name, revision in rerankers:
        reranker = CrossEncoderReranker(name, revision, device=args.device,
                                        batch_size=args.reranker_batch_size,
                                        fp16=args.device.startswith("cuda"))
        run_name = f"hybrid_{dense_label}_w{weight:g}_{label}_full_{args.split}"
        began = time.perf_counter()
        rows, latencies, manifest = run_variant(retrievers, caches, seconds, errors, questions,
                                                user, bm25_weight=weight, reranker=reranker,
                                                top_k=args.top_k, candidate_k=args.candidate_k)
        save(run_name, rows, latencies, manifest,
             fusion_weight_choice={"bm25_weight": weight, "chosen_by": choice})
        results[run_name] = rows
        print(f"{run_name}: {time.perf_counter() - began:.0f}s", flush=True)

    from src.evaluation.retrieval_metrics import evaluate_retrieval

    print(f"\n{'run':58} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'nDCG@10':>8}  vs BM25+ (R@10 better/worse)")
    for name, rows in results.items():
        o = evaluate_retrieval(rows, KS)["overall"]
        better, worse = compare(rows, bm25_rows)
        print(f"{name:58} {o['recall@5']['mean']:>6.3f} {o['recall@10']['mean']:>6.3f} "
              f"{o['recall@20']['mean']:>6.3f} {o['ndcg@10']['mean']:>8.3f}  {better:>3} / {worse}")


if __name__ == "__main__":
    main()

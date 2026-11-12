"""Time hybrid search with and without the reranker on this computer, as the web app runs it.

Uses the assembled data bundle and a few development questions; makes no model API calls.
The first query of each kind is a warm-up (model loading) and is not timed. Example:

    python scripts/time_reranker.py --count 20 --rerank-k 20 --rerank-max-windows 4

On macOS, set DYLD_LIBRARY_PATH as for the tests first (see docs/EVALUATION.md), or the
FAISS and PyTorch libraries can abort the process when both load.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import SearchRequest, UserContext  # noqa: E402
from src.evaluation.run_hybrid import reranker_identity  # noqa: E402
from src.evaluation.run_retrieval import load_questions  # noqa: E402
from src.retrieval.factory import add_retrieval_arguments, load_pair, settings_from_args  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402

BUNDLE = ROOT / ".cache" / "hybrid-artifacts"


def summary(seconds):
    ordered = sorted(seconds)
    return f"median {statistics.median(ordered):.2f}s, p95 {ordered[int(0.95 * (len(ordered) - 1))]:.2f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_retrieval_arguments(parser, weight=False)
    parser.set_defaults(corpus=BUNDLE / "documents.parquet", bm25_index=BUNDLE / "indexes/bm25",
                        dense_index=BUNDLE / "indexes/dense-bge-base")
    parser.add_argument("--questions", type=Path, default=ROOT / ".cache/corpus/questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation/splits.json")
    parser.add_argument("--count", type=int, default=20, help="Development questions to time")
    parser.add_argument("--top-k", type=int, default=5, help="Results per search (the app uses 5)")
    parser.add_argument("--reranker", default="bge-reranker-v2-m3", help="label or label@revision")
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--rerank-window-chars", type=int, default=2000)
    parser.add_argument("--rerank-max-windows", type=int, default=8)
    parser.add_argument("--rerank-weight", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    settings = settings_from_args(args, retrieval="hybrid")
    print("Loading the indexes (a few minutes)...", flush=True)
    bm25, dense = load_pair(settings)
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    _, name, revision = reranker_identity(args.reranker)
    config = settings.hybrid_config()
    plain = HybridRetriever(bm25, dense, config=config)
    reranked = HybridRetriever(bm25, dense, config=type(config)(
        **{**config.model_dump(), "rerank_weight": args.rerank_weight}),
        reranker=CrossEncoderReranker(
            name, revision, device=args.device, batch_size=args.batch_size,
            max_length=args.reranker_max_length, window_chars=args.rerank_window_chars,
            max_windows=args.rerank_max_windows, document_text=bm25.full_text))

    questions = load_questions(args.questions, args.splits, "development")[: args.count + 1]
    times = {"hybrid": [], "hybrid + reranker": []}
    for index, q in enumerate(questions):
        request = SearchRequest(query=q["question"], user=user, top_k=args.top_k)
        for label, retriever in (("hybrid", plain), ("hybrid + reranker", reranked)):
            began = time.perf_counter()
            retriever.search(request)
            if index:  # The first question warms up the models.
                times[label].append(time.perf_counter() - began)
        if index:
            print(f"{index:>3}/{args.count}  hybrid {times['hybrid'][-1]:.2f}s  "
                  f"reranked {times['hybrid + reranker'][-1]:.2f}s", flush=True)

    added = [r - h for h, r in zip(times["hybrid"], times["hybrid + reranker"])]
    print(f"\n{settings.rerank_k} candidates, up to {args.rerank_max_windows} windows of "
          f"{args.rerank_window_chars} characters, {args.reranker_max_length} tokens, on {args.device}:")
    for label, seconds in times.items():
        print(f"  {label:18} {summary(seconds)}")
    print(f"  {'added by reranker':18} {summary(added)}")


if __name__ == "__main__":
    main()

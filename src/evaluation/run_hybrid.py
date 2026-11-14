"""Score BM25+ / BGE hybrid variants (weighted RRF, optional cross-encoder) on one split.

Each question is searched once by each retriever (top --candidate-k documents);
every variant then runs HybridRetriever over those same saved results, so the
variants differ only in fusion weights and reranking. Use --device cuda for GPU inference.

Rerankers can score each candidate's best window across its whole document
(--rerank-window-chars, "MaxP") and be blended with the fused rank (--rerank-weights).
Each question's candidates are scored once and reused for every blend weight. With
several weights on the development split, a cross-validated run picks the weight on
the other folds for each question, so its score is not tuned on the questions it reports.
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.config import CORPUS_DIR, ROOT, RetrievalSettings
from src.contracts import SearchRequest, UserContext
from src.evaluation.run_retrieval import KS, build_summary, load_questions, write_run
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.factory import add_retrieval_arguments, load_pair, settings_from_args

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


class CachedReranker:
    """Scores each candidate once per question, so every blend weight, and every candidate
    count using the same reranker settings, reuses the scores. A candidate's score does not
    depend on the other candidates, so the order matches scoring them all together."""

    def __init__(self, reranker):
        self.reranker, self.identity = reranker, reranker.identity
        self.scored = {}  # (query, citation ID) -> (reranked item, windows scored)
        self.seconds = 0.0  # Time spent scoring, and the windows scored in it.
        self.windows_scored = 0
        self.windows_per_call = []  # Windows a setting needs per question, cached or not.

    def rerank(self, query, evidence, top_k):
        missing = [item for item in evidence if (query, item.citation_id) not in self.scored]
        if missing:
            began = time.perf_counter()
            reranked = self.reranker.rerank(query, missing, len(missing))
            self.seconds += time.perf_counter() - began
            counts = getattr(self.reranker, "last_window_counts", {})
            for item in reranked:
                windows = counts.get(item.citation_id, 1)
                self.scored[(query, item.citation_id)] = (item, windows)
                self.windows_scored += windows
        self.windows_per_call.append(sum(self.scored[(query, item.citation_id)][1]
                                         for item in evidence))
        ranked = sorted((self.scored[(query, item.citation_id)][0] for item in evidence),
                        key=lambda item: (-item.score, item.chunk.doc_id))
        return tuple(ranked[:top_k])

    def seconds_per_window(self):
        return self.seconds / self.windows_scored if self.windows_scored else 0.0


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
                reranker=None, top_k=max(KS), candidate_k=None, rerank_k=None, rrf_k=None,
                rerank_weight=None):
    """Rows, latencies and manifest for one fusion/rerank setting."""
    overrides = {name: value for name, value in (
        ('candidate_k', candidate_k), ('rerank_k', rerank_k), ('rrf_k', rrf_k))
        if value is not None}
    config = RetrievalSettings(bm25_weight=bm25_weight, **overrides).hybrid_config()
    if rerank_weight is not None:
        config = type(config)(**{**config.model_dump(), "rerank_weight": rerank_weight})
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


def reranker_identity(spec: str) -> tuple[str, str, str]:
    """"label" or "label@revision" -> (label, model name, exact revision)."""
    label, _, revision = spec.partition("@")
    if label not in RERANKERS:
        raise ValueError(f"Unknown reranker {label!r}; choose from {sorted(RERANKERS)}")
    if not revision:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(RERANKERS[label]).sha
    if not revision:
        raise ValueError(f"No revision available for reranker {label!r}")
    return label, RERANKERS[label], revision


def recall10(rows) -> float:
    from src.evaluation.retrieval_metrics import evaluate_retrieval

    score = evaluate_retrieval(rows, KS)["overall"]["recall@10"]["mean"]
    if score is None:
        raise ValueError("Recall@10 requires at least one labeled question")
    return score


def compare(rows, baseline):
    """Questions whose Recall@10 rose / fell versus the baseline rows."""
    from src.evaluation.retrieval_metrics import recall_at_k

    better = worse = 0
    for a, b in zip(rows, baseline):
        score = recall_at_k(a["retrieved_doc_ids"], a["relevant_doc_ids"], 10)
        baseline_score = recall_at_k(b["retrieved_doc_ids"], b["relevant_doc_ids"], 10)
        if score is None or baseline_score is None:
            continue
        change = score - baseline_score
        better, worse = better + (change > 0), worse + (change < 0)
    return better, worse


def cross_validated(rows_by_weight: dict[float, list[dict]], folds=5, seed=5980):
    """Rows where each question's blend weight was chosen on the other folds.

    For each fold, the weight with the highest mean Recall@10 on the remaining folds
    (then nDCG@10, then the lower weight, closer to the unreranked order) is used for
    that fold's questions. Returns the rows in the original order and the weight per fold.
    """
    from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k

    weights = sorted(rows_by_weight)
    order = [row["question_id"] for row in rows_by_weight[weights[0]]]
    shuffled = sorted(order)
    random.Random(seed).shuffle(shuffled)
    fold_of = {qid: i % folds for i, qid in enumerate(shuffled)}
    by_weight = {w: {row["question_id"]: row for row in rows} for w, rows in rows_by_weight.items()}

    def mean_score(metric, weight, ids):
        scores = [metric(by_weight[weight][q]["retrieved_doc_ids"],
                         by_weight[weight][q]["relevant_doc_ids"], 10) for q in ids]
        scores = [s for s in scores if s is not None]
        return sum(scores) / len(scores) if scores else 0.0

    chosen, picked = [], {}
    for fold in range(folds):
        train = [q for q in order if fold_of[q] != fold]
        best = max(weights, key=lambda w: (mean_score(recall_at_k, w, train),
                                           mean_score(ndcg_at_k, w, train), -w))
        chosen.append(best)
        picked.update({q: by_weight[best][q] for q in order if fold_of[q] == fold})
    return [picked[q] for q in order], chosen


def parse_depths(text):
    """"50:8,100:8,20:4" -> [(candidates reranked, windows per document), ...]."""
    depths = []
    for part in text.split(","):
        if not part.strip():
            continue
        try:
            candidates, windows = (int(v) for v in part.split(":"))
        except ValueError:
            raise argparse.ArgumentTypeError("Use candidates:windows pairs, e.g. 50:8,20:4") from None
        if not (1 <= candidates <= 200 and windows >= 1):
            raise argparse.ArgumentTypeError("Candidates must be 1-200 and windows at least 1")
        depths.append((candidates, windows))
    if not depths:
        raise argparse.ArgumentTypeError("Give at least one candidates:windows pair")
    return depths


def parse_weights(text):
    weights = [float(w) for w in text.split(",") if w.strip()]
    if not weights or any(not 0 <= w <= 1 for w in weights):
        raise argparse.ArgumentTypeError("Weights must be between 0 and 1")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_retrieval_arguments(parser, weight=False)
    parser.add_argument("--questions", type=Path, default=CORPUS_DIR / "questions.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation" / "splits.json")
    parser.add_argument("--split", choices=["development", "calibration", "final"],
                        default="development")
    parser.add_argument("--bm25-weights", type=parse_weights, default=[RetrievalSettings().bm25_weight],
                        help="BM25 RRF weights to score without reranking (BGE gets 1 - w)")
    parser.add_argument("--rerankers", default="",
                        help="Comma-separated rerankers (label or label@revision); empty for none")
    parser.add_argument("--rerank-bm25-weight", default="best",
                        help='Fusion weight before reranking: a number, or "best" (highest '
                             "Recall@10 among --bm25-weights on this split; use on development only)")
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--reranker-max-length", type=int, default=512,
                        help="Tokens per reranker input (query + window)")
    parser.add_argument("--rerank-weights", type=parse_weights, default=[1.0],
                        help="Reranker share of the final order: 1 = reranker order only; "
                             "lower values blend it with the fused rank. Comma-separated sweep")
    parser.add_argument("--rerank-window-chars", type=int, default=0,
                        help="MaxP: score windows of this many characters across each whole "
                             "document and keep the best; 0 = one query-relevant slice")
    parser.add_argument("--rerank-window-stride", type=int, default=0,
                        help="Characters between window starts (default: half a window)")
    parser.add_argument("--rerank-max-windows", type=int, default=8,
                        help="Most windows scored per document (the lead plus the best matches)")
    parser.add_argument("--rerank-headers", default="off",
                        help="Start each reranker window with the document's path header: off, "
                             "on, or off,on to compare both (needs --rerank-window-chars)")
    parser.add_argument("--rerank-depths", type=parse_depths, default=None,
                        help="Compare reranker settings in one run, as candidates:windows pairs "
                             "(e.g. 50:8,100:8,20:4): how many fused candidates are reranked and "
                             "the most windows per document. Default: --rerank-k and "
                             "--rerank-max-windows. Searches and shared scores are reused")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Folds for choosing the blend weight on development")
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--write-baselines", action="store_true",
                        help="Also save BM25+ alone and BGE alone (for a new split)")
    args = parser.parse_args()
    if args.top_k < max(KS):
        parser.error(f"--top-k must be at least {max(KS)} to score Recall@{max(KS)}")
    if min(args.rerank_window_chars, args.rerank_window_stride) < 0 or args.rerank_max_windows < 1 \
            or args.cv_folds < 2:
        parser.error("Window sizes must be non-negative, and window count and folds positive")
    if args.rerank_bm25_weight != "best":
        args.rerank_bm25_weight = parse_weights(args.rerank_bm25_weight)[0]
    rerankers = [reranker_identity(s.strip()) for s in args.rerankers.split(",") if s.strip()]

    from src.retrieval.corpus_dense import method_label
    from src.retrieval.reranker import CrossEncoderReranker

    questions = load_questions(args.questions, args.splits, args.split)
    started = time.perf_counter()
    settings = settings_from_args(args, retrieval='hybrid')
    bm25, dense = load_pair(settings)
    load_seconds = time.perf_counter() - started
    dense_label = method_label(dense.manifest["model"]["name"])
    bm25_tag = "_bm25hdr" if bm25.manifest.get("headers") else ""  # BM25 indexed with path headers.
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    print(f"Indexes loaded in {load_seconds:.0f}s; searching {len(questions)} questions...", flush=True)
    retrievers = (bm25, dense)
    caches, seconds, errors = search_once(retrievers, questions, user, max(args.top_k, settings.candidate_k))
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
        save(f"bm25{bm25_tag}_full_{args.split}", bm25_rows, bm25_latency, bm25.manifest)
        dense_rows, dense_latency = run_single(caches[1], seconds, errors, questions, args.top_k)
        save(f"dense_{dense_label}_full_{args.split}", dense_rows, dense_latency, dense.manifest)
        results[f"{dense_label} alone"] = dense_rows

    fusion: dict[float, float] = {}
    for weight in args.bm25_weights:
        name = f"hybrid_{dense_label}{bm25_tag}_w{weight:g}_full_{args.split}"
        rows, latencies, manifest = run_variant(retrievers, caches, seconds, errors, questions,
                                                user, bm25_weight=weight, top_k=args.top_k,
                                                candidate_k=settings.candidate_k,
                                                rerank_k=settings.rerank_k, rrf_k=settings.rrf_k)
        save(name, rows, latencies, manifest)
        results[name], fusion[weight] = rows, recall10(rows)

    weight, choice = args.rerank_bm25_weight, "given"
    if weight == "best" and rerankers:
        if args.split != "development":
            parser.error('--rerank-bm25-weight "best" tunes on the split; pass a number here')
        weight, choice = max(fusion, key=lambda w: fusion[w]), "best development Recall@10"
    windows = args.rerank_window_chars or None
    header_modes = [mode.strip() == "on" for mode in args.rerank_headers.split(",") if mode.strip()]
    if not header_modes or any(m.strip() not in ("on", "off") for m in args.rerank_headers.split(",") if m.strip()):
        parser.error("--rerank-headers takes off, on, or off,on")
    if any(header_modes) and not windows:
        parser.error("--rerank-headers on needs --rerank-window-chars")
    depths = args.rerank_depths or [(settings.rerank_k, args.rerank_max_windows)]
    defaults = (RetrievalSettings().rerank_k, 8)
    reranked_runs, shared, timing = [], {}, {}
    for (label, name, revision), (depth, max_windows), header in (
            (r, d, h) for r in rerankers for d in depths for h in header_modes):
        if depth < args.top_k:
            parser.error(f"{depth} candidates is below --top-k {args.top_k}: the hybrid passes "
                         "at least --top-k candidates, so the setting would not differ")
        if depth > settings.candidate_k * 2:
            parser.error(f"{depth} candidates exceeds the two {settings.candidate_k}-document "
                         "searches; raise --candidate-k")
        if (label, max_windows, header) not in shared:  # One scorer and cache per window setup.
            shared[(label, max_windows, header)] = CachedReranker(CrossEncoderReranker(
                name, revision, device=args.device, batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length, fp16=args.device.startswith("cuda"),
                window_chars=windows, window_stride=args.rerank_window_stride or None,
                max_windows=max_windows, document_text=bm25.full_text if windows else None,
                document_header=bm25.header if header else None))
        reranker = shared[(label, max_windows, header)]
        stem = (f"hybrid_{dense_label}{bm25_tag}_w{weight:g}_{label}"
                + (f"_maxp{windows}" if windows else "") + ("_rhdr" if header else ""))
        if (depth, max_windows if windows else 8) != defaults:  # Earlier runs keep their names.
            stem += f"_k{depth}" + (f"_win{max_windows}" if windows else "")
        by_weight, latency_by_weight, manifest = {}, {}, None
        calls_before = len(reranker.windows_per_call)
        for rerank_weight in args.rerank_weights:
            run_name = stem + (f"_blend{rerank_weight:g}" if rerank_weight < 1 else "") \
                + f"_full_{args.split}"
            began = time.perf_counter()
            rows, latencies, manifest = run_variant(
                retrievers, caches, seconds, errors, questions, user, bm25_weight=weight,
                reranker=reranker, top_k=args.top_k, candidate_k=settings.candidate_k,
                rerank_k=depth, rrf_k=settings.rrf_k, rerank_weight=rerank_weight)
            save(run_name, rows, latencies, manifest,
                 fusion_weight_choice={"bm25_weight": weight, "chosen_by": choice})
            results[run_name], by_weight[rerank_weight] = rows, rows
            latency_by_weight[rerank_weight] = latencies
            reranked_runs.append(run_name)
            # The first weight includes scoring every candidate; later ones reuse the scores.
            elapsed = time.perf_counter() - began
            if stem not in timing:
                calls = reranker.windows_per_call[calls_before:]
                timing[stem] = (reranker, sum(calls) / max(1, len(calls)), depth)
            print(f"{run_name}: {elapsed:.0f}s", flush=True)
        if len(by_weight) > 1 and args.split == "development":
            rows, chosen = cross_validated(by_weight, folds=args.cv_folds)
            # The first weight's latencies include scoring the candidates; later ones reuse them.
            first = latency_by_weight[args.rerank_weights[0]]
            run_name = f"{stem}_blendcv_full_{args.split}"
            save(run_name, rows, first, {**manifest, "config": {**manifest["config"],
                                                                 "rerank_weight": "cross-validated"}},
                 fusion_weight_choice={"bm25_weight": weight, "chosen_by": choice},
                 cross_validation={"folds": args.cv_folds, "seed": 5980,
                                   "weights": sorted(by_weight), "chosen_per_fold": chosen})
            results[run_name] = rows
            reranked_runs.append(run_name)
            print(f"{run_name}: weight chosen per fold {chosen}", flush=True)

    from src.evaluation.retrieval_metrics import (evaluate_retrieval, ndcg_at_k, paired_bootstrap,
                                                  recall_at_k)

    print(f"\n{'run':58} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'nDCG@10':>8}  vs BM25+ (R@10 better/worse)")
    for name, rows in results.items():
        o = evaluate_retrieval(rows, KS)["overall"]
        better, worse = compare(rows, bm25_rows)
        print(f"{name:58} {o['recall@5']['mean']:>6.3f} {o['recall@10']['mean']:>6.3f} "
              f"{o['recall@20']['mean']:>6.3f} {o['ndcg@10']['mean']:>8.3f}  {better:>3} / {worse}")

    if timing:
        print(f"\nEstimated reranking cost per question on this device ({args.device}): windows "
              "the setting scores per question x measured seconds per window (cache-independent):")
        costs = {}
        for stem, (reranker, windows, depth) in timing.items():
            seconds = windows * reranker.seconds_per_window()
            costs[stem] = {"candidates": depth, "windows_per_question": round(windows, 1),
                           "seconds_per_question": round(seconds, 3), "device": args.device}
            print(f"{stem:58} {depth:>4} candidates, {windows:6.1f} windows, ~{seconds:.2f}s")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"reranker_costs_{args.split}.json").write_text(json.dumps(costs, indent=2) + "\n")

    base = results.get(f"hybrid_{dense_label}{bm25_tag}_w{weight:g}_full_{args.split}") if rerankers else None
    if base is not None:
        print("\nChange versus the same hybrid without a reranker (95% bootstrap intervals):")
        for name in reranked_runs:
            r, n = (paired_bootstrap(results[name], base, metric=m) for m in (recall_at_k, ndcg_at_k))
            print(f"{name:58} R@10 {r['change']:+.3f} ({r['low']:+.3f} to {r['high']:+.3f}; "
                  f"{r['better']} better, {r['worse']} worse)  nDCG@10 {n['change']:+.3f} "
                  f"({n['low']:+.3f} to {n['high']:+.3f})")


if __name__ == "__main__":
    main()

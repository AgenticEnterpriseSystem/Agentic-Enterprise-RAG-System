"""Score BM25+ / BGE hybrid variants (weighted RRF, optional cross-encoder) on one split.

Each question is searched once by each retriever (top --candidate-k documents);
every variant then runs HybridRetriever over those same saved results, so the
variants differ only in fusion weights and reranking. Use --device cuda for GPU inference.
"""

import argparse
from dataclasses import replace
import json
import re
import time
from pathlib import Path

from src.config import CORPUS_DIR, ROOT, RetrievalSettings, EMBEDDING_MODELS, runtime_versions
from src.contracts import SearchRequest, UserContext
from src.evaluation.run_retrieval import KS, build_summary, git_state, load_questions, write_run
from src.evaluation.scoped_retrieval import (RunOutput, ScopedRetrievalRunner,
    _json, add_scope_arguments, file_sha256, label_summary,
    result_row as scoped_result_row, sanitized_error, scope_filters)
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
                reranker=None, top_k=max(KS), candidate_k=None, rerank_k=None, rrf_k=None):
    """Rows, latencies and manifest for one fusion/rerank setting."""
    overrides = {name: value for name, value in (
        ('candidate_k', candidate_k), ('rerank_k', rerank_k), ('rrf_k', rrf_k))
        if value is not None}
    config = RetrievalSettings(bm25_weight=bm25_weight, **overrides).hybrid_config()
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


def parse_weights(text):
    weights = [float(w) for w in text.split(",") if w.strip()]
    if not weights or any(not 0 <= w <= 1 for w in weights):
        raise argparse.ArgumentTypeError("BM25 weights must be between 0 and 1")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_retrieval_arguments(parser, weight=False)
    add_scope_arguments(parser)
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
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--write-baselines", action="store_true",
                        help="Also save BM25+ alone and BGE alone (for a new split)")
    args = parser.parse_args()
    if not max(KS) <= args.top_k <= 100:
        parser.error(f'--top-k must be between {max(KS)} and100')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if len(args.bm25_weights) != len(set(args.bm25_weights)):
        parser.error('Choose distinct fusion weights')
    specs = [value.strip() for value in args.rerankers.split(',') if value.strip()]
    if len(specs) > 1:
        parser.error('Run optional rerankers in separate processes; at most one pinned offline reranker fits this invocation')
    if specs and (not re.fullmatch(r'[^@]+@[0-9a-fA-F]{40}', specs[0])
                  or specs[0].partition('@')[0] not in RERANKERS):
        parser.error('Optional reranker requires a supported label@40hex immutable revision; no Hub lookup is performed')
    if specs and args.rerank_bm25_weight == 'best' and args.split != 'development':
        parser.error('Best-weight selection is development-only; pass a fixed numeric weight')
    if args.reranker_batch_size < 1:
        parser.error('--reranker-batch-size must be positive')

    outputs, summaries = {}, {}
    try:
        fixed_rerank_weight = None
        if args.rerank_bm25_weight != 'best':
            values = parse_weights(args.rerank_bm25_weight)
            if len(values) != 1:
                raise ValueError('One reranker fusion weight is required')
            fixed_rerank_weight = values[0]
        inputs = {'questions_sha256': file_sha256(args.questions), 'splits_sha256': file_sha256(args.splits)}
        questions = load_questions(args.questions, args.splits, args.split)
        selected = questions[:args.limit] if args.limit is not None else questions
        if not selected:
            raise ValueError('Empty question selection')
        settings = settings_from_args(args, retrieval='hybrid')
        filters = scope_filters(args)
        dense_header = _json(settings.dense_index / 'manifest.json', limit=65536)
        model_name = dense_header['model']['name']
        dense_label = next((label for label, value in EMBEDDING_MODELS.items()
                            if value['name'] == model_name), model_name.split('/')[-1])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', dense_label):
            raise ValueError('Unsafe model label')
        variants = {f'hybrid_{dense_label}_w{weight:g}_full_{args.split}': weight for weight in args.bm25_weights}
        names = list(variants)
        if args.write_baselines:
            names += [f'bm25_full_{args.split}', f'dense_{dense_label}_full_{args.split}']
        # Preflight collisions before loading indexes, models or credentials.
        possible_rerank_names = []
        if specs:
            reranker_label = specs[0].partition('@')[0]
            possible_weights = args.bm25_weights if fixed_rerank_weight is None else [fixed_rerank_weight]
            possible_rerank_names = [f'hybrid_{dense_label}_w{weight:g}_{reranker_label}_full_{args.split}'
                                     for weight in possible_weights]
        if any((args.output_dir / name).exists() for name in [*names, *possible_rerank_names]):
            raise FileExistsError('A requested result already exists')
        for name in names:
            outputs[name] = RunOutput(args.output_dir / name)
        started = time.perf_counter()
        with ScopedRetrievalRunner.open(settings, snapshot_file=args.snapshot_file,
                credentials_file=args.credentials_file, principal_kind=args.principal_kind,
                doc_ids_file=args.allowed_doc_ids_file, filters=filters,
                read_more_chars=args.read_more_chars) as runner:
            load_seconds = time.perf_counter() - started
            depth = max(args.top_k, settings.candidate_k)

            def append_question(question, destinations, *, reranker=None):
                began, context, labels, prepared, error = time.perf_counter(), None, None, None, None
                try:
                    context = runner.context()
                    labels = runner.labels(question, context)
                    prepared = runner.prepare(context, question['question'], depth=depth)
                    prepared = replace(prepared, seconds=time.perf_counter() - began)
                except Exception as failure:
                    error = failure
                setup_seconds = time.perf_counter() - began
                if prepared is not None:
                    setup_seconds -= prepared.seconds
                for name, spec in destinations.items():
                    variant_began = time.perf_counter()
                    observation, failed = None, error
                    if failed is None:
                        try:
                            if isinstance(spec, str):
                                observation = runner.baseline(prepared, method=spec, top_k=args.top_k)
                            else:
                                config = RetrievalSettings.model_validate({**settings.model_dump(), 'bm25_weight': spec}).hybrid_config()
                                observation = runner.hybrid(prepared, config=config, top_k=args.top_k, reranker=reranker)
                            runner.current(context)
                        except Exception as failure:
                            observation, failed = None, failure
                    row_labels = labels
                    if failed is not None and context is not None:
                        try:
                            runner.current(context)
                        except Exception as failure:
                            row_labels, failed = None, failure
                    row = scoped_result_row(question, observation, labels=row_labels,
                        error=sanitized_error(failed) if failed is not None else None)
                    elapsed = ((observation.seconds + setup_seconds) if observation is not None
                               else setup_seconds + (prepared.seconds if prepared else 0)
                                    + time.perf_counter() - variant_began)
                    outputs[name].append(row, elapsed)

            destinations = dict(variants)
            if args.write_baselines:
                destinations.update({f'bm25_full_{args.split}': 'bm25',
                                     f'dense_{dense_label}_full_{args.split}': 'dense'})
            for question in selected:
                append_question(question, destinations)

            def finish(name, *, weight=None, reranker=None, choice=None):
                if inputs != {'questions_sha256': file_sha256(args.questions), 'splits_sha256': file_sha256(args.splits)}:
                    raise ValueError('Evaluation inputs changed')
                manifest = runner.manifest()
                manifest.update(code=git_state(), runtime=runtime_versions())
                manifest['inputs'] = {**inputs, 'selected_questions': len(selected),
                    'split_questions': len(questions), 'partial_evaluation': len(selected) != len(questions)}
                manifest['variant'] = {'method': name, 'bm25_weight': weight,
                    'reranker': getattr(reranker, 'identity', None), 'chosen_by': choice}
                manifest['latency_policy'] = 'Actual shared candidate time plus this variant materialization/rerank/read_more time; inclusive, not a sum across variants'
                snapshot = manifest['snapshot']
                output = outputs[name]
                summary = build_summary(output.rows, output.latencies,
                    method=name.rsplit('_full_', 1)[0], split=args.split, top_k=args.top_k,
                    index={'method': 'postgres-store-bounded', **manifest['indexes'],
                           'config': manifest['variant']}, load_seconds=load_seconds,
                    evaluation=manifest, relevance=label_summary(output.rows),
                    corpus={'repo': None, 'commit': None, 'documents': snapshot['document_count'],
                            'content_sha256': snapshot['corpus_sha256'], 'snapshot_id': snapshot['snapshot_id']})
                runner.assert_ready()
                output.finish(summary, manifest)
                summaries[name] = summary['overall']

            for name in destinations:
                finish(name, weight=variants.get(name))
            if specs:
                from src.retrieval.reranker import CrossEncoderReranker
                label, model_name, revision = reranker_identity(specs[0])
                weight, choice = fixed_rerank_weight, 'given'
                if weight is None:
                    weight = max(args.bm25_weights,
                        key=lambda value: recall10(outputs[f'hybrid_{dense_label}_w{value:g}_full_{args.split}'].rows))
                    choice = 'best development authorized-reference Recall@10'
                name = f'hybrid_{dense_label}_w{weight:g}_{label}_full_{args.split}'
                outputs[name] = RunOutput(args.output_dir / name)
                reranker = CrossEncoderReranker(model_name, revision, device=args.device,
                    batch_size=args.reranker_batch_size, fp16=args.device.startswith('cuda'))
                # A second bounded pass preserves best-development selection
                # without retaining every question's candidates or text.
                for question in selected:
                    append_question(question, {name: weight}, reranker=reranker)
                finish(name, weight=weight, reranker=reranker, choice=choice)
    except Exception:
        parser.error('Scoped hybrid evaluation failed. Check private credentials, runtime database, exact snapshot, pinned offline models and unused result paths.')
    finally:
        for output in outputs.values():
            output.close()
    print(json.dumps({'split': args.split, 'results': summaries}, indent=2))


if __name__ == "__main__":
    main()

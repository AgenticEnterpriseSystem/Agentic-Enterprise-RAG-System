"""Run a full-corpus retrieval baseline over a question split and score it."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from statistics import median

from src.config import CORPUS_DIR, ROOT, RetrievalSettings, runtime_versions
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.scoped_retrieval import (RunOutput, ScopedRetrievalRunner,
    add_scope_arguments, file_sha256, label_summary, result_row as scoped_result_row,
    sanitized_error, scope_filters)
from src.retrieval.factory import add_retrieval_arguments, settings_from_args

KS = (5, 10, 20)


def load_questions(path: Path, splits_path: Path, split: str) -> list[dict]:
    wanted = set(json.loads(splits_path.read_text())["splits"][split])
    questions = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    selected = [q for q in questions if q["question_id"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("Split references questions missing from the question file")
    return sorted(selected, key=lambda q: q["question_id"])


def load_retriever(method: str, index: Path, device: str = RetrievalSettings().device, fp16=None):
    """Legacy library helper; the evaluation command never selects this loader.

    Full-resident FAISS is retained only for deliberately supervised reference
    work. Use compare_retrieval_backends under its memory/resource supervisor.
    """
    if method == "bm25":
        from src.retrieval.corpus_bm25 import CorpusBM25

        return CorpusBM25.load(index)
    if method == "dense":
        from src.retrieval.factory import load_dense

        return load_dense(RetrievalSettings(dense_index=index, device=device, fp16=fp16))
    raise ValueError(f"Unsupported method: {method}")


def run(retriever, questions: list[dict], top_k: int) -> tuple[list[dict], list[float]]:
    rows, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        error = None
        try:
            hits = retriever.rank(q["question"], top_k)
        except Exception as exc:  # Count failures as errors instead of dropping them.
            hits, error = [], f"{type(exc).__name__}: {exc}"
        latencies.append(time.perf_counter() - start)
        rows.append(
            {
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "source_types": q["source_types"],
                "relevant_doc_ids": q["expected_doc_ids"],
                "retrieved_doc_ids": [doc_id for doc_id, _ in hits],
                "scores": [round(score, 4) for _, score in hits],
                "error": error,
            }
        )
    return rows, latencies


def slice_summaries(rows: list[dict]) -> dict:
    """Extra slices from the evaluation protocol: reference count and source type."""
    groups = {
        "single_reference": [r for r in rows if len(r["relevant_doc_ids"]) == 1],
        "multiple_references": [r for r in rows if len(r["relevant_doc_ids"]) > 1],
    }
    for source in sorted({s for r in rows for s in r["source_types"]}):
        groups[f"source:{source}"] = [r for r in rows if source in r["source_types"]]
    return {
        name: evaluate_retrieval(g, KS)["overall"] for name, g in groups.items() if g
    }


def git_state() -> dict:
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    # Dirty means the results came from uncommitted code (only code paths count).
    changes, commit = git("status", "--porcelain", "--", "src", "requirements.txt"), git("rev-parse", "HEAD")
    available = changes is not None and bool(commit)
    return {"commit": commit or None, "dirty": bool(changes) if changes is not None else None,
            "available": available}


def build_summary(rows, latencies, *, method, split, top_k, index, load_seconds, **extra):
    """Scores plus the provenance recorded with every saved run."""
    metrics = evaluate_retrieval(rows, KS)
    ordered = sorted(latencies)
    corpus = extra.pop('corpus', None)
    if corpus is None:
        original = json.loads((ROOT / "evaluation" / "corpus_manifest.json").read_text())
        corpus = {k: original[k] for k in ("repo", "commit", "documents", "content_sha256")}
    return {
        "method": method,
        "split": split,
        "questions": len(rows),
        "top_k": top_k,
        "code": git_state(),
        "runtime": runtime_versions(),
        "corpus": corpus,
        "index": index,
        **extra,
        "timing_seconds": {
            "index_load": round(load_seconds, 2),
            "query_median": round(median(ordered), 4),
            "query_p95": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "total_queries": round(sum(ordered), 2),
        },
        "overall": metrics["overall"],
        "question_types": metrics["slices"],
        "other_slices": slice_summaries(rows),
    }


def write_run(output: Path, rows: list[dict], summary: dict):
    """Write predictions, the upstream answer format, and the summary."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    # Upstream EnterpriseRAG-Bench answer format (retrieval only, top 10 documents).
    (output / "upstream_answers.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "question_id": r["question_id"],
                    "document_ids": r["retrieved_doc_ids"][:10],
                }
            )
            + "\n"
            for r in rows
        )
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_retrieval_arguments(parser)
    add_scope_arguments(parser)
    parser.add_argument("--method", choices=["bm25", "dense"], default="bm25",
                        help='Dense selects the dense adapter of the accepted paired bundle; provide its BM25 metadata index too')
    parser.add_argument(
        "--index", type=Path, help="Default: .cache/corpus/indexes/<method>"
    )
    parser.add_argument(
        "--questions", type=Path, default=CORPUS_DIR / "questions.jsonl"
    )
    parser.add_argument(
        "--splits", type=Path, default=ROOT / "evaluation" / "splits.json"
    )
    parser.add_argument(
        "--split",
        choices=["development", "calibration", "final"],
        default="development",
    )
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not max(KS) <= args.top_k <= 100:
        parser.error(f"--top-k must be between {max(KS)} and100")
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = (
        args.output
        or ROOT / "evaluation" / "results" / f"{args.method}_full_{args.split}"
    )
    destination = None
    try:
        inputs = {'questions_sha256': file_sha256(args.questions), 'splits_sha256': file_sha256(args.splits)}
        questions = load_questions(args.questions, args.splits, args.split)
        selected = questions[:args.limit] if args.limit is not None else questions
        if not selected:
            raise ValueError('Empty question selection')
        settings = settings_from_args(args, retrieval='bm25' if args.method == 'bm25' else 'hybrid')
        if args.index is not None:
            field = 'bm25_index' if args.method == 'bm25' else 'dense_index'
            settings = RetrievalSettings.model_validate({**settings.model_dump(), field: args.index})
        filters = scope_filters(args)
        destination = RunOutput(output)  # Reserve before loading models or touching a principal.
        start = time.perf_counter()
        with ScopedRetrievalRunner.open(settings, snapshot_file=args.snapshot_file,
                credentials_file=args.credentials_file, principal_kind=args.principal_kind,
                doc_ids_file=args.allowed_doc_ids_file, filters=filters,
                read_more_chars=args.read_more_chars) as runner:
            load_seconds = time.perf_counter() - start
            for question in selected:
                began, labels, context = time.perf_counter(), None, None
                try:
                    context = runner.context()
                    labels = runner.labels(question, context)
                    observation = runner.search(context, question['question'], method=args.method, top_k=args.top_k)
                    runner.current(context)
                    row = scoped_result_row(question, observation, labels=labels)
                except Exception as error:
                    if context is not None:
                        try:
                            runner.current(context)
                        except Exception as current_error:
                            labels, error = None, current_error
                    row = scoped_result_row(question, labels=labels, error=sanitized_error(error))
                destination.append(row, time.perf_counter() - began)
            manifest = runner.manifest()
            if inputs != {'questions_sha256': file_sha256(args.questions), 'splits_sha256': file_sha256(args.splits)}:
                raise ValueError('Evaluation inputs changed')
            manifest['inputs'] = {**inputs, 'selected_questions': len(selected),
                'split_questions': len(questions), 'partial_evaluation': len(selected) != len(questions)}
            manifest.update(code=git_state(), runtime=runtime_versions())
            snapshot = manifest['snapshot']
            summary = build_summary(destination.rows, destination.latencies,
                method=args.method, split=args.split, top_k=args.top_k,
                index=manifest['indexes'][args.method], load_seconds=load_seconds,
                evaluation=manifest, relevance=label_summary(destination.rows),
                corpus={'repo': None, 'commit': None, 'documents': snapshot['document_count'],
                        'content_sha256': snapshot['corpus_sha256'], 'snapshot_id': snapshot['snapshot_id']})
            runner.assert_ready()
            destination.finish(summary, manifest)
    except Exception:
        parser.error('Scoped retrieval evaluation failed. Check private credentials, runtime database, exact snapshot, offline indexes/models and unused output path.')
    finally:
        if destination is not None:
            destination.close()
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("method", "split", "questions", "timing_seconds", "overall")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

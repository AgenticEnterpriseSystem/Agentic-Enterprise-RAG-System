"""Run a full-corpus retrieval baseline over a question split and score it."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from statistics import median

from src.config import CORPUS_DIR, ROOT, RetrievalSettings, runtime_versions
from src.evaluation.retrieval_metrics import evaluate_retrieval

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
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True
        ).stdout.strip()

    # Dirty means the results came from uncommitted code (only code paths count).
    changes = git("status", "--porcelain", "--", "src", "requirements.txt")
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(changes)}


def build_summary(rows, latencies, *, method, split, top_k, index, load_seconds, **extra):
    """Scores plus the provenance recorded with every saved run."""
    metrics = evaluate_retrieval(rows, KS)
    ordered = sorted(latencies)
    corpus = json.loads((ROOT / "evaluation" / "corpus_manifest.json").read_text())
    return {
        "method": method,
        "split": split,
        "questions": len(rows),
        "top_k": top_k,
        "code": git_state(),
        "runtime": runtime_versions(),
        "corpus": {
            k: corpus[k] for k in ("repo", "commit", "documents", "content_sha256")
        },
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
    output.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    parser.add_argument(
        "--index", type=Path, help="Default: .cache/corpus/indexes/<method>"
    )
    parser.add_argument("--device", default=RetrievalSettings().device, help="Query encoder device (dense)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
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
    if args.top_k < max(KS):
        parser.error(f"--top-k must be at least {max(KS)} to score Recall@{max(KS)}")
    output = (
        args.output
        or ROOT / "evaluation" / "results" / f"{args.method}_full_{args.split}"
    )
    questions = load_questions(args.questions, args.splits, args.split)
    start = time.perf_counter()
    defaults = RetrievalSettings()
    index = args.index or (defaults.bm25_index if args.method == "bm25" else defaults.dense_index)
    retriever = load_retriever(args.method, index, args.device, args.fp16)
    load_seconds = time.perf_counter() - start
    rows, latencies = run(retriever, questions, args.top_k)
    summary = build_summary(rows, latencies, method=args.method, split=args.split,
                            top_k=args.top_k, index=retriever.manifest,
                            load_seconds=load_seconds)
    write_run(output, rows, summary)
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

r"""Resumable, additive V3-scoring/V6-audit evaluator candidate. No Git actions.

From project root:
  .\.venv\Scripts\python.exe -m scripts.run_evaluator_candidate --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from src.evaluation.evaluator_candidate import evaluate_case
from src.evaluation.verified_judge import make_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "alex_evaluation"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(x, dict) or not isinstance(x.get("question_id"), str) for x in data):
        raise ValueError(f"{label} must be a JSON list of records with question_id: {path}")
    ids = [x["question_id"] for x in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {label}")
    return data


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def _manifest(args, selected: list[dict]) -> dict:
    files = {
        "answers": args.answers,
        "questions": args.questions,
        "score_prompt": args.score_prompt,
        "candidate_code": ROOT / "src/evaluation/evaluator_candidate.py",
        "verified_code": ROOT / "src/evaluation/verified_judge.py",
        "v6_audit_code": ROOT / "src/evaluation/verified_judge_v6.py",
        "runner_code": Path(__file__).resolve(),
    }
    if args.audit_mode == "all":
        files["audit_prompt"] = args.audit_prompt
    return {
        "evaluator": "v3-scoring-v6-review-candidate-1",
        "model": args.model,
        "audit_mode": args.audit_mode,
        "selected_ids": [a["question_id"] for a in selected],
        "sha256": {name: _hash(path) for name, path in files.items()},
        "note": "Resume is allowed only with identical selected inputs, prompts, model, and implementation.",
    }


def main(argv=None, provider=None) -> int:
    p = argparse.ArgumentParser(description="V3 baseline scorer + optional independent V6 review audit")
    p.add_argument("--answers", type=Path, default=DATA / "bm25_gptoss_answers_20.json")
    p.add_argument("--questions", type=Path, default=DATA / "questions_20.json")
    p.add_argument("--output", type=Path, default=DATA / "evaluator_candidate_live.json")
    p.add_argument("--score-prompt", type=Path, default=ROOT / "docs/evaluation_prompts/answer_grading_v3.md")
    p.add_argument("--audit-prompt", type=Path, default=ROOT / "docs/evaluation_prompts/citation_audit_v5.md")
    p.add_argument("--audit-mode", choices=("all", "none"), default="all")
    p.add_argument("--model", default="gpt-oss:20b-cloud")
    p.add_argument("--limit", type=int)
    p.add_argument("--question-id", action="append")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout-seconds", type=float, default=120)
    p.add_argument("--max-attempts", type=int, default=2)
    args = p.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    if args.max_attempts < 1 or args.timeout_seconds <= 0:
        p.error("--max-attempts and --timeout-seconds must be positive")
    if args.question_id and args.limit is not None:
        p.error("Choose --question-id or --limit, not both")
    inputs = {args.answers.resolve(), args.questions.resolve(), args.score_prompt.resolve()}
    if args.audit_mode == "all":
        inputs.add(args.audit_prompt.resolve())
    if args.output.resolve() in inputs:
        p.error("Output cannot overwrite an input")

    answers = _load(args.answers, "answers")
    questions = _load(args.questions, "questions")
    question_map = {q["question_id"]: q for q in questions}
    if args.question_id:
        wanted = set(args.question_id)
        unknown = wanted - {a["question_id"] for a in answers}
        if unknown:
            p.error("Unknown question IDs: " + ", ".join(sorted(unknown)))
        selected = [a for a in answers if a["question_id"] in wanted]
    else:
        selected = answers[:args.limit]
    if not selected:
        p.error("No cases selected")
    cases = {}
    for answer in selected:
        qid = answer["question_id"]
        if qid not in question_map:
            p.error(f"Missing reference question: {qid}")
        cases[qid] = make_case(answer, question_map[qid])
    score_prompt = args.score_prompt.read_text(encoding="utf-8")
    audit_prompt = args.audit_prompt.read_text(encoding="utf-8") if args.audit_mode == "all" else None
    if not score_prompt.strip() or (args.audit_mode == "all" and not audit_prompt.strip()):
        p.error("Selected prompts must not be empty")
    manifest = _manifest(args, selected)
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if args.dry_run:
        print(f"DRY RUN OK: {len(selected)} cases; model={args.model}; audit={args.audit_mode}; "
              f"{2 if audit_prompt else 1} calls/case without retries")
        print(f"Output: {args.output} (not written)")
        return 0

    if args.output.exists() and not args.resume:
        p.error("Output exists; use --resume for the identical experiment or choose a new output")
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            p.error("Unsafe resume: model, inputs, prompts, selected IDs, or code changed")
        saved = _load(args.output, "saved results") if args.output.exists() else []
        allowed = set(cases)
        if set(x["question_id"] for x in saved) - allowed:
            p.error("Saved results include out-of-scope IDs")
        results = {r["question_id"]: r for r in saved}
        if not args.output.exists():
            _atomic_json(args.output, [])  # Interrupted between manifest and first record.
    else:
        if args.resume and args.output.exists() and not manifest_path.exists():
            p.error("Unsafe resume: saved output has no manifest")
        if manifest_path.exists():
            p.error("Manifest already exists; choose another output or resume matching run")
        _atomic_json(manifest_path, manifest)
        _atomic_json(args.output, [])  # Always resumable, even before the first model call.
        results = {}

    if provider is None:
        from ollama import Client
        from src.generation.ollama_provider import OllamaProvider
        provider = OllamaProvider(model=args.model)
        provider.client = Client(host="http://localhost:11434", timeout=args.timeout_seconds)

    for i, answer in enumerate(selected, 1):
        qid = answer["question_id"]
        old = results.get(qid)
        done = (isinstance(old, dict) and old.get("scores") is not None
                and old.get("error") is None and (audit_prompt is None
                or (isinstance(old.get("audit"), dict) and not old.get("audit_error"))))
        if done:
            print(f"[{i}/{len(selected)}] {qid}: already complete", flush=True)
            continue
        start = time.perf_counter()
        result = evaluate_case(cases[qid], provider, score_prompt, audit_prompt,
                               max_attempts=args.max_attempts, existing=old)
        result["elapsed_seconds_last_attempt"] = round(time.perf_counter() - start, 2)
        results[qid] = result
        _atomic_json(args.output, [results[a["question_id"]] for a in selected if a["question_id"] in results])
        state = ("score FAILED" if result["scores"] is None else
                 "audit FAILED; score saved" if audit_prompt and result["audit"] is None else
                 "scored")
        print(f"[{i}/{len(selected)}] {qid}: {state}; review={result['review_priority']}; "
              f"elapsed={result['elapsed_seconds_last_attempt']}s", flush=True)

    incomplete = sum(1 for qid in cases if (
        qid not in results or results[qid].get("scores") is None or
        (audit_prompt is not None and results[qid].get("audit") is None)))
    flagged = sum(1 for r in results.values() if r.get("needs_review"))
    print(f"Saved {len(results)}/{len(selected)}; incomplete={incomplete}; review flags={flagged}; "
          f"output={args.output}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Evaluate saved RAG answers with an Ollama-compatible LLM judge."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.evaluation.answer_metrics import (
    AnswerEvaluationInput,
    EvidenceItem,
    evaluate_answer_with_usage,
    load_judge_prompt,
)
from src.generation.ollama_provider import OllamaProvider
from src.generation.generator import ProviderError


def load_records(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_results(path, results):
    """Write progress atomically to reduce corruption risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def index_by_id(records, label):
    indexed = {}
    for item in records:
        qid = item["question_id"]
        if qid in indexed:
            raise ValueError(f"Duplicate {label} question ID: {qid}")
        indexed[qid] = item
    return indexed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers."
    )
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama judge model; defaults to provider configuration.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing evaluation file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to the grading prompt Markdown file.",
    )
    args = parser.parse_args()
    prompt_template = (
        load_judge_prompt(args.prompt_file)
        if args.prompt_file
        else None
    )
    if args.output.resolve() in {
        args.answers.resolve(),
        args.questions.resolve(),
    }:
        parser.error("Output must differ from input files.")

    answers = load_records(args.answers)
    questions = load_records(args.questions)

    question_by_id = index_by_id(questions, "question")
    index_by_id(answers, "answer")

    selected = answers[:args.limit] if args.limit else answers
    if not selected:
        parser.error("No answers selected.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    missing = [
        a["question_id"]
        for a in selected
        if a["question_id"] not in question_by_id
    ]
    if missing:
        parser.error(
            f"{len(missing)} answers have no reference question: "
            + ", ".join(missing[:5])
        )

    if args.output.exists() and not args.resume:
        parser.error(
            f"Output already exists: {args.output}. "
            "Use --resume only for the same experiment."
        )

    if args.resume and args.output.exists():
        results = load_records(args.output)
        index_by_id(results, "saved evaluation")

        selected_ids = {a["question_id"] for a in selected}
        extra_ids = {
            r["question_id"] for r in results
        } - selected_ids
        if extra_ids:
            parser.error(
                "Existing results contain IDs outside this run: "
                + ", ".join(sorted(extra_ids)[:5])
            )
    else:
        results = []

    # Only successful scores or generation errors are considered done.
    results = [
        r for r in results
        if r.get("scores") is not None
        or r.get("generation_status") == "error"
    ]
    completed = {r["question_id"] for r in results}

    # Confirm the provider constructor supports a model override.
    provider = (
        OllamaProvider(model=args.model)
        if args.model
        else OllamaProvider()
    )

    for index, answer in enumerate(selected, start=1):
        qid = answer["question_id"]

        if qid in completed:
            print(f"[{index}/{len(selected)}] {qid}: already scored")
            continue

        if answer["status"] == "error":
            result = {
                "question_id": qid,
                "generation_status": "error",
                "scores": None,
                "usage": None,
                "error": "Answer generation failed.",
            }
        else:
            case = AnswerEvaluationInput(
                question_id=qid,
                question=answer["question"],
                generated_answer=answer["generated_answer"],
                reference_answer=question_by_id[qid]["gold_answer"],
                evidence=[
                    EvidenceItem(
                        citation_id=e["citation_id"],
                        text=e["text"],
                    )
                    for e in answer["evidence"]
                ],
                citation_ids=answer["citation_ids"],
            )

            try:
                evaluation = evaluate_answer_with_usage(
                    case,
                    provider,
                    prompt_template=prompt_template,
                )
                result = {
                    "question_id": qid,
                    "generation_status": answer["status"],
                    "scores": asdict(evaluation.scores),
                    "usage": evaluation.usage.model_dump(),
                    "error": None,
                }
            except (
                ValueError, KeyError, TypeError, ProviderError
            ) as exc:
                result = {
                    "question_id": qid,
                    "generation_status": answer["status"],
                    "scores": None,
                    "usage": None,
                    "error": str(exc),
                }

        # Replace rather than duplicate an existing failed record.
        results = [
            r for r in results if r["question_id"] != qid
        ]
        results.append(result)
        save_results(args.output, results)

        print(
            f"[{index}/{len(selected)}] {qid}: "
            + ("scored" if result["scores"] is not None
               else "not scored")
        )

    print(f"Saved {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()

"""Evaluate the 20 BM25+ / GPT-OSS benchmark results."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.answer_metrics import (
    AnswerEvaluationInput,
    EvidenceItem,
    evaluate_answer_with_usage,
)
from src.generation.ollama_provider import OllamaProvider
from src.generation.generator import ProviderError


FOLDER = Path("outputs/alex_evaluation")

ANSWERS_PATH = FOLDER / "bm25_gptoss_answers_20.json"
QUESTIONS_PATH = FOLDER / "questions_20.json"
OUTPUT_PATH = FOLDER / "bm25_gptoss_evaluation_20.json"


def save_results(results):
    """Save progress after each question."""
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    answers = json.loads(
        ANSWERS_PATH.read_text(encoding="utf-8")
    )

    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    if not 1 <= args.limit <= len(answers):
        parser.error(
            f"--limit must be between 1 and {len(answers)}"
        )

    references = {
        item["question_id"]: item["gold_answer"]
        for item in questions
    }

    # Resume previously completed evaluations if the file exists.
    if OUTPUT_PATH.exists():
        results = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    else:
        results = []

    # Keep successful evaluations and intentional generation-error records.
    # Remove failed judge evaluations so they can be retried.
    results = [
        item for item in results
        if item["scores"] is not None
        or item["generation_status"] == "error"
    ]

    completed_ids = {
        item["question_id"]
        for item in results
    }

    provider = OllamaProvider()

    for index, answer in enumerate(answers[:args.limit], start=1):
        question_id = answer["question_id"]

        if question_id in completed_ids:
            print(f"[{index}/{args.limit}] {question_id} - already saved")
            continue

        # A generation error is not a valid answer to judge.
        if answer["status"] == "error":
            result = {
                "question_id": question_id,
                "generation_status": "error",
                "scores": None,
                "usage": None,
                "error": "Answer generation failed.",
            }

        else:
            case = AnswerEvaluationInput(
                question_id=question_id,
                question=answer["question"],
                generated_answer=answer["generated_answer"],
                reference_answer=references[question_id],
                evidence=[
                    EvidenceItem(
                        citation_id=item["citation_id"],
                        text=item["text"],
                    )
                    for item in answer["evidence"]
                ],
                citation_ids=answer["citation_ids"],
            )

            try:
                evaluation = evaluate_answer_with_usage(
                    case,
                    provider,
                )

                result = {
                    "question_id": question_id,
                    "generation_status": answer["status"],
                    "scores": asdict(evaluation.scores),
                    "usage": evaluation.usage.model_dump(),
                    "error": None,
                }

            except (ValueError, KeyError, TypeError, ProviderError) as exc:
                result = {
                    "question_id": question_id,
                    "generation_status": answer["status"],
                    "scores": None,
                    "usage": None,
                    "error": str(exc),
                }

        results.append(result)
        completed_ids.add(question_id)

        save_results(results)

        status = (
            "scored"
            if result["scores"] is not None
            else "not scored"
        )

        print(
            f"[{index}/{args.limit}] "
            f"{question_id} - {status}"
        )

    print(f"\nSaved {len(results)} evaluation records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
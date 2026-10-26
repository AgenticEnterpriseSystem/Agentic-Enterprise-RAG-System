"""Generate RAG benchmark answers using GPT-OSS Cloud."""

import argparse
import json
from pathlib import Path

from src.contracts import Chunk, Evidence, GenerationRequest, UserContext
from src.generation.generator import GroundedGenerator, select_context, GenerationConfig
from src.generation.ollama_provider import OllamaProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=1,
        help="Maximum number of questions to generate (default: 1)"
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Generate only the specified question",
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")

    folder = Path("outputs/alex_evaluation")
    batch = json.loads(
        (folder / "bm25_cases_20.json").read_text(encoding="utf-8")
    )

    generator = GroundedGenerator(OllamaProvider())
    results = []

    selected_cases = (
    [case for case in batch if case["question_id"] == args.question_id]
    if args.question_id
    else batch[:args.limit]
    )

    if not selected_cases:
        parser.error(f"Question not found: {args.question_id}")

    for case in selected_cases:
        evidence = []

        for index, item in enumerate(case["evidence"]):
            content = item["text"]

            chunk = Chunk(
                chunk_id=f"benchmark-{index}",
                doc_id=item["doc_id"],
                source_type=item["source_type"],
                title=item["title"],
                text=content,
                start_char=0,
                end_char=len(content),
            )

            evidence.append(
                Evidence(
                    citation_id=item["citation_id"],
                    chunk=chunk,
                    score=1.0,
                    retrieval_method="bm25",
                )
            )

        user = UserContext(
            user_id="benchmark-evaluator",
            allowed_doc_ids=frozenset(item.chunk.doc_id for item in evidence),
        )

        request = GenerationRequest(
            question=case["question"],
            user=user,
            evidence=tuple(evidence),
        )

        # Record the exact passages sent to the model.
        selected = select_context(
            request, GenerationConfig().max_context_chars
        )

        answer, usage = generator.generate(request)

        results.append({
            "question_id": case["question_id"],
            "question_type": case["question_type"],
            "question": case["question"],
            "generated_answer": answer.text,
            "status": answer.status,
            "citation_ids": list(answer.citation_ids),
            "evidence": [
                {
                    "citation_id": item.citation_id,
                    "doc_id": item.chunk.doc_id,
                    "source_type": item.chunk.source_type,
                    "title": item.chunk.title,
                    "text": item.chunk.text,
                }
                for item in selected
            ],
            "usage": usage.model_dump(),
        })

        print(
            f"[{len(results)}/{args.limit}] "
            f"{case['question_id']} — {answer.status}"
        )

    if args.question_id:
        output = folder / f"bm25_gptoss_retry_{args.question_id}.json"
    else:
        output = folder / f"bm25_gptoss_answers_{args.limit}.json"

    with output.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved {len(results)} answers to {output}")


if __name__ == "__main__":
    main()
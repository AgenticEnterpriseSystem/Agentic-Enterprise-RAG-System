"""Answer-quality evaluation using the team's 0–2 rubric."""
import json
from dataclasses import dataclass
from src.contracts import Usage
from pathlib import Path

@dataclass
class AnswerScores:
    correctness: int
    completeness: int | None
    citation_support: int | None
    abstention_limits: int
    reasons: dict[str, str] | None = None

    def __post_init__(self):
        """Check that every score follows the evaluation rubric."""

        for name in (
        "correctness",
        "completeness",
        "citation_support",
        "abstention_limits",
        ):
            score = getattr(self, name)
            if score is None:
                if name in ("completeness", "citation_support"):
                    continue
                raise ValueError(f"{name} cannot be None")

            if type(score) is not int or score not in (0, 1, 2):
                raise ValueError(
                    f"{name} must be 0, 1, or 2"
                )

@dataclass
class EvidenceItem:
    citation_id: str
    text: str

@dataclass
class AnswerEvaluationInput:
    """Information needed to evaluate one generated answer."""

    question_id: str
    question: str
    generated_answer: str
    reference_answer: str
    evidence: list[EvidenceItem]
    citation_ids: list[str] | None = None

def parse_judge_scores(judge_response: dict) -> AnswerScores:
    """Convert a judge's response into validated scores and explanations."""

    reasons = judge_response.get("reasons")

    required_dimensions = (
        "correctness",
        "completeness",
        "citation_support",
        "abstention_limits",
    )

    if not isinstance(reasons, dict):
        raise ValueError("reasons must be a dictionary")

    for dimension in required_dimensions:
        if not isinstance(reasons.get(dimension), str):
            raise ValueError(f"reasons missing explanation for {dimension}")

        if not reasons[dimension].strip():
            raise ValueError(f"reasons cannot be empty for {dimension}")

    return AnswerScores(
        correctness=judge_response["correctness"],
        completeness=judge_response["completeness"],
        citation_support=judge_response["citation_support"],
        abstention_limits=judge_response["abstention_limits"],
        reasons=reasons,
    )

JUDGE_PROMPT = """
You are an independent evaluator of a question-answering system.

Evaluate the generated answer using:
1. The original question
2. The reference answer
3. The supporting evidence

Score each dimension from 0 to 2.

CORRECTNESS:
0 = Central claim is wrong.
1 = Main answer is right, but a substantive detail is wrong or unverifiable.
2 = All substantive claims are correct.

COMPLETENESS:
0 = Main information is missing.
1 = Some required details are missing.
2 = All required facts, constraints, and conflicts are covered.

CITATION SUPPORT:
0 = Central claims are unsupported.
1 = Some claims are unsupported or ambiguously cited.
2 = Every factual claim is supported by its citation.

ABSTENTION/LIMITS:
0 = Unsupported answer or unjustified refusal.
1 = Uncertainty is noted, but a key limitation is missed.
2 = Answers when justified; explains missing or conflicting evidence otherwise.

Return only a valid JSON object with these fields:
- correctness: integer 0-2
- completeness: integer 0-2 or null
- citation_support: integer 0-2 or null
- abstention_limits: integer 0-2
- reasons: an object containing a brief explanation for each score

Do not include Markdown formatting around the JSON.
Use null for completeness when there are no reference answer facts,
and for citation_support when there are no factual claims.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Generated answer citation IDs:
{citation_ids}

Supporting evidence:
{evidence}
"""

def evaluate_answer(case: AnswerEvaluationInput, provider) -> AnswerScores:
    """Evaluate an answer and return its validated scores."""
    result = evaluate_answer_with_usage(case, provider)
    return result.scores

def validate_citations(case: AnswerEvaluationInput) -> bool:
    """Check that all cited IDs exist in the supplied evidence."""

    if case.citation_ids is None:
        return False

    available_ids = {item.citation_id for item in case.evidence}

    return all(
        citation_id in available_ids
        for citation_id in case.citation_ids
    )

def evaluate_answer_safely(
    case: AnswerEvaluationInput,
    provider,
) -> tuple[AnswerScores | None, str | None]:
    """Evaluate an answer and report invalid judge responses."""

    try:
        scores = evaluate_answer(case, provider)
        return scores, None

    except (ValueError, KeyError, TypeError) as error:
        return None, f"Invalid judge response: {error}"

def evaluate_batch(cases, provider) -> list[dict]:
    """Evaluate multiple answers and retain judge usage information."""

    results = []

    for case in cases:
        try:
            evaluation = evaluate_answer_with_usage(case, provider)

            results.append({
                "question_id": case.question_id,
                "scores": evaluation.scores,
                "usage": evaluation.usage,
                "error": None,
            })

        except (ValueError, KeyError, TypeError) as exc:
            results.append({
                "question_id": case.question_id,
                "scores": None,
                "usage": None,
                "error": f"Invalid judge response: {exc}",
            })

    return results

@dataclass
class JudgeResult:
    """Store evaluation scores and the judge's API usage."""

    scores: AnswerScores
    usage: Usage

def evaluate_answer_with_usage(
    case: AnswerEvaluationInput,
    provider,
) -> JudgeResult:
    """Evaluate an answer and record the judge's token usage."""

    prompt = JUDGE_PROMPT.format(
        question=case.question,
        reference_answer=case.reference_answer,
        generated_answer=case.generated_answer,
        citation_ids=json.dumps(case.citation_ids),
        evidence=json.dumps([
            {
                "citation_id": item.citation_id,
                "text": item.text,
            }
            for item in case.evidence
        ]),
    )

    completion = provider.complete(
        system="You are an independent answer-quality evaluator.",
        prompt=prompt,
    )

    if completion.truncated:
        raise ValueError("Judge response was truncated")

    try:
        judge_response = json.loads(completion.text)

    except json.JSONDecodeError:
        debug_folder = Path("outputs/alex_evaluation/judge_debug")
        debug_folder.mkdir(parents=True, exist_ok=True)

        debug_path = debug_folder / f"{case.question_id}_raw.txt"
        debug_path.write_text(
            completion.text,
            encoding="utf-8",
        )

        print(f"Invalid judge JSON saved to: {debug_path}")
        raise

    scores = parse_judge_scores(judge_response)

    return JudgeResult(
        scores=scores,
        usage=completion.usage,
    )

def load_evaluation_cases(path: str) -> list[AnswerEvaluationInput]:
    """Load answer-evaluation cases from a JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    cases = []

    for item in data:
        evidence = [
            EvidenceItem(
                citation_id=doc["citation_id"],
                text=doc["text"],
            )
            for doc in item["evidence"]
        ]

        case = AnswerEvaluationInput(
            question_id=item["question_id"],
            question=item["question"],
            generated_answer=item["generated_answer"],
            reference_answer=item["reference_answer"],
            evidence=evidence,
            citation_ids=item["citation_ids"],
        )

        cases.append(case)

    return cases

from dataclasses import asdict


def save_evaluation_results(results: list[dict], path: str) -> None:
    """Save evaluation scores, usage, and errors to JSON."""

    output = []

    for result in results:
        scores = result["scores"]
        usage = result.get("usage")

        output.append({
            "question_id": result["question_id"],
            "scores": asdict(scores) if scores is not None else None,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "api_calls": usage.api_calls,
                "estimated_cost_usd": usage.estimated_cost_usd,
            } if usage is not None else None,
            "error": result["error"],
        })

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    Path(path).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
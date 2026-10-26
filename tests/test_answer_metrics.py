import pytest
from src.evaluation.answer_metrics import EvidenceItem
from src.evaluation.answer_metrics import AnswerScores


def test_valid_scores():
    scores = AnswerScores(
        correctness=2,
        completeness=1,
        citation_support=2,
        abstention_limits=2,
    )

    assert scores.correctness == 2
    assert scores.completeness == 1


def test_invalid_score():
    with pytest.raises(ValueError):
        AnswerScores(
            correctness=5,
            completeness=1,
            citation_support=2,
            abstention_limits=2,
        )


def test_not_applicable_scores():
    scores = AnswerScores(
        correctness=2,
        completeness=None,
        citation_support=None,
        abstention_limits=2,
    )

    assert scores.completeness is None
    assert scores.citation_support is None

def test_evaluation_input():
    from src.evaluation.answer_metrics import AnswerEvaluationInput

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Engineering update: Integration problems delayed Atlas.",
            )
        ],
    )

    assert case.question_id == "demo-1"
    assert case.generated_answer == "Integration issues."
    assert len(case.evidence) == 1

def test_evaluate_answer_with_fake_provider():
    import json

    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        evaluate_answer,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 1,
                    "citation_support": 2,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "The main claim is correct.",
                        "completeness": "The hardware delay is missing.",
                        "citation_support": "The cited evidence supports the claim.",
                        "abstention_limits": "The answer stays within the evidence.",
                    },
                }),
                usage=Usage(),
            )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Engineering update: Integration problems delayed Atlas.",
            )
        ],
    )

    result = evaluate_answer(case, FakeProvider())

    assert result.correctness == 2
    assert result.completeness == 1
    assert result.citation_support == 2
    assert result.abstention_limits == 2

def test_judge_explanations():
    from src.evaluation.answer_metrics import parse_judge_scores

    judge_response = {
        "correctness": 2,
        "completeness": 1,
        "citation_support": 2,
        "abstention_limits": 2,
        "reasons": {
            "correctness": "The answer is factually correct.",
            "completeness": "The answer omits the hardware delay.",
            "citation_support": "The cited evidence supports the claims.",
            "abstention_limits": "The answer does not make unsupported claims.",
        },
    }

    result = parse_judge_scores(judge_response)

    assert result.reasons is not None
    assert result.reasons["completeness"] == (
        "The answer omits the hardware delay."
    )

def test_invalid_judge_response():
    import pytest
    from src.evaluation.answer_metrics import parse_judge_scores

    judge_response = {
        "correctness": 3,
        "completeness": 2,
        "citation_support": 2,
        "abstention_limits": 2,
    }

    with pytest.raises(ValueError):
        parse_judge_scores(judge_response)

def test_validate_citations():
    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        validate_citations,
    )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues caused the delay.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Integration issues delayed Atlas.",
            )
        ],
        citation_ids=["doc-101"],
    )

    # The cited document exists.
    assert validate_citations(case) is True

    # The cited document does not exist.
    case.citation_ids = ["doc-999"]
    assert validate_citations(case) is False

    # Citation information is missing.
    case.citation_ids = None
    assert validate_citations(case) is False

    # An empty citation list contains no invalid IDs.
    case.citation_ids = []
    assert validate_citations(case) is True

def test_judge_receives_citation_ids():
    import json

    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        evaluate_answer,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            # Check what the evaluator sends to the LLM.
            assert '"doc-101"' in prompt
            assert '"doc-999"' in prompt
            assert "Integration issues delayed Atlas." in prompt

            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 1,
                    "citation_support": 0,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "The main claim is accurate.",
                        "completeness": "The hardware delay is missing.",
                        "citation_support": "The answer cites doc-999, which was not supplied.",
                        "abstention_limits": "The answer does not overstate the available evidence.",
                    },
                }),
                usage=Usage(),
            )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues caused the delay.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Integration issues delayed Atlas.",
            )
        ],
        citation_ids=["doc-999"],
    )

    result = evaluate_answer(case, FakeProvider())

    assert result.citation_support == 0

def test_malformed_judge_response():
    import pytest

    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        evaluate_answer,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text="This is not valid JSON.",
                usage=Usage(),
            )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues caused the delay.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Integration issues delayed Atlas.",
            )
        ],
        citation_ids=["doc-101"],
    )

    with pytest.raises(ValueError):
        evaluate_answer(case, FakeProvider())

def test_safe_evaluation_handles_invalid_json():
    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        evaluate_answer_safely,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text="This is not valid JSON.",
                usage=Usage(),
            )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues caused the delay.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Integration issues delayed Atlas.",
            )
        ],
        citation_ids=["doc-101"],
    )

    scores, error = evaluate_answer_safely(case, FakeProvider())

    assert scores is None
    assert error is not None
    assert "Invalid judge response" in error

def test_evaluate_batch_continues_after_failure():
    import json

    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        evaluate_batch,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system, prompt):
            self.calls += 1

            if self.calls == 2:
                return Completion(
                    text="Invalid JSON",
                    usage=Usage(),
                )

            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 1,
                    "citation_support": 2,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "The main claim is accurate.",
                        "completeness": "The hardware delay is missing.",
                        "citation_support": "The cited evidence supports the claim.",
                        "abstention_limits": "The answer stays within the evidence.",
                    },
                }),
                usage=Usage(),
            )

    cases = [
        AnswerEvaluationInput(
            question_id=f"demo-{i}",
            question="Why was Atlas delayed?",
            generated_answer="Integration issues.",
            reference_answer="Integration issues and late hardware.",
            evidence=[
                EvidenceItem(
                    citation_id="doc-101",
                    text="Integration issues delayed Atlas.",
                )
            ],
            citation_ids=["doc-101"],
        )
        for i in range(1, 4)
    ]

    provider = FakeProvider()
    results = evaluate_batch(cases, provider)

    assert len(results) == 3
    assert provider.calls == 3

    assert results[0]["scores"].correctness == 2
    assert results[0]["error"] is None

    assert results[1]["scores"] is None
    assert results[1]["error"] is not None

    assert results[2]["scores"].correctness == 2
    assert results[2]["error"] is None

def test_judge_requires_explanations():
    import pytest
    from src.evaluation.answer_metrics import parse_judge_scores

    judge_response = {
        "correctness": 2,
        "completeness": 1,
        "citation_support": 2,
        "abstention_limits": 2,
    }

    with pytest.raises(ValueError, match="reasons"):
        parse_judge_scores(judge_response)

def test_evaluate_answer_tracks_usage():
    import json

    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        evaluate_answer_with_usage,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 1,
                    "citation_support": 2,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "The main claim is accurate.",
                        "completeness": "A required detail is missing.",
                        "citation_support": "The cited evidence supports the claim.",
                        "abstention_limits": "The answer stays within the evidence.",
                    },
                }),
                usage=Usage(
                    input_tokens=1200,
                    output_tokens=150,
                    api_calls=2,
                    estimated_cost_usd=0.004,
                ),
            )

    case = AnswerEvaluationInput(
        question_id="demo-1",
        question="Why was Atlas delayed?",
        generated_answer="Integration issues.",
        reference_answer="Integration issues and late hardware.",
        evidence=[
            EvidenceItem(
                citation_id="doc-101",
                text="Integration issues delayed Atlas.",
            )
        ],
        citation_ids=["doc-101"],
    )

    result = evaluate_answer_with_usage(case, FakeProvider())

    assert result.scores.correctness == 2
    assert result.scores.completeness == 1
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 150
    assert result.usage.api_calls == 2
    assert result.usage.estimated_cost_usd == 0.004

def test_load_evaluation_cases():
    from src.evaluation.answer_metrics import load_evaluation_cases

    cases = load_evaluation_cases(
        "tests/fixtures/answer_evaluation_cases.json"
    )

    assert len(cases) == 2

    assert cases[0].question_id == "demo-1"
    assert cases[0].citation_ids == ["doc-101"]
    assert cases[0].evidence[0].citation_id == "doc-101"

    assert cases[1].question_id == "demo-2"
    assert cases[1].reference_answer == (
        "Project Orion was completed in April."
    )

def test_evaluate_batch_from_json():
    import json

    from src.evaluation.answer_metrics import (
        load_evaluation_cases,
        evaluate_batch,
    )
    from src.generation.generator import Completion
    from src.contracts import Usage

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 1,
                    "citation_support": 2,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "The answer is accurate.",
                        "completeness": "A detail is missing.",
                        "citation_support": "The citation supports the claim.",
                        "abstention_limits": "The answer stays within the evidence.",
                    },
                }),
                usage=Usage(),
            )

    cases = load_evaluation_cases(
        "tests/fixtures/answer_evaluation_cases.json"
    )

    results = evaluate_batch(cases, FakeProvider())

    assert len(results) == 2
    assert results[0]["question_id"] == "demo-1"
    assert results[1]["question_id"] == "demo-2"
    assert all(result["error"] is None for result in results)

def test_save_evaluation_results(tmp_path):
    import json

    from src.evaluation.answer_metrics import (
        AnswerScores,
        save_evaluation_results,
    )

    results = [
        {
            "question_id": "demo-1",
            "scores": AnswerScores(
                correctness=2,
                completeness=1,
                citation_support=2,
                abstention_limits=2,
                reasons={
                    "correctness": "The answer is accurate.",
                    "completeness": "A detail is missing.",
                    "citation_support": "The cited evidence supports the claim.",
                    "abstention_limits": "The answer stays within the evidence.",
                },
            ),
            "error": None,
        }
    ]

    output_path = tmp_path / "evaluation_results.json"

    save_evaluation_results(results, str(output_path))

    assert output_path.exists()

    saved_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(saved_data) == 1
    assert saved_data[0]["question_id"] == "demo-1"
    assert saved_data[0]["scores"]["correctness"] == 2
    assert saved_data[0]["scores"]["completeness"] == 1
    assert saved_data[0]["error"] is None

def test_save_evaluation_results_with_usage(tmp_path):
    import json

    from src.contracts import Usage
    from src.evaluation.answer_metrics import (
        AnswerScores,
        save_evaluation_results,
    )

    results = [
        {
            "question_id": "demo-1",
            "scores": AnswerScores(
                correctness=2,
                completeness=1,
                citation_support=2,
                abstention_limits=2,
            ),
            "usage": Usage(
                input_tokens=150,
                output_tokens=50,
                api_calls=1,
                estimated_cost_usd=0.002,
            ),
            "error": None,
        }
    ]

    output_path = tmp_path / "results.json"

    save_evaluation_results(results, str(output_path))

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved[0]["usage"]["input_tokens"] == 150
    assert saved[0]["usage"]["output_tokens"] == 50
    assert saved[0]["usage"]["api_calls"] == 1
    assert saved[0]["usage"]["estimated_cost_usd"] == 0.002

def test_real_benchmark_generation_integration():
    import json
    from pathlib import Path

    from src.contracts import Chunk, Evidence, GenerationRequest, Usage, UserContext
    from src.generation.generator import Completion, GroundedGenerator

    root = Path(__file__).resolve().parents[1]
    benchmark = root / ".cache" / "corpus" / "EnterpriseRAG-Bench"

    question_path = benchmark / "questions.jsonl"
    source_path = (
        benchmark
        / "generated_data"
        / "sources"
        / "github"
        / "pr-18421-multipart-file-validation-limits.json"
    )

    # Local benchmark data is optional for normal unit-test runs.
    if not question_path.exists() or not source_path.exists():
        import pytest
        pytest.skip("Local EnterpriseRAG-Bench data is unavailable")

    with question_path.open(encoding="utf-8") as file:
        question = next(
            json.loads(line)
            for line in file
            if json.loads(line)["question_id"] == "qst_0001"
        )

    document = json.loads(source_path.read_text(encoding="utf-8"))
    doc_id = document["dataset_doc_uuid"]

    # Match the corpus extraction logic.
    title, content = __import__(
        "src.data.corpus", fromlist=["extract_document"]
    ).extract_document(document)

    chunk = Chunk(
        chunk_id=f"{doc_id}:full",
        doc_id=doc_id,
        source_type="github",
        title=title,
        text=content,
        start_char=0,
        end_char=len(content),
    )

    evidence = Evidence(
        citation_id="source-1",
        chunk=chunk,
        score=97.9064,
        retrieval_method="bm25",
    )

    request = GenerationRequest(
        question=question["question"],
        user=UserContext(
            user_id="benchmark-test",
            allowed_doc_ids=frozenset({doc_id}),
        ),
        evidence=(evidence,),
    )

    class FakeProvider:
        def complete(self, system, prompt):
            return Completion(
                text=json.dumps(
                    {
                        "status": "answered",
                        "text": (
                            "The default limits are 10 MiB per file "
                            "and 50 MiB per request."
                        ),
                        "citation_ids": [selected[0].citation_id],
                    }
                ),
                usage=Usage(api_calls=0, estimated_cost_usd=0),
            )

    from src.generation.generator import select_context, GenerationConfig

    selected = select_context(request, GenerationConfig().max_context_chars)

    answer, usage = GroundedGenerator(FakeProvider()).generate(request)
    from src.evaluation.answer_metrics import (
        AnswerEvaluationInput,
        EvidenceItem,
        validate_citations,
    )

    evaluation_case = AnswerEvaluationInput(
        question_id=question["question_id"],
        question=question["question"],
        generated_answer=answer.text,
        reference_answer=question["gold_answer"],
        evidence=[
            EvidenceItem(
                citation_id=item.citation_id,
                text=item.chunk.text,
            )
            for item in selected
        ],
        citation_ids=list(answer.citation_ids),
    )

    assert validate_citations(evaluation_case)
    assert evaluation_case.reference_answer == question["gold_answer"]
    assert evaluation_case.generated_answer == answer.text

    from src.evaluation.answer_metrics import evaluate_answer

    class FakeJudge:
        def complete(self, system, prompt):
            return Completion(
                text=json.dumps({
                    "correctness": 2,
                    "completeness": 2,
                    "citation_support": 2,
                    "abstention_limits": 2,
                    "reasons": {
                        "correctness": "Both size limits are correct.",
                        "completeness": "Both required facts are included.",
                        "citation_support": "The cited document supports both limits.",
                        "abstention_limits": "The available evidence supports answering."
                    }
                }),
                usage=Usage(api_calls=0, estimated_cost_usd=0),
            )

    scores = evaluate_answer(evaluation_case, FakeJudge())

    assert scores.correctness == 2
    assert scores.completeness == 2
    assert scores.citation_support == 2
    assert scores.abstention_limits == 2

    assert answer.status == "answered"
    assert answer.citation_ids == (selected[0].citation_id,)
    assert "10 MiB" in answer.text
    assert "50 MiB" in answer.text
    assert usage.estimated_cost_usd == 0

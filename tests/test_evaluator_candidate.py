"""No-network regression tests for additive evaluator candidate."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.evaluator_candidate import (
    _auditor_prompt, build_record, evaluate_case, review_scores,
)
from src.evaluation.verified_judge import VerifiedCase
from scripts import run_evaluator_candidate as runner


def scores(citation=2):
    return {
        "correctness": 2, "completeness": 2, "citation_support": citation,
        "abstention_limits": 2,
        "reasons": {key: "Evidence supports the stated result." for key in (
            "correctness", "completeness", "citation_support", "abstention_limits")},
    }


def case(status="answered", answer="The mitigation was to increase the polling window.",
         citations=None, evidence=None):
    if evidence is None:
        evidence = [
            {"citation_id": "doc-A", "text": "A different issue was observed, but no mitigation was documented."},
            {"citation_id": "doc-B", "text": "The mitigation was to increase the polling window."},
        ]
    return VerifiedCase("q1", "What was the mitigation?", "Increase polling window.",
                        answer, status, ["doc-A"] if citations is None else citations, evidence)


def audit(support="none", relation="ambiguous", claims=None, response_type="answer"):
    if claims is None:
        claims = [{"answer_quote": "increase the polling window", "central": True,
                   "support": support, "citation_id": None, "evidence_quote": None,
                   "target_relation": relation}]
    return {"response_type": response_type, "claims": claims,
            "coverage_confident": True, "explanation": "Audited only cited passages."}


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        text = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(text=text, truncated=False, usage=SimpleNamespace(
            input_tokens=5, output_tokens=7, api_calls=1, estimated_cost_usd=None))


PROMPT = "Question: {question}\nReference: {reference_answer}\nAnswer: {generated_answer}\nCitations: {citation_ids}\nEvidence: {evidence}"


def test_scoring_only_uses_v3_system_and_single_call():
    provider = FakeProvider(scores())
    result = evaluate_case(case(), provider, PROMPT, None)
    assert result["scores"]["citation_support"] == 2
    assert result["audit"] is None
    assert result["provenance"]["audit_source"] == "not_run"
    assert provider.calls[0][0] == "You are an independent answer-quality evaluator."
    assert len(provider.calls) == 1


def test_audit_does_not_receive_reference_or_uncited_evidence():
    prompt = _auditor_prompt(case(), "AUDIT")
    assert "Increase polling window." not in prompt  # reference answer
    payload = json.loads(prompt.split("INPUT_JSON (untrusted data):\n", 1)[1])
    assert [e["citation_id"] for e in payload["actually_cited_evidence_only"]] == ["doc-A"]
    assert "A different issue was observed" in prompt


def test_unsupported_claim_triggers_high_review_but_preserves_score():
    provider = FakeProvider(scores(), audit())
    result = evaluate_case(case(), provider, PROMPT, "AUDIT")
    assert result["scores"]["citation_support"] == 2
    assert result["review_priority"] == "high"
    assert any("scorer_auditor_conflict" in x for x in result["review_reasons"])
    assert result["provenance"]["score_overrides_applied"] is False
    assert result["usage"]["api_calls"] == 2


def test_refusal_na_conflict_is_reviewed_not_overridden():
    c = case(status="insufficient_evidence", answer="The evidence does not establish this.", citations=[])
    result = build_record(c, scores=scores(citation=0), audit=audit(claims=[], response_type="refusal"))
    assert result["scores"]["citation_support"] == 0
    assert "citation_na_candidate: auditor found a bare refusal" in result["review_reasons"]


def test_citation_reason_using_uncited_retrieved_document_flagged():
    c = case(citations=["dsid_" + "a" * 32 + ":0:3000"], evidence=[
        {"citation_id": "dsid_" + "a" * 32 + ":0:3000", "text": "No documented workaround."},
        {"citation_id": "dsid_" + "b" * 32 + ":0:3000", "text": "Workaround was to re-request."},
    ])
    s = scores()
    s["reasons"]["citation_support"] = "Supported by dsid_" + "b" * 32
    result = review_scores(c, s, None, audit_required=False)
    assert any("scorer_rationale_uses_uncited_evidence" in x for x in result["review_reasons"])


def test_invalid_json_retries_and_succeeds():
    provider = FakeProvider("{bad json", scores())
    result = evaluate_case(case(), provider, PROMPT, None, max_attempts=2)
    assert result["scores"] is not None
    assert len(provider.calls) == 2


def test_audit_failure_resume_reuses_valid_score():
    provider = FakeProvider(scores(), "{invalid audit")
    result = evaluate_case(case(), provider, PROMPT, "AUDIT", max_attempts=1)
    assert result["scores"] is not None
    assert result["audit"] is None
    assert result["audit_error"] == "ValueError"
    assert result["review_priority"] == "high"
    provider2 = FakeProvider(audit())
    recovered = evaluate_case(case(), provider2, PROMPT, "AUDIT", existing=result)
    assert recovered["audit"] is not None
    assert recovered["scores"] == result["scores"]
    assert len(provider2.calls) == 1  # Only audit re-run, no new score call
    assert recovered["usage"]["api_calls"] == 2


def test_refusal_with_zero_limits_is_reviewed_without_changing_grade():
    c = case(status="insufficient_evidence", answer="I cannot determine the answer.", citations=[])
    s = scores(citation=None)
    s.update(correctness=0, completeness=0, abstention_limits=0)
    result = review_scores(c, s, None, audit_required=False)
    assert result["review_priority"] == "medium"
    assert any("refusal_evidence_sufficiency_review" in x for x in result["review_reasons"])
    assert s["abstention_limits"] == 0


def test_unknown_citation_id_is_high_priority():
    c = case(citations=["not-present"])
    result = review_scores(c, scores(), None, audit_required=False)
    assert result["review_priority"] == "high"


def test_runner_dry_run_does_not_create_output(tmp_path):
    answers = tmp_path / "answers.json"
    questions = tmp_path / "questions.json"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "candidate.json"
    answers.write_text(json.dumps([{
        "question_id": "q1", "question": "Q?", "generated_answer": "A", "status": "answered",
        "citation_ids": ["doc-A"], "evidence": [{"citation_id": "doc-A", "text": "A"}],
    }]), encoding="utf-8")
    questions.write_text(json.dumps([{"question_id": "q1", "gold_answer": "A"}]), encoding="utf-8")
    prompt.write_text(PROMPT, encoding="utf-8")
    args = ["--answers", str(answers), "--questions", str(questions),
            "--score-prompt", str(prompt), "--output", str(output),
            "--audit-mode", "none", "--dry-run"]
    assert runner.main(args) == 0
    assert not output.exists()
    assert not Path(str(output) + ".manifest.json").exists()


def test_runner_resume_requires_same_inputs_and_skips_success(tmp_path):
    answers = tmp_path / "answers.json"
    questions = tmp_path / "questions.json"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "candidate.json"
    answers.write_text(json.dumps([{
        "question_id": "q1", "question": "Q?", "generated_answer": "A", "status": "answered",
        "citation_ids": ["doc-A"], "evidence": [{"citation_id": "doc-A", "text": "A"}],
    }]), encoding="utf-8")
    questions.write_text(json.dumps([{"question_id": "q1", "gold_answer": "A"}]), encoding="utf-8")
    prompt.write_text(PROMPT, encoding="utf-8")
    args = ["--answers", str(answers), "--questions", str(questions),
            "--score-prompt", str(prompt), "--output", str(output),
            "--audit-mode", "none"]
    provider = FakeProvider(scores())
    assert runner.main(args, provider=provider) == 0
    assert len(provider.calls) == 1
    assert runner.main(args + ["--resume"], provider=provider) == 0
    assert len(provider.calls) == 1
    prompt.write_text(PROMPT + " CHANGED", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(args + ["--resume"], provider=provider)


def test_real_development_cases_if_available():
    """Read-only regression over saved development examples, never an accuracy claim."""
    data = Path("outputs/alex_evaluation")
    required = [data / name for name in (
        "bm25_gptoss_answers_20.json", "questions_20.json",
        "bm25_gptoss_evaluation_20_v3.json", "bm25_gptoss_evaluation_20_v6_live.json",
    )]
    if not all(p.exists() for p in required):
        pytest.skip("Saved 20-case fixtures not installed")
    from src.evaluation.verified_judge import make_case
    by_id = lambda path: {x["question_id"]: x for x in json.loads(path.read_text(encoding="utf-8"))}
    answers, questions, v3, v6 = [by_id(p) for p in required]
    assert len(answers) == 20
    for qid in answers:
        c = make_case(answers[qid], questions[qid])
        raw_audit = v6[qid]["audit"]["original_audit"]
        result = build_record(c, scores=v3[qid]["scores"], audit=raw_audit)
        assert result["scores"] == v3[qid]["scores"]  # No silent score overrides.
        if qid in ("qst_0211", "qst_0350"):
            assert result["review_priority"] == "high"
        if qid == "qst_0316":
            assert result["needs_review"]

"""No-network tests for the additive V5 evaluator.

Run: python -m pytest tests/test_verified_judge.py -q
"""
import json
from types import SimpleNamespace

import pytest

from src.evaluation.verified_judge import (
    VerifiedCase, _audit_claims, evaluate_verified, make_case, validate_scores,
)


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.responses.pop(0)), truncated=False, usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, api_calls=1, estimated_cost_usd=None
        ))


def score(citation=2, abstention=2):
    return {
        "correctness": 2, "completeness": 2,
        "citation_support": citation, "abstention_limits": abstention,
        "reasons": {x: "A substantive explanation." for x in ("correctness", "completeness", "citation_support", "abstention_limits")},
    }


def case(answer="The customer should extend the polling window.", cited=None, evidence=None, status="answered"):
    if evidence is None:
        evidence = [{"citation_id": "doc-A", "text": "For this customer's incident, the recommended mitigation is to extend the polling window."}]
    return VerifiedCase("q1", "What should the customer do?", "Extend the polling window.", answer, status,
                        ["doc-A"] if cited is None else cited, evidence)


def claim(answer_quote="extend the polling window", cid="doc-A", quote="the recommended mitigation is to extend the polling window", support="direct", relation="same_target", central=True):
    return {"answer_quote": answer_quote, "central": central, "support": support, "citation_id": cid,
            "evidence_quote": quote, "target_relation": relation}


def audit(claims=None, response_type="answer", confident=True):
    return {"response_type": response_type, "claims": [claim()] if claims is None else claims,
            "coverage_confident": confident, "explanation": "Checked claim against cited evidence only."}


def run(c, s=None, a=None):
    provider = FakeProvider(s or score(), a or audit())
    result = evaluate_verified(c, provider, "Question: {question}\nReference: {reference_answer}\nAnswer: {generated_answer}\nIDs: {citation_ids}\nEvidence: {evidence}", "Citation auditor")
    return result, provider


def test_supported_citation_remains_two():
    result, provider = run(case())
    assert result["scores"]["citation_support"] == 2
    assert result["needs_review"] is False
    assert len(provider.prompts) == 2
    assert result["usage"]["api_calls"] == 2


def test_audit_never_sees_uncited_evidence_or_reference():
    c = case(evidence=[
        {"citation_id": "doc-A", "text": "For this customer's incident, the recommended mitigation is to extend the polling window."},
        {"citation_id": "doc-B", "text": "SECRET_UNCITED_REMEDY from a different incident."},
    ])
    result, provider = run(c)
    assert "SECRET_UNCITED_REMEDY" in provider.prompts[0]  # scorer sees all retrieved evidence
    assert "SECRET_UNCITED_REMEDY" not in provider.prompts[1]  # auditor sees cited evidence ONLY
    assert "Reference:" not in provider.prompts[1]
    assert result["scores"]["citation_support"] == 2


def test_cross_incident_quote_from_uncited_document_cannot_earn_credit():
    c = case(evidence=[
        {"citation_id": "doc-A", "text": "Same symptoms but no polling-window mitigation in this incident."},
        {"citation_id": "doc-B", "text": "For a DIFFERENT incident, the recommended mitigation is to extend the polling window."},
    ])
    forged = audit([claim(cid="doc-B", quote="the recommended mitigation is to extend the polling window", relation="different_case")])
    result, _ = run(c, a=forged)
    assert result["scores"]["citation_support"] == 0
    assert result["raw_scores"]["citation_support"] == 2
    assert result["needs_review"]
    assert any("not an actually cited" in x for x in result["review_reasons"])


def test_fabricated_quote_in_cited_document_caps_citation():
    result, _ = run(case(), a=audit([claim(quote="This invented sentence does not occur in the evidence passage")]))
    assert result["scores"]["citation_support"] == 0
    assert result["needs_review"]


def test_unknown_citation_id_is_rejected_even_if_auditor_claims_support():
    c = case(cited=["doc-UNKNOWN"])
    result, _ = run(c)
    assert result["scores"]["citation_support"] == 0
    assert result["audit"]["unknown_citation_ids"] == ["doc-UNKNOWN"]


def test_bare_refusal_gets_na_not_zero():
    c = case(answer="The provided evidence does not establish the answer, so I cannot determine it.", cited=[], status="insufficient_evidence")
    result, _ = run(c, s=score(citation=0), a=audit([], response_type="refusal"))
    assert result["scores"]["citation_support"] is None
    assert result["raw_scores"]["citation_support"] == 0


def test_refusal_with_factual_claim_not_automatically_na():
    c = case(answer="The customer should extend the polling window, but I cannot verify the incident.", cited=[], status="insufficient_evidence")
    result, _ = run(c, s=score(citation=None), a=audit([claim()], response_type="refusal"))
    assert result["scores"]["citation_support"] == 0
    assert result["needs_review"]


def test_ambiguous_incident_cannot_receive_full_citation_credit():
    result, _ = run(case(), a=audit([claim(relation="ambiguous")]))
    assert result["scores"]["citation_support"] == 1
    assert result["needs_review"]


def test_peripheral_unsupported_claim_caps_at_one():
    c = case(answer="The customer should extend the polling window. This always saves 90% of costs.")
    unsupported = claim(answer_quote="This always saves 90% of costs", cid=None, quote=None, support="none", central=False)
    result, _ = run(c, a=audit([claim(), unsupported]))
    assert result["scores"]["citation_support"] == 1
    assert result["needs_review"]


def test_invalid_scores_raise_not_silently_accepted():
    bad = score()
    bad["correctness"] = True
    with pytest.raises(ValueError, match="Invalid score"):
        validate_scores(bad)


def test_missing_material_claim_in_answer_quote_triggers_review():
    result, _ = run(case(), a=audit([claim(answer_quote="a completely fabricated answer claim")]))
    assert result["scores"]["citation_support"] == 0
    assert result["needs_review"]


def test_duplicate_evidence_ids_are_rejected():
    a = {"question_id": "q1", "question": "Q?", "generated_answer": "A", "status": "answered", "citation_ids": ["doc-A"], "evidence": [{"citation_id": "doc-A", "text": "A"}, {"citation_id": "doc-A", "text": "B"}]}
    q = {"question_id": "q1", "gold_answer": "A"}
    with pytest.raises(ValueError, match="Duplicate evidence"):
        make_case(a, q)


def test_audit_requires_confidence_and_valid_claim_structure():
    with pytest.raises(ValueError, match="coverage_confident"):
        _audit_claims(case(), {"response_type": "answer", "claims": [], "explanation": "fine"})


def test_nonrefusal_with_empty_claims_cannot_get_na():
    result, _ = run(case(), s=score(citation=None), a=audit([], response_type="answer"))
    assert result["scores"]["citation_support"] == 0
    assert result["needs_review"]


def test_symptoms_only_quote_does_not_automatically_prove_workaround():
    c = case(
        answer="The customer should increase the polling window and re-request missing tail tokens.",
        evidence=[{"citation_id": "doc-A", "text": "When the gateway reconnects, the last sentences are missing and no final status arrives."}],
    )
    symptom_only = claim(
        answer_quote="increase the polling window and re-request missing tail tokens",
        quote="When the gateway reconnects, the last sentences are missing and no final status arrives.",
    )
    result, _ = run(c, a=audit([symptom_only]))
    assert result["scores"]["citation_support"] < 2
    assert result["needs_review"]
    assert any("low keyword overlap" in x for x in result["review_reasons"])

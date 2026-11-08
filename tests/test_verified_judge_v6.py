"""Offline tests for V6; no Ollama calls. Run: pytest tests/test_verified_judge_v6.py -q"""
import json
from types import SimpleNamespace

from src.evaluation.verified_judge import VerifiedCase
from src.evaluation.verified_judge_v6 import (
    quote_match, review_audit, review_saved_record, evaluate_verified_v6,
)


def case(answer="Use the known-good configuration.", cited=None, evidence=None, status="answered"):
    return VerifiedCase("q1", "How?", "Use the known-good configuration.", answer, status,
                        ["doc-A"] if cited is None else cited,
                        evidence or [{"citation_id": "doc-A", "text": "Use the known-good configuration."}])


def claim(a="Use the known-good configuration.", cid="doc-A", e="Use the known-good configuration.",
          support="direct", relation="same_target"):
    return {"answer_quote": a, "central": True, "support": support, "citation_id": cid,
            "evidence_quote": e, "target_relation": relation}


def audit(claims):
    return {"response_type": "answer", "coverage_confident": True, "explanation": "Checked", "claims": claims}


def scores(citation=2):
    return {"correctness": 2, "completeness": 2, "citation_support": citation,
            "abstention_limits": 2, "reasons": {k: "reason" for k in
            ("correctness", "completeness", "citation_support", "abstention_limits")}}


def record(audit_value=None, citation=2):
    return {"question_id": "q1", "scores": {**scores(citation), "citation_support": 0},
            "raw_scores": scores(citation), "audit": audit_value or audit([claim()]), "error": None}


def test_unicode_format_only_difference_is_not_false_citation_error():
    quote = "Extraction: Temp=0.0 + strict JSON + canonicalization rules: price normalization correctness ~96% (previously ~89% without canonicalization)."
    evidence = "* Extraction:\n  * Temp=0.0 + strict JSON + canonicalization rules: price normalization correctness ~96% (previously ~89% without canonicalization)."
    assert quote_match(quote, evidence) == "normalized"


def test_short_verbatim_quote_is_not_arbitrarily_rejected():
    assert quote_match("quantization switch", "Runtime hooks include quantization switch and KV hints.") == "literal"


def test_duplicate_claims_are_flagged_not_crashed():
    c = case()
    result = review_audit(c, audit([claim(), claim()]))
    assert result["audit_claims_checked"] == 1
    assert any("duplicate_claim" in x for x in result["review_reasons"])


def test_claim_misquotes_evidence_but_scores_unchanged():
    c = case()
    bad = claim(e="Completely fabricated supporting sentence.")
    r = review_saved_record(c, record(audit([bad])))
    assert r["scores"]["citation_support"] == 2
    assert r["original_v5_capped_scores"]["citation_support"] == 0
    assert r["needs_review"]
    assert any("evidence_quote_mismatch" in x for x in r["review_reasons"])


def test_claim_from_different_incident_is_flagged_not_auto_downgraded():
    c = case()
    different = claim(support="related", relation="different_case")
    r = review_saved_record(c, record(audit([different])))
    assert r["scores"]["citation_support"] == 2
    assert r["needs_review"] and r["review_priority"] == "high"


def test_wrong_passage_when_quote_in_other_cited_passage():
    c = case(cited=["doc-A", "doc-B"], evidence=[
        {"citation_id": "doc-A", "text": "No remedy in this paragraph."},
        {"citation_id": "doc-B", "text": "Use the known-good configuration."},
    ])
    r = review_audit(c, audit([claim(cid="doc-A")]))
    assert r["claim_checks"][0]["auditor_evidence_quote_match"] == "other_cited"
    assert any("wrong_passage" in x for x in r["review_reasons"])


def test_bare_refusal_with_meta_claim_is_na_candidate_not_unsupported():
    c = case(answer="The provided evidence does not contain the requested value.", cited=[], status="insufficient_evidence")
    a = audit([claim(a=c.generated_answer, cid=None, e=None, support="none", relation="ambiguous")])
    a["response_type"] = "refusal"
    r = review_saved_record(c, record(a, citation=0))
    assert r["audit"]["v6_checks"]["evidence_limitation_candidate"] is True
    assert any("citation_na_candidate" in x for x in r["review_reasons"])
    assert r["scores"]["citation_support"] == 0  # no silent change
    assert not any("semantic_review_claim" in x for x in r["review_reasons"])


def test_baseline_source_is_explicit_and_not_v5_derived():
    c = case()
    baseline = {"question_id": "q1", "scores": scores(citation=1)}
    r = review_saved_record(c, record(), baseline, "saved_v3")
    assert r["scores"]["citation_support"] == 1
    assert r["raw_scores"]["citation_support"] == 2
    assert r["score_source"] == "saved_v3"
    assert any("score_disagreement" in x for x in r["review_reasons"])


def test_failed_case_with_baseline_retains_score_and_flags_missing_audit():
    c = case()
    failed = {"question_id": "q1", "scores": None, "error": "ValueError"}
    baseline = {"question_id": "q1", "scores": scores()}
    r = review_saved_record(c, failed, baseline, "saved_v3")
    assert r["scores"]["citation_support"] == 2
    assert r["needs_review"] and r["review_priority"] == "high"
    assert r["audit_source"] == "missing"


def test_unknown_answer_citation_id_is_hard_review():
    c = case(cited=["doc-UNKNOWN"])
    r = review_saved_record(c, record())
    assert r["needs_review"] and r["review_priority"] == "high"
    assert any("invalid_answer_citation_ids" in x for x in r["review_reasons"])


class FakeProvider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return SimpleNamespace(text=json.dumps(self.responses.pop(0)), truncated=False,
                               usage=SimpleNamespace(input_tokens=3, output_tokens=5, api_calls=1, estimated_cost_usd=None))


def test_live_v6_uses_original_v3_system_and_cited_only_audit():
    c = case(evidence=[
        {"citation_id": "doc-A", "text": "Use the known-good configuration."},
        {"citation_id": "doc-B", "text": "SECRET_UNCITED"},
    ])
    provider = FakeProvider([scores(), audit([claim()])])
    result = evaluate_verified_v6(c, provider,
        "Q: {question}; Ref: {reference_answer}; A: {generated_answer}; C: {citation_ids}; E: {evidence}", "Audit")
    assert len(provider.calls) == 2
    assert provider.calls[0][0] == "You are an independent answer-quality evaluator."
    assert "SECRET_UNCITED" in provider.calls[0][1]
    assert "SECRET_UNCITED" not in provider.calls[1][1]
    assert result["scores"]["citation_support"] == 2


def test_live_duplicate_claims_do_not_fail_entire_case():
    c = case()
    provider = FakeProvider([scores(), audit([claim(), claim()])])
    result = evaluate_verified_v6(c, provider,
        "Q: {question}; Ref: {reference_answer}; A: {generated_answer}; C: {citation_ids}; E: {evidence}", "Audit")
    assert result["error"] is None
    assert result["scores"]["citation_support"] == 2
    assert result["needs_review"]
    assert any("duplicate_claim" in x for x in result["review_reasons"])


def test_live_different_case_remains_review_flag_not_score_cap():
    c = case()
    provider = FakeProvider([scores(), audit([claim(support="none", relation="different_case", cid=None, e=None)])])
    result = evaluate_verified_v6(c, provider,
        "Q: {question}; Ref: {reference_answer}; A: {generated_answer}; C: {citation_ids}; E: {evidence}", "Audit")
    assert result["scores"]["citation_support"] == 2
    assert result["needs_review"]
    assert result["review_priority"] == "high"

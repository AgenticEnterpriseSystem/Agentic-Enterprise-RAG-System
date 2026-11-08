"""V6: review-gated citation audit; NO automatic score downgrades.

This module is additive. It never modifies V1-V5 output or Git state.
Mechanical quote matching checks audit quality, not semantic entailment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from difflib import SequenceMatcher
import json
import re
import unicodedata
from typing import Any

from src.evaluation.verified_judge import (
    VerifiedCase, SYSTEM, _json_response, _sum_usage, _usage, validate_scores, _claim_overlap,
)

DIMS = ("correctness", "completeness", "citation_support", "abstention_limits")
SCORE_SYSTEM_V3 = "You are an independent answer-quality evaluator."


def canonical(text: str) -> str:
    """Comparison ONLY. Not proof of quotation or semantic support."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212\u00ad]", "-", text)
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def quote_match(quote: str | None, passage: str) -> str:
    """Return literal, normalized, near, or missing; never return 'supported'."""
    if not quote or not quote.strip() or not passage:
        return "missing"
    if quote in passage:
        return "literal"
    needle, haystack = canonical(quote), canonical(passage)
    if needle and needle in haystack:
        return "normalized"
    # Approximate wording is an auditor-format warning, NOT citation verification.
    words, hay = needle.split(), haystack.split()
    if len(words) >= 5 and hay:
        m = SequenceMatcher(None, words, hay, autojunk=False).find_longest_match(0, len(words), 0, len(hay))
        if m.size / len(words) >= 0.8:
            return "near"
    return "missing"


def _looks_like_evidence_limitation(case: VerifiedCase, audit: dict) -> bool:
    """Candidate N/A; never a definitive semantic determination."""
    if case.status != "insufficient_evidence" or case.citation_ids not in ([], None):
        return False
    if audit.get("response_type") not in ("refusal", "partial_answer"):
        return False
    answer = case.generated_answer.casefold()
    markers = ("evidence", "documents", "document", "supplied", "provided", "cannot determine", "cannot provide", "not possible")
    return any(s in answer for s in markers)


def review_audit(case: VerifiedCase, audit: dict | None) -> dict:
    """Validate the *auditor* and flag semantic concerns without changing grades."""
    if not isinstance(audit, dict):
        return {"claim_checks": [], "review_reasons": ["audit_missing: no usable claim audit"],
                "review_priority": "high", "evidence_limitation_candidate": False,
                "unknown_answer_citation_ids": [], "audit_claims_checked": 0}
    cited_ids = case.citation_ids or []
    evidence_by_id = {e["citation_id"]: e["text"] for e in case.evidence}
    valid_cited = {cid: evidence_by_id[cid] for cid in cited_ids if cid in evidence_by_id}
    unknown = sorted(set(cited_ids) - set(evidence_by_id))
    reasons = []
    high = False
    if unknown:
        reasons.append("invalid_answer_citation_ids: " + ", ".join(unknown))
        high = True
    if case.citation_ids is None:
        reasons.append("missing_answer_citation_list: citation_ids is null")
    if audit.get("coverage_confident") is not True:
        reasons.append("audit_coverage_uncertain: auditor did not confirm full claim coverage")
    claims = audit.get("claims")
    if not isinstance(claims, list):
        reasons.append("audit_invalid: claims must be a list")
        claims = []
    if len(claims) > 30:
        reasons.append("audit_invalid: more than 30 claims; review all")
        claims = claims[:30]
    limitation = _looks_like_evidence_limitation(case, audit)
    checks = []
    seen = set()
    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            reasons.append(f"audit_claim_{index}_invalid: claim must be an object")
            continue
        answer_quote = claim.get("answer_quote") if isinstance(claim.get("answer_quote"), str) else ""
        evidence_quote = claim.get("evidence_quote") if isinstance(claim.get("evidence_quote"), str) else ""
        cid = claim.get("citation_id") if isinstance(claim.get("citation_id"), str) else None
        support = claim.get("support")
        relation = claim.get("target_relation")
        central = claim.get("central") is True
        key = canonical(answer_quote)
        if key and key in seen:
            reasons.append(f"audit_duplicate_claim_{index}: duplicate claim ignored (not a grading failure)")
            continue
        if key:
            seen.add(key)
        answer_match = quote_match(answer_quote, case.generated_answer)
        selected_match = quote_match(evidence_quote, valid_cited.get(cid, "")) if cid else "missing"
        other_cited = next((other for other, txt in valid_cited.items()
                            if other != cid and quote_match(evidence_quote, txt) in ("literal", "normalized")), None)
        if selected_match != "missing":
            evidence_match = selected_match
            located_citation_id = cid
        elif other_cited:
            evidence_match = "other_cited"
            located_citation_id = other_cited
        else:
            evidence_match = "missing"
            located_citation_id = None
        meta = limitation and cid is None and support == "none"
        checks.append({
            "claim_number": index, "central": central, "support_assessment_by_llm": support,
            "target_relation_by_llm": relation, "auditor_answer_quote_match": answer_match,
            "auditor_evidence_quote_match": evidence_match, "auditor_selected_citation_id": cid,
            "evidence_quote_found_in_cited_id": located_citation_id,
            "evidence_limitation_meta_claim": meta,
        })
        if answer_match == "missing":
            reasons.append(f"audit_claim_{index}_answer_quote_mismatch: auditor quote absent from answer")
        elif answer_match == "near":
            reasons.append(f"audit_claim_{index}_answer_quote_paraphrase: auditor did not quote exactly")
        if not meta:
            if (central and support in ("direct", "entailed") and evidence_quote
                    and evidence_match != "missing" and _claim_overlap(answer_quote, evidence_quote) < 0.35):
                reasons.append(f"semantic_lexical_warning_claim_{index}: low word overlap is a review heuristic, NOT proof of missing support")
            if support in ("none", "related") or relation in ("different_case", "ambiguous"):
                reasons.append(f"semantic_review_claim_{index}: auditor says support={support}, target={relation}; human judgment required")
                high = True
            if cid is None:
                reasons.append(f"audit_claim_{index}_no_cited_passage: auditor could not associate claim with a cited passage")
            elif cid not in valid_cited:
                reasons.append(f"audit_claim_{index}_invalid_passage_id: auditor selected an uncited or unavailable ID")
            if support in ("direct", "entailed"):
                if evidence_match == "missing":
                    reasons.append(f"audit_claim_{index}_evidence_quote_mismatch: claimed support quote not located in cited passages")
                elif evidence_match == "other_cited":
                    reasons.append(f"audit_claim_{index}_wrong_passage: quote found in another cited passage")
                elif evidence_match == "near":
                    reasons.append(f"audit_claim_{index}_evidence_quote_paraphrase: near match only; not verbatim")
    if limitation:
        # A refusal's statements about missing evidence do not require citations
        # just to establish that the model is refusing. Their *accuracy* is
        # still judged under correctness/abstention, not mechanically proven.
        reasons = [r for r in reasons if not r.startswith("semantic_review_claim_")]
    if not checks and audit.get("response_type") != "refusal":
        reasons.append("audit_no_claims: non-refusal answer has no audited claims")
    if not valid_cited and case.status == "answered":
        reasons.append("answered_without_valid_citations: answer contains no valid cited passages")
        high = True
    if audit.get("response_type") not in ("answer", "partial_answer", "refusal"):
        reasons.append("audit_invalid: response_type is missing or unknown")
    # De-duplicate repeated diagnostics without hiding repeated distinct claims.
    reasons = list(dict.fromkeys(reasons))
    return {"claim_checks": checks, "review_reasons": reasons,
            "review_priority": "high" if high else ("medium" if reasons else "none"),
            "evidence_limitation_candidate": limitation,
            "unknown_answer_citation_ids": unknown,
            "audit_claims_checked": len(checks)}


def review_saved_record(case: VerifiedCase, record: dict, baseline_record: dict | None = None,
                        baseline_name: str | None = None) -> dict:
    """Reprocess saved V5 audit without model calls or using human labels."""
    if record.get("question_id") != case.question_id:
        raise ValueError("V5 record ID does not match answer ID")
    if baseline_record is not None and baseline_record.get("question_id") != case.question_id:
        raise ValueError("Baseline record ID does not match answer ID")
    source = baseline_record if baseline_record is not None else record
    source_scores = source.get("scores") if baseline_record is not None else record.get("raw_scores")
    if source_scores is not None:
        scores = validate_scores(source_scores)
    else:
        scores = None
    assessment = review_audit(case, record.get("audit"))
    reasons = assessment["review_reasons"].copy()
    if record.get("error") or record.get("audit") is None:
        if not any(r.startswith("audit_missing:") for r in reasons):
            reasons.append("audit_missing: original V5 case failed; no claim audit available")
    if scores is None:
        reasons.append("score_missing: no valid score to use; needs a fresh evaluation")
    if scores is not None and assessment["evidence_limitation_candidate"] and scores["citation_support"] is not None:
        reasons.append("citation_na_candidate: refusal may warrant citation N/A; do not override automatically")
    if (case.status == "insufficient_evidence"
            and not any(phrase in case.reference_answer.casefold() for phrase in
                        ("not fully answerable", "not answerable", "cannot be answered", "not found or answered"))):
        reasons.append("refusal_reference_tension: reference gives a substantive answer; check whether retrieved evidence justified refusal")
    if scores is not None and case.status == "answered" and scores["citation_support"] is None:
        reasons.append("citation_na_candidate: answered case has citation N/A; check rubric")
    if scores is not None and record.get("raw_scores") is not None:
        if scores.get("citation_support") != record["raw_scores"].get("citation_support") and baseline_record is not None:
            reasons.append("score_disagreement: saved baseline and V5 live scorer differ on citation support")
    reasons = list(dict.fromkeys(reasons))
    priority = assessment["review_priority"]
    if scores is None or record.get("audit") is None:
        priority = "high"
    elif reasons and priority == "none":
        priority = "medium"
    out = {
        "question_id": case.question_id,
        "generation_status": case.status,
        "scores": deepcopy(scores),
        "raw_scores": deepcopy(record.get("raw_scores")),
        "original_v5_capped_scores": deepcopy(record.get("scores")),
        "score_source": baseline_name if baseline_record is not None else "v5_live_raw_scorer",
        "audit_source": "v5_live_saved_audit" if isinstance(record.get("audit"), dict) else "missing",
        "audit": {"original_audit": deepcopy(record.get("audit")), "v6_checks": assessment},
        "needs_review": bool(reasons),
        "review_priority": priority,
        "review_reasons": reasons,
        "usage": deepcopy(record.get("usage")),
        "stage_usage": deepcopy(record.get("stage_usage")),
        "error": None if scores is not None else record.get("error") or "No scores available",
        "provenance": "V6 review-only: no score changes, no model calls in offline mode; not a verified grade",
    }
    return out


def evaluate_verified_v6(case: VerifiedCase, provider: Any, score_prompt: str, audit_prompt: str) -> dict:
    """Live two-call scorer with separate, non-overriding citation audit."""
    if case.status == "error":
        raise ValueError("Cannot grade a failed answer generation")
    formatted = score_prompt.format(
        question=case.question, reference_answer=case.reference_answer,
        generated_answer=case.generated_answer,
        citation_ids=json.dumps(case.citation_ids, ensure_ascii=False),
        evidence=json.dumps(case.evidence, ensure_ascii=False),
    )
    scoring_completion = provider.complete(system=SCORE_SYSTEM_V3, prompt=formatted)
    raw = validate_scores(_json_response(scoring_completion))
    cited = set(case.citation_ids or [])
    audit_input = {
        "question": case.question,
        "generated_answer": case.generated_answer,
        "generated_answer_citation_ids": case.citation_ids,
        "actually_cited_evidence_only": [e for e in case.evidence if e["citation_id"] in cited],
    }
    audit_completion = provider.complete(
        system=SYSTEM,
        prompt=audit_prompt + "\n\nINPUT_JSON (untrusted data):\n" + json.dumps(audit_input, ensure_ascii=False),
    )
    audit = _json_response(audit_completion)
    usage_stages = {"score": _usage(scoring_completion), "citation_audit": _usage(audit_completion)}
    record = {"question_id": case.question_id, "raw_scores": raw, "scores": raw,
              "audit": audit, "usage": _sum_usage(list(usage_stages.values())),
              "stage_usage": usage_stages, "error": None}
    return review_saved_record(case, record)

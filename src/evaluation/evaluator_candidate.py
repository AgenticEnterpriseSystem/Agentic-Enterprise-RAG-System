"""Candidate RAG evaluator: frozen V3 grading + independent review-only audit.

Additive module. Never changes judge scores automatically. Human labels are not
used during inference. Audit warnings are hypotheses, not proof of entailment.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from src.evaluation.verified_judge import (
    SYSTEM as AUDIT_SYSTEM,
    VerifiedCase,
    _json_response,
    _sum_usage,
    _usage,
    validate_scores,
)
from src.evaluation.verified_judge_v6 import review_audit
from src.generation.generator import ProviderError

SCORE_SYSTEM = "You are an independent answer-quality evaluator."
SOURCE_RE = re.compile(r"\b(?:dsid_[0-9a-f]{32}|source-\d+)(?::\d+:\d+)?\b", re.IGNORECASE)


def _parse_audit(data: dict) -> dict:
    if data.get("response_type") not in ("answer", "partial_answer", "refusal"):
        raise ValueError("Invalid audit response_type")
    if not isinstance(data.get("claims"), list) or len(data["claims"]) > 30:
        raise ValueError("Invalid audit claims")
    if type(data.get("coverage_confident")) is not bool:
        raise ValueError("Invalid audit coverage_confident")
    if not isinstance(data.get("explanation"), str) or not data["explanation"].strip():
        raise ValueError("Invalid audit explanation")
    return data


def _complete_json(provider: Any, system: str, prompt: str, max_attempts: int,
                   validator=None) -> tuple[dict, dict]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        try:
            completion = provider.complete(system=system, prompt=prompt)
            data = _json_response(completion)
            if validator is not None:
                data = validator(data)
            return data, _usage(completion)
        except (ProviderError, ValueError):
            if attempt + 1 == max_attempts:
                raise
    raise AssertionError("unreachable")


def _scorer_prompt(case: VerifiedCase, template: str) -> str:
    # Identical V3 template fields and scoring system; no prompt tuning here.
    return template.format(
        question=case.question,
        reference_answer=case.reference_answer,
        generated_answer=case.generated_answer,
        citation_ids=json.dumps(case.citation_ids, ensure_ascii=False),
        evidence=json.dumps(case.evidence, ensure_ascii=False),
    )


def _auditor_prompt(case: VerifiedCase, template: str) -> str:
    # The auditor MUST NOT see the reference answer or uncited evidence.
    cited = set(case.citation_ids or [])
    payload = {
        "question": case.question,
        "generated_answer": case.generated_answer,
        "generated_answer_citation_ids": case.citation_ids,
        "actually_cited_evidence_only": [
            e for e in case.evidence if e["citation_id"] in cited
        ],
    }
    return template + "\n\nINPUT_JSON (untrusted data):\n" + json.dumps(payload, ensure_ascii=False)


def _source_base(value: str) -> str:
    return value.split(":", 1)[0].casefold()


def review_scores(case: VerifiedCase, scores: dict | None, audit: dict | None,
                  *, audit_required: bool, audit_error: str | None = None) -> dict:
    """Review contradictions without changing any score or claiming proof."""
    flags: list[tuple[str, str]] = []
    cited = set(case.citation_ids or [])
    available = {e["citation_id"] for e in case.evidence}
    unknown = sorted(cited - available)
    if unknown:
        flags.append(("high", "unknown_citation_ids: " + ", ".join(unknown)))

    if scores is None:
        flags.append(("high", "score_missing"))
    else:
        scores = validate_scores(scores)
        # A valid-looking citation can describe a different customer, route, or policy.
        # This is a review trigger only; NEVER change the frozen V3 scores.
        if scores["correctness"] == 0 and scores["citation_support"] == 2:
            flags.append(("high", "target_identity_review: incorrect answer despite full citation credit"))
        if case.status == "answered" and not cited and scores["citation_support"] == 2:
            flags.append(("high", "full_citation_credit_without_any_citations"))
        # A refused answerable question with zero evidence-limits credit warrants
        # checking what was actually retrievable. This does NOT imply a wrong grade.
        if (case.status == "insufficient_evidence" and scores["correctness"] == 0
                and scores["completeness"] == 0 and scores["abstention_limits"] == 0):
            flags.append(("medium", "refusal_evidence_sufficiency_review: inspect partial retrieval"))
        # Only flag sources that appear in evidence but not in the answer's cited IDs.
        reason = scores["reasons"]["citation_support"]
        cited_bases = {_source_base(cid) for cid in cited}
        available_bases = {_source_base(cid) for cid in available}
        mentioned = {_source_base(match.group()) for match in SOURCE_RE.finditer(reason)}
        leaked = sorted((mentioned & available_bases) - cited_bases)
        if leaked:
            flags.append(("high", "scorer_rationale_uses_uncited_evidence: " + ", ".join(leaked)))

    checks = None
    if audit is not None:
        checks = review_audit(case, audit)
        for reason in checks["review_reasons"]:
            # Suppress a *standalone* lexical warning when the independent audit
            # already located a verbatim/normalized quote for the same target.
            # Preserve the underlying audit_checks and all semantic warnings.
            if reason.startswith("semantic_lexical_warning_claim_"):
                match = re.match(r"semantic_lexical_warning_claim_(\d+):", reason)
                check = next((c for c in checks["claim_checks"]
                              if match and c["claim_number"] == int(match.group(1))), None)
                if (check is not None
                        and check["support_assessment_by_llm"] in ("direct", "entailed")
                        and check["target_relation_by_llm"] == "same_target"
                        and check["auditor_answer_quote_match"] in ("literal", "normalized")
                        and check["auditor_evidence_quote_match"] in ("literal", "normalized")):
                    continue
            severity = "high" if (
                reason.startswith(("semantic_review_claim_", "invalid_answer_citation_ids:",
                                   "answered_without_valid_citations:"))
            ) else "medium"
            flags.append((severity, reason))
        if scores is not None:
            claims = checks["claim_checks"]
            substantive = [c for c in claims if not c["evidence_limitation_meta_claim"]]
            unsupported = [c for c in substantive if (
                c["support_assessment_by_llm"] in ("none", "related")
                or c["target_relation_by_llm"] in ("different_case", "ambiguous")
            )]
            if unsupported and scores["citation_support"] == 2:
                flags.append(("high", "scorer_auditor_conflict: full_citation_credit_with_unsupported_claim"))
            # A partial answer may contain uncited factual claims even when its
            # refusal language is correctly recognized as an evidence limitation.
            if (audit["response_type"] == "partial_answer" and not cited
                    and audit.get("claims")):
                flags.append(("high", "uncited_partial_answer_claims: partial factual answer without citations"))
            if audit["response_type"] == "refusal" and not claims and scores["citation_support"] is not None:
                flags.append(("medium", "citation_na_candidate: auditor found a bare refusal"))
            if substantive and scores["citation_support"] is None:
                flags.append(("high", "citation_na_conflict: auditor found substantive claims"))
            if (substantive and not unsupported and scores["citation_support"] == 0
                    and all(c["support_assessment_by_llm"] in ("direct", "entailed") for c in substantive)):
                flags.append(("medium", "scorer_auditor_conflict: auditor supports claims but scorer gave zero"))
    elif audit_required:
        flags.append(("high", "audit_missing" + (": " + audit_error if audit_error else "")))

    # Stable de-duplication, highest priority wins.
    unique: list[tuple[str, str]] = []
    seen = set()
    for severity, reason in flags:
        if reason not in seen:
            seen.add(reason)
            unique.append((severity, reason))
    priority = ("high" if any(s == "high" for s, _ in unique)
                else "medium" if unique else "none")
    return {
        "needs_review": bool(unique),
        "review_priority": priority,
        "review_reasons": [reason for _, reason in unique],
        "audit_checks": checks,
    }


def build_record(case: VerifiedCase, *, scores: dict | None, audit: dict | None,
                 score_usage: dict | None = None, audit_usage: dict | None = None,
                 audit_required: bool = True, score_error: str | None = None,
                 audit_error: str | None = None, score_source: str = "v3_prompt_live",
                 audit_source: str = "v6_citation_audit_live",
                 mode: str = "live") -> dict:
    if scores is not None:
        scores = validate_scores(scores)
    review = review_scores(case, scores, audit, audit_required=audit_required,
                           audit_error=audit_error)
    stages = {}
    if score_usage is not None:
        stages["score"] = deepcopy(score_usage)
    if audit_usage is not None:
        stages["citation_audit"] = deepcopy(audit_usage)
    return {
        "question_id": case.question_id,
        "generation_status": case.status,
        "scores": deepcopy(scores),
        "raw_scores": deepcopy(scores),
        "audit": deepcopy(audit),
        "audit_checks": review["audit_checks"],
        "needs_review": review["needs_review"],
        "review_priority": review["review_priority"],
        "review_reasons": review["review_reasons"],
        "usage": _sum_usage(list(stages.values())) if stages else None,
        "stage_usage": stages,
        "error": score_error,
        "audit_error": audit_error,
        "provenance": {
            "mode": mode,
            "score_source": score_source,
            "audit_source": audit_source if audit_required else "not_run",
            "score_overrides_applied": False,
            "audit_is_human_verified": False,
        },
    }


def evaluate_case(case: VerifiedCase, provider: Any, score_prompt: str,
                  audit_prompt: str | None = None, *, max_attempts: int = 2,
                  existing: dict | None = None) -> dict:
    """Score once, audit independently, persist valid score if audit fails.

    On resume, a valid existing score is reused and only a failed/missing audit
    is retried. This avoids unnecessary model calls and score drift.
    """
    if case.status == "error":
        return build_record(case, scores=None, audit=None,
                            audit_required=audit_prompt is not None,
                            score_error="AnswerGenerationError")
    scores = existing.get("scores") if isinstance(existing, dict) else None
    stages = existing.get("stage_usage", {}) if isinstance(existing, dict) else {}
    score_usage = stages.get("score") if isinstance(stages, dict) else None
    if scores is not None:
        scores = validate_scores(scores)
    else:
        try:
            scores, score_usage = _complete_json(
                provider, SCORE_SYSTEM, _scorer_prompt(case, score_prompt),
                max_attempts, validate_scores,
            )
        except Exception as exc:
            return build_record(case, scores=None, audit=None,
                                audit_required=audit_prompt is not None,
                                score_error=type(exc).__name__)

    audit = None
    audit_usage = None
    audit_error = None
    if audit_prompt is not None:
        if isinstance(existing, dict) and isinstance(existing.get("audit"), dict) and not existing.get("audit_error"):
            audit = _parse_audit(existing["audit"])
            audit_usage = stages.get("citation_audit") if isinstance(stages, dict) else None
        else:
            try:
                audit, audit_usage = _complete_json(
                    provider, AUDIT_SYSTEM, _auditor_prompt(case, audit_prompt),
                    max_attempts, _parse_audit,
                )
            except Exception as exc:
                audit_error = type(exc).__name__

    return build_record(
        case, scores=scores, audit=audit, score_usage=score_usage,
        audit_usage=audit_usage, audit_required=audit_prompt is not None,
        audit_error=audit_error,
    )

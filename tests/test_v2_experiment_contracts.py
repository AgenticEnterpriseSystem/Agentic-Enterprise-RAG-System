"""Scientific handoff regressions; no exporter, model, database, or judge calls."""

import hashlib
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.contracts import Answer, Chunk, Trace, Usage
from src.contracts_v2 import AggregateEvidence, PassageEvidence, QueryResponseV2, ScopeBinding, SnapshotRef
from src.evaluation.experiment_contracts import (
    CodeProvenance, ExperimentManifest, GenerationBudgets, ModelProvenance,
    RetrievalProvenance, RuntimeCaseExport, RuntimeFailure, ScoringInput,
    ScoringReference, runtime_case_digest, validate_paired_arms,
    validate_resume_compatibility, validate_runtime_exports,
)

NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)
SHA = "a" * 64
OTHER_SHA = "b" * 64
COMMIT = "c" * 40
SNAPSHOT = SnapshotRef(snapshot_id="corpus-v1", corpus_sha256=SHA,
                       catalog_sha256=OTHER_SHA, document_count=2)
SCOPE = ScopeBinding(scope_id="benchmark", user_id="reviewer", snapshot=SNAPSHOT,
                     policy_revision=1, session_version=1)


def update(model, **changes):
    return type(model).model_validate({**model.model_dump(), **changes})


def manifest(workflow="fixed", **changes):
    values = dict(experiment_id="comparison", manifest_id=workflow, workflow=workflow,
        created_at=NOW, code=CodeProvenance(commit=COMMIT), split="development",
        split_manifest_sha256=SHA, question_set_sha256=SHA, question_ids=("q1", "q2"),
        scope=SCOPE, scope_fingerprint_sha256=SHA,
        generation_model=ModelProvenance(provider="scripted", model_id="model-v1"),
        generator_prompt_sha256=SHA, critic_prompt_sha256=SHA if workflow == "adaptive" else None,
        comparison_config_sha256=SHA,
        retrieval=RetrievalProvenance(retriever_id="hybrid", snapshot=SNAPSHOT,
            index_manifest_sha256=SHA, config_sha256=SHA),
        budgets=GenerationBudgets(top_k=5, max_question_chars=8000,
            max_context_chars=24000, max_output_tokens=1024, max_run_tokens=16000,
            max_run_seconds=120, max_run_cost_usd=.1), max_rounds=3 if workflow == "adaptive" else 1)
    return ExperimentManifest(**{**values, **changes})


def passage(snapshot_id=SNAPSHOT.snapshot_id):
    return PassageEvidence(citation_id="doc:0:8", snapshot_id=snapshot_id, revision_id="revision-v1",
        chunk=Chunk(chunk_id="doc:0:8", doc_id="doc", source_type="jira", title="Ticket",
                    text="Resolved", start_char=0, end_char=8), score=1, retrieval_method="hybrid")


def case(owner="fixed", qid="q1", *, failed=False, evidence=None, **changes):
    selected = (passage(),) if evidence is None else evidence
    question = "What is the status?" if qid == "q1" else "What happened?"
    run_id = f"{owner}-{qid}"
    response = QueryResponseV2(snapshot=SNAPSHOT,
        answer=Answer(status="error" if failed else "answered",
            text="Model unavailable" if failed else "Resolved.",
            citation_ids=() if failed else (selected[0].citation_id,)),
        sources=() if failed else selected,
        trace=Trace(run_id=run_id, stop_reason="error" if failed else "single_pass",
                    usage=Usage(api_calls=1, estimated_cost_usd=None)))
    values = dict(manifest_id=owner, run_id=run_id, question_id=qid, question=question,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(), response=response,
        generation_evidence=() if failed else selected,
        error=RuntimeFailure(code="provider_unavailable", stage="generate") if failed else None)
    return RuntimeCaseExport(**{**values, **changes})


def test_export_round_trip_keeps_failures_unknown_costs_and_exact_generation_evidence():
    fixed = manifest()
    cases = (case(), case(qid="q2", failed=True))
    validate_runtime_exports(fixed, cases, require_complete=True)
    restored = RuntimeCaseExport.model_validate_json(cases[1].model_dump_json())
    assert restored == cases[1]
    assert restored.response.trace.usage.estimated_cost_usd is None
    assert restored.error.code == "provider_unavailable"
    assert cases[0].generation_evidence == cases[0].response.sources
    assert ExperimentManifest.model_validate_json(fixed.model_dump_json()) == fixed


def test_partial_resume_is_distinct_from_a_missing_complete_outcome():
    validate_runtime_exports(manifest(), ())
    validate_runtime_exports(manifest(), (case(),))
    with pytest.raises(ValueError, match="missing declared"):
        validate_runtime_exports(manifest(), (case(),), require_complete=True)
    validate_resume_compatibility(manifest(), manifest(created_at=datetime(2026, 11, 3, tzinfo=timezone.utc)))


@pytest.mark.parametrize("field,value", [
    ("manifest_id", "another-run"), ("split", "final"), ("question_ids", ("q2", "q1")),
    ("scope_fingerprint_sha256", OTHER_SHA), ("generator_prompt_sha256", OTHER_SHA),
    ("code", CodeProvenance(commit="d" * 40)),
])
def test_resume_rejects_changed_identity_or_scientific_settings(field, value):
    with pytest.raises(ValueError, match="Resume changes"):
        validate_resume_compatibility(manifest(), manifest(**{field: value}))


def test_frozen_split_and_full_source_commit_are_required():
    for split in ("development", "calibration", "final"):
        assert manifest(split=split).split == split
    for split in ("test", "train", "FINAL"):
        with pytest.raises(ValidationError):
            manifest(split=split)
    for commit in ("da970e8", "g" * 40, "c" * 39, "C" * 40):
        with pytest.raises(ValidationError):
            CodeProvenance(commit=commit)
    with pytest.raises(ValidationError, match="diff hash"):
        CodeProvenance(commit=COMMIT, dirty=True)
    with pytest.raises(ValidationError, match="diff hash"):
        CodeProvenance(commit=COMMIT, diff_sha256=SHA)
    assert CodeProvenance(commit=COMMIT, dirty=True, diff_sha256=SHA).dirty


def test_selection_snapshot_and_arm_metadata_cannot_contradict_the_manifest():
    with pytest.raises(ValidationError, match="Duplicate manifest"):
        manifest(question_ids=("q1", "q1"))
    other = update(SNAPSHOT, catalog_sha256="d" * 64)
    with pytest.raises(ValidationError, match="same snapshot"):
        manifest(retrieval=update(manifest().retrieval, snapshot=other))
    with pytest.raises(ValidationError, match="one round"):
        manifest(max_rounds=2)
    with pytest.raises(ValidationError, match="critic prompt"):
        manifest("adaptive", critic_prompt_sha256=None)


@pytest.mark.parametrize("extra", ["reference_answer", "gold_answer", "answer_facts", "reference"])
def test_reference_fields_are_forbidden_in_runtime_exports(extra):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeCaseExport.model_validate({**case().model_dump(), extra: "Scoring only"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentManifest.model_validate({**manifest().model_dump(), extra: "Scoring only"})


def test_nested_contracts_are_immutable_and_do_not_retain_mutable_input_lists():
    ids = ["q1", "q2"]
    frozen = manifest(question_ids=ids)
    ids.append("q3")
    assert frozen.question_ids == ("q1", "q2")
    for obj, field, value in ((frozen, "split", "final"), (frozen.budgets, "top_k", 10),
        (frozen.scope.snapshot, "document_count", 999), (case().response.answer, "text", "Changed")):
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(obj, field, value)
    json.dumps(ExperimentManifest.model_json_schema())
    json.dumps(RuntimeCaseExport.model_json_schema())
    json.dumps(ScoringInput.model_json_schema())


def test_duplicate_question_and_run_ids_reject_while_error_outcomes_remain_valid():
    first = case()
    with pytest.raises(ValueError, match="Duplicate question or run"):
        validate_runtime_exports(manifest(), (first, first))
    second = case(qid="q2")
    reused = update(second, run_id=first.run_id,
                    response=update(second.response, trace=update(second.response.trace, run_id=first.run_id)))
    with pytest.raises(ValueError, match="Duplicate question or run"):
        validate_runtime_exports(manifest(), (first, reused))
    with pytest.raises(ValueError, match="another manifest"):
        validate_runtime_exports(manifest(), (case(owner="adaptive"),))
    with pytest.raises(ValueError, match="frozen selection"):
        validate_runtime_exports(manifest(), (case(qid="outside"),))


def test_invalid_runtime_metadata_is_not_confused_with_a_valid_failed_case():
    failed = case(failed=True)
    validate_runtime_exports(manifest(), (failed,))
    with pytest.raises(ValidationError, match="run ID"):
        update(failed, run_id="another-run")
    with pytest.raises(ValidationError, match="Question text"):
        update(failed, question_sha256=OTHER_SHA)
    with pytest.raises(ValidationError, match="failure metadata"):
        update(failed, error=None)
    with pytest.raises(ValidationError, match="failure metadata"):
        update(case(), error=failed.error)


def test_error_trace_and_error_answer_must_agree_in_both_directions():
    succeeded = case()
    with pytest.raises(ValidationError, match="error trace"):
        update(succeeded, response=update(succeeded.response,
            trace=update(succeeded.response.trace, stop_reason="error")))
    failed = case(failed=True)
    with pytest.raises(ValidationError, match="error trace"):
        update(failed, response=update(failed.response,
            trace=update(failed.response.trace, stop_reason="single_pass")))


def test_direct_clarification_preserves_citations_without_inventing_generation_input():
    original = case()
    clarification = update(original.response,
        answer=Answer(status="insufficient_evidence", text="Which incident do you mean?",
                      citation_ids=original.response.answer.citation_ids),
        trace=update(original.response.trace, stop_reason="clarification"))
    exported = update(original, response=clarification, generation_evidence=())
    validate_runtime_exports(manifest(), (exported,))
    assert exported.response.answer.citation_ids and not exported.generation_evidence
    with pytest.raises(ValidationError, match="no generation input"):
        update(exported, generation_evidence=original.generation_evidence)


def test_public_helpers_revalidate_copied_cases_instead_of_trusting_frozen_types():
    copied = case().model_copy(update={"question": "Substituted text with the original hash"})
    with pytest.raises(ValidationError, match="Question text"):
        validate_runtime_exports(manifest(), (copied,))
    with pytest.raises(ValidationError, match="Question text"):
        validate_paired_arms(manifest(), manifest("adaptive"), (copied,),
                             (case(owner="adaptive"),), require_complete=False)
    with pytest.raises(ValidationError, match="Question text"):
        runtime_case_digest(copied)


def test_public_helpers_revalidate_copied_manifest_and_nested_metadata():
    bad_rounds = manifest().model_copy(update={"max_rounds": 2})
    with pytest.raises(ValidationError, match="one round"):
        validate_runtime_exports(bad_rounds, ())
    with pytest.raises(ValidationError, match="one round"):
        validate_resume_compatibility(manifest(), bad_rounds)
    with pytest.raises(ValidationError, match="one round"):
        validate_paired_arms(bad_rounds, manifest("adaptive"), (), (), require_complete=False)
    nested = manifest().model_copy(update={
        "code": CodeProvenance(commit=COMMIT).model_copy(update={"commit": "short"})})
    with pytest.raises(ValidationError):
        validate_runtime_exports(nested, ())


def test_generation_evidence_must_match_answer_and_snapshot():
    with pytest.raises(ValidationError, match="outside the recorded"):
        update(case(), generation_evidence=())
    with pytest.raises(ValidationError, match="Duplicate generation"):
        update(case(), generation_evidence=(passage(), passage()))
    with pytest.raises(ValidationError, match="another snapshot"):
        update(case(), generation_evidence=(passage("other-snapshot"),))
    changed = update(passage(), chunk=update(passage().chunk, text="Deferred"))
    with pytest.raises(ValidationError, match="evidence disagree"):
        update(case(), generation_evidence=(changed,))
    other = update(SNAPSHOT, catalog_sha256="d" * 64)
    with pytest.raises(ValueError, match="another snapshot"):
        validate_runtime_exports(manifest(), (update(case(), response=update(case().response, snapshot=other)),))


def test_zero_aggregate_keeps_query_scope_provenance_and_validates_as_runtime_evidence():
    zero = AggregateEvidence(citation_id="aggregate-zero", artifact_id="artifact-zero", scope=SCOPE,
        template_id="ticket-count", template_version="1", count=0, lineage_sha256=SHA,
        coverage="verified_records", created_at=NOW)
    exported = case(evidence=(zero,))
    validate_runtime_exports(manifest(), (exported,))
    roundtrip = RuntimeCaseExport.model_validate_json(exported.model_dump_json())
    assert roundtrip.generation_evidence[0].count == 0
    assert roundtrip.generation_evidence[0].scope == SCOPE
    other_user = update(zero, scope=update(SCOPE, user_id="different-user"))
    with pytest.raises(ValueError, match="another permission scope"):
        validate_runtime_exports(manifest(), (case(evidence=(other_user,)),))


def test_pairs_allow_different_answers_order_and_retained_failure_but_not_unmatched_cases():
    left, right = manifest(), manifest("adaptive")
    fixed_cases = (case(), case(qid="q2", failed=True))
    adaptive_cases = (case(owner="adaptive", qid="q2"), case(owner="adaptive"))
    validate_paired_arms(left, right, fixed_cases, adaptive_cases)
    with pytest.raises(ValueError, match="missing declared"):
        validate_paired_arms(left, right, fixed_cases, adaptive_cases[:1])
    with pytest.raises(ValueError, match="unmatched question"):
        validate_paired_arms(left, right, fixed_cases[:1], adaptive_cases[:1], require_complete=False)


@pytest.mark.parametrize("field,value", [
    ("code", CodeProvenance(commit="d" * 40)),
    ("scope", update(SCOPE, policy_revision=2)),
    ("generation_model", ModelProvenance(provider="scripted", model_id="different-model")),
    ("generator_prompt_sha256", OTHER_SHA), ("comparison_config_sha256", OTHER_SHA),
    ("router_prompt_sha256", OTHER_SHA), ("temperature", .5), ("split", "calibration"),
    ("budgets", update(manifest().budgets, max_context_chars=12000)),
    ("retrieval", update(manifest().retrieval, config_sha256=OTHER_SHA)),
    ("retrieval", update(manifest().retrieval, index_manifest_sha256=OTHER_SHA)),
    ("question_ids", ("q1",)),
])
def test_pairs_reject_incompatible_scope_model_retrieval_evidence_budgets_and_config(field, value):
    with pytest.raises(ValueError, match="shared configuration or question selection"):
        validate_paired_arms(manifest(), manifest("adaptive", **{field: value}), (), (), require_complete=False)


def test_pair_identity_and_question_text_cannot_be_reused_or_substituted():
    with pytest.raises(ValueError, match="distinct manifest"):
        validate_paired_arms(manifest(), manifest("adaptive", manifest_id="fixed"), (), ())
    with pytest.raises(ValueError, match="fixed and adaptive"):
        validate_paired_arms(manifest("adaptive"), manifest(), (), ())
    left, right = case(), case(owner="adaptive")
    right = update(right, run_id=left.run_id,
                   response=update(right.response, trace=update(right.response.trace, run_id=left.run_id)))
    with pytest.raises(ValueError, match="reused across paired"):
        validate_paired_arms(manifest(), manifest("adaptive"), (left,), (right,), require_complete=False)
    question = "A substituted question"
    right = update(case(owner="adaptive"), question=question,
                   question_sha256=hashlib.sha256(question.encode()).hexdigest())
    with pytest.raises(ValueError, match="question text differs"):
        validate_paired_arms(manifest(), manifest("adaptive"), (left,), (right,), require_complete=False)


def test_scoring_inputs_are_a_separate_exact_hash_join_and_allow_absent_references():
    runtime = case()
    reference = ScoringReference(question_id=runtime.question_id, question_sha256=runtime.question_sha256,
        reference_set_sha256=SHA, reference_answer="The ticket was resolved.",
        answer_facts=("Resolved",), expected_doc_ids=("doc",))
    scoring = ScoringInput(runtime_case=runtime, runtime_case_sha256=runtime_case_digest(runtime), reference=reference)
    assert ScoringInput.model_validate_json(scoring.model_dump_json()) == scoring
    assert "reference_answer" not in runtime.model_dump_json()
    with pytest.raises(ValidationError, match="another question"):
        update(scoring, reference=update(reference, question_id="q2"))
    with pytest.raises(ValidationError, match="different runtime case"):
        update(scoring, runtime_case_sha256=OTHER_SHA)
    absent = update(reference, reference_answer=None, answer_facts=(), expected_doc_ids=())
    assert update(scoring, reference=absent).reference.reference_answer is None

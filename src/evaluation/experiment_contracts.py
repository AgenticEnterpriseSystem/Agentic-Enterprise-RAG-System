"""Immutable experiment handoff contracts, without runners or judge execution.

Runtime exports deliberately contain no reference-answer fields. Scoring inputs
join an exact runtime export with separate references only after generation.
These schemas check declared provenance, not authorization or on-disk hashes.
"""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, model_validator

from src.contracts import (
    AggregateEvidence, Contract, GroundedEvidence, Identifier, PassageEvidence, QueryResponse,
    ScopeBinding, Sha256, SnapshotRef, StructuredRecord,
)

EXPERIMENT_SCHEMA_VERSION = "2"
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
QuestionText = Annotated[str, Field(min_length=1, max_length=8000, pattern=r"\S")]
Split = Literal["development", "calibration", "final"]


class CodeProvenance(Contract):
    commit: GitCommit
    dirty: StrictBool = False
    diff_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def dirty_diff_is_explicit(self):
        if self.dirty != (self.diff_sha256 is not None):
            raise ValueError("Dirty source requires a diff hash; clean source must omit it")
        return self


class ModelProvenance(Contract):
    provider: Identifier
    model_id: Identifier
    revision: Identifier | None = None


class RetrievalProvenance(Contract):
    retriever_id: Identifier
    snapshot: SnapshotRef
    index_manifest_sha256: Sha256
    config_sha256: Sha256
    embedding_model: ModelProvenance | None = None


class GenerationBudgets(Contract):
    """Shared evidence/output/run limits; rounds are the declared arm difference."""

    top_k: PositiveInt
    max_question_chars: PositiveInt
    max_context_chars: PositiveInt
    max_output_tokens: PositiveInt
    max_run_tokens: PositiveInt
    max_run_seconds: Annotated[float, Field(gt=0)]
    max_run_cost_usd: Annotated[float, Field(ge=0)]
    retries: NonnegativeInt = 0


class ExperimentManifest(Contract):
    """One immutable arm/repetition with a frozen selection and shared settings.

    The comparison hash identifies shared configuration, excluding the intended
    workflow/round/critic differences. File hashes are declarations that a future
    exporter must verify against its actual files; this DTO performs no I/O.
    """

    schema_version: Literal["2"] = EXPERIMENT_SCHEMA_VERSION
    experiment_id: Identifier
    manifest_id: Identifier
    workflow: Literal["fixed", "adaptive"]
    repetition: PositiveInt = 1
    created_at: AwareDatetime
    code: CodeProvenance
    split: Split
    split_manifest_sha256: Sha256
    question_set_sha256: Sha256
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    scope: ScopeBinding
    scope_fingerprint_sha256: Sha256
    generation_model: ModelProvenance
    generator_prompt_sha256: Sha256
    critic_prompt_sha256: Sha256 | None = None
    router_prompt_sha256: Sha256 | None = None
    comparison_config_sha256: Sha256
    retrieval: RetrievalProvenance
    budgets: GenerationBudgets
    temperature: Annotated[float, Field(ge=0)] = 0
    max_rounds: PositiveInt

    @model_validator(mode="after")
    def consistent_selection_and_snapshot(self):
        if len(self.question_ids) != len(set(self.question_ids)):
            raise ValueError("Duplicate manifest question IDs")
        if self.scope.snapshot != self.retrieval.snapshot:
            raise ValueError("Retrieval and permission scope must bind the same snapshot")
        if self.workflow == "fixed" and self.max_rounds != 1:
            raise ValueError("Fixed workflow must declare one round")
        if self.workflow == "adaptive" and self.critic_prompt_sha256 is None:
            raise ValueError("Adaptive workflow requires its critic prompt hash")
        return self


class RuntimeFailure(Contract):
    """Sanitized failure metadata; failures remain cases, never missing rows."""

    code: Identifier
    stage: Identifier
    retryable: StrictBool = False
    message: Annotated[str, Field(min_length=1, max_length=500)] | None = None


class RuntimeCaseExport(Contract):
    """Exact runtime outcome and generation evidence, with no gold references."""

    schema_version: Literal["2"] = EXPERIMENT_SCHEMA_VERSION
    manifest_id: Identifier
    run_id: Identifier
    question_id: Identifier
    question: QuestionText
    question_sha256: Sha256
    response: QueryResponse
    generation_evidence: tuple[GroundedEvidence, ...] = ()
    error: RuntimeFailure | None = None

    @model_validator(mode="after")
    def consistent_runtime_outcome(self):
        if self.response.snapshot is None:
            raise ValueError("Runtime exports require a snapshot-bound response")
        if self.run_id != self.response.trace.run_id:
            raise ValueError("Export run ID differs from its response trace")
        if self.question_sha256 != hashlib.sha256(self.question.encode("utf-8")).hexdigest():
            raise ValueError("Question text does not match its hash")
        failed = self.response.answer.status == "error"
        if failed != (self.error is not None):
            raise ValueError("Error outcomes require failure metadata, and only error outcomes may have it")
        if failed != (self.response.trace.stop_reason == "error"):
            raise ValueError("Error answer and error trace must agree")
        supplied = {item.citation_id: item for item in self.generation_evidence}
        if len(supplied) != len(self.generation_evidence):
            raise ValueError("Duplicate generation evidence citation IDs")
        for item in self.generation_evidence:
            _validate_evidence_snapshot(item, self.response.snapshot)
        returned = {item.citation_id: item for item in self.response.sources}
        if any(returned[cid] != item for cid, item in supplied.items() if cid in returned):
            raise ValueError("Generation and response evidence disagree for the same citation")
        if self.response.trace.stop_reason == "clarification":
            # The current agent returns the critic's distinguishing question
            # directly. Its citations resolve through response.sources, without
            # pretending those excerpts were sent to a generator.
            if self.response.answer.status != "insufficient_evidence" or self.generation_evidence:
                raise ValueError("Clarification requires insufficient evidence and no generation input")
        elif not set(self.response.answer.citation_ids) <= supplied.keys():
            raise ValueError("Answer cites evidence outside the recorded generation input")
        return self


class ScoringReference(Contract):
    """Scoring-only reference data; unknown reference facts remain absent."""

    question_id: Identifier
    question_sha256: Sha256
    reference_set_sha256: Sha256
    reference_answer: Identifier | None = None
    answer_facts: tuple[Identifier, ...] = ()
    expected_doc_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_reference_documents(self):
        if len(self.expected_doc_ids) != len(set(self.expected_doc_ids)):
            raise ValueError("Duplicate reference document IDs")
        return self


class ScoringInput(Contract):
    """External scoring handoff; never a runtime-generation request."""

    schema_version: Literal["2"] = EXPERIMENT_SCHEMA_VERSION
    runtime_case: RuntimeCaseExport
    runtime_case_sha256: Sha256
    reference: ScoringReference

    @model_validator(mode="after")
    def exact_reference_join(self):
        if (self.reference.question_id != self.runtime_case.question_id
                or self.reference.question_sha256 != self.runtime_case.question_sha256):
            raise ValueError("Reference belongs to another question")
        if self.runtime_case_sha256 != runtime_case_digest(self.runtime_case):
            raise ValueError("Scoring input refers to a different runtime case")
        return self


def runtime_case_digest(case: RuntimeCaseExport) -> str:
    """Hash canonical, versioned JSON with all fields, including explicit nulls."""
    case = RuntimeCaseExport.model_validate(case.model_dump())
    payload = json.dumps(case.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_runtime_exports(manifest: ExperimentManifest,
                             cases: tuple[RuntimeCaseExport, ...], *,
                             require_complete: bool = False) -> None:
    """Validate partial resumable exports or a complete arm, retaining errors."""
    # Frozen Pydantic objects can still be made with model_copy(update=...),
    # which bypasses their validators. Revalidate the serialized boundary rather
    # than trusting the apparent type of an object supplied by a caller.
    manifest = ExperimentManifest.model_validate(manifest.model_dump())
    cases = tuple(RuntimeCaseExport.model_validate(case.model_dump()) for case in cases)
    question_ids, run_ids = set(), set()
    declared = set(manifest.question_ids)
    for case in cases:
        if case.manifest_id != manifest.manifest_id:
            raise ValueError("Case belongs to another manifest")
        if case.question_id not in declared:
            raise ValueError("Case question is outside the frozen selection")
        if case.question_id in question_ids or case.run_id in run_ids:
            raise ValueError("Duplicate question or run identity in exported cases")
        question_ids.add(case.question_id)
        run_ids.add(case.run_id)
        if case.response.snapshot != manifest.scope.snapshot:
            raise ValueError("Case belongs to another snapshot")
        for item in (*case.response.sources, *case.generation_evidence):
            _validate_evidence_snapshot(item, manifest.scope.snapshot)
            if isinstance(item, (AggregateEvidence, StructuredRecord)) and item.scope != manifest.scope:
                raise ValueError("Native evidence belongs to another permission scope")
    if require_complete and question_ids != declared:
        raise ValueError("Export is missing declared question outcomes")


def validate_resume_compatibility(original: ExperimentManifest,
                                  resumed: ExperimentManifest) -> None:
    """Resume the same manifest only; an invocation timestamp may differ.

    Preserve original creation metadata when writing the resumed manifest. All
    scientific/identity/selection settings, including order, must be unchanged.
    Failed cases are already completed attempts and are not silently retried.
    """
    original = ExperimentManifest.model_validate(original.model_dump())
    resumed = ExperimentManifest.model_validate(resumed.model_dump())
    before = original.model_dump(exclude={"created_at"})
    after = resumed.model_dump(exclude={"created_at"})
    changed = sorted(key for key in before if before[key] != after[key])
    if changed:
        raise ValueError("Resume changes frozen manifest fields: " + ", ".join(changed))


def validate_paired_arms(fixed: ExperimentManifest, adaptive: ExperimentManifest,
                         fixed_cases: tuple[RuntimeCaseExport, ...],
                         adaptive_cases: tuple[RuntimeCaseExport, ...], *,
                         require_complete: bool = True) -> None:
    """Match fixed/adaptive outcomes by question, allowing errors and run order.

    Rounds, workflow, critic prompt, manifest ID, and creation time are intended
    arm differences. Shared generator/router, source, scope, retrieval, budgets,
    and comparison settings must match; answer/evidence content may improve.
    """
    fixed = ExperimentManifest.model_validate(fixed.model_dump())
    adaptive = ExperimentManifest.model_validate(adaptive.model_dump())
    fixed_cases = tuple(RuntimeCaseExport.model_validate(case.model_dump()) for case in fixed_cases)
    adaptive_cases = tuple(RuntimeCaseExport.model_validate(case.model_dump()) for case in adaptive_cases)
    if fixed.workflow != "fixed" or adaptive.workflow != "adaptive":
        raise ValueError("Pair requires fixed and adaptive manifests in that order")
    if fixed.manifest_id == adaptive.manifest_id:
        raise ValueError("Paired arms require distinct manifest identities")
    shared_fields = (
        "schema_version", "experiment_id", "repetition", "code", "split",
        "split_manifest_sha256", "question_set_sha256", "scope",
        "scope_fingerprint_sha256", "generation_model", "generator_prompt_sha256",
        "router_prompt_sha256", "comparison_config_sha256", "retrieval", "budgets",
        "temperature",
    )
    changed = [name for name in shared_fields if getattr(fixed, name) != getattr(adaptive, name)]
    if changed or set(fixed.question_ids) != set(adaptive.question_ids):
        raise ValueError("Paired arms differ in shared configuration or question selection: "
                         + ", ".join(changed))
    validate_runtime_exports(fixed, fixed_cases, require_complete=require_complete)
    validate_runtime_exports(adaptive, adaptive_cases, require_complete=require_complete)
    left = {case.question_id: case for case in fixed_cases}
    right = {case.question_id: case for case in adaptive_cases}
    if left.keys() != right.keys():
        raise ValueError("Paired arms have unmatched question outcomes")
    if {case.run_id for case in fixed_cases} & {case.run_id for case in adaptive_cases}:
        raise ValueError("A runtime run identity cannot be reused across paired arms")
    if any(left[qid].question_sha256 != right[qid].question_sha256 for qid in left):
        raise ValueError("Paired question text differs")


def _validate_evidence_snapshot(item: PassageEvidence | AggregateEvidence | StructuredRecord,
                                snapshot: SnapshotRef) -> None:
    if isinstance(item, PassageEvidence):
        if item.snapshot_id != snapshot.snapshot_id:
            raise ValueError("Generation passage belongs to another snapshot")
    elif item.scope.snapshot != snapshot:
        raise ValueError("Native evidence belongs to another snapshot")

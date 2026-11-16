"""Offline CAL-01 packets for native fact-checklist stopping decisions.

This module never runs a model or establishes document authorization. It checks
stored runtime captures and the producer's scoped authorization attestation.
Retrieval rankings, development fixtures, fixed traces and upstream answers are
not human critic labels. A freeze produced here is always an unaccepted proposal;
human origin and final acceptance remain explicit reviewer/operator attestations.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, model_validator

from src.contracts import (
    AggregateEvidence, Contract, GroundedEvidence, Identifier, PassageEvidence,
    ScopeBinding, Sha256, StructuredRecord,
)
from src.evaluation.experiment_contracts import (
    ExperimentManifest, ModelProvenance, RuntimeCaseExport, runtime_case_digest,
)
from src.evaluation.experiment_inputs import (
    _json_loads, _questions, _read_json, _split_membership, canonical_json_sha256,
    file_sha256, scientific_settings_sha256,
    question_set_sha256, RuntimeQuestion,
)
from src.generation.generator import evidence_payload

PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Index = Annotated[int, Field(strict=True, ge=0)]
FactStatus = Literal["supported", "missing", "conflicting", "ambiguous"]
ExpectedAction = Literal["answer", "read_more", "research", "clarify", "stop_budget", "stop_no_new_evidence"]


class CalibrationQuestion(Contract):
    question_id: Identifier
    question: Annotated[str, Field(min_length=1, max_length=8000)]
    question_type: Identifier


class CalibrationSelection(Contract):
    format: Literal["rag-critic-calibration-selection-v1"] = "rag-critic-calibration-selection-v1"
    split: Literal["calibration"] = "calibration"
    method: Literal["sorted-question-type-round-robin-v1"] = "sorted-question-type-round-robin-v1"
    questions_file_sha256: Sha256
    split_file_sha256: Sha256
    calibration_population_ids: tuple[Identifier, ...] = Field(min_length=1)
    questions: tuple[CalibrationQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_selection(self):
        ids = [q.question_id for q in self.questions]
        if (len(ids) != len(set(ids)) or len(self.calibration_population_ids) != len(set(self.calibration_population_ids))
                or not set(ids) <= set(self.calibration_population_ids)):
            raise ValueError("Selection must contain unique calibration members")
        return self


def select_calibration_questions(question_file: Path, split_file: Path, count: int) -> CalibrationSelection:
    """Outcome-independent category coverage; never takes a split/final override."""
    questions, questions_sha = _questions(question_file)
    split, split_sha = _read_json(split_file)
    membership = _split_membership(split, questions)
    population = set(membership["calibration"])
    if type(count) is not int or not 1 <= count <= len(population):
        raise ValueError("Count must fit the calibration population")
    if question_file.suffix.lower() == ".jsonl":
        rows = [_json_loads(line) for line in question_file.read_bytes().splitlines() if line.strip()]
    else:
        rows, _ = _read_json(question_file)
    categories = {row["question_id"]: row.get("question_type") for row in rows}
    groups = defaultdict(list)
    for qid in sorted(population):
        category = categories[qid]
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Calibration questions require explicit question types")
        groups[category].append(CalibrationQuestion(**questions[qid].model_dump(), question_type=category))
    chosen = []
    while len(chosen) < count:
        for category in sorted(groups):
            if groups[category] and len(chosen) < count:
                chosen.append(groups[category].pop(0))
    return CalibrationSelection(questions_file_sha256=questions_sha, split_file_sha256=split_sha,
        calibration_population_ids=tuple(sorted(population)), questions=tuple(chosen))


def recheck_selection(selection: CalibrationSelection, question_file: Path, split_file: Path) -> None:
    actual = select_calibration_questions(question_file, split_file, len(selection.questions))
    if actual != selection:
        raise ValueError("Frozen calibration selection or question/split files changed")


class CriticInputCapture(Contract):
    """Exact pre-provider context; planned is not evidence of SDK dispatch.

    payload_sha256 hashes canonical semantic JSON, not the raw serialized wire
    bytes. Request and selected evidence differ when the critic trims passages.
    """
    request_evidence: tuple[GroundedEvidence, ...]
    selected_evidence: tuple[GroundedEvidence, ...]
    payload_sha256: Sha256
    provider_call_planned: StrictBool


class AuthorizationReceipt(Contract):
    producer: Literal["AuthenticatedRuntime"]
    scope: ScopeBinding
    effective_bitmap_sha256: Sha256
    capture_artifact_sha256: Sha256
    checked_at: AwareDatetime


def capture_artifact_sha256(captures: Mapping[int, CriticInputCapture]) -> str:
    return canonical_json_sha256([{ "trace_step_index": index, "input": capture.model_dump(mode="json")}
        for index, capture in sorted(captures.items())])


def _checked_evidence(evidence, scope):
    ids = [item.citation_id for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("Captured evidence citation IDs must be unique")
    for item in evidence:
        if isinstance(item, PassageEvidence):
            if item.snapshot_id != scope.snapshot.snapshot_id or item.chunk.metadata_origin == "synthetic":
                raise ValueError("Calibration requires snapshot-bound nonfixture passages")
            if item.citation_id != item.chunk.chunk_id or item.citation_id != f"{item.chunk.doc_id}:{item.chunk.start_char}:{item.chunk.end_char}":
                raise ValueError("Captured passages require exact canonical citation offsets")
        elif item.scope != scope or (isinstance(item, AggregateEvidence) and item.coverage != "verified_records"):
            raise ValueError("Captured native evidence scope/coverage differs")


def _selected_from_request(selected, request):
    for item in selected:
        if item in request:
            continue
        if not isinstance(item, PassageEvidence):
            raise ValueError("Native critic inputs must be exact authorized request items")
        valid = False
        for parent in request:
            if not isinstance(parent, PassageEvidence):
                continue
            a, b = parent.chunk, item.chunk
            if (parent.snapshot_id == item.snapshot_id and parent.revision_id == item.revision_id
                    and a.doc_id == b.doc_id and a.source_type == b.source_type and a.title == b.title
                    and a.timestamp == b.timestamp and a.metadata_origin == b.metadata_origin
                    and a.start_char <= b.start_char < b.end_char <= a.end_char
                    and a.text[b.start_char - a.start_char:b.end_char - a.start_char] == b.text):
                valid = True
        if not valid:
            raise ValueError("Selected critic passage is not an exact authorized request subrange")


class CriticEpisode(Contract):
    format: Literal["rag-critic-episode-v1"] = "rag-critic-episode-v1"
    episode_id: Identifier
    manifest: ExperimentManifest
    runtime_case: RuntimeCaseExport
    runtime_case_sha256: Sha256
    critic_model: ModelProvenance
    trace_step_index: Index
    inputs: CriticInputCapture
    authorization: AuthorizationReceipt

    @property
    def decision(self):
        return self.runtime_case.response.trace.steps[self.trace_step_index].critic

    @property
    def disposition(self):
        return "native_checklist" if self.decision and self.decision.facts else "empty_authorized_context"

    @model_validator(mode="after")
    def exact_runtime_capture(self):
        manifest, case = self.manifest, self.runtime_case
        if (self.episode_id != f"{case.run_id}:critic:{self.trace_step_index}"
                or manifest.split != "calibration" or manifest.workflow != "adaptive"
                or self.critic_model != manifest.generation_model
                or case.manifest_id != manifest.manifest_id or case.question_id not in manifest.question_ids
                or case.response.snapshot != manifest.scope.snapshot
                or self.authorization.scope != manifest.scope
                or runtime_case_digest(case) != self.runtime_case_sha256):
            raise ValueError("Critic episodes require exact calibration adaptive runtime identity")
        steps = case.response.trace.steps
        if self.trace_step_index >= len(steps):
            raise ValueError("Critic trace index is absent")
        step = steps[self.trace_step_index]
        if step.tool != "critic" or step.critic is None:
            raise ValueError("An episode requires an actual successful critic decision")
        if step.round_number > manifest.max_rounds:
            raise ValueError("Critic episode exceeds the declared round budget")
        request, selected = self.inputs.request_evidence, self.inputs.selected_evidence
        if tuple(item.citation_id for item in request) != step.evidence_ids:
            raise ValueError("Captured critic request differs from its ordered trace inputs")
        _checked_evidence(request, manifest.scope); _checked_evidence(selected, manifest.scope)
        _selected_from_request(selected, request)
        payload = {"question": case.question, "evidence": [evidence_payload(item) for item in selected]}
        if (self.inputs.payload_sha256 != canonical_json_sha256(payload)
                or self.inputs.provider_call_planned != bool(selected)):
            raise ValueError("Captured critic payload/planned-call identity differs")
        decision = step.critic
        if selected:
            if not decision.facts:
                raise ValueError("Nonempty native critic episodes require a fact checklist; confidence is not calibration")
            ids = {item.citation_id for item in selected}
            docs = {item.chunk.doc_id for item in selected if isinstance(item, PassageEvidence)}
            if (not set(decision.read_more_doc_ids) <= docs
                    or any(not set(fact.citation_ids) <= ids or any(claim.citation_id not in ids for claim in fact.claims)
                           for fact in decision.facts)):
                raise ValueError("Critic checklist references evidence outside the actual selected input")
            for fact in decision.facts:
                matching = tuple(dict.fromkeys(claim.citation_id for claim in fact.claims if claim.answers_requested_fact))
                if (fact.status != "missing" and not fact.claims or tuple(fact.citation_ids) != matching
                        or fact.status == "supported" and not matching
                        or fact.status in {"conflicting", "ambiguous"} and len({(claim.citation_id, claim.statement)
                            for claim in fact.claims if claim.answers_requested_fact}) < 2):
                    raise ValueError("Native fact status/citations must agree with its scoped evidence claims")
            sufficient = all(fact.status == "supported" for fact in decision.facts) and not decision.clarification_question
            gaps = tuple(fact.fact for fact in decision.facts if fact.status != "supported")
            if (sufficient != decision.sufficient or decision.missing_facts != gaps
                    or decision.sufficient and (decision.read_more_doc_ids or decision.suggested_queries)):
                raise ValueError("Native sufficiency must be derived from facts and clarification")
        elif decision.facts or decision.sufficient:
            raise ValueError("Empty authorized context cannot have a supported checklist")
        return self


def import_critic_episodes(selection: CalibrationSelection, manifest: ExperimentManifest,
                          runtime_case: RuntimeCaseExport, captures: Mapping[int, CriticInputCapture], *,
                          authorization_receipt: AuthorizationReceipt,
                          critic_model: ModelProvenance) -> tuple[CriticEpisode, ...]:
    """Import actual successful critic steps only, with exact indexed captures.

    A run stopping before a critic decision has no eligible episode and is not
    transformed into one. Its original budget/error events remain in the case.
    """
    manifest = ExperimentManifest.model_validate(manifest.model_dump(mode="json"))
    runtime_case = RuntimeCaseExport.model_validate(runtime_case.model_dump(mode="json"))
    selected = {q.question_id: q for q in selection.questions}
    if (runtime_case.question_id not in selected or selected[runtime_case.question_id].question != runtime_case.question
            or manifest.split_manifest_sha256 != selection.split_file_sha256
            or not set(manifest.question_ids) <= selected.keys()
            or manifest.split != "calibration" or manifest.workflow != "adaptive"):
        raise ValueError("Stored case is outside the calibration-only selection")
    expected_questions = [RuntimeQuestion(question_id=qid, question=selected[qid].question) for qid in manifest.question_ids]
    if question_set_sha256(expected_questions) != manifest.question_set_sha256:
        raise ValueError("Manifest selected question digest differs from the calibration sample")
    checked = {}
    for index, value in captures.items():
        if type(index) is not int or index < 0:
            raise ValueError("Capture keys must be exact trace step indices")
        checked[index] = CriticInputCapture.model_validate(value.model_dump(mode="json") if isinstance(value, CriticInputCapture) else value)
    if capture_artifact_sha256(checked) != authorization_receipt.capture_artifact_sha256:
        raise ValueError("Authorization receipt belongs to different captured inputs")
    steps = runtime_case.response.trace.steps
    if any(index >= len(steps) or steps[index].tool not in {"critic", "critic:error", "critic:budget"} for index in checked):
        raise ValueError("Capture refers to a noncritic trace step")
    indices = [index for index, step in enumerate(steps) if step.tool == "critic" and step.critic is not None]
    if not set(indices) <= checked.keys():
        raise ValueError("Exact provider inputs are missing; final sources cannot reconstruct them")
    return tuple(CriticEpisode(episode_id=f"{runtime_case.run_id}:critic:{index}", manifest=manifest, runtime_case=runtime_case,
        runtime_case_sha256=runtime_case_digest(runtime_case), critic_model=critic_model,
        trace_step_index=index, inputs=checked[index], authorization=authorization_receipt) for index in indices)


def episode_timeline(episode: CriticEpisode) -> list[dict]:
    """Ordered real events, plus an explicitly identified trace terminal event."""
    trace = episode.runtime_case.response.trace
    events = []
    for index, step in enumerate(trace.steps):
        tool = step.tool
        kind = ("read_more" if tool == "read_more" else "research" if tool == "text_search"
            else "budget" if tool.endswith(":budget") else tool if tool in {"route", "critic", "generate"} else "tool")
        events.append({"event_id": f"{episode.runtime_case.run_id}:step:{index}", "kind": kind,
            "trace_step_index": index, **step.model_dump(mode="json")})
    events.append({"event_id": f"{episode.runtime_case.run_id}:terminal", "kind": "stop",
        "derived_from": "response.trace.stop_reason", "stop_reason": trace.stop_reason,
        "usage": trace.usage.model_dump(mode="json"), "budgets": episode.manifest.budgets.model_dump(mode="json")})
    return events


def episode_observations(episode: CriticEpisode) -> dict:
    """Facts about the trace, never automatically assigned human mistake labels."""
    steps = episode.runtime_case.response.trace.steps
    following = []
    terminal = True
    for step in steps[episode.trace_step_index + 1:]:
        if step.tool == "critic" and step.critic is not None:
            terminal = False
            break
        following.append(step)
    seen, repeated = set(), False
    for step in steps:
        if step.tool in {"text_search", "read_more"}:
            repeated |= bool(seen.intersection(step.evidence_ids))
            seen.update(step.evidence_ids)
    actions = set()
    if any(step.tool == "read_more" for step in following): actions.add("read_more")
    if any(step.tool in {"route", "text_search"} for step in following): actions.add("research")
    if any(step.tool == "generate" for step in following): actions.add("answer")
    stop = episode.runtime_case.response.trace.stop_reason
    if terminal and stop == "clarification": actions.add("clarify")
    if terminal and stop == "no_new_evidence": actions.add("stop_no_new_evidence")
    budget = any(step.tool.endswith(":budget") for step in following) or (terminal and stop == "budget")
    if budget: actions.add("stop_budget")
    return {"observed_follow_up_actions": sorted(actions), "repeated_retrieval_evidence_observed": repeated,
        "budget_stop_observed": budget, "terminal_critic_episode": terminal,
        "legacy_confidence_is_not_a_calibration_target": True}


class HumanFactJudgment(Contract):
    critic_fact_index: Index | None = None  # None records a requested fact omitted by the critic.
    requested_fact: Identifier
    status: FactStatus
    citation_ids: tuple[Identifier, ...] = ()


class HumanJudgment(Contract):
    episode_id: Identifier
    episode_sha256: Sha256
    reviewer_id: Identifier | None = None
    reviewed_at: AwareDatetime | None = None
    label_source: Literal["fresh_human_review"] | None = None
    human_attestation: StrictBool = False
    expected_facts: tuple[HumanFactJudgment, ...] = ()
    expected_action: ExpectedAction | None = None
    repeated_evidence_mistake: StrictBool | None = None
    budget_stop_mistake: StrictBool | None = None
    rationale: Identifier | None = None

    @property
    def complete(self):
        return bool(self.reviewer_id and self.reviewed_at and self.label_source == "fresh_human_review"
            and self.human_attestation and self.expected_facts and self.expected_action
            and self.repeated_evidence_mistake is not None and self.budget_stop_mistake is not None and self.rationale)


class ReviewPacket(Contract):
    format: Literal["rag-critic-calibration-review-packet-v1"] = "rag-critic-calibration-review-packet-v1"
    selection: CalibrationSelection
    episodes: tuple[CriticEpisode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def linked_calibration(self):
        selected = {q.question_id: q.question for q in self.selection.questions}
        ids = [e.episode_id for e in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate critic episode identities")
        for episode in self.episodes:
            if (selected.get(episode.runtime_case.question_id) != episode.runtime_case.question
                    or episode.manifest.split_manifest_sha256 != self.selection.split_file_sha256):
                raise ValueError("Episode differs from the frozen calibration sample")
        return self


def blank_human_judgments(packet: ReviewPacket) -> tuple[HumanJudgment, ...]:
    return tuple(HumanJudgment(episode_id=e.episode_id,
        episode_sha256=canonical_json_sha256(e.model_dump(mode="json"))) for e in packet.episodes)


def _write_new(path: Path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def export_review_packet(packet: ReviewPacket, output: Path) -> dict:
    packet = ReviewPacket.model_validate(packet.model_dump(mode="json"))
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(output / "packet.json", packet.model_dump(mode="json"))
    _write_new(output / "human_judgments.json", [row.model_dump(mode="json") for row in blank_human_judgments(packet)])
    _write_new(output / "human_judgment_schema.json", HumanJudgment.model_json_schema())
    _write_new(output / "timeline.json", {e.episode_id: episode_timeline(e) for e in packet.episodes})
    _write_new(output / "observations.json", {e.episode_id: episode_observations(e) for e in packet.episodes})
    return {"packet_sha256": file_sha256(output / "packet.json"), "human_labels_present": False,
        "calibration_accepted": False, "episode_count": len(packet.episodes)}


def analyze_reviews(packet: ReviewPacket, judgments: Sequence[HumanJudgment]) -> dict:
    """No labels means unavailable metrics, never zero observed mistakes.

    Metrics count reviewer/episode judgments. Multiple reviewers are disclosed;
    this routine makes no inter-rater agreement or calibrated-confidence claim.
    """
    packet = ReviewPacket.model_validate(packet.model_dump(mode="json"))
    rows = tuple(HumanJudgment.model_validate(row.model_dump(mode="json") if isinstance(row, HumanJudgment) else row)
                 for row in judgments)
    episodes = {e.episode_id: e for e in packet.episodes}
    keys, complete, covered, opinions = set(), [], set(), defaultdict(set)
    errors = Counter(); applicable = Counter()
    metric_names = ("false_sufficient", "false_insufficient", "conflict_mistakes", "ambiguity_mistakes",
                    "missing_fact_mistakes", "repeated_evidence_mistakes", "budget_stop_mistakes", "follow_up_action_mistakes")
    for row in rows:
        if row.episode_id not in episodes:
            raise ValueError("Human judgment references an unknown episode")
        episode = episodes[row.episode_id]
        if row.episode_sha256 != canonical_json_sha256(episode.model_dump(mode="json")):
            raise ValueError("Human judgment belongs to different runtime evidence/decision")
        if not row.complete:
            continue
        key = (row.episode_id, row.reviewer_id)
        if key in keys:
            raise ValueError("Duplicate reviewer/episode judgment")
        keys.add(key); covered.add(row.episode_id)
        facts = episode.decision.facts
        indices = [fact.critic_fact_index for fact in row.expected_facts if fact.critic_fact_index is not None]
        if len(indices) != len(set(indices)) or set(indices) != set(range(len(facts))):
            raise ValueError("Human review must cover each critic fact exactly once, plus any omitted facts")
        supplied = {item.citation_id for item in episode.inputs.selected_evidence}
        for fact in row.expected_facts:
            if fact.critic_fact_index is not None and fact.requested_fact != facts[fact.critic_fact_index].fact:
                raise ValueError("Indexed human judgments must identify the exact critic fact label")
            if (not set(fact.citation_ids) <= supplied
                    or (fact.status in {"supported", "conflicting"} and not fact.citation_ids)):
                raise ValueError("Human fact judgments must cite the actual critic inputs")
        human_sufficient = all(f.status == "supported" for f in row.expected_facts)
        if (row.expected_action == "answer" and not human_sufficient
                or row.expected_action == "clarify" and not any(f.status == "ambiguous" for f in row.expected_facts)):
            raise ValueError("Human action contradicts the requested-fact judgments")
        opinions[row.episode_id].add(canonical_json_sha256({"facts": [f.model_dump(mode="json") for f in row.expected_facts],
            "action": row.expected_action, "repeated": row.repeated_evidence_mistake, "budget": row.budget_stop_mistake}))
        observed = episode_observations(episode)
        if row.budget_stop_mistake and not observed["budget_stop_observed"]:
            raise ValueError("Cannot judge a budget-stop mistake without an observed budget stop")
        if row.repeated_evidence_mistake and not observed["repeated_retrieval_evidence_observed"]:
            raise ValueError("Cannot judge repeated evidence without an observed repeated retrieval")
        if episode.disposition == "native_checklist":
            errors["false_sufficient"] += int(episode.decision.sufficient and not human_sufficient)
            errors["false_insufficient"] += int(not episode.decision.sufficient and human_sufficient)
            for name, status in (("conflict_mistakes", "conflicting"), ("ambiguity_mistakes", "ambiguous")):
                errors[name] += int(any((f.status == status) != (facts[f.critic_fact_index].status == status
                    if f.critic_fact_index is not None else False) for f in row.expected_facts))
            errors["missing_fact_mistakes"] += int(any(f.status == "missing" and (f.critic_fact_index is None
                or facts[f.critic_fact_index].status != "missing") for f in row.expected_facts))
        errors["repeated_evidence_mistakes"] += int(row.repeated_evidence_mistake)
        errors["budget_stop_mistakes"] += int(row.budget_stop_mistake)
        applicable["repeated_evidence_mistakes"] += int(observed["repeated_retrieval_evidence_observed"])
        applicable["budget_stop_mistakes"] += int(observed["budget_stop_observed"])
        errors["follow_up_action_mistakes"] += int(row.expected_action not in observed["observed_follow_up_actions"])
        complete.append(row)
    native_n = sum(episodes[row.episode_id].disposition == "native_checklist" for row in complete)
    denominators = {name: applicable[name] if name in {"repeated_evidence_mistakes", "budget_stop_mistakes"}
        else len(complete) if name == "follow_up_action_mistakes" else native_n for name in metric_names}
    sampled = {q.question_id for q in packet.selection.questions}
    captured = {e.runtime_case.question_id for e in packet.episodes}
    return {"format": "rag-critic-calibration-analysis-v1", "packet_sha256": canonical_json_sha256(packet.model_dump(mode="json")),
        "judgments_sha256": canonical_json_sha256([row.model_dump(mode="json") for row in rows]),
        "episode_count": len(episodes), "human_review_count": len(complete), "native_checklist_review_count": native_n,
        "reviewer_ids": sorted({row.reviewer_id for row in complete}),
        "reviewers_per_episode": {key: sum(row.episode_id == key for row in complete) for key in sorted(episodes)},
        "unreviewed_episode_ids": sorted(episodes.keys() - covered), "uncaptured_question_ids": sorted(sampled - captured),
        "unresolved_disagreement_episode_ids": sorted(key for key, values in opinions.items() if len(values) > 1),
        "human_review_complete": covered == episodes.keys() and captured == sampled and bool(complete),
        "metrics": {name: {"mistakes": errors[name] if denominators[name] else None, "denominator": denominators[name]}
            for name in metric_names}, "confidence_calibration_performed": False, "inter_rater_agreement_claimed": False,
        "human_origin_is_reviewer_attested_not_authenticated": True, "accepted": False}


class HumanReviewReport(Contract):
    label_source: Literal["fresh_human_review"]
    reviewer_id: Identifier
    reviewed_at: AwareDatetime
    human_attestation: StrictBool
    packet_sha256: Sha256
    judgments_sha256: Sha256
    analysis_sha256: Sha256
    decision: Literal["ready_for_configuration_freeze", "changes_required"]
    findings_and_justification: Identifier

    @model_validator(mode="after")
    def actual_human_attestation(self):
        if not self.human_attestation:
            raise ValueError("Human findings review must be explicitly attested")
        return self


def build_freeze_proposal(packet: ReviewPacket, judgments: Sequence[HumanJudgment], review: HumanReviewReport, *,
                          manifests: Sequence[ExperimentManifest], source_file_sha256: Mapping[str, str],
                          calibration_artifact: Path) -> dict:
    """Require fresh complete attested human work; never grant final acceptance.

    calibration_artifact must already contain this exact packet, labels, analysis
    and human findings report. Root separately reviews/accepts the proposal.
    The relative artifact shape matches experiment_inputs._final_gate.
    """
    packet = ReviewPacket.model_validate(packet.model_dump(mode="json"))
    judgments = tuple(HumanJudgment.model_validate(row.model_dump(mode="json") if isinstance(row, HumanJudgment) else row)
                      for row in judgments)
    analysis = analyze_reviews(packet, judgments)
    review = HumanReviewReport.model_validate(review.model_dump(mode="json"))
    if (not analysis["human_review_complete"] or not analysis["native_checklist_review_count"]
            or analysis["unresolved_disagreement_episode_ids"]
            or review.decision != "ready_for_configuration_freeze"
            or review.packet_sha256 != analysis["packet_sha256"]
            or review.judgments_sha256 != analysis["judgments_sha256"]
            or review.analysis_sha256 != canonical_json_sha256(analysis)):
        raise ValueError("Configuration freeze requires complete actual human labels and reviewed findings")
    required = {"src/agents/search_critic.py", "src/agents/search_agent.py", "src/contracts.py"}
    if not required <= source_file_sha256.keys():
        raise ValueError("Freeze must pin critic, orchestration and evidence contracts")
    for name, digest in source_file_sha256.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Source versions must be relative paths with exact SHA256 values")
    checked = [ExperimentManifest.model_validate(m.model_dump(mode="json")) for m in manifests]
    settings = {}
    adaptive_settings = {scientific_settings_sha256(e.manifest) for e in packet.episodes}
    critic_models = {e.critic_model for e in packet.episodes}
    if len(adaptive_settings) != 1 or len(critic_models) != 1:
        raise ValueError("Calibration episodes mix scientific settings or critic models")
    for manifest in checked:
        if manifest.workflow in settings:
            raise ValueError("Freeze requires one settings identity per workflow")
        if manifest.workflow == "adaptive" and scientific_settings_sha256(manifest) not in adaptive_settings:
            raise ValueError("Adaptive final settings differ from the reviewed calibration settings")
        settings[manifest.workflow] = scientific_settings_sha256(manifest)
    if "adaptive" not in settings:
        raise ValueError("Critic calibration must bind adaptive scientific settings")
    adaptive = next(manifest for manifest in checked if manifest.workflow == "adaptive")
    shared = ("code", "scope", "scope_fingerprint_sha256", "generation_model", "generator_prompt_sha256",
              "router_prompt_sha256", "comparison_config_sha256", "retrieval", "budgets", "temperature")
    if any(any(getattr(manifest, name) != getattr(adaptive, name) for name in shared)
           for manifest in checked if manifest.workflow == "fixed"):
        raise ValueError("Fixed/adaptive freeze templates differ in shared scientific settings")
    bundle = {"packet": packet.model_dump(mode="json"), "judgments": [row.model_dump(mode="json") for row in judgments],
        "analysis": analysis, "review": review.model_dump(mode="json")}
    actual, _ = _read_json(calibration_artifact)
    if actual != bundle:
        raise ValueError("Calibration artifact differs from the exact human-reviewed bundle")
    return {"format": "rag-critic-configuration-freeze-proposal-v1", "accepted": False,
        "operator_acceptance_required": True, "settings_sha256": settings,
        "calibration_artifact": calibration_artifact.name, "calibration_sha256": file_sha256(calibration_artifact),
        "calibration_selection": packet.selection.model_dump(mode="json"),
        "source_file_sha256": dict(source_file_sha256),
        "critic_models": [m.model_dump(mode="json") for m in critic_models],
        "scientific_manifests": [m.model_dump(mode="json") for m in checked],
        "human_reviewer_ids": analysis["reviewer_ids"], "human_review_report_sha256": canonical_json_sha256(review.model_dump(mode="json"))}

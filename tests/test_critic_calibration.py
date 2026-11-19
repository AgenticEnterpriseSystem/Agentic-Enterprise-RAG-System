"""Offline protocol tests. All review rows here are synthetic UNIT TEST metadata.

No fixture judgments are exported to the actual calibration work directory and
no accepted freeze can be produced by this module, even in positive tests.
"""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess

import pytest

from src.contracts import (
    Answer, Chunk, CriticDecision, EvidenceClaim, FactAssessment, PassageEvidence,
    QueryResponse, ScopeBinding, SnapshotRef, Trace, TraceStep, Usage,
)
from src.evaluation.critic_calibration import (
    AuthorizationReceipt, CriticInputCapture, HumanFactJudgment, HumanJudgment,
    CriticEpisode,
    HumanReviewReport, ReviewPacket, analyze_reviews, blank_human_judgments,
    build_freeze_proposal, capture_artifact_sha256, episode_timeline,
    export_review_packet, import_critic_episodes, recheck_selection, select_calibration_questions,
)
from src.evaluation.experiment_contracts import (
    CodeProvenance, ExperimentManifest, GenerationBudgets, ModelProvenance,
    RetrievalProvenance, RuntimeCaseExport,
)
from src.evaluation.experiment_inputs import (
    RuntimeQuestion, _final_gate, canonical_json_sha256, file_sha256, question_set_sha256,
)

NOW = "2026-11-14T12:00:00+00:00"


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Offline CAL protocol checks cannot execute external operations")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "run", denied)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def selection_files(tmp_path, count=1):
    rows = [{"question_id": "q_dev", "question": "Development fixture?", "question_type": "basic"},
        {"question_id": "q_final", "question": "Untouched final fixture?", "question_type": "basic"}]
    rows += [{"question_id": f"q_cal_{i:02}", "question": f"What does note {i} say?", "question_type": "basic" if i % 2 else "semantic"}
        for i in range(count)]
    questions = write(tmp_path / "questions.json", rows)
    splits = write(tmp_path / "splits.json", {"splits": {"development": ["q_dev"], "final": ["q_final"],
        "calibration": [r["question_id"] for r in rows[2:]]}})
    return questions, splits


def passage(doc, text, start=0):
    cid = f"{doc}:{start}:{start + len(text)}"
    return PassageEvidence(citation_id=cid, snapshot_id="fixture-snapshot", revision_id="b" * 64,
        retrieval_method="hybrid", score=1,
        chunk=Chunk(chunk_id=cid, doc_id=doc, source_type="github", title="Fixture note",
            text=text, start_char=start, end_char=start + len(text), metadata_origin="upstream"))


def decision(status, item):
    return CriticDecision(sufficient=status == "supported", confidence=0,
        reason="Synthetic unit-test checklist", missing_facts=() if status == "supported" else ("limit",),
        facts=(FactAssessment(fact="limit", status=status,
            citation_ids=(item.citation_id,) if status != "missing" else (),
            claims=(EvidenceClaim(citation_id=item.citation_id, statement="10 MiB", scope_reason="requested limit",
                answers_requested_fact=True),) if status != "missing" else ()),),
        read_more_doc_ids=(item.chunk.doc_id,) if status == "missing" else ())


@pytest.fixture
def captured(tmp_path):
    questions, splits = selection_files(tmp_path)
    selection = select_calibration_questions(questions, splits, 1)
    question = selection.questions[0]
    snapshot = SnapshotRef(snapshot_id="fixture-snapshot", corpus_sha256="a" * 64,
        catalog_sha256="b" * 64, document_count=2)
    scope = ScopeBinding(scope_id="fixture-scope", user_id="fixture-user", snapshot=snapshot,
        policy_revision=7, session_version=0)
    model = ModelProvenance(provider="fixture", model_id="fixture-checklist-model", revision="fixture-v1")
    manifest = ExperimentManifest(experiment_id="fixture-calibration", manifest_id="fixture-adaptive",
        workflow="adaptive", created_at=NOW, code=CodeProvenance(commit="a" * 40), split="calibration",
        split_manifest_sha256=selection.split_file_sha256,
        question_set_sha256=question_set_sha256([RuntimeQuestion(question_id=question.question_id, question=question.question)]),
        question_ids=(question.question_id,), scope=scope, scope_fingerprint_sha256="c" * 64,
        generation_model=model, generator_prompt_sha256="d" * 64, critic_prompt_sha256="e" * 64,
        router_prompt_sha256="f" * 64, comparison_config_sha256="1" * 64,
        retrieval=RetrievalProvenance(retriever_id="fixture-hybrid", snapshot=snapshot,
            index_manifest_sha256="2" * 64, config_sha256="3" * 64),
        budgets=GenerationBudgets(top_k=5, max_question_chars=8000, max_context_chars=24000,
            max_output_tokens=1024, max_run_tokens=16000, max_run_seconds=120, max_run_cost_usd=.05), max_rounds=3)
    a, b = passage("doc-a", "Background only."), passage("doc-a", "The limit is 10 MiB.", 20)
    d1, d2 = decision("missing", a), decision("supported", b)
    steps = (TraceStep(round_number=1, query=question.question, tool="text_search", evidence_ids=(a.citation_id,), elapsed_ms=1),
        TraceStep(round_number=1, query=question.question, tool="critic", evidence_ids=(a.citation_id,), critic=d1, elapsed_ms=2),
        TraceStep(round_number=2, query=question.question, tool="read_more", evidence_ids=(b.citation_id,), elapsed_ms=3),
        TraceStep(round_number=2, query="limit note", tool="text_search", evidence_ids=(a.citation_id,), elapsed_ms=4),
        TraceStep(round_number=2, query=question.question, tool="critic", evidence_ids=(a.citation_id, b.citation_id), critic=d2, elapsed_ms=5),
        TraceStep(round_number=2, query=question.question, tool="generate", evidence_ids=(b.citation_id,), elapsed_ms=6))
    response = QueryResponse(snapshot=snapshot, answer=Answer(status="answered", text="10 MiB", citation_ids=(b.citation_id,)),
        sources=(a, b), trace=Trace(run_id="fixture-run", steps=steps, stop_reason="sufficient", usage=Usage(estimated_cost_usd=.01)))
    import hashlib
    case = RuntimeCaseExport(manifest_id=manifest.manifest_id, run_id="fixture-run", question_id=question.question_id,
        question=question.question, question_sha256=hashlib.sha256(question.question.encode()).hexdigest(),
        response=response, generation_evidence=(b,))
    from src.generation.generator import evidence_payload
    captures = {index: CriticInputCapture(request_evidence=request, selected_evidence=selected,
        payload_sha256=canonical_json_sha256({"question": case.question, "evidence": [evidence_payload(e) for e in selected]}),
        provider_call_planned=True) for index, request, selected in [(1, (a,), (a,)), (4, (a, b), (b,))]}
    receipt = AuthorizationReceipt(producer="AuthenticatedRuntime", scope=scope, effective_bitmap_sha256="4" * 64,
        capture_artifact_sha256=capture_artifact_sha256(captures), checked_at=NOW)
    episodes = import_critic_episodes(selection, manifest, case, captures,
        authorization_receipt=receipt, critic_model=model)
    return selection, manifest, case, captures, receipt, episodes


def test_selection_is_deterministic_balanced_calibration_only_and_rechecks_bytes(tmp_path):
    questions, splits = selection_files(tmp_path, 6)
    selected = select_calibration_questions(questions, splits, 4)
    assert [q.question_id for q in selected.questions] == ["q_cal_01", "q_cal_00", "q_cal_03", "q_cal_02"]
    assert all(q.question_id.startswith("q_cal") for q in selected.questions)
    assert selected == select_calibration_questions(questions, splits, 4)
    recheck_selection(selected, questions, splits)
    questions.write_text(questions.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"): recheck_selection(selected, questions, splits)


@pytest.mark.parametrize("count", [True, 0, -1, 99])
def test_invalid_counts_do_not_select_final_or_dev(tmp_path, count):
    questions, splits = selection_files(tmp_path)
    with pytest.raises(ValueError): select_calibration_questions(questions, splits, count)


def test_disjoint_split_validation_and_no_gold_or_reference_in_runtime_selection(tmp_path):
    questions, splits = selection_files(tmp_path)
    rows = json.loads(questions.read_text()); rows[2].update(answer="MUST NOT FORWARD", expected_doc_ids=["gold"])
    write(questions, rows)
    selected = select_calibration_questions(questions, splits, 1)
    assert "MUST NOT FORWARD" not in selected.model_dump_json()
    split = json.loads(splits.read_text()); split["splits"]["final"].append("q_cal_00"); write(splits, split)
    with pytest.raises(ValueError, match="disjoint"): select_calibration_questions(questions, splits, 1)


def test_exact_episode_keeps_read_more_research_and_terminal_budget_identities(captured):
    _, _, _, _, _, episodes = captured
    assert len(episodes) == 2 and episodes[0].decision.sufficient is False and episodes[1].decision.sufficient is True
    timeline = episode_timeline(episodes[0])
    assert timeline[2]["kind"] == "read_more" and timeline[3]["kind"] == "research"
    assert timeline[2]["event_id"] == "fixture-run:step:2"
    assert timeline[-1]["derived_from"] == "response.trace.stop_reason"
    assert timeline[-1]["budgets"]["max_run_cost_usd"] == .05
    assert all(e.disposition == "native_checklist" for e in episodes)


@pytest.mark.parametrize("change", ["fixed", "final", "missing_capture", "wrong_index", "wrong_payload", "wrong_scope", "wrong_question_digest"])
def test_fixed_final_or_unproven_inputs_cannot_be_relabelled_critic_calibration(captured, change):
    selection, manifest, case, captures, receipt, episodes = captured
    if change == "fixed": manifest = manifest.model_copy(update={"workflow": "fixed", "max_rounds": 1, "critic_prompt_sha256": None})
    elif change == "final": manifest = manifest.model_copy(update={"split": "final"})
    elif change == "missing_capture": captures = {1: captures[1]}
    elif change == "wrong_index": captures = {0: captures[1], 4: captures[4]}
    elif change == "wrong_payload": captures = {1: captures[1].model_copy(update={"payload_sha256": "0" * 64}), 4: captures[4]}
    elif change == "wrong_scope": receipt = receipt.model_copy(update={"scope": receipt.scope.model_copy(update={"user_id": "other"})})
    else: manifest = manifest.model_copy(update={"question_set_sha256": "0" * 64})
    receipt = receipt.model_copy(update={"capture_artifact_sha256": capture_artifact_sha256(captures)})
    with pytest.raises(ValueError): import_critic_episodes(selection, manifest, case, captures,
        authorization_receipt=receipt, critic_model=episodes[0].critic_model)


def test_authorized_trimmed_subrange_is_exact_and_forged_text_rejected(captured):
    selection, manifest, case, captures, receipt, episodes = captured
    parent = captures[1].request_evidence[0]
    from src.generation.generator import evidence_payload
    child = passage(parent.chunk.doc_id, parent.chunk.text[2:8], parent.chunk.start_char + 2)
    capture = captures[1].model_copy(update={"selected_evidence": (child,), "payload_sha256": canonical_json_sha256(
        {"question": case.question, "evidence": [evidence_payload(child)]})})
    # The first decision is missing and contains no unsupported selected citations.
    captures = {1: capture, 4: captures[4]}
    receipt = receipt.model_copy(update={"capture_artifact_sha256": capture_artifact_sha256(captures)})
    actual = import_critic_episodes(selection, manifest, case, captures, authorization_receipt=receipt,
        critic_model=episodes[0].critic_model)
    assert actual[0].inputs.selected_evidence[0] == child
    forged = passage(parent.chunk.doc_id, "FORGED", 2)
    captures[1] = capture.model_copy(update={"selected_evidence": (forged,), "payload_sha256": canonical_json_sha256(
        {"question": case.question, "evidence": [evidence_payload(forged)]})})
    receipt = receipt.model_copy(update={"capture_artifact_sha256": capture_artifact_sha256(captures)})
    with pytest.raises(ValueError, match="subrange"): import_critic_episodes(selection, manifest, case, captures,
        authorization_receipt=receipt, critic_model=episodes[0].critic_model)


def packet(captured):
    return ReviewPacket(selection=captured[0], episodes=captured[-1])


def filled_fixture_rows(value):
    """SYNTHETIC metadata exercising a human-schema shape; never actual labels."""
    rows = []
    for episode, blank in zip(value.episodes, blank_human_judgments(value)):
        fact = episode.decision.facts[0]
        rows.append(blank.model_copy(update={"reviewer_id": "SYNTHETIC-UNIT-TEST-REVIEWER", "reviewed_at": datetime.fromisoformat(NOW),
            "label_source": "fresh_human_review", "human_attestation": True,
            "expected_facts": (HumanFactJudgment(critic_fact_index=0, requested_fact="limit", status=fact.status,
                citation_ids=fact.citation_ids),), "expected_action": "answer" if episode.decision.sufficient else "read_more",
            "repeated_evidence_mistake": False, "budget_stop_mistake": False,
            "rationale": "Synthetic unit-test label, not an actual human calibration finding."}))
    return rows


def test_review_export_labels_are_blank_exclusive_and_unscored(captured, tmp_path):
    value = packet(captured); output = tmp_path / "new-packet"
    result = export_review_packet(value, output)
    rows = json.loads((output / "human_judgments.json").read_text())
    assert result["human_labels_present"] is False and result["calibration_accepted"] is False
    assert all(row["reviewer_id"] is None and row["expected_action"] is None and not row["human_attestation"] for row in rows)
    analysis = analyze_reviews(value, [HumanJudgment.model_validate(r) for r in rows])
    assert analysis["human_review_complete"] is False
    assert analysis["metrics"]["false_sufficient"] == {"mistakes": None, "denominator": 0}
    with pytest.raises(FileExistsError): export_review_packet(value, output)


def test_real_label_schema_rejects_ai_and_development_fixture_conversion(captured):
    blank = blank_human_judgments(packet(captured))[0].model_dump(mode="json")
    with pytest.raises(ValueError): HumanJudgment.model_validate({**blank, "label_source": "ai_qualitative"})
    with pytest.raises(ValueError): HumanJudgment.model_validate({**blank, "expected_sufficient": True, "fixture": True})


def test_false_sufficient_insufficient_conflict_ambiguity_and_explicit_stop_mistakes(captured):
    value = packet(captured); rows = filled_fixture_rows(value)
    # First critic says missing although the reviewer marks the supplied passage supported.
    cid = value.episodes[0].inputs.selected_evidence[0].citation_id
    rows[0] = rows[0].model_copy(update={"expected_facts": (HumanFactJudgment(critic_fact_index=0,
        requested_fact="limit", status="supported", citation_ids=(cid,)),), "expected_action": "answer",
        "repeated_evidence_mistake": True, "budget_stop_mistake": False})
    # Second critic says supported; reviewer judges the available text ambiguous.
    rows[1] = rows[1].model_copy(update={"expected_facts": (HumanFactJudgment(critic_fact_index=0,
        requested_fact="limit", status="ambiguous", citation_ids=rows[1].expected_facts[0].citation_ids),),
        "expected_action": "clarify"})
    result = analyze_reviews(value, rows)
    assert result["human_review_complete"] is True and result["accepted"] is False
    for name in ["false_sufficient", "false_insufficient", "ambiguity_mistakes", "repeated_evidence_mistakes"]:
        assert result["metrics"][name] == {"mistakes": 1, "denominator": 2}
    assert result["metrics"]["budget_stop_mistakes"] == {"mistakes": None, "denominator": 0}
    assert result["confidence_calibration_performed"] is False


@pytest.mark.parametrize("change", ["episode_hash", "missing_fact", "foreign_citation", "duplicate", "contradictory_action"])
def test_invalid_human_rows_cannot_become_findings(captured, change):
    value = packet(captured); rows = filled_fixture_rows(value)
    if change == "episode_hash": rows[0] = rows[0].model_copy(update={"episode_sha256": "0" * 64})
    elif change == "missing_fact": rows[0] = rows[0].model_copy(update={"expected_facts": (HumanFactJudgment(
        critic_fact_index=8, requested_fact="other", status="missing"),)})
    elif change == "foreign_citation": rows[0] = rows[0].model_copy(update={"expected_facts": (HumanFactJudgment(
        critic_fact_index=0, requested_fact="limit", status="supported", citation_ids=("foreign",)),)})
    elif change == "duplicate": rows.append(rows[0])
    else: rows[0] = rows[0].model_copy(update={"expected_action": "answer"})
    with pytest.raises(ValueError): analyze_reviews(value, rows)


def freeze_fixture(value, rows, tmp_path):
    analysis = analyze_reviews(value, rows)
    review = HumanReviewReport(label_source="fresh_human_review", reviewer_id="SYNTHETIC-UNIT-TEST-REVIEWER",
        reviewed_at=NOW, human_attestation=True, packet_sha256=analysis["packet_sha256"],
        judgments_sha256=analysis["judgments_sha256"], analysis_sha256=canonical_json_sha256(analysis),
        decision="ready_for_configuration_freeze", findings_and_justification="Synthetic unit test only; no real findings.")
    bundle = {"packet": value.model_dump(mode="json"), "judgments": [r.model_dump(mode="json") for r in rows],
        "analysis": analysis, "review": review.model_dump(mode="json")}
    path = write(tmp_path / "SYNTHETIC-NOT-ACTUAL-CALIBRATION.json", bundle)
    pins = {name: "a" * 64 for name in ["src/agents/search_critic.py", "src/agents/search_agent.py", "src/contracts.py"]}
    return review, path, pins


def test_freeze_proposal_requires_human_work_and_never_grants_final_acceptance(captured, tmp_path):
    value = packet(captured); rows = filled_fixture_rows(value)
    review, path, pins = freeze_fixture(value, rows, tmp_path)
    result = build_freeze_proposal(value, rows, review, manifests=[value.episodes[0].manifest],
        source_file_sha256=pins, calibration_artifact=path)
    assert result["accepted"] is False and result["operator_acceptance_required"] is True
    assert result["calibration_sha256"] == file_sha256(path)
    freeze_path = write(tmp_path / "UNACCEPTED-FREEZE-PROPOSAL.json", result)
    with pytest.raises(ValueError, match="operator acceptance"):
        _final_gate(value.episodes[0].manifest, freeze_path, {})
    with pytest.raises(ValueError, match="human labels"):
        build_freeze_proposal(value, blank_human_judgments(value), review,
            manifests=[value.episodes[0].manifest], source_file_sha256=pins, calibration_artifact=path)


@pytest.mark.parametrize("change", ["artifact", "prompt", "policy", "source", "review_hash", "unresolved_disagreement"])
def test_freeze_rejects_unreviewed_settings_findings_or_disagreement(captured, tmp_path, change):
    value = packet(captured); rows = filled_fixture_rows(value)
    review, path, pins = freeze_fixture(value, rows, tmp_path)
    manifest = value.episodes[0].manifest
    if change == "artifact": write(path, {"fixture": "not the reviewed labels"})
    elif change == "prompt": manifest = manifest.model_copy(update={"critic_prompt_sha256": "0" * 64})
    elif change == "policy": manifest = manifest.model_copy(update={"scope": manifest.scope.model_copy(update={"policy_revision": 8})})
    elif change == "source": pins = {}
    elif change == "review_hash": review = review.model_copy(update={"analysis_sha256": "0" * 64})
    else:
        other = rows[0].model_copy(update={"reviewer_id": "SYNTHETIC-SECOND-REVIEWER", "repeated_evidence_mistake": True})
        rows.append(other)
        review, path, pins = freeze_fixture(value, rows, tmp_path)
    with pytest.raises(ValueError): build_freeze_proposal(value, rows, review, manifests=[manifest],
        source_file_sha256=pins, calibration_artifact=path)


def test_budget_stop_is_scored_only_when_observed_and_reviewed(captured):
    selection, manifest, case, captures, receipt, episodes = captured
    response = case.response.model_copy(update={"answer": Answer(status="insufficient_evidence", text="Budget reached."),
        "trace": case.response.trace.model_copy(update={"stop_reason": "budget"})})
    case = case.model_copy(update={"response": response})
    episodes = import_critic_episodes(selection, manifest, case, captures,
        authorization_receipt=receipt, critic_model=episodes[0].critic_model)
    value = ReviewPacket(selection=selection, episodes=episodes)
    rows = filled_fixture_rows(value)
    rows[1] = rows[1].model_copy(update={"budget_stop_mistake": True})
    report = analyze_reviews(value, rows)
    assert report["metrics"]["budget_stop_mistakes"] == {"mistakes": 1, "denominator": 1}
    ordinary = packet(captured)
    rows = filled_fixture_rows(ordinary)
    rows[0] = rows[0].model_copy(update={"budget_stop_mistake": True})
    with pytest.raises(ValueError, match="without an observed budget stop"): analyze_reviews(ordinary, rows)


def test_empty_context_is_explicitly_retained_without_confidence_or_native_fact_metrics(captured):
    selection, manifest, case, captures, receipt, episodes = captured
    empty = CriticDecision(sufficient=False, confidence=0, reason="No authorized evidence fits.",
        missing_facts=("Evidence answering the question",), suggested_queries=(case.question,))
    steps = list(case.response.trace.steps)
    steps[1] = steps[1].model_copy(update={"critic": empty})
    response = case.response.model_copy(update={"trace": case.response.trace.model_copy(update={"steps": tuple(steps)})})
    case = case.model_copy(update={"response": response})
    captures = dict(captures)
    captures[1] = captures[1].model_copy(update={"selected_evidence": (), "provider_call_planned": False,
        "payload_sha256": canonical_json_sha256({"question": case.question, "evidence": []})})
    receipt = receipt.model_copy(update={"capture_artifact_sha256": capture_artifact_sha256(captures)})
    episodes = import_critic_episodes(selection, manifest, case, captures, authorization_receipt=receipt,
        critic_model=episodes[0].critic_model)
    assert episodes[0].disposition == "empty_authorized_context"
    value = ReviewPacket(selection=selection, episodes=episodes)
    rows = filled_fixture_rows(packet(captured))
    rows = [row.model_copy(update={"episode_sha256": blank.episode_sha256})
        for row, blank in zip(rows, blank_human_judgments(value))]
    rows[0] = rows[0].model_copy(update={
        "expected_facts": (HumanFactJudgment(requested_fact="limit", status="missing"),)})
    report = analyze_reviews(value, rows)
    assert report["human_review_count"] == 2 and report["native_checklist_review_count"] == 1
    assert report["metrics"]["false_sufficient"]["denominator"] == 1


def test_captured_budget_failure_without_decision_is_not_a_fabricated_critic_episode(captured):
    selection, manifest, case, captures, receipt, episodes = captured
    step = TraceStep(round_number=1, query=case.question, tool="critic:budget", elapsed_ms=1)
    response = case.response.model_copy(update={"answer": Answer(status="insufficient_evidence", text="Budget reached."),
        "trace": case.response.trace.model_copy(update={"steps": (step,), "stop_reason": "budget"})})
    case = case.model_copy(update={"response": response, "generation_evidence": ()})
    receipt = receipt.model_copy(update={"capture_artifact_sha256": capture_artifact_sha256({})})
    assert import_critic_episodes(selection, manifest, case, {}, authorization_receipt=receipt,
        critic_model=episodes[0].critic_model) == ()


def test_conflict_disagreement_can_be_identified_but_not_automatically_resolved(captured):
    value = packet(captured); rows = filled_fixture_rows(value)
    old = rows[1].expected_facts[0]
    rows[1] = rows[1].model_copy(update={"expected_action": "research",
        "expected_facts": (old.model_copy(update={"status": "conflicting"}),)})
    report = analyze_reviews(value, rows)
    assert report["metrics"]["conflict_mistakes"] == {"mistakes": 1, "denominator": 2}


def test_freeze_pins_matching_fixed_template_and_rejects_unequal_shared_budget(captured, tmp_path):
    value = packet(captured); rows = filled_fixture_rows(value)
    review, path, pins = freeze_fixture(value, rows, tmp_path)
    adaptive = value.episodes[0].manifest
    fixed = adaptive.model_copy(update={"manifest_id": "fixture-fixed", "workflow": "fixed", "max_rounds": 1,
        "critic_prompt_sha256": None})
    result = build_freeze_proposal(value, rows, review, manifests=[fixed, adaptive],
        source_file_sha256=pins, calibration_artifact=path)
    assert set(result["settings_sha256"]) == {"fixed", "adaptive"} and result["accepted"] is False
    fixed = fixed.model_copy(update={"budgets": fixed.budgets.model_copy(update={"top_k": 6})})
    with pytest.raises(ValueError, match="shared scientific"): build_freeze_proposal(value, rows, review,
        manifests=[fixed, adaptive], source_file_sha256=pins, calibration_artifact=path)


def test_supported_fact_requires_an_applicable_claim_not_only_unrelated_background(captured, monkeypatch):
    episode = captured[-1][1]
    original = episode.decision.facts[0]
    unrelated = original.claims[0].model_copy(update={"answers_requested_fact": False,
        "scope_reason": "Different requested metric or event"})
    # Deliberately exercise unchecked nested DTOs, as well as parsed JSON. The
    # normalized production critic would mark this unsupported combination missing.
    invalid_fact = original.model_copy(update={"claims": (unrelated,), "citation_ids": ()})
    bad_decision = episode.decision.model_copy(update={"facts": (invalid_fact,)})
    steps = list(episode.runtime_case.response.trace.steps)
    steps[episode.trace_step_index] = steps[episode.trace_step_index].model_copy(update={"critic": bad_decision})
    bad_case = episode.runtime_case.model_copy(update={"response": episode.runtime_case.response.model_copy(
        update={"trace": episode.runtime_case.response.trace.model_copy(update={"steps": tuple(steps)})})})
    # Isolate this decision guard from the independent runtime-export parser,
    # which also rejects malformed supported/citation DTOs on the ordinary path.
    import src.evaluation.critic_calibration as calibration
    monkeypatch.setattr(calibration, "runtime_case_digest", lambda _case: episode.runtime_case_sha256)
    with pytest.raises(ValueError, match="scoped evidence claims"):
        CriticEpisode(**{**episode.model_dump(), "runtime_case": bad_case,
            "runtime_case_sha256": episode.runtime_case_sha256})


def test_human_index_cannot_silently_label_a_different_requested_fact(captured):
    value = packet(captured); rows = filled_fixture_rows(value)
    wrong = rows[1].expected_facts[0].model_copy(update={"requested_fact": "Unrelated rollout approval"})
    rows[1] = rows[1].model_copy(update={"expected_facts": (wrong,)})
    with pytest.raises(ValueError, match="exact critic fact label"):
        analyze_reviews(value, rows)


def test_critic_model_is_bound_to_the_current_shared_runtime_provider(captured):
    selection, manifest, case, captures, receipt, _ = captured
    with pytest.raises(ValueError, match="runtime identity"):
        import_critic_episodes(selection, manifest, case, captures, authorization_receipt=receipt,
            critic_model=ModelProvenance(provider="unattested", model_id="different-model"))

"""Offline canonical-to-RC2 projections and exact result joins.

Only tiny synthetic experiment files are used. Socket connections fail closed;
the imported candidate parser is pinned source, not a provider or judge runner.
"""

import hashlib
import json
from pathlib import Path
import socket
import stat
from types import SimpleNamespace

import pytest

from src.contracts import AggregateEvidence, Answer, Chunk, PassageEvidence, Usage
from src.evaluation import evaluator_bridge as bridge
from src.evaluation.evaluator_candidate import evaluate_case
from src.evaluation.experiment_contracts import ScoringInput, ScoringReference, runtime_case_digest
from src.evaluation.experiment_io import RunDirectory, export_scoring_inputs
from src.evaluation.verified_judge import make_case
from tests.test_experiment_inputs import make_verified_inputs_fixture
from tests.test_experiment_io import case_for, file_bytes, references, replace, write_complete_run
from tests.test_native_evidence_contracts import record


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("Offline bridge tests must not make network connections")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def verified_fixture(tmp_path):
    return make_verified_inputs_fixture(tmp_path / "verified")


def packet(case, *, answer="SCORING_ONLY_GOLD_SENTINEL", reference_hash="a" * 64):
    return ScoringInput(runtime_case=case, runtime_case_sha256=runtime_case_digest(case),
        reference=ScoringReference(question_id=case.question_id,
            question_sha256=case.question_sha256, reference_set_sha256=reference_hash,
            reference_answer=answer, answer_facts=("SCORING_ONLY_FACT_SENTINEL",)))


def aggregate(manifest, *, count=0, scope=None):
    return AggregateEvidence(citation_id="aggregate:zero", artifact_id="zero",
        scope=scope or manifest.scope, template_id="jira-count", template_version="1",
        count=count, lineage_sha256="b" * 64, coverage="synthetic_fixture",
        created_at=manifest.created_at)


def with_returned(case, evidence):
    return replace(case, response=replace(case.response,
        sources=(*case.response.sources, evidence),
        contract_version="3" if evidence.kind == "record" else case.response.contract_version))


def bundle_fixture(tmp_path, verified_fixture, *, outcomes=None):
    manifest, inputs = verified_fixture
    run = write_complete_run(tmp_path / "run", manifest, inputs, outcomes=outcomes)
    refs = tmp_path / "references.jsonl"
    references(refs, inputs)
    scoring = tmp_path / "scoring"
    export_scoring_inputs(run, refs, scoring)
    output = tmp_path / "projection"
    metadata = bridge.export_projection(run.path, scoring / "scoring_inputs.jsonl", output)
    return run, scoring, output, metadata


def write_candidate_results(tmp_path, metadata, rows, *, audit_mode="none"):
    results, manifest = tmp_path / "results.json", tmp_path / "judge-manifest.json"
    results.write_text(json.dumps(rows) + "\n")
    evaluator = metadata["evaluator"]
    answers = json.loads((tmp_path / "projection" / "answers.json").read_text())
    hashes = {
        "answers": metadata["files"]["answers.json"],
        "questions": metadata["files"]["questions.json"],
        "score_prompt": evaluator["files"][evaluator["score_prompt"]]["sha256"],
        "candidate_code": evaluator["files"]["src/evaluation/evaluator_candidate.py"]["sha256"],
        "verified_code": evaluator["files"]["src/evaluation/verified_judge.py"]["sha256"],
        "v6_audit_code": evaluator["files"]["src/evaluation/verified_judge_v6.py"]["sha256"],
        "runner_code": bridge.EVALUATOR_RUNNER_SHA256}
    if audit_mode == "all":
        hashes["audit_prompt"] = evaluator["files"][evaluator["audit_prompt"]]["sha256"]
    manifest.write_text(json.dumps({"evaluator": "v3-scoring-v6-review-candidate-1",
        "model": "offline-fixture-no-provider", "audit_mode": audit_mode,
        "selected_ids": [row["question_id"] for row in answers], "sha256": hashes}) + "\n")
    return results, manifest


def test_projection_preserves_exact_unicode_text_ids_and_separates_reference(verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    text = "  café\nΔ throughput: 10 MB—not 10,000 MB.\n"
    cid = "doc-one:0:" + str(len(text))
    evidence = PassageEvidence(citation_id=cid, snapshot_id=manifest.scope.snapshot.snapshot_id,
        revision_id=case.generation_evidence[0].revision_id, score=.5, retrieval_method="hybrid",
        chunk=Chunk(chunk_id=cid, doc_id="doc-one", source_type="jira", text=text,
            start_char=0, end_char=len(text), metadata_origin="synthetic"))
    answer_text = "café: 10 MB.\nSee " + cid
    case = replace(case, response=replace(case.response,
        answer=Answer(status="answered", text=answer_text, citation_ids=(cid,)), sources=(evidence,)),
        generation_evidence=(evidence,))
    scoring = packet(case)
    before = scoring.model_dump_json()
    row = bridge.project_scoring_input(scoring, manifest)
    assert row["disposition"] == "eligible"
    assert row["answer"] == {"question_id": case.question_id, "question": case.question,
        "generated_answer": answer_text, "status": "answered", "citation_ids": [cid],
        "evidence": [{"citation_id": cid, "text": text}]}
    assert row["reference"] == {"question_id": case.question_id,
        "gold_answer": "SCORING_ONLY_GOLD_SENTINEL"}
    assert "SCORING_ONLY_" not in json.dumps(row["answer"])
    assert "SCORING_ONLY_" not in json.dumps(row["identity"])
    assert row["identity"] == {"manifest_id": manifest.manifest_id, "run_id": case.run_id,
        "question_id": case.question_id, "question_sha256": case.question_sha256,
        "runtime_case_sha256": runtime_case_digest(case)}
    assert make_case(row["answer"], row["reference"]).evidence == row["answer"]["evidence"]
    assert scoring.model_dump_json() == before
    canonical = json.dumps(case.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert row["identity"]["runtime_case_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert row["identity"]["runtime_case_sha256"] != hashlib.sha256(canonical + b"\n").hexdigest()


def test_only_generation_evidence_is_projected_not_extra_returned_sources(verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    extra = case.generation_evidence[0]
    extra = replace(extra, citation_id="other:0:4", chunk=Chunk(chunk_id="other:0:4",
        doc_id="other", source_type="jira", text="GOLD", start_char=0, end_char=4))
    case = with_returned(case, extra)
    row = bridge.project_scoring_input(packet(case), manifest)
    assert len(case.response.sources) == 2
    assert row["answer"]["evidence"] == [{"citation_id": case.generation_evidence[0].citation_id,
        "text": case.generation_evidence[0].chunk.text}]
    assert "GOLD" not in json.dumps(row["answer"])


@pytest.mark.parametrize("tamper", ["answer", "evidence", "reference_hash", "manifest", "snapshot"])
def test_unchecked_copies_cannot_bypass_canonical_validation(verified_fixture, tamper):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    scoring = packet(case)
    if tamper == "answer":
        changed = case.model_copy(update={"response": case.response.model_copy(update={
            "answer": case.response.answer.model_copy(update={"text": "Tampered answer"})})})
        scoring = scoring.model_copy(update={"runtime_case": changed})
    elif tamper == "evidence":
        changed = case.generation_evidence[0].model_copy(update={"score": .25})
        scoring = scoring.model_copy(update={"runtime_case": case.model_copy(update={
            "generation_evidence": (changed,)})})
    elif tamper == "reference_hash":
        scoring = scoring.model_copy(update={"reference": scoring.reference.model_copy(update={
            "question_sha256": "f" * 64})})
    elif tamper == "manifest":
        manifest = manifest.model_copy(update={"manifest_id": "wrong-manifest"})
    else:
        snap = manifest.scope.snapshot.model_copy(update={"snapshot_id": "wrong-snapshot"})
        manifest = manifest.model_copy(update={"scope": manifest.scope.model_copy(update={"snapshot": snap}),
            "retrieval": manifest.retrieval.model_copy(update={"snapshot": snap})})
    with pytest.raises(ValueError):
        bridge.project_scoring_input(scoring, manifest)


@pytest.mark.parametrize("kind", ["zero_count", "record_with_missing_field"])
def test_native_returned_evidence_remains_unscored_and_exact(verified_fixture, kind):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    native = (aggregate(manifest) if kind == "zero_count" else
        record(SimpleNamespace(binding=manifest.scope), doc="doc-one"))
    case = with_returned(case, native)
    scoring = packet(case)
    before = scoring.model_dump_json()
    row = bridge.project_scoring_input(scoring, manifest)
    assert row["disposition"] == "native_evidence_unsupported"
    assert row["answer"] is None and row["reference"] is None
    assert scoring.model_dump_json() == before
    if kind == "zero_count":
        assert scoring.runtime_case.response.sources[-1].count == 0
    else:
        assert scoring.runtime_case.response.sources[-1].priority is None


def test_native_scope_mismatch_is_integrity_failure_not_an_unsupported_case(verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    wrong = replace(manifest.scope, user_id="another-user", scope_id="another-scope")
    case = with_returned(case, aggregate(manifest, scope=wrong))
    with pytest.raises(ValueError):
        bridge.project_scoring_input(packet(case), manifest)


@pytest.mark.parametrize("outcome,expected", [
    ("clarification", "clarification"), ("error", "runtime_error"), ("budget", "no_generation_input")])
def test_non_generated_outcomes_preserve_identity_status_and_failure(verified_fixture, outcome, expected):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs, outcome=outcome)
    row = bridge.project_scoring_input(packet(case), manifest)
    assert row["disposition"] == expected
    assert row["generation_status"] == case.response.answer.status
    assert row["stop_reason"] == case.response.trace.stop_reason
    assert row["answer"] is None and row["reference"] is None
    assert row["runtime_failure"] == (case.error.model_dump(mode="json") if case.error else None)


def test_reference_answer_absence_is_not_replaced_by_facts(verified_fixture):
    manifest, inputs = verified_fixture
    row = bridge.project_scoring_input(packet(case_for(manifest, inputs), answer=None), manifest)
    assert row["disposition"] == "reference_answer_missing"
    assert row["answer"] is None and row["reference"] is None


def test_generated_abstention_is_eligible_with_original_status(verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    case = replace(case, response=replace(case.response,
        answer=Answer(status="insufficient_evidence", text="Missing the resolution."),
        trace=replace(case.response.trace, stop_reason="no_new_evidence")))
    row = bridge.project_scoring_input(packet(case), manifest)
    assert row["disposition"] == "eligible"
    assert row["answer"]["status"] == "insufficient_evidence"
    assert row["stop_reason"] == "no_new_evidence"


@pytest.mark.parametrize("duplicate", ["question", "run"])
def test_batch_rejects_duplicate_question_or_run_ids(verified_fixture, duplicate):
    manifest, inputs = verified_fixture
    first = case_for(manifest, inputs)
    second = (first if duplicate == "question" else
        case_for(manifest, inputs, manifest.question_ids[1], run_id=first.run_id))
    with pytest.raises(ValueError):
        bridge.project_batch(manifest, (packet(first), packet(second)))


def test_missing_case_requires_explicit_partial_and_keeps_selection_order(verified_fixture):
    manifest, inputs = verified_fixture
    scoring = packet(case_for(manifest, inputs, manifest.question_ids[1]))
    with pytest.raises(ValueError):
        bridge.project_batch(manifest, (scoring,))
    rows = bridge.project_batch(manifest, (scoring,), require_complete=False)
    assert [r["identity"]["question_id"] for r in rows] == list(manifest.question_ids)
    assert rows[0]["disposition"] == "missing_committed_case"
    assert rows[0]["identity"]["runtime_case_sha256"] is None
    assert rows[1]["disposition"] == "eligible"


def test_repeated_question_ids_are_separate_across_manifest_batches(verified_fixture):
    manifest, inputs = verified_fixture
    first = packet(case_for(manifest, inputs))
    other = replace(manifest, manifest_id="other-arm")
    second = packet(replace(first.runtime_case, manifest_id=other.manifest_id, run_id="other-run",
        response=replace(first.runtime_case.response,
            trace=replace(first.runtime_case.response.trace, run_id="other-run"))))
    a = bridge.project_batch(manifest, (first,), require_complete=False)
    b = bridge.project_batch(other, (second,), require_complete=False)
    assert a[0]["identity"]["question_id"] == b[0]["identity"]["question_id"]
    assert a[0]["identity"] != b[0]["identity"]
    with pytest.raises(ValueError):
        bridge.project_batch(manifest, (first, second), require_complete=False)


def test_one_batch_cannot_mix_reference_sets(verified_fixture):
    manifest, inputs = verified_fixture
    packets = tuple(packet(case_for(manifest, inputs, qid), reference_hash=str(i + 1) * 64)
        for i, qid in enumerate(manifest.question_ids))
    with pytest.raises(ValueError):
        bridge.project_batch(manifest, packets)


def test_export_pins_candidate_parser_and_writes_private_immutable_bundle(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    run = write_complete_run(tmp_path / "run", manifest, inputs, reverse=True)
    refs = tmp_path / "references.jsonl"
    references(refs, inputs)
    scoring = tmp_path / "scoring"
    export_scoring_inputs(run, refs, scoring)
    before = file_bytes(run.path), file_bytes(scoring), refs.read_bytes()
    output = tmp_path / "projection"
    metadata = bridge.export_projection(run.path, scoring / "scoring_inputs.jsonl", output)
    assert metadata["selected_cases"] == metadata["committed_cases"] == metadata["eligible_cases"] == 2
    assert metadata["judge_calls"] == 0 and metadata["quality_scored"] is False
    assert metadata["evaluator"]["upstream_commit"] == bridge.EVALUATOR_COMMIT
    assert metadata["candidate_shape_preflight_passed"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (output / "scoring_inputs.jsonl").read_bytes() == (scoring / "scoring_inputs.jsonl").read_bytes()
    assert (output / "runtime_manifest.json").read_bytes() == (run.path / "manifest.json").read_bytes()
    answers = json.loads((output / "answers.json").read_text())
    questions = json.loads((output / "questions.json").read_text())
    assert [row["question_id"] for row in answers] == list(manifest.question_ids)
    assert "SCORING_ONLY_" not in (output / "answers.json").read_text()
    for answer, reference in zip(answers, questions, strict=True):
        assert make_case(answer, reference).question_id == answer["question_id"]
    exported = file_bytes(output)
    with pytest.raises((ValueError, OSError)):
        bridge.export_projection(run.path, scoring / "scoring_inputs.jsonl", output)
    assert file_bytes(output) == exported
    assert (file_bytes(run.path), file_bytes(scoring), refs.read_bytes()) == before


@pytest.mark.parametrize("corruption", ["duplicate_json_field", "tampered_packet", "omitted_committed"])
def test_bad_scoring_input_rejects_before_output_publication(tmp_path, verified_fixture, corruption):
    manifest, inputs = verified_fixture
    run = write_complete_run(tmp_path / "run", manifest, inputs)
    refs = tmp_path / "references.jsonl"
    references(refs, inputs)
    scoring = tmp_path / "scoring"
    export_scoring_inputs(run, refs, scoring)
    path = scoring / "scoring_inputs.jsonl"
    lines = path.read_text().splitlines()
    if corruption == "duplicate_json_field":
        lines[0] = lines[0].replace('{', '{"schema_version":"2",', 1)
    elif corruption == "tampered_packet":
        row = json.loads(lines[0]); row["runtime_case"]["response"]["answer"]["text"] = "Tampered"
        lines[0] = json.dumps(row)
    else:
        lines.pop()
    path.write_text("\n".join(lines) + "\n")
    before = file_bytes(run.path), file_bytes(scoring)
    output = tmp_path / "projection"
    with pytest.raises(ValueError):
        bridge.export_projection(run.path, path, output, allow_partial=True)
    assert not output.exists()
    assert (file_bytes(run.path), file_bytes(scoring)) == before


@pytest.mark.parametrize("corruption", ["duplicate", "unknown", "wrong_input_hash"])
def test_result_binding_rejects_bad_identity_or_input_hash(tmp_path, verified_fixture, corruption):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    qid = verified_fixture[0].question_ids[0]
    rows = [{"question_id": qid, "generation_status": "answered",
        "error": "Judge unavailable", "scores": None}]
    if corruption == "duplicate": rows *= 2
    elif corruption == "unknown": rows[0]["question_id"] = "not-selected"
    results, judge = write_candidate_results(tmp_path, metadata, rows)
    if corruption == "wrong_input_hash":
        value = json.loads(judge.read_text()); value["sha256"]["answers"] = "f" * 64
        judge.write_text(json.dumps(value))
    before = file_bytes(output), results.read_bytes(), judge.read_bytes()
    destination = tmp_path / "bound-results.json"
    with pytest.raises(ValueError):
        bridge.bind_candidate_results(output, results, judge, destination)
    assert not destination.exists()
    assert (file_bytes(output), results.read_bytes(), judge.read_bytes()) == before


def test_missing_and_failed_judges_retain_case_denominator_and_original_results(tmp_path, verified_fixture):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    qid = verified_fixture[0].question_ids[0]
    failure = {"question_id": qid, "generation_status": "answered",
        "error": "Judge unavailable", "scores": None,
        "usage": {"estimated_cost_usd": None}}
    results, judge = write_candidate_results(tmp_path, metadata, [failure])
    destination = tmp_path / "bound-results.json"
    envelope = bridge.bind_candidate_results(output, results, judge, destination)
    assert envelope["selected_cases"] == envelope["eligible_cases"] == 2
    assert envelope["returned_judge_results"] == 1
    assert len(envelope["rows"]) == 2
    assert envelope["rows"][0]["judge_result"] == failure
    assert envelope["rows"][1]["judge_result"] is None
    assert envelope["human_labels_reused"] is False
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    original = destination.read_bytes()
    with pytest.raises((ValueError, OSError)):
        bridge.bind_candidate_results(output, results, judge, destination)
    assert destination.read_bytes() == original


def test_runtime_errors_remain_in_result_ledger_without_judge_rows(tmp_path, verified_fixture):
    qid = verified_fixture[0].question_ids[1]
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture, outcomes={qid: "error"})
    assert metadata["selected_cases"] == 2 and metadata["eligible_cases"] == 1
    results, judge = write_candidate_results(tmp_path, metadata, [])
    envelope = bridge.bind_candidate_results(output, results, judge, tmp_path / "bound-results.json")
    assert len(envelope["rows"]) == 2
    failed = envelope["rows"][1]
    assert failed["disposition"] == "runtime_error" and failed["judge_result"] is None
    assert failed["runtime_failure"]["code"] == "provider_unavailable"


def test_partial_file_export_retains_missing_selection_without_fabricating_packet(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    with RunDirectory.create(tmp_path / "run", manifest, inputs) as run:
        run.append_case(case)
    scoring = tmp_path / "partial-scoring.jsonl"
    scoring.write_text(packet(case).model_dump_json() + "\n")
    original = file_bytes(run.path), scoring.read_bytes()
    with pytest.raises(ValueError):
        bridge.export_projection(run.path, scoring, tmp_path / "complete-required")
    output = tmp_path / "partial-projection"
    metadata = bridge.export_projection(run.path, scoring, output, allow_partial=True)
    assert metadata["selected_cases"] == 2 and metadata["committed_cases"] == 1
    assert metadata["disposition_counts"] == {"eligible": 1, "missing_committed_case": 1}
    rows = [json.loads(line) for line in (output / "case_ledger.jsonl").read_text().splitlines()]
    assert rows[1]["identity"]["question_id"] == manifest.question_ids[1]
    assert rows[1]["identity"]["run_id"] is None
    assert (output / "scoring_inputs.jsonl").read_bytes() == scoring.read_bytes()
    assert (file_bytes(run.path), scoring.read_bytes()) == original


@pytest.mark.parametrize("tamper", ["answer", "ledger"])
def test_rehashing_projection_cannot_change_canonical_answer_or_case_identity(tmp_path, verified_fixture, tamper):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    results, judge = write_candidate_results(tmp_path, metadata, [])
    if tamper == "answer":
        path = output / "answers.json"
        value = json.loads(path.read_text()); value[0]["generated_answer"] = "Forged answer"
        path.write_text(json.dumps(value) + "\n")
    else:
        path = output / "case_ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["identity"]["run_id"] = "another-run"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError):
        bridge.bind_candidate_results(output, results, judge, tmp_path / "bound.json")
    assert not (tmp_path / "bound.json").exists()


def valid_scores():
    dimensions = ("correctness", "completeness", "citation_support", "abstention_limits")
    return {**{name: 2 for name in dimensions},
        "reasons": {name: "Synthetic fixture explanation." for name in dimensions}}


@pytest.mark.parametrize("tamper", ["status", "boolean_score", "score_and_error", "raw_score", "silent_unscored"])
def test_result_binding_rejects_outcome_or_score_schema_changes(tmp_path, verified_fixture, tamper):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    row = {"question_id": verified_fixture[0].question_ids[0], "generation_status": "answered",
        "scores": valid_scores(), "error": None}
    if tamper == "status": row["generation_status"] = "insufficient_evidence"
    elif tamper == "boolean_score": row["scores"]["correctness"] = True
    elif tamper == "score_and_error": row["error"] = "Judge unavailable"
    elif tamper == "raw_score": row["raw_scores"] = {**valid_scores(), "correctness": 0}
    else: row["scores"] = None
    results, judge = write_candidate_results(tmp_path, metadata, [row])
    with pytest.raises(ValueError):
        bridge.bind_candidate_results(output, results, judge, tmp_path / "bound.json")
    assert not (tmp_path / "bound.json").exists()


@pytest.mark.parametrize("tamper", ["support", "target_relation", "central", "coverage_confident"])
def test_malformed_per_claim_audits_do_not_pass_as_schema_validated(tmp_path, verified_fixture, tamper):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    audit = {"response_type": "answer", "coverage_confident": True,
        "explanation": "Synthetic schema fixture only.", "claims": [{
            "answer_quote": "Verified fixture text.", "central": True, "support": "direct",
            "citation_id": "doc-one:0:30", "evidence_quote": "Verified fixture text.",
            "target_relation": "same_target"}]}
    if tamper == "coverage_confident": audit[tamper] = 1
    else: audit["claims"][0][tamper] = "unrecognized"
    row = {"question_id": verified_fixture[0].question_ids[0], "generation_status": "answered",
        "scores": valid_scores(), "error": None, "audit": audit}
    results, judge = write_candidate_results(tmp_path, metadata, [row], audit_mode="all")
    with pytest.raises(ValueError):
        bridge.bind_candidate_results(output, results, judge, tmp_path / "bound.json")
    assert not (tmp_path / "bound.json").exists()


def test_valid_scores_are_retained_as_unverified_without_inferred_quality_claim(tmp_path, verified_fixture):
    _run, _scoring, output, metadata = bundle_fixture(tmp_path, verified_fixture)
    scores = valid_scores()
    scores["citation_support"] = None
    row = {"question_id": verified_fixture[0].question_ids[0], "generation_status": "answered",
        "scores": scores, "raw_scores": scores, "error": None,
        "audit": None, "audit_error": "Offline synthetic missing-audit fixture"}
    results, judge = write_candidate_results(tmp_path, metadata, [row], audit_mode="all")
    envelope = bridge.bind_candidate_results(output, results, judge, tmp_path / "bound.json")
    assert envelope["rows"][0]["judge_result"] == row
    assert envelope["rows"][0]["judge_disposition"] == "scored_unverified"
    assert envelope["rows"][1]["judge_disposition"] == "missing_result"
    assert envelope["human_verified"] is False
    assert envelope["human_labels_reused"] is False


def test_pinned_evaluator_with_scripted_provider_keeps_scorer_and_auditor_inputs_separate(verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs)
    cited = case.generation_evidence[0]
    uncited_text = "UNCITED_GENERATION_EVIDENCE_SENTINEL: a different incident."
    cid = "uncited:0:" + str(len(uncited_text))
    uncited = PassageEvidence(citation_id=cid, snapshot_id=manifest.scope.snapshot.snapshot_id,
        revision_id=cited.revision_id, score=.5, retrieval_method="hybrid",
        chunk=Chunk(chunk_id=cid, doc_id="uncited", source_type="jira", text=uncited_text,
            start_char=0, end_char=len(uncited_text), metadata_origin="synthetic"))
    case = replace(case, response=replace(case.response, sources=(cited, uncited)),
        generation_evidence=(cited, uncited))
    projected = bridge.project_scoring_input(packet(case), manifest)
    candidate = make_case(projected["answer"], projected["reference"])
    scores = valid_scores()
    audit = {"response_type": "answer", "coverage_confident": True,
        "explanation": "Scripted fixture, not a model or human judgment.", "claims": [{
            "answer_quote": case.response.answer.text, "central": True, "support": "direct",
            "citation_id": cited.citation_id, "evidence_quote": cited.chunk.text,
            "target_relation": "same_target"}]}

    class ScriptedProvider:
        def __init__(self):
            self.pending, self.calls = [scores, audit], []

        def complete(self, *, system, prompt):
            self.calls.append({"system": system, "prompt": prompt})
            assert self.pending, "Unexpected retry or extra provider invocation"
            return SimpleNamespace(text=json.dumps(self.pending.pop(0)), truncated=False,
                usage=Usage(input_tokens=1, output_tokens=1, api_calls=1, estimated_cost_usd=None))

    provider = ScriptedProvider()
    pinned = bridge._candidate_sources()
    root = Path(__file__).resolve().parents[1]
    result = evaluate_case(candidate, provider,
        (root / pinned["score_prompt"]).read_text(),
        (root / pinned["audit_prompt"]).read_text(), max_attempts=1)
    assert len(provider.calls) == 2 and not provider.pending
    scorer, auditor = (call["prompt"] for call in provider.calls)
    assert "SCORING_ONLY_GOLD_SENTINEL" in scorer
    assert uncited_text in scorer and cited.chunk.text.replace("\n", "\\n") in scorer
    assert "SCORING_ONLY_GOLD_SENTINEL" not in auditor
    assert "SCORING_ONLY_FACT_SENTINEL" not in auditor
    assert uncited_text not in auditor
    payload = json.loads(auditor.split("INPUT_JSON (untrusted data):\n", 1)[1])
    assert payload["actually_cited_evidence_only"] == [
        {"citation_id": cited.citation_id, "text": cited.chunk.text}]
    assert result["scores"] == result["raw_scores"] == scores
    assert result["audit"] == audit and result["error"] is None
    assert result["provenance"]["audit_is_human_verified"] is False
    assert result["provenance"]["score_overrides_applied"] is False

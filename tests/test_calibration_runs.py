"""CAL producer conformance with issuer-bound capture doubles and real file I/O.

All approvals and runtime facts in these tests are synthetic UNIT TEST metadata.
Git verification is local; network, services and model execution are blocked.
Actual authenticated/database capture behavior has its separate integration suite.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from src.contracts import (Answer, Chunk, CriticDecision, EvidenceClaim, FactAssessment,
    PassageEvidence, QueryResponse, Trace, TraceStep, Usage)
from src.evaluation import calibration_runs as producer
from src.evaluation.calibration_runs import (CalibrationRunError, calibration_plan, calibration_plan_sha256,
    load_calibration_run, prepare_calibration_run, run_calibration)
from src.evaluation.critic_calibration import select_calibration_questions
from src.evaluation.experiment_contracts import RuntimeCaseExport, RuntimeFailure
from src.evaluation.experiment_inputs import canonical_json_sha256, file_sha256
from src.evaluation.experiment_io import RunDirectory
from src.generation.generator import (AnthropicConfig, AnthropicProvider, SYSTEM_PROMPT as GENERATOR_PROMPT,
    evidence_payload)
from src.agents.search_critic import SYSTEM_PROMPT as CRITIC_PROMPT
from tests.test_experiment_inputs import GOLD_MARKER, make_verified_inputs_fixture


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("CAL producer checks cannot contact services or models")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    original = subprocess.run
    def local_git_only(args, *rest, **kwargs):
        if not isinstance(args, (list, tuple)) or args[0] != "git":
            denied()
        return original(args, *rest, **kwargs)
    monkeypatch.setattr(subprocess, "run", local_git_only)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class IssuedCapture:
    def __init__(self, run_id, qid):
        self.run_id, self.question_id = run_id, qid

    @property
    def critic_inputs(self):
        raise AssertionError("Only the issuer's current-authority export may supply critic inputs")


class RuntimeDouble:
    def __init__(self, profile):
        self.profile, self.calls, self.exports, self.issued = profile, [], [], set()
        self.errors, self.budgets = set(), set()
        self.lost_response = self.revoke_at_export = self.missing_capture = False
        self.after_export = None
        self.context = SimpleNamespace(binding=producer.ScopeBinding.model_validate(profile["scope"]),
            scope=SimpleNamespace(bitmap_sha256="9" * 64), assert_bound=lambda: None)
        self.resolver = SimpleNamespace(resolve=lambda _identity: self.context, assert_current=lambda _context: None)
        self.provider = None

    def experiment_profile(self, body, identity):
        assert body.mode == "agentic" and GOLD_MARKER not in body.model_dump_json()
        return deepcopy(self.profile)

    def query_with_capture(self, body, identity, *, experiment):
        assert body.mode == "agentic" and experiment.manifest.workflow == "adaptive"
        assert experiment.manifest.split == "calibration" and body.question == next(
            q.question for q in experiment.inputs.questions if q.question_id == experiment.question_id)
        self.calls.append((body, experiment))
        if self.lost_response:
            raise ConnectionError("PRIVATE_FAILURE_SENTINEL")
        capture = IssuedCapture("fixture-run-" + str(len(self.calls)), experiment.question_id)
        self.issued.add(capture)
        return capture

    def _case(self, capture, manifest, verified_inputs):
        assert capture in self.issued
        question = next(q for q in verified_inputs.questions if q.question_id == capture.question_id)
        text = "Verified fixture fact."
        cid = "doc-one:0:" + str(len(text))
        item = PassageEvidence(citation_id=cid, snapshot_id=manifest.scope.snapshot.snapshot_id,
            revision_id="b" * 64, retrieval_method="hybrid", score=1,
            chunk=Chunk(chunk_id=cid, doc_id="doc-one", source_type="jira", text=text,
                start_char=0, end_char=len(text), metadata_origin="upstream"))
        decision = CriticDecision(sufficient=True, confidence=0, reason="Synthetic conformance fact checklist",
            facts=(FactAssessment(fact="fixture fact", status="supported", citation_ids=(cid,),
                claims=(EvidenceClaim(citation_id=cid, statement=text, scope_reason="requested fixture fact",
                    answers_requested_fact=True),)),))
        error, budget = question.question_id in self.errors, question.question_id in self.budgets
        tool = "critic:error" if error else "critic:budget" if budget else "critic"
        step = TraceStep(round_number=1, query=question.question, tool=tool, elapsed_ms=0,
            evidence_ids=(cid,) if not error else (), critic=decision if tool == "critic" else None,
            usage=Usage(estimated_cost_usd=0))
        response = QueryResponse(snapshot=manifest.scope.snapshot, sources=(item,),
            answer=Answer(status="error" if error else "insufficient_evidence" if budget else "answered",
                text="Synthetic conformance outcome", citation_ids=() if error or budget else (cid,)),
            trace=Trace(run_id=capture.run_id, steps=(step,), stop_reason="error" if error else "budget" if budget else "sufficient",
                usage=Usage(estimated_cost_usd=0)))
        return RuntimeCaseExport(manifest_id=manifest.manifest_id, run_id=capture.run_id,
            question_id=question.question_id, question=question.question,
            question_sha256=hashlib.sha256(question.question.encode()).hexdigest(), response=response,
            generation_evidence=() if error or budget else (item,),
            error=RuntimeFailure(code="runtime_error", stage="runtime") if error else None), item

    def export_capture(self, capture, *, manifest, verified_inputs, question_id, directory):
        case, item = self._case(capture, manifest, verified_inputs)
        directory.append_case(case)
        self.exports.append((capture, case, item))
        if self.after_export: self.after_export()
        return case

    def export_critic_inputs(self, capture):
        assert capture in self.issued
        if self.revoke_at_export:
            raise PermissionError("PRIVATE_REVOCATION_SENTINEL")
        _capture, case, item = next(value for value in self.exports if value[0] is capture)
        unavailable = self.missing_capture or case.error is not None
        return ({"trace_step_index": 0, "critic_call_ordinal": 1,
            "disposition": "input_unavailable" if unavailable else "captured",
            "request_evidence": None if unavailable else (item,),
            "selected_evidence": None if unavailable else (item,),
            "payload_sha256": None if unavailable else canonical_json_sha256({"question": case.question,
                "evidence": [evidence_payload(item)]}), "provider_call_planned": None if unavailable else True},)


def prepared(tmp_path, *, live=False):
    base, verified = make_verified_inputs_fixture(tmp_path / "base", workflow="adaptive", split="calibration")
    files = verified.files
    selection = select_calibration_questions(files.questions, files.splits, 2)
    manifest = base.model_dump(mode="json")
    index = json.loads(files.index_manifest.read_text())
    config = {"method": "store-hybrid", "top_k": 5, "router": {"kind": "text_router", "version": "1"},
        "filters": {"source_types": [], "after": None, "before": None}}
    manifest["retrieval"].update(retriever_id="store-hybrid", index_manifest_sha256=canonical_json_sha256(index),
        config_sha256=canonical_json_sha256(config))
    manifest["generation_model"] = {"provider": "anthropic" if live else "fixture", "model_id": "fixture-model", "revision": None}
    manifest["budgets"]["max_run_cost_usd"] = .05 if live else 0
    manifest["generator_prompt_sha256"] = hashlib.sha256(GENERATOR_PROMPT.encode()).hexdigest()
    manifest["critic_prompt_sha256"] = hashlib.sha256(CRITIC_PROMPT.encode()).hexdigest()
    profile = {key: manifest[key] for key in producer._PROFILE_FIELDS}
    profile.update(scope=manifest["scope"], source_files={"source.py": file_sha256(files.source_repo / "source.py")},
        index_manifest=index, retrieval_config=config)
    actual = RuntimeDouble(profile)
    comparison = {"experiment_id": "fixture-cal-producer"}
    if live:
        actual.provider = AnthropicProvider(AnthropicConfig(model="fixture-model", allow_paid=True, budget_usd=.10,
            input_usd_per_million=1, output_usd_per_million=5), client=object())
        comparison["generation"] = {"provider": "anthropic", "model": "fixture-model", "budget_usd": .10,
            "input_price_per_million": 1, "output_price_per_million": 5, "max_output_tokens": 1024, "timeout_seconds": 30}
    sources = {name: getattr(files, name) for name in ("questions", "splits", "corpus", "catalog", "source_repo")}
    run = prepare_calibration_run(actual, object(), selection, sources=sources,
        prepared_dir=tmp_path / "prepared", run_dir=tmp_path / "canonical", comparison=comparison)
    return run, actual


def approved_fixture(run, tmp_path, *, cap="0", approved=False):
    output = tmp_path / "capture-output"
    plan = calibration_plan(run, cap, output)
    value = {"format": "rag-critic-calibration-approval-v1", "approved": approved,
        "approval_source": "explicit_user_affirmative_reply", "user_reply": "SYNTHETIC UNIT TEST ONLY, NOT USER APPROVAL",
        "root_execution_review_complete": True, "plan_sha256": calibration_plan_sha256(plan),
        "maximum_selected_cases": plan["maximum_selected_cases"], "question_ids": plan["manifest"]["question_ids"],
        "generation_model": plan["manifest"]["generation_model"], "critic_prompt_sha256": plan["manifest"]["critic_prompt_sha256"],
        "maximum_rounds": plan["manifest"]["max_rounds"], "maximum_case_cost_usd": plan["maximum_case_cost_usd"],
        "maximum_total_cost_usd": plan["maximum_total_cost_usd"], "execution_context": plan["execution_context"],
        "calibration_runs_allowed": True, "automatic_retry_allowed": False, "final_runs_allowed": False,
        "judging_allowed": False, "sdk_max_retries": 0,
        "egress": {"destination": "Anthropic API" if cap != "0" else "fixture", "question": True,
            "retrieved_permitted_fictional_corpus_excerpts": True}}
    path = tmp_path / "SYNTHETIC-NOT-USER-APPROVAL.json"; write(path, value)
    return path, file_sha256(path), output


def execute_fixture(run, actual, tmp_path, *, cap="0", approved=True, opened=None):
    path, digest, output = approved_fixture(run, tmp_path, cap=cap, approved=approved)
    @contextmanager
    def default_open(_run):
        yield actual, object()
    return run_calibration(run, open_runtime=opened or default_open, approval_file=path, approval_sha256=digest,
        max_total_cost_usd=cap, output_dir=output, allow_paid=cap != "0")


def test_preparation_is_reference_free_non_generating_and_reloads_exact_inputs(tmp_path):
    run, actual = prepared(tmp_path)
    assert actual.calls == [] and run.manifest.workflow == "adaptive" and run.manifest.split == "calibration"
    assert run.manifest.retrieval.retriever_id == "store-hybrid"
    for path in (tmp_path / "prepared").iterdir():
        assert GOLD_MARKER not in path.read_text()
    loaded = load_calibration_run(tmp_path / "prepared")
    assert loaded.manifest == run.manifest and loaded.inputs.hashes == run.inputs.hashes
    assert not run.directory.exists()


def test_missing_approval_fails_before_opening_runtime_or_creating_any_output(tmp_path):
    run, actual = prepared(tmp_path)
    def forbidden(_run): pytest.fail("Approval must precede runtime factory")
    with pytest.raises(CalibrationRunError, match="affirmative"):
        execute_fixture(run, actual, tmp_path, approved=False, opened=forbidden)
    assert not (tmp_path / "capture-output").exists() and actual.calls == []


def test_zero_provider_execution_captures_canonical_cases_and_exact_critic_episodes_without_labels(tmp_path):
    run, actual = prepared(tmp_path)
    result = execute_fixture(run, actual, tmp_path)
    assert result["passed"] and result["canonical_cases"] == result["critic_episodes"] == 2
    assert result["human_labels_created"] is False and result["calibration_accepted"] is False
    cases = RunDirectory.load(run.directory).load_cases(require_complete=True)
    assert len(cases) == 2 and all(case.response.trace.usage.api_calls == 0 for case in cases)
    output = tmp_path / "capture-output"
    journal = json.loads((output / "journal.json").read_text())
    assert Decimal(journal["reserved_usd"]) == 0 and all(item["state"] == "committed" for item in journal["attempts"])
    assert len(list((output / "captures").glob("*.episodes.json"))) == 2
    assert not list(output.rglob("human*")) and not list(output.rglob("*freeze*"))
    assert all(GOLD_MARKER not in path.read_text() for path in output.rglob("*.json"))


@pytest.mark.parametrize("outcome", ["error", "budget"])
def test_error_and_budget_cases_without_a_critic_decision_are_retained_not_fabricated(tmp_path, outcome):
    run, actual = prepared(tmp_path)
    getattr(actual, "errors" if outcome == "error" else "budgets").add(run.manifest.question_ids[0])
    result = execute_fixture(run, actual, tmp_path)
    assert result["passed"] and result["canonical_cases"] == 2
    assert result["cases_without_critic_decision"] == 1 and result["critic_episodes"] == 1
    assert json.loads((tmp_path / "capture-output/captures/case_000001.episodes.json").read_text()) == []


def test_lost_response_keeps_single_durable_intent_and_prohibits_replay(tmp_path):
    run, actual = prepared(tmp_path); actual.lost_response = True
    result = execute_fixture(run, actual, tmp_path)
    assert not result["passed"] and result["attempt_count"] == 1 and result["spend_may_be_unknown"]
    assert len(actual.calls) == 1 and result["failure_type"] == "ConnectionError"
    path = tmp_path / "capture-output/journal.json"; before = path.read_bytes()
    with pytest.raises(CalibrationRunError, match="Existing outputs"):
        execute_fixture(run, actual, tmp_path)
    assert path.read_bytes() == before and len(actual.calls) == 1
    assert "PRIVATE_FAILURE_SENTINEL" not in (tmp_path / "capture-output/summary.json").read_text()


@pytest.mark.parametrize("kind", ["revoked_export", "missing_capture"])
def test_canonical_case_survives_private_capture_failure_without_retry(tmp_path, kind):
    run, actual = prepared(tmp_path)
    actual.revoke_at_export = kind == "revoked_export"; actual.missing_capture = kind == "missing_capture"
    result = execute_fixture(run, actual, tmp_path)
    assert not result["passed"] and result["canonical_cases"] == 1 and len(actual.calls) == 1
    assert len(RunDirectory.load(run.directory).load_cases()) == 1
    journal = json.loads((tmp_path / "capture-output/journal.json").read_text())
    assert journal["attempts"][0]["state"] == "runtime_committed"
    assert "PRIVATE_REVOCATION_SENTINEL" not in (tmp_path / "capture-output/summary.json").read_text()


def test_runtime_profile_change_fails_before_any_query(tmp_path):
    run, actual = prepared(tmp_path)
    @contextmanager
    def changed(_run):
        actual.profile["budgets"]["max_run_tokens"] += 1
        yield actual, object()
    result = execute_fixture(run, actual, tmp_path, opened=changed)
    assert not result["passed"] and result["attempt_count"] == 0 and actual.calls == []


def test_changed_scientific_source_and_changed_selection_fail_before_runtime_open(tmp_path):
    run, actual = prepared(tmp_path)
    path, digest, output = approved_fixture(run, tmp_path, approved=True)
    (run.inputs.files.source_repo / "source.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError):
        run_calibration(run, open_runtime=lambda _run: pytest.fail("No runtime"), approval_file=path,
            approval_sha256=digest, max_total_cost_usd=0, output_dir=output)
    assert not output.exists() and actual.calls == []


@pytest.mark.parametrize("field,value", [("maximum_selected_cases", 1), ("maximum_total_cost_usd", "1"),
    ("automatic_retry_allowed", True), ("final_runs_allowed", True), ("judging_allowed", True),
    ("maximum_rounds", True), ("generation_model", {"provider": "fixture", "model_id": "other"})])
def test_approval_cannot_expand_case_model_budget_or_execution_scope(tmp_path, field, value):
    run, actual = prepared(tmp_path)
    path, digest, output = approved_fixture(run, tmp_path, approved=True)
    approval = json.loads(path.read_text()); approval[field] = value; write(path, approval)
    with pytest.raises(ValueError): run_calibration(run,
        open_runtime=lambda _run: pytest.fail("No runtime before exact approval"), approval_file=path,
        approval_sha256=file_sha256(path), max_total_cost_usd=0, output_dir=output)
    assert not output.exists()


def test_live_cap_is_conservatively_reserved_before_each_case_without_sdk_execution(tmp_path):
    run, actual = prepared(tmp_path, live=True)
    actual.lost_response = True
    result = execute_fixture(run, actual, tmp_path, cap="0.10")
    assert not result["passed"] and result["reserved_cost_usd"] == "0.05"
    assert result["attempt_count"] == 1 and actual.provider.reserved_usd == 0
    assert actual.provider.client is not None  # opaque fixture client; never invoked
    journal = json.loads((tmp_path / "capture-output/journal.json").read_text())
    assert journal["attempts"][0]["reservation_usd"] == "0.05"


@pytest.mark.parametrize("cap", ["0.04", "0.09", "0.11", "NaN", True])
def test_live_cap_must_fit_all_cases_and_equal_managed_provider_budget(tmp_path, cap):
    run, actual = prepared(tmp_path, live=True)
    with pytest.raises(ValueError): calibration_plan(run, cap, tmp_path / "capture-output")
    assert actual.calls == []


def test_duplicate_unavailable_capture_indices_are_rejected():
    case = SimpleNamespace(response=SimpleNamespace(trace=SimpleNamespace(steps=(SimpleNamespace(tool="critic:budget"),))))
    row = {"request_evidence": None, "selected_evidence": None, "payload_sha256": None, "provider_call_planned": None,
        "trace_step_index": 0, "critic_call_ordinal": 1, "disposition": "input_unavailable"}
    with pytest.raises(CalibrationRunError, match="Repeated"):
        producer._serialize_records((row, {**row, "critic_call_ordinal": 2}), case)


def test_false_capture_order_cannot_be_associated_with_another_trace_event():
    case = SimpleNamespace(response=SimpleNamespace(trace=SimpleNamespace(steps=(
        SimpleNamespace(tool="text_search"), SimpleNamespace(tool="critic:budget")))))
    row = {"request_evidence": None, "selected_evidence": None, "payload_sha256": None, "provider_call_planned": None,
        "trace_step_index": 0, "critic_call_ordinal": 1, "disposition": "input_unavailable"}
    with pytest.raises(CalibrationRunError, match="ordered critic trace"):
        producer._serialize_records((row,), case)


def test_prepared_case_reservation_is_durable_before_query_dispatch(tmp_path):
    run, actual = prepared(tmp_path, live=True)
    original_query = actual.query_with_capture
    def checked_query(*args, **kwargs):
        journal = json.loads((tmp_path / "capture-output/journal.json").read_text())
        assert journal["attempts"][-1]["state"] == "intent"
        assert Decimal(journal["reserved_usd"]) == Decimal("0.05") * len(journal["attempts"])
        return original_query(*args, **kwargs)
    actual.query_with_capture = checked_query
    result = execute_fixture(run, actual, tmp_path, cap="0.10")
    assert result["passed"] and result["reserved_cost_usd"] == "0.10" and len(actual.calls) == 2


def test_changed_approval_after_first_case_stops_without_second_dispatch(tmp_path):
    run, actual = prepared(tmp_path)
    path, digest, output = approved_fixture(run, tmp_path, approved=True)
    actual.after_export = lambda: path.write_text('{"approved":false}')
    @contextmanager
    def opened(_run): yield actual, object()
    result = run_calibration(run, open_runtime=opened, approval_file=path, approval_sha256=digest,
        max_total_cost_usd=0, output_dir=output)
    assert not result["passed"] and len(actual.calls) == 1 and result["attempt_count"] == 1
    assert len(RunDirectory.load(run.directory).load_cases()) == 1


def test_new_output_and_canonical_ancestors_are_fsynced_before_first_query(tmp_path, monkeypatch):
    run, actual = prepared(tmp_path)
    run = replace(run, directory=tmp_path / "new-canonical-parent" / "nested" / "run")
    output = tmp_path / "new-output-parent" / "nested" / "captures"
    path, _, _ = approved_fixture(run, tmp_path, approved=True)
    approval = json.loads(path.read_text())
    approval["plan_sha256"] = calibration_plan_sha256(calibration_plan(run, 0, output))
    write(path, approval)
    synced = []
    original_sync = producer._sync
    def record_sync(path):
        synced.append(Path(path)); original_sync(path)
    monkeypatch.setattr(producer, "_sync", record_sync)
    original_query = actual.query_with_capture
    def checked_query(*args, **kwargs):
        for parent in (tmp_path, output.parent.parent, output.parent, output,
                       run.directory.parent.parent, run.directory.parent, run.directory):
            assert parent in synced
        return original_query(*args, **kwargs)
    actual.query_with_capture = checked_query
    @contextmanager
    def opened(_run): yield actual, object()
    result = run_calibration(run, open_runtime=opened, approval_file=path, approval_sha256=file_sha256(path),
        max_total_cost_usd=0, output_dir=output)
    assert result["passed"] and len(actual.calls) == 2


def test_parent_fsync_failure_prevents_any_query(tmp_path, monkeypatch):
    run, actual = prepared(tmp_path)
    def fail(path):
        if Path(path) == tmp_path: raise OSError("SYNTHETIC FSYNC FAILURE")
    monkeypatch.setattr(producer, "_sync", fail)
    with pytest.raises(OSError): execute_fixture(run, actual, tmp_path)
    assert actual.calls == []

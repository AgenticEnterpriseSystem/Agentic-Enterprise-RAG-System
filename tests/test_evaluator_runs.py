"""Guarded RC2 execution with finite scripted providers and no network access."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from src.agents.budget import RunBudgetExceeded, current_call_budget
from src.contracts import AggregateEvidence, Usage
from src.evaluation import evaluator_runs as runs
from src.evaluation.experiment_inputs import verify_inputs
from src.evaluation.experiment_io import RunDirectory, export_scoring_inputs
from src.evaluation.evaluator_bridge import export_projection
from src.generation.generator import Completion, ProviderError
from tests.test_experiment_inputs import make_verified_inputs_fixture
from tests.test_experiment_io import case_for, references, replace


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("Scripted evaluator tests cannot connect to services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def bundle(path, *, name="fixed", outcomes=None, native=()):
    manifest, inputs = make_verified_inputs_fixture(path / "fixture")
    manifest = replace(manifest, manifest_id=name)
    inputs = verify_inputs(manifest, inputs.files)
    with RunDirectory.create(path / "runtime", manifest, inputs) as runtime:
        for qid in manifest.question_ids:
            case = case_for(manifest, inputs, qid,
                outcome=(outcomes or {}).get(qid, "answered"), run_id=name + "-" + qid)
            if qid in native:
                count = AggregateEvidence(citation_id="aggregate:zero", artifact_id="zero", scope=manifest.scope,
                    template_id="count", template_version="1", count=0, lineage_sha256="a" * 64,
                    coverage="synthetic_fixture", created_at=manifest.created_at)
                case = replace(case, response=replace(case.response, sources=(*case.response.sources, count)))
            runtime.append_case(case)
    refs = path / "references.jsonl"; references(refs, inputs)
    export_scoring_inputs(runtime.path, refs, path / "scoring")
    output = path / "projection"
    export_projection(runtime.path, path / "scoring/scoring_inputs.jsonl", output)
    return output


@pytest.fixture
def projection(tmp_path):
    return bundle(tmp_path / "source")


def settings(**changes):
    return runs.JudgeSettings(model_id="fixture-haiku", input_usd_per_million="1", output_usd_per_million="5",
        max_output_tokens=1024, max_tokens=16000, timeout_seconds=30,
        score_cost_cap_usd="0.02", audit_cost_cap_usd="0.02", case_cost_cap_usd="0.05",
        total_cost_cap_usd=changes.pop("total_cost_cap_usd", "0.10"), **changes)


def plan_for(projection, tmp_path, *, selection=None, extra_source_files=None, **plan_options):
    return runs.prepare_evaluator_plan([projection], tmp_path / "execution", settings=settings(),
        selected_question_ids=selection, execution_context="fixture", extra_source_files=extra_source_files, **plan_options)


def approval(plan, tmp_path):
    path = tmp_path / "approval.json"
    value = {"format": "rag-evaluator-approval-v1", "approved": True,
        "approval_source": "explicit_user_affirmative_reply", "user_reply": "SYNTHETIC TEST ONLY",
        "plan_sha256": runs.evaluator_plan_sha256(plan), "provider": "anthropic",
        "model_id": plan.settings.model_id, "execution_context": plan.execution_context,
        "maximum_selected_cases": len(plan.cases), "maximum_total_cost_usd": str(plan.settings.total_cost_cap_usd),
        "automatic_retry_allowed": False, "max_attempts_per_stage": 1, "sdk_max_retries": 0,
        "generation_runs_allowed": False, "adaptive_runs_allowed": False,
        "scoring_allowed": True, "citation_audit_allowed": True,
        "egress": {"destination": "Anthropic API", "questions": True, "generated_answers": True,
            "permitted_fictional_corpus_excerpts": True, "reference_answers_for_scoring": True}}
    path.write_text(json.dumps(value) + "\n")
    return path, runs._sha(path.read_bytes())


def score():
    dimensions = ("correctness", "completeness", "citation_support", "abstention_limits")
    return {**{name: 2 for name in dimensions}, "reasons": {name: "Scripted fixture only." for name in dimensions}}


def audit():
    return {"response_type": "answer", "coverage_confident": True, "explanation": "Scripted fixture only.",
        "claims": [{"answer_quote": "Verified fixture text.", "central": True, "support": "direct",
            "citation_id": "doc-one:0:30", "evidence_quote": "Fixture\nVerified fixture text.",
            "target_relation": "same_target"}]}


class Factory:
    def __init__(self, plan, behavior=None):
        self.plan, self.behavior, self.calls, self.admissions, self.closed = plan, behavior or {}, [], [], 0

    def __call__(self, admission):
        self.admissions.append(admission)
        output = Path(self.plan.output_directory)
        journal = json.loads((output / "journal.json").read_text())
        assert journal["cases"][-1]["identity"] == admission.identity.model_dump(mode="json")
        assert journal["cases"][-1]["stages"][admission.stage]["state"] == "intent"
        assert Decimal(journal["reserved_usd"]) >= self.plan.settings.case_cost_cap_usd
        owner = self
        class Provider:
            def complete(self, *, system, prompt):
                owner.calls.append((admission, system, prompt))
                budget = current_call_budget()
                assert budget is not None and budget.remaining_cost_usd == float(admission.max_cost_usd)
                value = owner.behavior.get((admission.identity.question_id, admission.stage))
                if isinstance(value, BaseException): raise value
                if callable(value): return value(admission, system, prompt)
                payload = value if value is not None else score() if admission.stage == "score" else audit()
                return Completion(json.dumps(payload), Usage(input_tokens=100, output_tokens=50,
                    api_calls=2, estimated_cost_usd=.001))
            def close(self): owner.closed += 1
        return Provider()


def execute(plan, tmp_path, factory):
    path, sha = approval(plan, tmp_path)
    return runs.run_evaluator_plan(plan, approval_file=path, approval_sha256=sha, provider_factory=factory)


def test_two_stage_execution_has_durable_admission_and_exact_binding(projection, tmp_path):
    plan = plan_for(projection, tmp_path); factory = Factory(plan)
    before = {p.name: p.read_bytes() for p in projection.iterdir()}
    receipt = execute(plan, tmp_path, factory)
    assert len(factory.calls) == 4 and factory.closed == 4
    assert [a.stage for a in factory.admissions] == ["score", "citation_audit", "score", "citation_audit"]
    assert "SCORING_ONLY_GOLD_SENTINEL" in factory.calls[0][2]
    assert "SCORING_ONLY_GOLD_SENTINEL" not in factory.calls[1][2]
    assert receipt["reserved_usd"] == "0.10" and receipt["logical_provider_calls"] == 4
    assert receipt["reported_usage"]["api_calls"] == 8
    assert receipt["known_reported_cost_decimal"] == "0.004" and receipt["unknown_cost_stages"] == 0
    assert receipt["human_verified"] is False and receipt["execution_context"] == "fixture"
    output = Path(plan.output_directory)
    fixture_record = json.loads((output / "cases/case_000001.json").read_text())
    assert fixture_record["provenance"]["mode"] == "fixture"
    assert fixture_record["provenance"]["score_source"] == "rc2_prompt_fixture"
    assert fixture_record["provenance"]["audit_source"] == "v5_cited_only_audit_fixture"
    assert "live" not in json.dumps(fixture_record["provenance"])
    bound = runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")
    assert bound["source_and_journal_verified"] is True and bound["committed_results"] == 2
    assert {p.name: p.read_bytes() for p in projection.iterdir()} == before
    for p in output.rglob("*"):
        if p.is_file(): assert p.stat().st_mode & 0o777 == 0o600
    assert not (output / ".writer.lock").exists()


@pytest.mark.parametrize("field", ["approved", "plan_sha256", "maximum_total_cost_usd", "automatic_retry_allowed", "egress"])
def test_missing_or_broadened_approval_never_constructs_provider(projection, tmp_path, field):
    plan = plan_for(projection, tmp_path); factory = Factory(plan)
    path, _sha = approval(plan, tmp_path); value = json.loads(path.read_text())
    value[field] = ({"destination": "somewhere"} if field == "egress" else False if field == "approved" else
        True if field == "automatic_retry_allowed" else "f" * 64 if field == "plan_sha256" else "0.20")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=runs._sha(path.read_bytes()), provider_factory=factory)
    assert not factory.calls and not factory.admissions and not Path(plan.output_directory).exists()


def test_output_namespace_cannot_be_replayed_even_after_success(projection, tmp_path):
    plan = plan_for(projection, tmp_path); factory = Factory(plan)
    path, sha = approval(plan, tmp_path)
    runs.run_evaluator_plan(plan, approval_file=path, approval_sha256=sha, provider_factory=factory)
    count = len(factory.calls)
    with pytest.raises(ValueError): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=sha, provider_factory=factory)
    assert len(factory.calls) == count


def test_bad_score_keeps_failed_attempt_usage_and_never_retries_or_audits_it(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    bad = score(); bad["correctness"] = True
    factory = Factory(plan, {("q1", "score"): bad})
    receipt = execute(plan, tmp_path, factory)
    first = json.loads((Path(plan.output_directory) / "cases/case_000001.json").read_text())
    assert first["scores"] is None and first["error"] == "ValueError"
    assert first["stage_usage"]["score"]["estimated_cost_usd"] == .001
    assert "citation_audit" not in first["stage_usage"]
    assert receipt["logical_provider_calls"] == 3 and receipt["known_reported_cost_decimal"] == "0.003"


def test_malformed_audit_preserves_valid_score_and_both_stage_usage(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    bad = audit(); bad["claims"][0]["support"] = "UNRECOGNIZED"
    factory = Factory(plan, {("q1", "citation_audit"): bad})
    receipt = execute(plan, tmp_path, factory)
    first = json.loads((Path(plan.output_directory) / "cases/case_000001.json").read_text())
    assert first["scores"] == first["raw_scores"] == score()
    assert first["audit"] is None and first["audit_error"] == "ValueError"
    assert set(first["stage_usage"]) == {"score", "citation_audit"}
    assert receipt["coverage_counts"]["audit_error"] == 1 and receipt["human_verified"] is False


def test_provider_failure_keeps_unknown_cost_and_incomplete_token_usage(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    factory = Factory(plan, {("q1", "score"): ProviderError(usage=Usage(api_calls=2, estimated_cost_usd=None))})
    receipt = execute(plan, tmp_path, factory)
    first = json.loads((Path(plan.output_directory) / "cases/case_000001.json").read_text())
    assert first["scores"] is None and first["usage"]["estimated_cost_usd"] is None
    assert first["stage_usage"]["score"]["token_counts_complete"] is False
    assert receipt["unknown_cost_stages"] == 1 and receipt["reserved_usd"] == "0.10"


def test_unexpected_failure_stops_other_cases_and_preserves_full_coverage(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    factory = Factory(plan, {("q1", "score"): RuntimeError("Synthetic unknown failure")})
    receipt = execute(plan, tmp_path, factory)
    assert receipt["fatal_error"] == "RuntimeError" and receipt["committed_results"] == 1
    assert receipt["coverage_counts"] == {"score_error": 1, "not_dispatched": 1}
    assert receipt["unknown_cost_stages"] == 1 and receipt["reported_usage"]["api_calls"] is None
    assert receipt["reserved_usd"] == "0.05" and len(factory.calls) == 1
    output = Path(plan.output_directory)
    assert runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")["committed_results"] == 1


def test_runtime_errors_are_unsupported_for_dispatch_but_remain_in_denominator(tmp_path):
    projection = bundle(tmp_path / "source", outcomes={"q2": "error"})
    with pytest.raises(ValueError): plan_for(projection, tmp_path, selection={"fixed": ["q2"]})
    plan = plan_for(projection, tmp_path)
    receipt = execute(plan, tmp_path, Factory(plan))
    assert len(plan.cases) == 1 and receipt["coverage_counts"]["unsupported"] == 1
    coverage = [json.loads(line) for line in (Path(plan.output_directory) / "coverage.jsonl").read_text().splitlines()]
    assert len(coverage) == 2 and coverage[1]["runtime_disposition"] == "runtime_error"


def test_zero_aggregate_native_case_is_not_flattened_or_selected_for_paid_dispatch(tmp_path):
    projection = bundle(tmp_path / "source", native=("q1",))
    with pytest.raises(ValueError): plan_for(projection, tmp_path, selection={"fixed": ["q1"]})
    plan = plan_for(projection, tmp_path); factory = Factory(plan)
    receipt = execute(plan, tmp_path, factory)
    assert len(plan.cases) == 1 and plan.cases[0].question_id == "q2"
    assert len(factory.calls) == 2 and receipt["coverage_counts"]["unsupported"] == 1


def test_completion_without_usage_keeps_unknown_counters_and_cost(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    def unknown(*_args): return SimpleNamespace(text=json.dumps(score()), truncated=False)
    receipt = execute(plan, tmp_path, Factory(plan, {("q1", "score"): unknown}))
    output = Path(plan.output_directory)
    first = json.loads((output / "cases/case_000001.json").read_text())
    assert first["scores"] == score() and first["stage_usage"]["score"]["api_calls"] is None
    assert first["usage"]["estimated_cost_usd"] is None and receipt["unknown_cost_stages"] == 1
    assert receipt["reported_usage"]["input_tokens"] is None
    runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")


def test_stage_overspend_is_retained_and_stops_further_dispatch(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    def over(*_args): return Completion(json.dumps(score()), Usage(input_tokens=100, output_tokens=50,
        api_calls=2, estimated_cost_usd=.03))
    factory = Factory(plan, {("q1", "score"): over}); receipt = execute(plan, tmp_path, factory)
    assert receipt["fatal_error"] == "EvaluatorRunError" and len(factory.calls) == 1
    assert receipt["known_reported_cost_decimal"] == "0.03"
    output = Path(plan.output_directory)
    with pytest.raises(ValueError): runs.validate_guarded_result_manifest(projection,
        output / "bundle_000001.results.json", output / "bundle_000001.manifest.json")


def test_interruption_keeps_uncertain_intent_and_forbids_replay(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    factory = Factory(plan, {("q1", "score"): KeyboardInterrupt()})
    path, sha = approval(plan, tmp_path)
    with pytest.raises(KeyboardInterrupt): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=sha, provider_factory=factory)
    output = Path(plan.output_directory); journal = json.loads((output / "journal.json").read_text())
    assert journal["reserved_usd"] == "0.05" and journal["cases"][0]["state"] == "intent"
    assert (output / "stages/case_000001_score.intent.json").exists()
    assert not (output / "receipt.json").exists()
    with pytest.raises(ValueError): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=sha, provider_factory=factory)
    assert len(factory.calls) == 1


def test_repeated_question_ids_stay_separate_per_manifest(tmp_path):
    a, b = bundle(tmp_path / "a", name="arm-a"), bundle(tmp_path / "b", name="arm-b")
    plan = runs.prepare_evaluator_plan([a, b], tmp_path / "execution", settings=settings(total_cost_cap_usd="0.20"),
        execution_context="fixture")
    receipt = execute(plan, tmp_path, Factory(plan))
    assert len(plan.cases) == 4 and receipt["committed_results"] == 4
    output = Path(plan.output_directory)
    for index, projection in enumerate((a, b), 1):
        bound = runs.validate_guarded_result_manifest(projection, output / f"bundle_{index:06d}.results.json",
            output / f"bundle_{index:06d}.manifest.json")
        assert bound["committed_results"] == 2


def test_caps_and_missing_required_source_pins_fail_before_execution(projection, tmp_path):
    raw = settings().model_dump(mode="json")
    for change in ({"score_cost_cap_usd": "0.04"}, {"max_output_tokens": 20000}, {"input_usd_per_million": True}):
        with pytest.raises(ValueError): runs.JudgeSettings.model_validate(raw | change)
    with pytest.raises(ValueError): runs.prepare_evaluator_plan([projection], tmp_path / "too-small",
        settings=settings(total_cost_cap_usd="0.05"), execution_context="fixture")
    plan = plan_for(projection, tmp_path)
    plan = plan.model_copy(update={"source_files": {}})
    path, sha = approval(plan, tmp_path)
    with pytest.raises(ValueError): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=sha, provider_factory=lambda _: pytest.fail("Provider constructed"))


def test_source_tampering_after_factory_cannot_dispatch_or_drop_reservation(projection, tmp_path):
    source = tmp_path / "adapter.py"; source.write_text("SAFE = True\n")
    plan = plan_for(projection, tmp_path, extra_source_files={"test-adapter": source})
    factory = Factory(plan)
    def changed(admission):
        result = factory(admission); source.write_text("CHANGED = True\n"); return result
    path, sha = approval(plan, tmp_path)
    with pytest.raises(ValueError): runs.run_evaluator_plan(plan, approval_file=path,
        approval_sha256=sha, provider_factory=changed)
    journal = json.loads((Path(plan.output_directory) / "journal.json").read_text())
    assert journal["reserved_usd"] == "0.05" and not factory.calls
    assert not (Path(plan.output_directory) / "receipt.json").exists()


def test_score_cleanup_failure_blocks_audit_and_every_later_case(projection, tmp_path):
    plan = plan_for(projection, tmp_path)
    factory = Factory(plan)
    def with_bad_cleanup(admission):
        provider = factory(admission)
        def broken_close():
            factory.closed += 1
            raise RuntimeError("Synthetic cleanup failure")
        provider.close = broken_close
        return provider
    receipt = execute(plan, tmp_path, with_bad_cleanup)
    assert receipt["fatal_error"] == "ProviderCleanupError"
    assert [item.stage for item in factory.admissions] == ["score"]
    assert len(factory.calls) == 1 and factory.closed == 1
    output = Path(plan.output_directory)
    first = json.loads((output / "cases/case_000001.json").read_text())
    assert first["scores"] == score() and first["audit_error"] == "EvaluatorRunError"
    assert set(first["stage_usage"]) == {"score"}
    assert not (output / "stages/case_000001_citation_audit.intent.json").exists()
    assert receipt["coverage_counts"] == {"audit_error": 1, "not_dispatched": 1}


def test_new_output_and_missing_parent_entries_are_fsynced_before_factory(projection, tmp_path, monkeypatch):
    output = tmp_path / "new-parent" / "new-child" / "execution"
    plan = runs.prepare_evaluator_plan([projection], output, settings=settings(), execution_context="fixture")
    synced = []
    original = runs._fsync_directory
    def observed(path):
        synced.append(Path(path)); original(path)
    monkeypatch.setattr(runs, "_fsync_directory", observed)
    factory = Factory(plan)
    def checked(admission):
        assert all(path in synced for path in (tmp_path, tmp_path / "new-parent", output.parent))
        return factory(admission)
    receipt = execute(plan, tmp_path, checked)
    assert receipt["committed_results"] == 2


def test_live_plan_requires_explicit_native_output_and_stop_policy(projection, tmp_path):
    fixture = plan_for(projection, tmp_path)
    assert fixture.output_protocol == "text_json_v1" and fixture.stop_on_invalid_response is False
    live = runs.prepare_evaluator_plan([projection], tmp_path / "live", settings=settings())
    assert live.output_protocol == "native_schema_v1" and live.stop_on_invalid_response is True
    assert live.max_response_bytes == 262144
    for changed in ({"output_protocol": "text_json_v1"}, {"stop_on_invalid_response": False},
            {"max_response_bytes": True}, {"max_response_bytes": 0}, {"max_response_bytes": 1048577}):
        with pytest.raises(ValueError):
            runs.EvaluatorPlan.model_validate(live.model_dump(mode="json") | changed)
    native_fixture = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    assert runs.evaluator_plan_sha256(fixture) != runs.evaluator_plan_sha256(native_fixture)


@pytest.mark.parametrize("kind,code", [
    ("malformed", "invalid_json"), ("fence", "invalid_json"), ("script", "invalid_json"),
    ("duplicate", "invalid_json"), ("bom", "invalid_json"), ("deep_json", "invalid_json"),
    ("score_shape", "invalid_score_schema"), ("unpaired_escape", "invalid_score_schema"),
    ("extra_field", "invalid_score_schema"), ("truncated", "response_truncated"),
    ("over_bound", "response_too_large"),
])
def test_native_invalid_response_is_retained_before_parse_and_stops_batch(projection, tmp_path, monkeypatch, kind, code):
    good = json.dumps(score())
    raw = {"malformed": '{"correctness":', "fence": "```json\n" + good + "\n```",
        "script": "<script>PRIVATE_RAW_SENTINEL()</script>",
        "duplicate": good.replace('"correctness": 2', '"correctness": 2, "correctness": 1'),
        "bom": "\ufeff" + good, "deep_json": "[" * 2000 + "0" + "]" * 2000,
        "unpaired_escape": good.replace("Scripted fixture only.", r"\ud800", 1),
        "score_shape": json.dumps(score() | {"correctness": True}),
        "extra_field": json.dumps(score() | {"unexpected": "PRIVATE_RAW_SENTINEL"}),
        "truncated": good, "over_bound": good}[kind]
    bound = 32 if kind == "over_bound" else 262144
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1",
        stop_on_invalid_response=True, max_response_bytes=bound)
    output = Path(plan.output_directory)
    raw_path = output / "stages/case_000001_score.response.txt"
    original = runs._response_payload
    def checked(retained, **kwargs):
        assert raw_path.is_file() and raw_path.read_bytes() == retained
        assert raw_path.stat().st_mode & 0o777 == 0o600
        return original(retained, **kwargs)
    monkeypatch.setattr(runs, "_response_payload", checked)
    def invalid(*_args):
        return Completion(raw, Usage(input_tokens=100, output_tokens=50, api_calls=2,
            estimated_cost_usd=.001), truncated=kind == "truncated")
    factory = Factory(plan, {("q1", "score"): invalid})
    receipt = execute(plan, tmp_path, factory)
    assert len(factory.calls) == len(factory.admissions) == 1
    assert factory.admissions[0].output_protocol == "native_schema_v1"
    assert receipt["fatal_error"] == "InvalidEvaluatorResponse"
    assert receipt["coverage_counts"] == {"score_error": 1, "not_dispatched": 1}
    assert receipt["reserved_usd"] == "0.05" and receipt["known_reported_cost_decimal"] == "0.001"
    assert receipt["reported_usage"]["api_calls"] == 2 and receipt["raw_response_artifacts"] == 1
    assert receipt["validation_failure_counts"] == {code: 1}
    first = json.loads((output / "cases/case_000001.json").read_text())
    assert first["scores"] is None and first["error"] == "ValueError"
    diagnosis = first["stage_diagnostics"]["score"]
    assert diagnosis["validation_failure_code"] == code
    assert diagnosis["response_sha256"] == runs._sha(raw.encode())
    assert diagnosis["raw_response"] == {"path": str(raw_path.relative_to(output)),
        "sha256": runs._sha(raw.encode()[:bound]), "size_bytes": min(len(raw.encode()), bound),
        "total_size_bytes": len(raw.encode()), "complete": len(raw.encode()) <= bound}
    assert raw_path.read_bytes() == raw.encode()[:bound]
    assert receipt["artifacts"][str(raw_path.relative_to(output))] == runs._sha(raw_path.read_bytes())
    assert not (output / "stages/case_000001_citation_audit.intent.json").exists()
    assert all("PRIVATE_RAW_SENTINEL" not in path.read_text() for path in output.rglob("*.json"))
    assert runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")["committed_results"] == 1


def test_native_invalid_audit_keeps_score_and_usage_without_dispatching_later_case(projection, tmp_path):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    bad = audit(); bad["claims"][0]["support"] = "INVENTED"
    factory = Factory(plan, {("q1", "citation_audit"): bad})
    receipt = execute(plan, tmp_path, factory)
    output = Path(plan.output_directory)
    first = json.loads((output / "cases/case_000001.json").read_text())
    assert first["scores"] == first["raw_scores"] == score()
    assert first["audit"] is None and first["audit_error"] == "ValueError"
    assert receipt["fatal_error"] == "InvalidEvaluatorResponse" and len(factory.calls) == 2
    assert receipt["known_reported_cost_decimal"] == "0.002" and receipt["reported_usage"]["api_calls"] == 4
    assert receipt["raw_response_artifacts"] == 2
    assert receipt["validation_failure_counts"] == {"invalid_audit_schema": 1}
    assert receipt["coverage_counts"] == {"audit_error": 1, "not_dispatched": 1}
    runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")


def test_native_success_keeps_exact_private_raw_responses_and_binds_them(projection, tmp_path):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    receipt = execute(plan, tmp_path, Factory(plan))
    output = Path(plan.output_directory)
    assert receipt["output_protocol"] == "native_schema_v1" and receipt["stop_on_invalid_response"] is True
    assert receipt["raw_response_artifacts"] == 4 and receipt["validation_failure_counts"] == {}
    for ordinal in (1, 2):
        for stage, payload in (("score", score()), ("citation_audit", audit())):
            name = f"stages/case_{ordinal:06d}_{stage}.response.txt"
            assert (output / name).read_bytes() == json.dumps(payload).encode()
            assert receipt["artifacts"][name] == runs._sha((output / name).read_bytes())
    runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")


@pytest.mark.parametrize("changed", ["raw_bytes", "failure_code", "raw_descriptor", "raw_matches_different_score", "over_bound"])
def test_raw_response_and_classification_tampering_fail_binding(projection, tmp_path, changed):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True,
        max_response_bytes=1024)
    execute(plan, tmp_path, Factory(plan))
    output = Path(plan.output_directory)
    name = "stages/case_000001_score.outcome.json"
    path = output / name
    value = json.loads(path.read_text())
    raw_name = value["raw_response"]["path"]
    raw_path = output / raw_name
    if changed in {"raw_bytes", "raw_matches_different_score"}:
        raw_path.write_bytes(json.dumps(score() | {"correctness": 1}).encode())
    if changed == "over_bound": raw_path.write_bytes(b"x" * 1025)
    if changed == "failure_code": value["validation_failure_code"] = "invalid_json"
    if changed == "raw_descriptor": value["raw_response"]["total_size_bytes"] += 1
    if changed == "raw_matches_different_score":
        value["raw_response"]["sha256"] = value["response_sha256"] = runs._sha(raw_path.read_bytes())
    path.write_bytes(runs._bytes(value) + b"\n")
    # Even a repinned artifact inventory cannot invent the raw validation fact.
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][name] = runs._sha(path.read_bytes())
    receipt["artifacts"][raw_name] = runs._sha(raw_path.read_bytes())
    receipt_path.write_bytes(runs._bytes(receipt) + b"\n")
    manifest_path = output / "bundle_000001.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution"]["references"]["receipt.json"]["sha256"] = runs._sha(receipt_path.read_bytes())
    manifest_path.write_bytes(runs._bytes(manifest) + b"\n")
    with pytest.raises(ValueError):
        runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json", manifest_path)


def test_raw_response_recheck_prevents_any_later_provider_after_tampering(projection, tmp_path):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    factory = Factory(plan)
    def changed(admission):
        provider = factory(admission)
        if admission.stage == "score":
            def close():
                factory.closed += 1
                (Path(plan.output_directory) / "stages/case_000001_score.response.txt").write_text("TAMPERED")
            provider.close = close
        return provider
    with pytest.raises(ValueError): execute(plan, tmp_path, changed)
    assert len(factory.calls) == len(factory.admissions) == 1
    output = Path(plan.output_directory)
    assert not (output / "stages/case_000001_citation_audit.intent.json").exists()
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("stage", ["score", "citation_audit"])
def test_rawless_provider_failure_binds_only_absent_failed_stage_result(projection, tmp_path, stage):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    error = ProviderError(usage=Usage(api_calls=2, estimated_cost_usd=None))
    receipt = execute(plan, tmp_path, Factory(plan, {("q1", stage): error}))
    output = Path(plan.output_directory)
    first = json.loads((output / "cases/case_000001.json").read_text())
    assert first["stage_diagnostics"][stage]["raw_response"] is None
    if stage == "score":
        assert first["scores"] is first["raw_scores"] is None and first["error"] == "ProviderError"
    else:
        assert first["scores"] == score() and first["audit"] is None and first["audit_error"] == "ProviderError"
    assert receipt["unknown_cost_stages"] == 1
    runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")


@pytest.mark.parametrize("changed", ["score", "raw_scores_only", "citation_audit", "score_missing_error", "audit_missing_error"])
def test_repinned_rawless_stage_cannot_preserve_grades_or_omit_error(projection, tmp_path, changed):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    execute(plan, tmp_path, Factory(plan))
    output = Path(plan.output_directory)
    stage = "citation_audit" if changed in {"citation_audit", "audit_missing_error"} else "score"
    outcome_name = f"stages/case_000001_{stage}.outcome.json"
    outcome = json.loads((output / outcome_name).read_text())
    raw_name = outcome["raw_response"]["path"]
    outcome.update(response_sha256=None, raw_response=None, response_truncated=None,
        validation_failure_code=None, error="ProviderError")
    (output / outcome_name).write_bytes(runs._bytes(outcome) + b"\n")
    case_name = "cases/case_000001.json"
    case = json.loads((output / case_name).read_text())
    case["stage_diagnostics"][stage] = runs._diagnostics(outcome)
    if stage == "score":
        case["error"] = "ProviderError"
        if changed in {"raw_scores_only", "score_missing_error"}: case["scores"] = None
        if changed == "score_missing_error": case.update(raw_scores=None, error=None)
    else:
        case["audit_error"] = "ProviderError"
        if changed == "audit_missing_error": case.update(audit=None, audit_error=None)
    (output / case_name).write_bytes(runs._bytes(case) + b"\n")
    results_name = "bundle_000001.results.json"
    results = json.loads((output / results_name).read_text()); results[0] = case
    (output / results_name).write_bytes(runs._bytes(results) + b"\n")
    journal_path = output / "journal.json"
    journal = json.loads(journal_path.read_text())
    journal["cases"][0]["result_sha256"] = runs._sha((output / case_name).read_bytes())
    journal_path.write_bytes(runs._bytes(journal) + b"\n")
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text()); del receipt["artifacts"][raw_name]
    receipt["raw_response_artifacts"] -= 1
    for name in (outcome_name, case_name, results_name, "journal.json"):
        receipt["artifacts"][name] = runs._sha((output / name).read_bytes())
    receipt_path.write_bytes(runs._bytes(receipt) + b"\n")
    manifest_path = output / "bundle_000001.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("journal.json", "receipt.json"):
        manifest["execution"]["references"][name]["sha256"] = runs._sha((output / name).read_bytes())
    manifest_path.write_bytes(runs._bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="rawless failed stage"):
        runs.validate_guarded_result_manifest(projection, output / results_name, manifest_path)


@pytest.mark.parametrize("changed", ["invalid_raw_scores", "blank_score_error", "blank_audit_error"])
def test_repinned_invalid_raw_stage_cannot_preserve_raw_grades_or_blank_errors(projection, tmp_path, changed):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    stage = "citation_audit" if changed == "blank_audit_error" else "score"
    bad = audit() if stage == "citation_audit" else score()
    if stage == "citation_audit": bad["claims"][0]["support"] = "INVALID"
    else: bad["correctness"] = True
    execute(plan, tmp_path, Factory(plan, {("q1", stage): bad}))
    output = Path(plan.output_directory)
    case_name, results_name = "cases/case_000001.json", "bundle_000001.results.json"
    case = json.loads((output / case_name).read_text())
    if changed == "invalid_raw_scores": case["raw_scores"] = score()
    elif changed == "blank_score_error": case["error"] = "  "
    else: case["audit_error"] = "  "
    (output / case_name).write_bytes(runs._bytes(case) + b"\n")
    (output / results_name).write_bytes(runs._bytes([case]) + b"\n")
    journal_path = output / "journal.json"
    journal = json.loads(journal_path.read_text())
    journal["cases"][0]["result_sha256"] = runs._sha((output / case_name).read_bytes())
    journal_path.write_bytes(runs._bytes(journal) + b"\n")
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name in (case_name, results_name, "journal.json"):
        receipt["artifacts"][name] = runs._sha((output / name).read_bytes())
    receipt_path.write_bytes(runs._bytes(receipt) + b"\n")
    manifest_path = output / "bundle_000001.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("journal.json", "receipt.json"):
        manifest["execution"]["references"][name]["sha256"] = runs._sha((output / name).read_bytes())
    manifest_path.write_bytes(runs._bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="Invalid raw response was promoted"):
        runs.validate_guarded_result_manifest(projection, output / results_name, manifest_path)


@pytest.mark.parametrize("changed", ["case_result", "stage_intent", "journal", "result_array", "manifest_path_escape"])
def test_guarded_binding_rejects_execution_tampering(projection, tmp_path, changed):
    plan = plan_for(projection, tmp_path); execute(plan, tmp_path, Factory(plan))
    output = Path(plan.output_directory)
    if changed == "case_result": path = output / "cases/case_000001.json"
    elif changed == "stage_intent": path = output / "stages/case_000001_score.intent.json"
    elif changed == "journal": path = output / "journal.json"
    elif changed == "result_array": path = output / "bundle_000001.results.json"
    else:
        path = output / "bundle_000001.manifest.json"
        value = json.loads(path.read_text()); value["execution"]["references"]["plan.json"]["path"] = "../plan.json"
        path.write_text(json.dumps(value))
    if changed != "manifest_path_escape": path.write_text('{"tampered":true}')
    with pytest.raises(ValueError): runs.validate_guarded_result_manifest(projection,
        output / "bundle_000001.results.json", output / "bundle_000001.manifest.json")


@pytest.mark.parametrize("stage", ["score", "citation_audit"])
@pytest.mark.parametrize("error_type", [ProviderError, RunBudgetExceeded])
@pytest.mark.parametrize("known_cost", [False, True])
def test_stop_policy_halts_provider_and_budget_failures_even_with_known_cost(
        projection, tmp_path, stage, error_type, known_cost):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    usage = Usage(input_tokens=23, output_tokens=7, api_calls=2,
        estimated_cost_usd=.000058 if known_cost else None)
    factory = Factory(plan, {("q1", stage): error_type(usage=usage)})
    receipt = execute(plan, tmp_path, factory)
    output = Path(plan.output_directory)
    result = json.loads((output / "cases/case_000001.json").read_text())
    assert receipt["fatal_error"] == error_type.__name__
    assert receipt["committed_results"] == 1 and receipt["reserved_usd"] == "0.05"
    assert [a.stage for a in factory.admissions] == (["score"] if stage == "score" else ["score", "citation_audit"])
    assert factory.closed == len(factory.calls) == len(factory.admissions)
    assert result["execution_identity"] == plan.cases[0].model_dump(mode="json")
    attempted = result["stage_usage"][stage]
    assert attempted == {"input_tokens": 23, "output_tokens": 7, "api_calls": 2,
        "estimated_cost_usd": .000058 if known_cost else None, "token_counts_complete": False}
    assert result["stage_diagnostics"][stage]["raw_response"] is None
    assert result["scores"] == (None if stage == "score" else score())
    assert result["audit"] is None
    assert result["error" if stage == "score" else "audit_error"] == error_type.__name__
    assert not (output / "cases/case_000002.json").exists()
    if stage == "score":
        assert not (output / "stages/case_000001_citation_audit.intent.json").exists()
    assert receipt["coverage_counts"]["not_dispatched"] == 1
    assert receipt["unknown_cost_stages"] == (0 if known_cost else 1)
    assert runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")["committed_results"] == 1


@pytest.mark.parametrize("stage", ["score", "citation_audit"])
@pytest.mark.parametrize("missing", ["estimated_cost_usd", "api_calls", "input_tokens", "output_tokens", "all"])
def test_stop_policy_keeps_valid_raw_grade_but_halts_incomplete_usage(projection, tmp_path, stage, missing):
    plan = plan_for(projection, tmp_path, output_protocol="native_schema_v1", stop_on_invalid_response=True)
    usage = {"input_tokens": 23, "output_tokens": 7, "api_calls": 2, "estimated_cost_usd": .000058}
    if missing != "all": usage[missing] = None
    raw = json.dumps(score() if stage == "score" else audit())
    def unknown(*_args):
        fields = {} if missing == "all" else {"usage": usage}
        return SimpleNamespace(text=raw, truncated=False, **fields)
    factory = Factory(plan, {("q1", stage): unknown})
    receipt = execute(plan, tmp_path, factory)
    output = Path(plan.output_directory)
    result = json.loads((output / "cases/case_000001.json").read_text())
    assert receipt["fatal_error"] == "IncompleteEvaluatorUsage"
    assert receipt["committed_results"] == 1 and receipt["reserved_usd"] == "0.05"
    assert [a.stage for a in factory.admissions] == (["score"] if stage == "score" else ["score", "citation_audit"])
    assert len(factory.calls) == len(factory.admissions) == factory.closed
    assert result["scores"] == result["raw_scores"] == score()
    assert result["execution_identity"] == plan.cases[0].model_dump(mode="json")
    assert result["audit"] == (None if stage == "score" else audit())
    assert result["error"] is None
    assert result["audit_error"] == ("EvaluatorRunError" if stage == "score" else None)
    retained = result["stage_diagnostics"][stage]["raw_response"]
    assert retained["complete"] is True and (output / retained["path"]).read_text() == raw
    attempted = result["stage_usage"][stage]
    assert attempted["estimated_cost_usd"] is None if missing in {"all", "estimated_cost_usd"} else attempted["estimated_cost_usd"] == .000058
    assert attempted["api_calls"] is None if missing in {"all", "api_calls"} else attempted["api_calls"] == 2
    assert not (output / "cases/case_000002.json").exists()
    assert receipt["coverage_counts"]["not_dispatched"] == 1
    if stage == "score":
        assert not (output / "stages/case_000001_citation_audit.intent.json").exists()
    assert runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")["committed_results"] == 1


def _repin_fixture_stop_policy(output):
    """Change a completed permissive fixture into an otherwise consistent strict one.

    This deliberately updates every outer hash, so only the admission-history
    policy can detect the continued stages, rather than a stale file checksum.
    """
    def read(name): return json.loads((output / name).read_text())
    def write(name, value): (output / name).write_bytes(runs._bytes(value) + b"\n")
    plan = read("plan.json"); plan["stop_on_invalid_response"] = True
    write("plan.json", plan)
    plan_hash = runs.evaluator_plan_sha256(runs.EvaluatorPlan.model_validate(plan))
    grant = read("approval.json"); grant["plan_sha256"] = plan_hash; write("approval.json", grant)
    journal = read("journal.json"); journal["plan_sha256"] = plan_hash; write("journal.json", journal)
    receipt = read("receipt.json")
    receipt.update(plan_sha256=plan_hash, stop_on_invalid_response=True,
        approval_sha256=runs._sha((output / "approval.json").read_bytes()), fatal_error="IncompleteEvaluatorUsage")
    for name in receipt["artifacts"]:
        receipt["artifacts"][name] = runs._sha((output / name).read_bytes())
    write("receipt.json", receipt)
    manifest = read("bundle_000001.manifest.json"); manifest["execution"]["plan_sha256"] = plan_hash
    for name, reference in manifest["execution"]["references"].items():
        reference["sha256"] = runs._sha((output / name).read_bytes())
    write("bundle_000001.manifest.json", manifest)


@pytest.mark.parametrize("boundary", ["audit_after_unknown_score", "case_after_unknown_audit", "case_after_known_provider_failure"])
def test_stop_policy_binding_rejects_fully_repinned_continuation(projection, tmp_path, boundary):
    only_one = boundary == "audit_after_unknown_score"
    plan = plan_for(projection, tmp_path, selection={"fixed": ["q1"]} if only_one else None,
        output_protocol="native_schema_v1", stop_on_invalid_response=False)
    stage = "citation_audit" if boundary == "case_after_unknown_audit" else "score"
    if boundary == "case_after_known_provider_failure":
        behavior = ProviderError(usage=Usage(input_tokens=23, output_tokens=7, api_calls=2, estimated_cost_usd=.000058))
    else:
        def behavior(*_args):
            return SimpleNamespace(text=json.dumps(score() if stage == "score" else audit()), truncated=False)
    factory = Factory(plan, {("q1", stage): behavior})
    receipt = execute(plan, tmp_path, factory)
    assert receipt["fatal_error"] is None
    assert len(factory.calls) == (2 if only_one else 3 if boundary == "case_after_known_provider_failure" else 4)
    output = Path(plan.output_directory)
    # This same history remains valid under the explicitly permissive fixture policy.
    runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
        output / "bundle_000001.manifest.json")
    _repin_fixture_stop_policy(output)
    with pytest.raises(ValueError, match="continued after a provider failure or incomplete usage"):
        runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
            output / "bundle_000001.manifest.json")


@pytest.mark.parametrize("missing", ["input_tokens", "output_tokens"])
def test_stop_policy_binding_rejects_missing_tokens_despite_repinned_complete_flag(projection, tmp_path, missing):
    plan = plan_for(projection, tmp_path, selection={"fixed": ["q1"]},
        output_protocol="native_schema_v1", stop_on_invalid_response=False)
    usage = {"input_tokens": 23, "output_tokens": 7, "api_calls": 2, "estimated_cost_usd": .000058}
    usage[missing] = None
    def unknown(*_args): return SimpleNamespace(text=json.dumps(score()), truncated=False, usage=usage)
    factory = Factory(plan, {("q1", "score"): unknown})
    execute(plan, tmp_path, factory)
    assert [a.stage for a in factory.admissions] == ["score", "citation_audit"]
    output = Path(plan.output_directory)
    def read(name): return json.loads((output / name).read_text())
    def write(name, value): (output / name).write_bytes(runs._bytes(value) + b"\n")
    outcome_name = "stages/case_000001_score.outcome.json"
    outcome = read(outcome_name); outcome["usage"]["token_counts_complete"] = True; write(outcome_name, outcome)
    result_name = "cases/case_000001.json"
    result = read(result_name)
    result["stage_usage"]["score"]["token_counts_complete"] = True
    result["usage"] = runs._sum_usage(list(result["stage_usage"].values()))
    write(result_name, result); write("bundle_000001.results.json", [result])
    journal = read("journal.json")
    journal["cases"][0]["result_sha256"] = runs._sha((output / result_name).read_bytes())
    write("journal.json", journal)
    receipt = read("receipt.json")
    receipt["reported_usage"] = result["usage"]; write("receipt.json", receipt)
    _repin_fixture_stop_policy(output)
    with pytest.raises(ValueError, match="continued after a provider failure or incomplete usage"):
        runs.validate_guarded_result_manifest(projection, output / "bundle_000001.results.json",
            output / "bundle_000001.manifest.json")

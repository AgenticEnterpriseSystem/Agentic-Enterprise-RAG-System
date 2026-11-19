"""End-to-end offline joins from the guarded runner through the RC2 bridge."""

import hashlib
import json
from pathlib import Path
import socket

import pytest

from src.evaluation import evaluator_bridge as bridge
from src.evaluation import evaluator_runs as runs
from src.evaluation.experiment_io import RunIntegrityError
from tests.test_evaluator_runs import Factory, bundle, execute, plan_for


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Guarded binding tests cannot access the network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def finished(tmp_path, *, behavior=None):
    projection = bundle(tmp_path / "source")
    plan = plan_for(projection, tmp_path)
    factory = Factory(plan, behavior)
    execute(plan, tmp_path, factory)
    output = Path(plan.output_directory)
    return projection, plan, output, output / "bundle_000001.results.json", output / "bundle_000001.manifest.json"


def test_guarded_results_bind_with_actual_runner_and_explicit_fixture_context(tmp_path):
    projection, plan, output, results, manifest = finished(tmp_path)
    source = json.loads(manifest.read_text())
    assert source["evaluator"] == "guarded-rc2-v1"
    assert source["sha256"]["runner_code"] == hashlib.sha256(Path(runs.__file__).read_bytes()).hexdigest()
    assert source["sha256"]["runner_code"] != bridge.EVALUATOR_RUNNER_SHA256
    bound = bridge.bind_candidate_results(projection, results, manifest, tmp_path / "bound.json")
    assert bound["guarded_execution"]["source_and_journal_verified"] is True
    assert bound["guarded_execution"]["execution_context"] == "fixture"
    assert bound["human_verified"] is False
    assert bound["returned_judge_results"] == 2
    assert all(row["judge_disposition"] == "scored_unverified" for row in bound["rows"])


@pytest.mark.parametrize("filename", ["journal.json", "receipt.json", "plan.json",
                                      "approval.json", "bundle_000001.results.json"])
def test_any_changed_execution_record_is_rejected_by_bridge(tmp_path, filename):
    projection, plan, output, results, manifest = finished(tmp_path)
    path = output / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RunIntegrityError):
        bridge.bind_candidate_results(projection, results, manifest, tmp_path / "never-bound.json")
    assert not (tmp_path / "never-bound.json").exists()


def test_guarded_result_cannot_be_relabelled_as_old_cloud_runner(tmp_path):
    projection, plan, output, results, manifest = finished(tmp_path)
    source = json.loads(manifest.read_text())
    source["evaluator"] = "v3-scoring-v6-review-candidate-1"
    source["sha256"]["runner_code"] = bridge.EVALUATOR_RUNNER_SHA256
    manifest.write_text(json.dumps(source))
    with pytest.raises(RunIntegrityError, match="legacy runner provenance"):
        bridge.bind_candidate_results(projection, results, manifest, tmp_path / "never-bound.json")


def test_partial_audit_error_keeps_score_and_its_exact_case(tmp_path):
    behavior = {("q1", "citation_audit"): ValueError("scripted malformed audit")}
    projection, plan, output, results, manifest = finished(tmp_path, behavior=behavior)
    bound = bridge.bind_candidate_results(projection, results, manifest, tmp_path / "bound.json")
    first = next(row for row in bound["rows"] if row["identity"]["question_id"] == "q1")
    assert first["judge_result"]["scores"] is not None
    assert first["judge_result"]["audit"] is None
    assert first["judge_result"]["audit_error"]
    assert first["identity"]["runtime_case_sha256"] == plan.cases[0].runtime_case_sha256
    assert bound["human_verified"] is False

"""Verified, offline experiment persistence and separate scoring handoff.

Fixtures use genuine tiny files and a deterministic temporary Git repository.
No inference, benchmark corpus, database or external judge is invoked.
"""

from dataclasses import asdict, replace as dataclass_replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation import experiment_io as experiment_io_module
from src.contracts import Answer, Chunk, PassageEvidence, QueryResponse, Trace, Usage
from src.evaluation.experiment_contracts import (
    ExperimentManifest, RuntimeCaseExport, RuntimeFailure, ScoringInput, runtime_case_digest,
)
from src.evaluation.experiment_io import (
    ExperimentIOError, RunDirectory, RunIntegrityError, WriterLockError,
    export_scoring_inputs, to_answer_evaluation_input, validate_pair,
)
from src.evaluation.experiment_inputs import verify_inputs
from tests.test_experiment_inputs import (
    FIXTURE_CANONICAL_TEXT, FIXTURE_DOCUMENT_ID, GOLD_MARKER, make_verified_inputs_fixture,
)


def replace(model, **changes):
    return type(model).model_validate(model.model_dump() | changes)


@pytest.fixture
def verified_fixture(tmp_path):
    return make_verified_inputs_fixture(tmp_path / "verified-inputs")


def passage(manifest, inputs):
    entry = json.loads(inputs.files.catalog.read_text().splitlines()[0])
    text = FIXTURE_CANONICAL_TEXT
    key = f"{FIXTURE_DOCUMENT_ID}:0:{len(text)}"
    return PassageEvidence(citation_id=key, snapshot_id=manifest.scope.snapshot.snapshot_id,
        revision_id=entry["revision_id"], chunk=Chunk(chunk_id=key, doc_id=FIXTURE_DOCUMENT_ID,
        source_type="jira", title="Fixture", text=text, start_char=0, end_char=len(text),
        metadata_origin="synthetic"), score=1, retrieval_method="hybrid")


def case_for(manifest, inputs, question_id=None, *, outcome="answered", run_id=None):
    question_id = question_id or manifest.question_ids[0]
    question = next(item.question for item in inputs.questions if item.question_id == question_id)
    run_id = run_id or "run-" + manifest.workflow + "-" + question_id
    selected = (passage(manifest, inputs),)
    if outcome == "error":
        answer = Answer(status="error", text="The provider was unavailable.")
        reason, sources, generation = "error", (), ()
        failure = RuntimeFailure(code="provider_unavailable", stage="critic", retryable=True)
    elif outcome == "budget":
        answer = Answer(status="insufficient_evidence", text="Search budget reached.")
        reason, sources, generation, failure = "budget", selected, (), None
    elif outcome == "clarification":
        answer = Answer(status="insufficient_evidence", text="Which incident do you mean?",
                        citation_ids=(selected[0].citation_id,))
        reason, sources, generation, failure = "clarification", selected, (), None
    else:
        answer = Answer(status="answered", text="Verified fixture text.", citation_ids=(selected[0].citation_id,))
        reason = "single_pass" if manifest.workflow == "fixed" else "sufficient"
        sources, generation, failure = selected, selected, None
    return RuntimeCaseExport(manifest_id=manifest.manifest_id, run_id=run_id,
        question_id=question_id, question=question, question_sha256=hashlib.sha256(question.encode()).hexdigest(),
        response=QueryResponse(snapshot=manifest.scope.snapshot, answer=answer, sources=sources,
            trace=Trace(run_id=run_id, stop_reason=reason, usage=Usage(api_calls=1, estimated_cost_usd=None))),
        generation_evidence=generation, error=failure)


def file_bytes(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def write_complete_run(path, manifest, inputs, *, reverse=False, outcomes=None):
    ids = tuple(reversed(manifest.question_ids)) if reverse else manifest.question_ids
    with RunDirectory.create(path, manifest, inputs) as run:
        for qid in ids:
            run.append_case(case_for(manifest, inputs, qid, outcome=(outcomes or {}).get(qid, "answered")))
    return RunDirectory.load(path)


def references(path, inputs, *, omit_answers=False):
    rows = [{"question_id": q.question_id, "question": q.question,
             **({"reference_answer": None} if omit_answers else {"reference_answer": "SCORING_ONLY_GOLD_SENTINEL",
                 "answer_facts": ["SCORING_ONLY_FACT_SENTINEL"], "expected_doc_ids": [FIXTURE_DOCUMENT_ID]})}
            for q in inputs.questions]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return rows


def assert_no_gold(path):
    forbidden = {"reference_answer", "gold_answer", "answer_facts", "expected_doc_ids", "reference"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for file in path.rglob("*"):
        if file.is_file() and file.suffix == ".json":
            content = file.read_text()
            assert "SCORING_ONLY_GOLD_SENTINEL" not in content
            assert "SCORING_ONLY_FACT_SENTINEL" not in content
            visit(json.loads(content))


def test_exclusive_creation_does_not_overwrite_existing_files(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "existing"
    path.mkdir()
    (path / "keep.txt").write_text("Existing user data")
    before = file_bytes(path)
    with pytest.raises((FileExistsError, ExperimentIOError)):
        RunDirectory.create(path, manifest, inputs)
    assert file_bytes(path) == before


@pytest.mark.parametrize("destination", ["unignored", "git_metadata"])
def test_run_output_cannot_dirty_source_or_write_git_metadata(verified_fixture, destination):
    manifest, inputs = verified_fixture
    path = (inputs.files.source_repo / "unignored-run" if destination == "unignored" else
            inputs.files.source_repo / ".git" / "unsafe-run")
    with pytest.raises(ExperimentIOError):
        RunDirectory.create(path, manifest, inputs)
    assert not path.exists()
    assert inputs.recheck(manifest).hashes == inputs.hashes


def test_gitignored_output_resumes_without_dirtying_source_or_recording_input_paths(verified_fixture):
    manifest, inputs = verified_fixture
    path = inputs.files.source_repo / "ignored" / "new-run"
    with RunDirectory.create(path, manifest, inputs) as run:
        run.append_case(case_for(manifest, inputs))
    with RunDirectory.resume(path, manifest, inputs) as resumed:
        assert resumed.pending_question_ids == manifest.question_ids[1:]
    ledger_text = (path / "inputs.json").read_text()
    ledger = json.loads(ledger_text)
    assert set(ledger) == {"schema_version", "hashes", "manifest_settings_sha256", "questions"}
    assert all(set(question) == {"question_id", "question"} for question in ledger["questions"])
    assert str(inputs.files.source_repo.parent) not in ledger_text
    assert GOLD_MARKER not in ledger_text
    assert inputs.recheck(manifest).hashes == inputs.hashes
    assert_no_gold(path)


def test_writer_lock_contends_but_readonly_snapshot_never_releases_another_writer(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs) as writer:
        writer.append_case(case_for(manifest, inputs))
        before = file_bytes(path)
        reader = RunDirectory.load(path)
        assert len(reader.load_cases()) == 1
        with pytest.raises(ExperimentIOError):
            reader.append_case(case_for(manifest, inputs, manifest.question_ids[1]))
        reader.close()
        assert file_bytes(path) == before
        with pytest.raises(WriterLockError):
            RunDirectory.resume(path, manifest, inputs)
    with RunDirectory.resume(path, manifest, inputs) as resumed:
        assert resumed.pending_question_ids == manifest.question_ids[1:]


def test_out_of_order_cases_are_atomic_ordinal_records_and_resume_preserves_creation(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs) as writer:
        assert writer.pending_question_ids == manifest.question_ids
        second = case_for(manifest, inputs, manifest.question_ids[1])
        target = writer.append_case(second)
        assert target.parent == path / "cases" and target.name == "case_000002.json"
        assert writer.pending_question_ids == (manifest.question_ids[0],)
        with pytest.raises(ValueError):
            writer.load_cases(require_complete=True)
    resumed_manifest = replace(manifest, created_at=manifest.created_at + timedelta(hours=1))
    with RunDirectory.resume(path, resumed_manifest, inputs) as resumed:
        assert resumed.manifest.created_at == manifest.created_at
        resumed.append_case(case_for(manifest, inputs))
        assert resumed.pending_question_ids == ()
    restored = RunDirectory.load(path).load_cases(require_complete=True)
    assert tuple(case.question_id for case in restored) == manifest.question_ids
    assert restored[1] == second


def test_question_ids_are_data_not_output_paths(tmp_path):
    ids = ("../escape", "/absolute/question", "..\\windows\\路径")
    manifest, inputs = make_verified_inputs_fixture(tmp_path / "inputs", question_ids=ids)
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs) as run:
        targets = [run.append_case(case_for(manifest, inputs, qid)) for qid in ids]
    assert {target.name for target in targets} == {"case_000001.json", "case_000002.json", "case_000003.json"}
    assert all(target.parent == path / "cases" for target in targets)
    assert not (tmp_path / "escape").exists()
    assert tuple(case.question_id for case in RunDirectory.load(path).load_cases(require_complete=True)) == ids


@pytest.mark.parametrize("outcome", ["error", "budget"])
def test_recorded_failures_and_budget_stops_are_completed_attempts_not_retried(tmp_path, verified_fixture, outcome):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    case = case_for(manifest, inputs, outcome=outcome)
    with RunDirectory.create(path, manifest, inputs) as run:
        run.append_case(case)
        before = file_bytes(path)
        with pytest.raises(ValueError):
            run.append_case(case)
        assert file_bytes(path) == before
    with RunDirectory.resume(path, manifest, inputs) as run:
        assert manifest.question_ids[0] not in run.pending_question_ids
        restored = run.load_cases()[0]
        assert restored == case and restored.response.trace.usage.estimated_cost_usd is None
        assert restored.generation_evidence == ()
        if outcome == "error":
            assert restored.error.retryable is True


def test_clarification_keeps_assessed_sources_without_inventing_generation_input(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    case = case_for(manifest, inputs, outcome="clarification")
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs) as run:
        run.append_case(case)
    restored = RunDirectory.load(path).load_cases()[0]
    assert restored.response.answer.citation_ids == (case.response.sources[0].citation_id,)
    assert restored.response.sources == case.response.sources
    assert restored.generation_evidence == ()
    assert runtime_case_digest(restored) == runtime_case_digest(case)


@pytest.mark.parametrize("failure", ["copied_invalid_case", "substituted_question", "duplicate_run", "duplicate_question"])
def test_append_revalidates_models_and_verified_question_identity_without_partial_outputs(tmp_path, verified_fixture, failure):
    manifest, inputs = verified_fixture
    with RunDirectory.create(tmp_path / "run", manifest, inputs) as run:
        first = case_for(manifest, inputs)
        run.append_case(first)
        second = case_for(manifest, inputs, manifest.question_ids[1])
        if failure == "copied_invalid_case":
            bad = second.model_copy(update={"question": "Changed without a matching hash"})
        elif failure == "substituted_question":
            text = "A validly hashed question absent from the frozen input"
            bad = replace(second, question=text, question_sha256=hashlib.sha256(text.encode()).hexdigest())
        elif failure == "duplicate_run":
            bad = replace(second, run_id=first.run_id, response=replace(second.response,
                trace=replace(second.response.trace, run_id=first.run_id)))
        else:
            bad = first
        before = file_bytes(tmp_path / "run")
        with pytest.raises(ValueError):
            run.append_case(bad)
        assert file_bytes(tmp_path / "run") == before


def test_copied_invalid_manifest_is_revalidated_before_persistence(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    bad = manifest.model_copy(update={"max_rounds": 2})
    with pytest.raises(ValueError):
        RunDirectory.create(tmp_path / "invalid-run", bad, inputs)
    assert not (tmp_path / "invalid-run" / "manifest.json").exists()


@pytest.mark.parametrize("corruption", ["truncated", "digest", "ordinal", "question_id", "unexpected", "partial_temp", "manifest", "inputs"])
def test_readonly_load_rejects_corrupt_headers_cases_or_uncommitted_outputs(tmp_path, verified_fixture, corruption):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs) as run:
        case_path = run.append_case(case_for(manifest, inputs))
    if corruption == "truncated":
        case_path.write_text('{"case":')
    elif corruption in {"digest", "ordinal", "question_id"}:
        payload = json.loads(case_path.read_text())
        if corruption == "digest":
            payload["case"]["response"]["answer"]["text"] = "Corrupted answer"
        elif corruption == "ordinal":
            payload["ordinal"] = 2
        else:
            payload["question_id"] = "outside-frozen-selection"
        case_path.write_text(json.dumps(payload))
    elif corruption == "unexpected":
        (path / "cases" / "case_999999.json").write_text(case_path.read_text())
    elif corruption == "partial_temp":
        (path / "cases" / "case_000002.json.partial").write_text('{"case":')
    else:
        header = path / ("manifest.json" if corruption == "manifest" else "inputs.json")
        header.write_text(header.read_text() + "\n ")
    with pytest.raises((RunIntegrityError, ValueError)):
        RunDirectory.load(path)


def test_stale_writer_lock_requires_manual_review_instead_of_silent_retry(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs):
        pass
    (path / ".writer.lock").write_text("abandoned synthetic lock")
    with pytest.raises(WriterLockError):
        RunDirectory.resume(path, manifest, inputs)


def rewrite_input_ledger_and_reseal(path, change):
    """Keep the outer byte checksum valid to test internal manifest bindings."""
    ledger_path = path / "inputs.json"
    ledger = json.loads(ledger_path.read_text())
    change(ledger)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True, ensure_ascii=False) + "\n")
    integrity_path = path / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["inputs_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity) + "\n")


@pytest.mark.parametrize("role", ["corpus", "splits", "index_manifest", "source_repo", "generator_prompt"])
def test_reader_binds_input_fingerprints_to_manifest_after_outer_digest_is_recomputed(
        tmp_path, verified_fixture, role):
    manifest, inputs = verified_fixture
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, inputs):
        pass
    rewrite_input_ledger_and_reseal(path, lambda ledger: ledger["hashes"].__setitem__(role, "f" * 64))
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


@pytest.mark.parametrize("role", ["critic_prompt", "router_prompt"])
def test_reader_rejects_undeclared_optional_prompts_even_with_a_valid_outer_digest(
        tmp_path, verified_fixture, role):
    manifest, inputs = verified_fixture
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, inputs):
        pass
    rewrite_input_ledger_and_reseal(path, lambda ledger: ledger["hashes"].__setitem__(role, "f" * 64))
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


@pytest.mark.parametrize("role", ["critic_prompt", "router_prompt"])
@pytest.mark.parametrize("change", ["hash", "missing"])
def test_reader_requires_declared_optional_prompt_presence_and_matching_hash(tmp_path, role, change):
    manifest, inputs = make_verified_inputs_fixture(tmp_path / "inputs", workflow="adaptive")
    if role == "router_prompt":
        router = inputs.files.generator_prompt.parent / "router.txt"
        router.write_text("Route only permitted structured and text queries.\n")
        manifest = replace(manifest, router_prompt_sha256=hashlib.sha256(router.read_bytes()).hexdigest())
        inputs = verify_inputs(manifest, dataclass_replace(inputs.files, router_prompt=router))
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, inputs):
        pass
    def corrupt(ledger):
        if change == "missing":
            ledger["hashes"].pop(role)
        else:
            ledger["hashes"][role] = "f" * 64
    rewrite_input_ledger_and_reseal(path, corrupt)
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


@pytest.mark.parametrize("missing", ["final_freeze", "calibration_artifact"])
def test_final_reader_requires_both_freeze_and_calibration_fingerprints(tmp_path, missing):
    manifest, inputs = make_verified_inputs_fixture(tmp_path / "inputs", split="final")
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, inputs):
        pass
    rewrite_input_ledger_and_reseal(path, lambda ledger: ledger["hashes"].pop(missing))
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


def test_nonfinal_reader_rejects_a_gated_calibration_artifact_but_allows_opaque_freeze(
        tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    freeze = inputs.files.generator_prompt.parent / "draft-freeze.json"
    freeze.write_text(json.dumps({"accepted": False, "note": "Opaque draft; calibration not established"}))
    verified = verify_inputs(manifest, dataclass_replace(inputs.files, final_freeze=freeze))
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, verified):
        pass
    assert RunDirectory.load(path).manifest.split == "development"
    ledger = json.loads((path / "inputs.json").read_text())
    assert "final_freeze" in ledger["hashes"] and "calibration_artifact" not in ledger["hashes"]
    rewrite_input_ledger_and_reseal(path, lambda value: value["hashes"].__setitem__("calibration_artifact", "f" * 64))
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


def test_reader_binds_whitelisted_question_text_to_manifest_even_before_cases_exist(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "runtime"
    with RunDirectory.create(path, manifest, inputs):
        pass
    rewrite_input_ledger_and_reseal(path, lambda ledger: ledger["questions"][0].__setitem__(
        "question", "A substituted question with the same ID"))
    with pytest.raises(RunIntegrityError):
        RunDirectory.load(path)


@pytest.mark.parametrize("after_publication", [False, True])
def test_interrupted_case_publication_never_leaves_a_case_and_index_disagreeing(
        tmp_path, verified_fixture, monkeypatch, after_publication):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    case = case_for(manifest, inputs)
    with RunDirectory.create(path, manifest, inputs) as writer:
        with monkeypatch.context() as patch:
            if after_publication:
                original = experiment_io_module._sync_directory
                pending = [True]
                def sync(directory):
                    if Path(directory) == path / "cases" and pending[0]:
                        pending[0] = False
                        raise OSError("Injected failure after atomic publication")
                    return original(directory)
                patch.setattr(experiment_io_module, "_sync_directory", sync)
            else:
                def reject_publication(*args, **kwargs):
                    raise OSError("Injected failure before atomic publication")
                patch.setattr(experiment_io_module.os, "link", reject_publication)
            with pytest.raises(OSError, match="Injected failure"):
                writer.append_case(case)
        cases = writer.load_cases()
        assert cases == ((case,) if after_publication else ())
        assert (case.question_id in writer.pending_question_ids) is not after_publication
        assert not any(file.name.startswith(".tmp-") for file in (path / "cases").iterdir())
    with RunDirectory.resume(path, manifest, inputs) as resumed:
        if after_publication:
            with pytest.raises(ValueError):
                resumed.append_case(case)
        else:
            resumed.append_case(case)


@pytest.mark.parametrize("field", ["questions", "splits", "corpus", "catalog", "index_manifest",
                                    "retrieval_config", "comparison_config", "scope", "runtime_config", "generator_prompt"])
def test_resume_rechecks_actual_input_bytes_not_only_declared_hashes(tmp_path, verified_fixture, field):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs):
        pass
    before = file_bytes(path)
    source = getattr(inputs.files, field)
    source.write_bytes(source.read_bytes() + b"\n ")
    with pytest.raises(ValueError):
        RunDirectory.resume(path, manifest, inputs)
    assert file_bytes(path) == before


def test_resume_rechecks_source_checkout_and_frozen_runtime_configuration(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    path = tmp_path / "run"
    with RunDirectory.create(path, manifest, inputs):
        pass
    changed = replace(manifest, budgets=replace(manifest.budgets, max_context_chars=12345))
    with pytest.raises(ValueError):
        RunDirectory.resume(path, changed, inputs)
    tracked = subprocess.check_output(["git", "-C", str(inputs.files.source_repo), "ls-files"], text=True).splitlines()
    source = inputs.files.source_repo / tracked[0]
    source.write_bytes(source.read_bytes() + b"\nmodified source\n")
    with pytest.raises(ValueError):
        RunDirectory.resume(path, manifest, inputs)


def test_pairs_allow_different_order_and_retained_unknown_or_failed_outcomes(tmp_path):
    fixed, fi = make_verified_inputs_fixture(tmp_path / "fixed-inputs", workflow="fixed")
    adaptive, ai = make_verified_inputs_fixture(tmp_path / "adaptive-inputs", workflow="adaptive")
    left = write_complete_run(tmp_path / "fixed", fixed, fi, reverse=True,
                              outcomes={fixed.question_ids[1]: "error"})
    right = write_complete_run(tmp_path / "adaptive", adaptive, ai,
                               outcomes={adaptive.question_ids[1]: "clarification"})
    validate_pair(left, right)
    assert left.load_cases(require_complete=True)[1].error is not None
    assert right.load_cases(require_complete=True)[1].generation_evidence == ()
    with pytest.raises(ValueError):
        validate_pair(right, left)


def test_pairs_reject_unmatched_outcomes_and_reused_run_identity(tmp_path):
    fixed, fi = make_verified_inputs_fixture(tmp_path / "fixed-inputs", workflow="fixed")
    adaptive, ai = make_verified_inputs_fixture(tmp_path / "adaptive-inputs", workflow="adaptive")
    with RunDirectory.create(tmp_path / "fixed", fixed, fi) as left:
        left.append_case(case_for(fixed, fi, run_id="shared-run"))
    with RunDirectory.create(tmp_path / "adaptive", adaptive, ai) as right:
        right.append_case(case_for(adaptive, ai, adaptive.question_ids[1]))
    with pytest.raises(ValueError):
        validate_pair(tmp_path / "fixed", tmp_path / "adaptive", require_complete=False)
    with RunDirectory.resume(tmp_path / "adaptive", adaptive, ai) as right:
        right.append_case(case_for(adaptive, ai, run_id="shared-run"))
    with RunDirectory.resume(tmp_path / "fixed", fixed, fi) as left:
        left.append_case(case_for(fixed, fi, fixed.question_ids[1]))
    with pytest.raises(ValueError):
        validate_pair(tmp_path / "fixed", tmp_path / "adaptive")


def test_scoring_is_a_separate_raw_reference_hash_and_exact_runtime_digest_join(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    runtime = tmp_path / "runtime"
    run = write_complete_run(runtime, manifest, inputs)
    reference_file = tmp_path / "references.jsonl"
    references(reference_file, inputs)
    before = file_bytes(runtime)
    scoring = export_scoring_inputs(run, reference_file, tmp_path / "scoring")
    assert len(scoring) == len(manifest.question_ids)
    raw_reference_sha = hashlib.sha256(reference_file.read_bytes()).hexdigest()
    for item, case in zip(scoring, run.load_cases(require_complete=True)):
        assert isinstance(item, ScoringInput)
        assert item.runtime_case_sha256 == runtime_case_digest(case)
        assert item.reference.reference_set_sha256 == raw_reference_sha
        assert item.reference.question_id == case.question_id
        assert item.reference.reference_answer == "SCORING_ONLY_GOLD_SENTINEL"
        legacy = to_answer_evaluation_input(item)
        assert legacy.question_id == case.question_id
        assert legacy.reference_answer == item.reference.reference_answer
        assert [(entry.citation_id, entry.text) for entry in legacy.evidence] == [
            (source.citation_id, source.chunk.text) for source in case.generation_evidence]
    assert file_bytes(runtime) == before
    assert_no_gold(runtime)
    assert (tmp_path / "scoring" / "scoring_inputs.jsonl").is_file()
    with pytest.raises((FileExistsError, ExperimentIOError)):
        export_scoring_inputs(run, reference_file, tmp_path / "scoring")
    with pytest.raises(ValueError):
        export_scoring_inputs(run, reference_file, runtime / "scoring")


@pytest.mark.parametrize("change", ["text", "question_id", "duplicate", "missing"])
def test_scoring_reference_identity_changes_or_missing_rows_reject(tmp_path, verified_fixture, change):
    manifest, inputs = verified_fixture
    run = write_complete_run(tmp_path / "runtime", manifest, inputs)
    path = tmp_path / "references.jsonl"
    rows = references(path, inputs)
    if change == "text":
        rows[0]["question"] = "Substituted reference question"
    elif change == "question_id":
        rows[0]["question_id"] = "different-question"
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        export_scoring_inputs(run, path, tmp_path / "scoring")
    assert_no_gold(tmp_path / "runtime")


def test_absent_gold_stays_absent_and_legacy_adapter_does_not_invent_provenance(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    run = write_complete_run(tmp_path / "runtime", manifest, inputs)
    path = tmp_path / "references.jsonl"
    references(path, inputs, omit_answers=True)
    scoring = export_scoring_inputs(run, path, tmp_path / "scoring")
    assert all(item.reference.reference_answer is None for item in scoring)
    legacy = QueryResponse(answer=Answer(status="error", text="Unavailable"),
        trace=Trace(run_id=scoring[0].runtime_case.run_id, stop_reason="error"))
    forged_runtime = scoring[0].runtime_case.model_copy(update={"response": legacy})
    forged_scoring = scoring[0].model_copy(update={"runtime_case": forged_runtime})
    with pytest.raises(ValueError):
        to_answer_evaluation_input(forged_scoring)


def offline_cli(*arguments):
    program = '''
import importlib.abc, json, runpy, socket, sys
class OfflineOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','sentence_transformers','faiss','bm25s','sqlalchemy','psycopg','anthropic'}:
            raise AssertionError('Heavy/inference/database import: ' + fullname)
sys.meta_path.insert(0, OfflineOnly())
def no_network(*args, **kwargs):
    raise AssertionError('Network forbidden in experiment IO CLI')
socket.socket.connect = no_network
sys.argv = ['experiment_io', *json.loads(sys.argv[1])]
runpy.run_module('src.evaluation.experiment_io', run_name='__main__')
'''
    return subprocess.run([sys.executable, "-c", program, json.dumps(arguments)],
                          text=True, capture_output=True, timeout=30)


def test_cli_help_preflight_validation_and_scoring_are_offline_and_lightweight(tmp_path, verified_fixture):
    manifest, inputs = verified_fixture
    manifest_file, input_file = tmp_path / "manifest.json", tmp_path / "input-files.json"
    manifest_file.write_text(manifest.model_dump_json())
    input_file.write_text(json.dumps({key: str(value) if value is not None else None
                                     for key, value in asdict(inputs.files).items()}))
    run_path = tmp_path / "runtime"
    write_complete_run(run_path, manifest, inputs)
    reference_file = tmp_path / "references.jsonl"
    references(reference_file, inputs)
    commands = [("--help",), ("preflight", "--manifest", str(manifest_file), "--inputs", str(input_file)),
                ("validate", "--run", str(run_path), "--require-complete"),
                ("export-scoring", "--run", str(run_path), "--references", str(reference_file),
                 "--output", str(tmp_path / "scoring"))]
    for arguments in commands:
        result = offline_cli(*arguments)
        assert result.returncode == 0, result.stderr
    assert_no_gold(run_path)

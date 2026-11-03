"""Real tiny-file/Git provenance checks; no corpus/index/model/DB loaders."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from src.contracts import ScopeBinding, SnapshotRef
from src.evaluation.experiment_contracts import (
    ExperimentManifest, GenerationBudgets, ModelProvenance, RetrievalProvenance,
)
from src.evaluation.experiment_inputs import (
    InputFiles, RuntimeQuestion, canonical_json_sha256, capture_code_provenance,
    file_sha256, question_set_sha256, scientific_settings_sha256,
    select_runtime_questions, verify_inputs,
)
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint

FIXTURE_DOCUMENT_ID = "doc-one"
FIXTURE_CANONICAL_TEXT = "Fixture\nVerified fixture text."
GOLD_MARKER = "GOLD_SHOULD_STAY_SCORING_ONLY"


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _git(repo, *args):
    environment = {**os.environ,
        "GIT_AUTHOR_NAME": "Fixture", "GIT_COMMITTER_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "2026-11-03T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-11-03T00:00:00+00:00"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
        "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment).stdout


def make_verified_inputs_fixture(tmp_path: Path, *, workflow="fixed", split="development",
                                  question_ids=None, source_repo=None):
    """Shared IO-test fixture: real inputs and deterministic source Git history."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "inputs"
    folder.mkdir()
    repo = tmp_path / "source" if source_repo is None else source_repo
    if source_repo is None:
        repo.mkdir()
        _git(repo, "init", "-q", "--initial-branch=main")
        (repo / "source.py").write_text("VALUE = 1\n")
        (repo / ".gitignore").write_text("ignored/\n")
        _git(repo, "add", "source.py", ".gitignore")
        _git(repo, "commit", "-qm", "Fixture source")
    selected = tuple(question_ids) if question_ids is not None else ("q1", "q2")
    partitions = {name: [] for name in ("development", "calibration", "final")}
    partitions[split] = list(selected)
    used = set(selected)
    for name in partitions:
        if name == split:
            continue
        qid = name + "-fixture"
        while qid in used:
            qid += "-extra"
        partitions[name] = [qid]
        used.add(qid)
    rows = [{"question_id": qid, "question": "Question about " + qid,
             "gold_answer": GOLD_MARKER, "expected_doc_ids": [FIXTURE_DOCUMENT_ID],
             "question_type": "basic", "source_types": ["jira"]}
            for qid in sorted(used, reverse=True)]
    question_path = folder / "questions.json"
    _write_json(question_path, rows)
    split_path = folder / "splits.json"
    _write_json(split_path, {"version": 1, "splits": partitions})
    corpus = folder / "corpus.jsonl"
    _write_json(corpus, {"doc_id": FIXTURE_DOCUMENT_ID, "source_type": "jira",
                         "title": "Fixture", "content": "Verified fixture text."})
    entry = CatalogEntry(doc_id=FIXTURE_DOCUMENT_ID, revision_id=canonical_revision_id(FIXTURE_CANONICAL_TEXT),
        position=0, source_type="jira", text_length=len(FIXTURE_CANONICAL_TEXT), metadata_origin="synthetic")
    catalog_path = folder / "catalog.jsonl"
    _write_json(catalog_path, entry.model_dump(mode="json"))
    snapshot = SnapshotRef(snapshot_id="fixture-snapshot", corpus_sha256=file_sha256(corpus),
                           catalog_sha256=catalog_fingerprint((entry,), expected_count=1), document_count=1)
    binding = ScopeBinding(scope_id="fixture-scope", user_id="fixture-user", snapshot=snapshot,
                           policy_revision=1, session_version=1)
    scope = {"binding": binding.model_dump(mode="json"),
             "allowed_doc_ids": [FIXTURE_DOCUMENT_ID], "origin": "synthetic"}
    scope_path = folder / "scope.json"
    _write_json(scope_path, scope)
    index = folder / "index.json"
    _write_json(index, {"fixture": True, "documents": 1, "corpus_sha256": snapshot.corpus_sha256})
    retrieval_value = {"method": "fixture-hybrid", "candidate_k": 5}
    comparison_value = {"generator": "fixture", "evidence_budget": 24000}
    retrieval_path, comparison_path = folder / "retrieval.json", folder / "comparison.json"
    _write_json(retrieval_path, retrieval_value)
    _write_json(comparison_path, comparison_value)
    generator = folder / "generator.txt"
    generator.write_text("Answer using only supplied evidence.\n", encoding="utf-8")
    critic = folder / "critic.txt" if workflow == "adaptive" else None
    if critic:
        critic.write_text("Check the requested facts.\n", encoding="utf-8")
    runtime_path = folder / "runtime.json"
    values = dict(experiment_id="fixture-comparison", manifest_id=workflow, workflow=workflow,
        created_at=datetime(2026, 11, 3, tzinfo=timezone.utc), code=capture_code_provenance(repo),
        split=split, split_manifest_sha256=file_sha256(split_path),
        question_set_sha256=question_set_sha256(tuple(RuntimeQuestion(question_id=qid, question="Question about " + qid)
                                                     for qid in selected)),
        question_ids=selected, scope=binding, scope_fingerprint_sha256=canonical_json_sha256(scope),
        generation_model=ModelProvenance(provider="fixture", model_id="fixture-generator"),
        generator_prompt_sha256=file_sha256(generator), critic_prompt_sha256=file_sha256(critic) if critic else None,
        comparison_config_sha256=canonical_json_sha256(comparison_value),
        retrieval=RetrievalProvenance(retriever_id="fixture-hybrid", snapshot=snapshot,
            index_manifest_sha256=file_sha256(index), config_sha256=canonical_json_sha256(retrieval_value)),
        budgets=GenerationBudgets(top_k=5, max_question_chars=8000, max_context_chars=24000,
            max_output_tokens=1024, max_run_tokens=16000, max_run_seconds=120, max_run_cost_usd=.1),
        max_rounds=3 if workflow == "adaptive" else 1)
    manifest = ExperimentManifest(**values)
    _write_json(runtime_path, {name: manifest.model_dump(mode="json")[name] for name in (
        "generation_model", "retrieval", "budgets", "temperature", "workflow", "max_rounds")})
    freeze_path = None
    if split == "final":
        calibration = folder / "calibration.json"
        _write_json(calibration, {"fixture": True, "note": "Not a claim of completed calibration"})
        freeze_path = folder / "freeze.json"
        _write_json(freeze_path, {"accepted": True, "settings_sha256": {workflow: scientific_settings_sha256(manifest)},
            "calibration_artifact": calibration.name, "calibration_sha256": file_sha256(calibration)})
    files = InputFiles(question_path, split_path, corpus, catalog_path, index, retrieval_path,
                       comparison_path, scope_path, runtime_path, generator, repo,
                       critic_prompt=critic, final_freeze=freeze_path)
    return manifest, verify_inputs(manifest, files)


def test_real_inputs_verify_and_runtime_selection_whitelists_fields_and_preserves_order(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path, question_ids=("q2", "q1"))
    assert tuple(q.question_id for q in verified.questions) == ("q2", "q1")
    assert GOLD_MARKER not in json.dumps([q.model_dump() for q in verified.questions])
    assert all(set(q.model_dump()) == {"question_id", "question"} for q in verified.questions)
    counted = select_runtime_questions(verified.files.questions, verified.files.splits, "development", count=1)
    assert counted[0].question_id == "q1"
    assert question_set_sha256(verified.questions) == question_set_sha256(tuple(reversed(verified.questions)))
    assert verified.recheck(manifest).hashes == verified.hashes
    with pytest.raises(TypeError):
        verified.hashes["corpus"] = "changed"
    with pytest.raises(FrozenInstanceError):
        verified.files.corpus = tmp_path / "other"


def test_source_provenance_covers_staged_unstaged_untracked_and_ignored_files(tmp_path):
    _, verified = make_verified_inputs_fixture(tmp_path)
    repo = verified.files.source_repo
    clean = capture_code_provenance(repo)
    assert not clean.dirty and clean.diff_sha256 is None and len(clean.commit) == 40
    (repo / "ignored").mkdir()
    (repo / "ignored" / "not-source.txt").write_text("ignored")
    assert capture_code_provenance(repo) == clean
    source = repo / "source.py"
    source.write_text("VALUE = 'PRIVATE_SOURCE_CONTENT'\n")
    unstaged = capture_code_provenance(repo)
    assert unstaged.dirty and "PRIVATE_SOURCE_CONTENT" not in unstaged.model_dump_json()
    _git(repo, "add", "source.py")
    staged = capture_code_provenance(repo)
    assert staged.diff_sha256 != unstaged.diff_sha256
    binary = repo / "new-source.bin"
    binary.write_bytes(b"\x00first")
    first = capture_code_provenance(repo)
    binary.write_bytes(b"\x00second")
    assert capture_code_provenance(repo).diff_sha256 != first.diff_sha256


@pytest.mark.parametrize("change", ["duplicate-source", "missing-split-id", "overlap", "blank", "too-long"])
def test_invalid_source_or_split_never_becomes_a_runtime_selection(tmp_path, change):
    _, verified = make_verified_inputs_fixture(tmp_path)
    rows = json.loads(verified.files.questions.read_text())
    splits = json.loads(verified.files.splits.read_text())
    if change == "duplicate-source": rows.append(rows[0])
    elif change == "missing-split-id": splits["splits"]["final"] = ["absent"]
    elif change == "overlap": splits["splits"]["final"] = ["q1"]
    elif change == "blank": rows[0]["question"] = "  "
    else: rows[0]["question"] = "x" * 8001
    _write_json(verified.files.questions, rows)
    _write_json(verified.files.splits, splits)
    with pytest.raises(ValueError):
        select_runtime_questions(verified.files.questions, verified.files.splits, "development")


@pytest.mark.parametrize("count", [True, 1.0, 0, 3])
def test_count_is_strict_and_explicit_selection_must_belong_to_its_split(tmp_path, count):
    _, verified = make_verified_inputs_fixture(tmp_path)
    with pytest.raises(ValueError):
        select_runtime_questions(verified.files.questions, verified.files.splits, "development", count=count)
    with pytest.raises(ValueError):
        select_runtime_questions(verified.files.questions, verified.files.splits, "development", question_ids=("absent",))


def test_jsonl_question_inputs_use_the_same_whitelist(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    rows = json.loads(verified.files.questions.read_text())
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    files = replace(verified.files, questions=path)
    assert verify_inputs(manifest, files).questions == verified.questions


@pytest.mark.parametrize("role", ["corpus", "generator_prompt", "index_manifest", "retrieval_config", "comparison_config", "scope"])
def test_altered_actual_inputs_fail_even_when_manifest_is_unchanged(tmp_path, role):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    path = getattr(verified.files, role)
    if role in {"corpus", "generator_prompt"}: path.write_bytes(path.read_bytes() + b"changed")
    else: _write_json(path, {"changed": True})
    with pytest.raises(ValueError):
        verified.recheck(manifest)


def test_recheck_compares_raw_observed_files_even_when_canonical_question_values_match(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    rows = json.loads(verified.files.questions.read_text())
    rows[0]["gold_answer"] = "A different scoring-only reference"
    _write_json(verified.files.questions, rows)
    current = verify_inputs(manifest, verified.files)
    assert current.questions == verified.questions
    assert question_set_sha256(current.questions) == manifest.question_set_sha256
    with pytest.raises(ValueError, match="changed since"):
        verified.recheck(manifest)
    (verified.files.source_repo / "new.py").write_text("untracked change")
    with pytest.raises(ValueError, match="source provenance"):
        verify_inputs(manifest, verified.files)


@pytest.mark.parametrize("field,value", [("top_k", True), ("top_k", 1.0)])
def test_effective_config_rejects_python_equal_but_invalid_strict_budget_values(tmp_path, field, value):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    config = json.loads(verified.files.runtime_config.read_text())
    config["budgets"][field] = value
    _write_json(verified.files.runtime_config, config)
    with pytest.raises(ValidationError):
        verify_inputs(manifest, verified.files)


def test_effective_config_requires_exact_declared_shape_model_and_rounds(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    original = json.loads(verified.files.runtime_config.read_text())
    for changed in ({**original, "unexpected": True}, {**original, "max_rounds": True},
                    {**original, "generation_model": {"provider": "other", "model_id": "other"}}):
        _write_json(verified.files.runtime_config, changed)
        with pytest.raises(ValueError):
            verify_inputs(manifest, verified.files)


def test_catalog_streaming_verifies_count_order_and_canonical_fingerprint(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    entry = json.loads(verified.files.catalog.read_text())
    for content in ("", json.dumps({**entry, "position": 1}) + "\n",
                    json.dumps({**entry, "text_length": 1}) + "\n"):
        verified.files.catalog.write_text(content)
        with pytest.raises(ValueError):
            verify_inputs(manifest, verified.files)


def test_scope_digest_cannot_replace_its_embedded_binding(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    scope = json.loads(verified.files.scope.read_text())
    scope["binding"]["policy_revision"] = 2
    _write_json(verified.files.scope, scope)
    changed = ExperimentManifest.model_validate({**manifest.model_dump(),
        "scope_fingerprint_sha256": canonical_json_sha256(scope)})
    with pytest.raises(ValueError, match="binding differs"):
        verify_inputs(changed, verified.files)


@pytest.mark.parametrize("declaration", [
    {"corpus_sha256": "a" * 64}, {"catalog_sha256": "b" * 64},
    {"document_count": 2}, {"documents": True}, {"snapshot": {"snapshot_id": "other"}},
    {"bm25": {"stats": {"documents": 2}}}, {"dense": {"corpus_sha256": "a" * 64}},
    {"complete": False}, {"dense": {"complete": False}},
])
def test_known_index_identity_cannot_be_hidden_by_rehashing_a_contradictory_manifest(tmp_path, declaration):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    _write_json(verified.files.index_manifest, declaration)
    raw = manifest.model_dump()
    raw["retrieval"]["index_manifest_sha256"] = file_sha256(verified.files.index_manifest)
    changed = ExperimentManifest.model_validate(raw)
    with pytest.raises(ValueError, match="Index (metadata|manifest)"):
        verify_inputs(changed, verified.files)


def test_strict_json_rejects_nonfinite_and_duplicate_keys_and_not_json_values(tmp_path):
    _, verified = make_verified_inputs_fixture(tmp_path)
    for number in ("NaN", "1e999"):
        verified.files.questions.write_text('[{"question_id":"q1","question":"Q","gold_answer":' + number + '}]')
        with pytest.raises(ValueError):
            select_runtime_questions(verified.files.questions, verified.files.splits, "development")
    verified.files.questions.write_text('[{"question_id":"q1","question":"Q","question":"Changed"}]')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        select_runtime_questions(verified.files.questions, verified.files.splits, "development")
    for value in (float("nan"), {1: "non-string-key"}, Path("not-json")):
        with pytest.raises(ValueError):
            canonical_json_sha256(value)


def test_final_selection_requires_explicit_freeze_with_calibration_hash_and_containment(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path, split="final")
    with pytest.raises(ValueError, match="Final selection"):
        select_runtime_questions(verified.files.questions, verified.files.splits, "final")
    assert select_runtime_questions(verified.files.questions, verified.files.splits, "final", allow_final=True)
    with pytest.raises(ValueError):
        verify_inputs(manifest, replace(verified.files, final_freeze=None))
    path = verified.files.final_freeze
    original = json.loads(path.read_text())
    variants = ({**original, "accepted": 1}, {**original, "settings_sha256": {"fixed": "a" * 64}},
                {**original, "calibration_sha256": "a" * 64},
                {**original, "calibration_artifact": "../outside.json"})
    for changed in variants:
        _write_json(path, changed)
        with pytest.raises(ValueError):
            verify_inputs(manifest, verified.files)


def test_scientific_freeze_excludes_run_selection_and_includes_effective_science(tmp_path):
    manifest, verified = make_verified_inputs_fixture(tmp_path)
    changed = ExperimentManifest.model_validate({**manifest.model_dump(), "manifest_id": "other", "experiment_id": "other",
        "repetition": 2, "split": "calibration", "question_ids": ("other",),
        "question_set_sha256": "a" * 64, "split_manifest_sha256": "b" * 64})
    assert scientific_settings_sha256(changed) == scientific_settings_sha256(manifest)
    changed = ExperimentManifest.model_validate({**manifest.model_dump(), "temperature": .5})
    assert scientific_settings_sha256(changed) != scientific_settings_sha256(manifest)

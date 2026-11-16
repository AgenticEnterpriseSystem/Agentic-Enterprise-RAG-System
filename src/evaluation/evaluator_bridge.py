"""Offline passage-evaluator projections with exact canonical case identities.

This module never constructs a provider or runs a judge. Reference answers are
projected separately after generation. Unsupported outcomes stay in the ledger;
their canonical scoring packets are copied unchanged, not flattened into prose.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.contracts import PassageEvidence
from src.evaluation.experiment_contracts import (
    ExperimentManifest, ScoringInput, runtime_case_digest, validate_runtime_exports,
)
from src.evaluation.experiment_io import (
    ExperimentIOError, RunDirectory, RunIntegrityError, _bytes, _checked, _json,
    _output_location, _read, _regular, _sha, _write_new,
)

BRIDGE_VERSION = "1"
EVALUATOR_COMMIT = "38fb5088cb479eeb89b74175168b60872e047b25"
EVALUATOR_RUNNER_SHA256 = "310543a9a67a61989c77e85622208d902e5ed045b6899dab7f3a608c289dbff2"
# Git object hashes pin the reviewed upstream bytes, including prompt contents.
EVALUATOR_BLOBS = {
    "src/evaluation/verified_judge.py": "43721493f53e16f5f2718bb870d662596c5bb88d",
    "src/evaluation/verified_judge_v6.py": "2f7543386f1fe7f302a989ca9c247f01ba6a408c",
    "src/evaluation/evaluator_candidate.py": "73e7991dd7bc0c9a39c7f103a0954dfc7d6dc5e4",
    "docs/evaluation_prompts/answer_grading_rc2.md": "2c2c39f0b1f6ccac90bbb37ca02de170a5c11af5",
    "docs/evaluation_prompts/citation_audit_v5.md": "c660cf44aaf6dcc34bd0a099087397da7ff7c9ae",
}


def project_scoring_input(scoring: ScoringInput, manifest: ExperimentManifest) -> dict:
    """Project one validated case, without making any factual quality judgment.

    Revalidate even frozen DTOs because model_copy(update=...) bypasses validators.
    Integrity failures raise; unsupported cases return explicit unscored outcomes.
    """
    scoring = _checked(ScoringInput, scoring, "scoring input")
    manifest = _checked(ExperimentManifest, manifest, "experiment manifest")
    case = scoring.runtime_case
    validate_runtime_exports(manifest, (case,))
    result = {
        "identity": {
            "manifest_id": case.manifest_id,
            "run_id": case.run_id,
            "question_id": case.question_id,
            "question_sha256": case.question_sha256,
            "runtime_case_sha256": runtime_case_digest(case),
        },
        "generation_status": case.response.answer.status,
        "stop_reason": case.response.trace.stop_reason,
        "runtime_failure": case.error.model_dump(mode="json") if case.error else None,
        "disposition": "eligible",
        "answer": None,
        "reference": None,
    }
    if case.error is not None:
        result["disposition"] = "runtime_error"
    elif case.response.trace.stop_reason == "clarification":
        result["disposition"] = "clarification"
    elif any(not isinstance(item, PassageEvidence)
             for item in (*case.response.sources, *case.generation_evidence)):
        result["disposition"] = "native_evidence_unsupported"
    elif not case.generation_evidence:
        result["disposition"] = "no_generation_input"
    elif scoring.reference.reference_answer is None:
        result["disposition"] = "reference_answer_missing"
    else:
        result["answer"] = {
            "question_id": case.question_id,
            "question": case.question,
            "generated_answer": case.response.answer.text,
            "status": case.response.answer.status,
            "citation_ids": list(case.response.answer.citation_ids),
            "evidence": [{"citation_id": item.citation_id, "text": item.chunk.text}
                         for item in case.generation_evidence],
        }
        result["reference"] = {"question_id": case.question_id,
                               "gold_answer": scoring.reference.reference_answer}
    return result


def project_batch(manifest: ExperimentManifest, packets: tuple[ScoringInput, ...], *,
                  require_complete: bool = True) -> tuple[dict, ...]:
    """One manifest per batch; retain its selection order and missing outcomes."""
    manifest = _checked(ExperimentManifest, manifest, "experiment manifest")
    packets = tuple(_checked(ScoringInput, item, "scoring input") for item in packets)
    validate_runtime_exports(manifest, tuple(item.runtime_case for item in packets),
                             require_complete=require_complete)
    references = {item.reference.reference_set_sha256 for item in packets}
    if len(references) > 1:
        raise RunIntegrityError("One projection batch requires one reference set")
    by_id = {item.runtime_case.question_id: item for item in packets}
    return tuple(project_scoring_input(by_id[qid], manifest) if qid in by_id else {
        "identity": {"manifest_id": manifest.manifest_id, "question_id": qid,
                     "run_id": None, "question_sha256": None, "runtime_case_sha256": None},
        "generation_status": None, "stop_reason": None, "runtime_failure": None,
        "disposition": "missing_committed_case", "answer": None, "reference": None,
    } for qid in manifest.question_ids)


def _candidate_sources() -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for name, expected in EVALUATOR_BLOBS.items():
        path = root / name
        if not _regular(path):
            raise RunIntegrityError("Reviewed evaluator source is unavailable")
        payload = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
        if blob != expected:
            raise RunIntegrityError("Evaluator source differs from the reviewed version")
        sources[name] = {"git_blob_sha1": blob, "sha256": _sha(payload)}
    return {"upstream_commit": EVALUATOR_COMMIT, "files": sources,
            "score_prompt": "docs/evaluation_prompts/answer_grading_rc2.md",
            "audit_prompt": "docs/evaluation_prompts/citation_audit_v5.md"}


def _projection_payloads(ledger):
    return {"answers.json": _bytes([row["answer"] for row in ledger if row["disposition"] == "eligible"]),
            "questions.json": _bytes([row["reference"] for row in ledger if row["disposition"] == "eligible"]),
            "case_ledger.jsonl": b"".join(_bytes({key: value for key, value in row.items()
                if key not in {"answer", "reference"}}) for row in ledger)}


def _validate_audit_schema(audit):
    """Check the promised JSON schema; this does not establish claim support."""
    if (not isinstance(audit, dict) or set(audit) != {"response_type", "claims", "coverage_confident", "explanation"}
            or audit["response_type"] not in {"answer", "partial_answer", "refusal"}
            or type(audit["coverage_confident"]) is not bool
            or not isinstance(audit["explanation"], str) or not audit["explanation"].strip()
            or not isinstance(audit["claims"], list) or len(audit["claims"]) > 30):
        raise RunIntegrityError("Invalid citation audit schema")
    for claim in audit["claims"]:
        if (not isinstance(claim, dict) or set(claim) != {"answer_quote", "central", "support",
                "citation_id", "evidence_quote", "target_relation"}
                or not isinstance(claim["answer_quote"], str) or not claim["answer_quote"].strip()
                or type(claim["central"]) is not bool
                or claim["support"] not in {"direct", "entailed", "related", "none"}
                or claim["target_relation"] not in {"same_target", "general_rule", "different_case", "ambiguous"}
                or any(value is not None and not isinstance(value, str)
                       for value in (claim["citation_id"], claim["evidence_quote"]))):
            raise RunIntegrityError("Invalid citation audit claim schema")


def export_projection(runtime_dir: str | Path, scoring_file: str | Path,
                      output_dir: str | Path, *, allow_partial: bool = False) -> dict:
    """Verify a stored run and publish a new private offline evaluator bundle.

    The input must include every committed case, even errors and unsupported
    outcomes. Missing selections require explicit opt-in and remain unscored.
    Existing files, runtime writer locks and paid journals are never modified.
    """
    run = RunDirectory.load(runtime_dir)
    cases = run.load_cases(require_complete=not allow_partial)
    scoring_path = Path(scoring_file).resolve()
    if not _regular(scoring_path):
        raise RunIntegrityError("Scoring input must be a regular file")
    scoring_bytes = scoring_path.read_bytes()
    packets = tuple(_checked(ScoringInput, _json(line), "scoring input")
                    for line in scoring_bytes.splitlines() if line.strip())
    ledger = project_batch(run.manifest, packets, require_complete=not allow_partial)
    actual = {case.question_id: runtime_case_digest(case) for case in cases}
    projected = {item.runtime_case.question_id: item.runtime_case_sha256 for item in packets}
    if actual != projected:
        raise RunIntegrityError("Scoring packets differ from the committed runtime cases")
    evaluator = _candidate_sources()
    # Import only the reviewed pure case parser, never a provider or cloud runner.
    from src.evaluation.verified_judge import make_case
    for row in ledger:
        if row["disposition"] == "eligible":
            make_case(row["answer"], row["reference"])
    manifest_bytes = (run.path / "manifest.json").read_bytes()
    payloads = _projection_payloads(ledger)
    payloads["scoring_inputs.jsonl"] = scoring_bytes
    payloads["runtime_manifest.json"] = manifest_bytes
    counts = dict(Counter(row["disposition"] for row in ledger))
    metadata = {
        "schema_version": "1", "bridge_version": BRIDGE_VERSION,
        "bridge_sha256": _sha(Path(__file__).read_bytes()),
        "experiment_id": run.manifest.experiment_id, "manifest_id": run.manifest.manifest_id,
        "workflow": run.manifest.workflow, "repetition": run.manifest.repetition,
        "runtime_manifest_sha256": _sha(manifest_bytes),
        "scoring_input_sha256": _sha(scoring_bytes),
        "reference_set_sha256": packets[0].reference.reference_set_sha256 if packets else None,
        "selected_cases": len(ledger), "committed_cases": len(packets),
        "eligible_cases": counts.get("eligible", 0), "disposition_counts": counts,
        "allow_partial": allow_partial, "candidate_shape_preflight_passed": True,
        "quality_scored": False, "judge_calls": 0, "evaluator": evaluator,
        "scope_validation": "Declared offline integrity only; no fresh authorization check",
        "files": {name: _sha(payload) for name, payload in payloads.items()},
    }
    output = Path(output_dir).absolute()
    runtime = run.path.resolve()
    if output.resolve() == runtime or runtime in output.resolve().parents:
        raise ExperimentIOError("Projection output must be separate from the runtime directory")
    output = _output_location(output)
    output.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        _write_new(output / name, payload)
    # Written last: an interrupted bundle has no completion manifest.
    _write_new(output / "manifest.json", _bytes(metadata))
    return metadata


def bind_candidate_results(bundle_dir: str | Path, results_file: str | Path,
                           evaluator_manifest: str | Path, output_file: str | Path) -> dict:
    """Bind external results to a verified bundle without reusing human labels.

    This verifies exact projection hashes, not judge quality. Missing/failed judge
    results remain null. No numeric average or completion status is inferred.
    """
    bundle = Path(bundle_dir)
    metadata, metadata_sha = _read(bundle / "manifest.json")
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != "1"
            or metadata.get("bridge_version") != BRIDGE_VERSION):
        raise RunIntegrityError("Unsupported evaluator projection bundle")
    required = {"answers.json", "questions.json", "case_ledger.jsonl",
                "scoring_inputs.jsonl", "runtime_manifest.json"}
    if not isinstance(metadata.get("files"), dict) or set(metadata["files"]) != required:
        raise RunIntegrityError("Incomplete projection file ledger")
    for name, expected in metadata["files"].items():
        if not _regular(bundle / name) or _sha((bundle / name).read_bytes()) != expected:
            raise RunIntegrityError("Projection file changed after export")
    # Rebuild the projection from its canonical packets rather than trusting
    # caller-editable ledger identities or self-declared file hashes alone.
    raw_manifest, runtime_sha = _read(bundle / "runtime_manifest.json")
    manifest = _checked(ExperimentManifest, raw_manifest, "experiment manifest")
    scoring_bytes = (bundle / "scoring_inputs.jsonl").read_bytes()
    packets = tuple(_checked(ScoringInput, _json(line), "scoring input")
                    for line in scoring_bytes.splitlines() if line.strip())
    if type(metadata.get("allow_partial")) is not bool:
        raise RunIntegrityError("Projection completeness must be explicit")
    projected = project_batch(manifest, packets, require_complete=not metadata["allow_partial"])
    for name, payload in _projection_payloads(projected).items():
        if (bundle / name).read_bytes() != payload:
            raise RunIntegrityError("Projection differs from its canonical packets")
    if (metadata.get("manifest_id") != manifest.manifest_id
            or metadata.get("experiment_id") != manifest.experiment_id
            or metadata.get("workflow") != manifest.workflow
            or metadata.get("repetition") != manifest.repetition
            or metadata.get("runtime_manifest_sha256") != runtime_sha
            or metadata.get("scoring_input_sha256") != _sha(scoring_bytes)
            or metadata.get("selected_cases") != len(projected)
            or metadata.get("committed_cases") != len(packets)
            or metadata.get("eligible_cases") != sum(row["disposition"] == "eligible" for row in projected)
            or metadata.get("disposition_counts") != dict(Counter(row["disposition"] for row in projected))
            or metadata.get("reference_set_sha256") != (packets[0].reference.reference_set_sha256 if packets else None)):
        raise RunIntegrityError("Projection manifest differs from its canonical inputs")
    evaluator = _candidate_sources()
    if metadata.get("evaluator") != evaluator:
        raise RunIntegrityError("Projection evaluator differs from the reviewed source")
    result_rows, results_sha = _read(Path(results_file))
    judge_manifest, judge_sha = _read(Path(evaluator_manifest))
    if not isinstance(result_rows, list) or not isinstance(judge_manifest, dict):
        raise RunIntegrityError("Candidate results must be an array with a manifest")
    # Candidate manifests bind their input bytes. Labels/paths alone are not joins.
    # The guarded local runner has its own real source and durable execution
    # provenance. It must never impersonate the older upstream runner hash.
    guarded_execution = None
    evaluator_kind = judge_manifest.get("evaluator")
    if evaluator_kind == "guarded-rc2-v1":
        from src.evaluation import evaluator_runs
        try:
            guarded_execution = evaluator_runs.validate_guarded_result_manifest(
                bundle, Path(results_file), Path(evaluator_manifest))
        except (ValueError, OSError, KeyError, TypeError):
            raise RunIntegrityError("Guarded judge execution differs from its frozen records") from None
        if (guarded_execution.get("source_and_journal_verified") is not True
                or guarded_execution.get("human_verified") is not False):
            raise RunIntegrityError("Guarded judge execution was not verified")
        runner_sha = _sha(Path(evaluator_runs.__file__).read_bytes())
    elif evaluator_kind == "v3-scoring-v6-review-candidate-1":
        runner_sha = EVALUATOR_RUNNER_SHA256
    else:
        raise RunIntegrityError("Unknown evaluator execution provenance")
    hashes = judge_manifest.get("sha256", {})
    expected = {"answers": metadata["files"]["answers.json"],
        "questions": metadata["files"]["questions.json"],
        "score_prompt": evaluator["files"][evaluator["score_prompt"]]["sha256"],
        "candidate_code": evaluator["files"]["src/evaluation/evaluator_candidate.py"]["sha256"],
        "verified_code": evaluator["files"]["src/evaluation/verified_judge.py"]["sha256"],
        "v6_audit_code": evaluator["files"]["src/evaluation/verified_judge_v6.py"]["sha256"],
        "runner_code": runner_sha}
    if judge_manifest.get("audit_mode") == "all":
        expected["audit_prompt"] = evaluator["files"][evaluator["audit_prompt"]]["sha256"]
    elif judge_manifest.get("audit_mode") != "none":
        raise RunIntegrityError("Unknown candidate audit mode")
    if (hashes != expected or not isinstance(judge_manifest.get("model"), str)
            or not judge_manifest["model"].strip()):
        raise RunIntegrityError("Judge manifest does not bind the reviewed RC2 inputs and code")
    ledger = [{key: value for key, value in row.items() if key not in {"answer", "reference"}}
              for row in projected]
    eligible = {row["identity"]["question_id"] for row in ledger if row["disposition"] == "eligible"}
    selected = judge_manifest.get("selected_ids")
    if (not isinstance(selected, list) or any(not isinstance(qid, str) for qid in selected)
            or len(selected) != len(set(selected)) or not set(selected) <= eligible):
        raise RunIntegrityError("Invalid selected judge question IDs")
    statuses = {row["identity"]["question_id"]: row["generation_status"] for row in ledger}
    from src.evaluation.verified_judge import validate_scores
    by_id = {}
    for row in result_rows:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise RunIntegrityError("Invalid candidate result identity")
        provenance = row.get("provenance", {})
        if (evaluator_kind == "v3-scoring-v6-review-candidate-1"
                and isinstance(provenance, dict)
                and provenance.get("execution_runner") == "guarded-rc2-v1"):
            raise RunIntegrityError("Guarded results cannot claim legacy runner provenance")
        qid = row["question_id"]
        if qid not in selected or qid in by_id:
            raise RunIntegrityError("Unknown, unsupported or duplicate candidate result")
        if row.get("generation_status") != statuses[qid]:
            raise RunIntegrityError("Judge result changed the generation outcome")
        if row.get("scores") is not None:
            validate_scores(row["scores"])
            if row.get("error") is not None:
                raise RunIntegrityError("Scored result also declares a scoring failure")
            if "raw_scores" in row and row["raw_scores"] != row["scores"]:
                raise RunIntegrityError("Candidate scores differ from the unmodified raw scores")
        elif not isinstance(row.get("error"), str) or not row["error"].strip():
            raise RunIntegrityError("Unscored result requires an explicit judge failure")
        if row.get("audit") is not None:
            if judge_manifest["audit_mode"] != "all":
                raise RunIntegrityError("Audit returned when audit mode was disabled")
            _validate_audit_schema(row["audit"])
        if any(row.get(key) is not None and not isinstance(row[key], str) for key in ("error", "audit_error")):
            raise RunIntegrityError("Invalid judge failure metadata")
        by_id[qid] = row
    envelope = {"schema_version": "1", "projection_manifest_sha256": metadata_sha,
        "candidate_results_sha256": results_sha, "candidate_manifest_sha256": judge_sha,
        "candidate_manifest": judge_manifest, "human_labels_reused": False,
        "selected_cases": len(ledger), "eligible_cases": len(eligible),
        "selected_judge_cases": len(selected), "human_verified": False,
        "audit_claim_schema_validated": True,
        "returned_judge_results": len(by_id), "rows": [
            {**row, "judge_result": by_id.get(row["identity"]["question_id"]),
             "judge_disposition": ("unsupported" if row["disposition"] != "eligible" else
                "not_selected" if row["identity"]["question_id"] not in selected else
                "missing_result" if row["identity"]["question_id"] not in by_id else
                "judge_error" if by_id[row["identity"]["question_id"]].get("error") else "scored_unverified")}
            for row in ledger]}
    if guarded_execution is not None:
        envelope["guarded_execution"] = guarded_execution
    output = _output_location(Path(output_file).absolute())
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_new(output, _bytes(envelope))
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project", help="Verify and project inputs; never run a model")
    project.add_argument("--run", type=Path, required=True)
    project.add_argument("--scoring-inputs", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--allow-partial", action="store_true")
    bind = commands.add_parser("bind-results", help="Bind already-produced results without model calls")
    bind.add_argument("--bundle", type=Path, required=True)
    bind.add_argument("--results", type=Path, required=True)
    bind.add_argument("--evaluator-manifest", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "project":
            result = export_projection(args.run, args.scoring_inputs, args.output,
                                       allow_partial=args.allow_partial)
            print(json.dumps({key: result[key] for key in ("selected_cases", "committed_cases",
                "eligible_cases", "disposition_counts", "judge_calls", "quality_scored")}))
        else:
            result = bind_candidate_results(args.bundle, args.results, args.evaluator_manifest, args.output)
            print(json.dumps({key: result[key] for key in ("selected_cases", "eligible_cases", "returned_judge_results")}))
    except (ValueError, OSError, KeyError, TypeError):
        parser.exit(2, "Evaluator handoff validation failed; inputs and existing outputs were preserved.\n")


if __name__ == "__main__":
    main()

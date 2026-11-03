"""Offline immutable experiment records, verified resume, and scoring handoff.

No runner, provider, index or database is loaded here. Validating an export is
not authentication or a quality score. Actual inputs are checked at create and
resume; read-only loading checks stored integrity and the frozen DTO contracts.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import TYPE_CHECKING
from uuid import uuid4

from src.contracts import AggregateEvidence
from src.evaluation.experiment_contracts import (
    ExperimentManifest, RuntimeCaseExport, ScoringInput, ScoringReference,
    runtime_case_digest, validate_paired_arms, validate_resume_compatibility,
    validate_runtime_exports,
)

if TYPE_CHECKING:
    from src.evaluation.answer_metrics import AnswerEvaluationInput
    from src.evaluation.experiment_inputs import VerifiedInputs

SCHEMA_VERSION = "2"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_CASE_NAME = re.compile(r"case_[0-9]{6,}\.json\Z")
_HASH_ROLES = frozenset(("questions", "splits", "corpus", "catalog", "index_manifest",
    "retrieval_config", "comparison_config", "scope", "runtime_config", "generator_prompt",
    "source_repo", "critic_prompt", "router_prompt", "final_freeze", "calibration_artifact"))
_REQUIRED_HASH_ROLES = _HASH_ROLES - {"critic_prompt", "router_prompt", "final_freeze", "calibration_artifact"}


class ExperimentIOError(ValueError):
    """A content-free experiment input/output boundary failure."""


class RunIntegrityError(ExperimentIOError):
    """Stored records are incomplete, inconsistent or modified."""


class WriterLockError(ExperimentIOError):
    """A writer is active or a crashed writer requires operator review."""


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RunIntegrityError("Duplicate JSON object fields")
        value[key] = item
    return value


def _reject_constant(_value):
    raise RunIntegrityError("Non-finite JSON values are invalid")


def _json(payload):
    try:
        return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError):
        raise RunIntegrityError("Invalid or partial JSON record") from None


def _regular(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _read(path):
    if not _regular(path):
        raise RunIntegrityError("Required experiment record is missing or not a regular file")
    try:
        payload = path.read_bytes()
    except OSError:
        raise RunIntegrityError("Experiment record could not be read") from None
    return _json(payload), _sha(payload)


def _checked(model, value, label):
    try:
        raw = value.model_dump(warnings=False) if isinstance(value, model) else value
        return model.model_validate(raw)
    except (ValueError, TypeError):
        raise RunIntegrityError("Invalid " + label) from None


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path, payload):
    """Fsync a private temp file, publish by atomic no-replace hard link."""
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _sync_directory(path.parent)


def _output_location(path, source_repo=None):
    """Generated output must not dirty its source checkout or overwrite files."""
    path = Path(path).absolute()
    if source_repo is None:
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        found = subprocess.run(["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True, timeout=10, check=False)
        source_repo = Path(found.stdout.strip()) if found.returncode == 0 else None
    if source_repo is not None:
        repo = Path(source_repo).resolve()
        try:
            relative = path.resolve().relative_to(repo)
        except ValueError:
            return path
        if relative.parts and relative.parts[0] == ".git":
            raise ExperimentIOError("Experiment output cannot be placed in Git metadata")
        ignored = subprocess.run(["git", "-C", str(repo), "check-ignore", "--quiet", "--", str(relative)],
                                 capture_output=True, timeout=10, check=False)
        if ignored.returncode != 0:
            raise ExperimentIOError("Output inside the source checkout must be git-ignored; use an external run directory")
    return path


def _verified_record(manifest, verified_inputs):
    from src.evaluation.experiment_inputs import VerifiedInputs

    if not isinstance(verified_inputs, VerifiedInputs):
        raise ExperimentIOError("Create/resume requires verified experiment inputs")
    verified = verified_inputs.recheck(manifest)
    record = {"schema_version": SCHEMA_VERSION, "hashes": dict(verified.hashes),
        "manifest_settings_sha256": verified.manifest_settings_sha256,
        "questions": [question.model_dump(mode="json") for question in verified.questions]}
    _check_inputs(record, manifest)
    return verified, record


def _check_inputs(record, manifest):
    from src.evaluation.experiment_inputs import (
        RuntimeQuestion, canonical_json_sha256, question_set_sha256,
        scientific_settings_sha256,
    )

    if (not isinstance(record, dict) or set(record) != {"schema_version", "hashes", "manifest_settings_sha256", "questions"}
            or record["schema_version"] != SCHEMA_VERSION):
        raise RunIntegrityError("Invalid verified input ledger")
    hashes = record["hashes"]
    if (not isinstance(hashes, dict) or not _REQUIRED_HASH_ROLES <= hashes.keys()
            or not hashes.keys() <= _HASH_ROLES
            or any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in hashes.values())):
        raise RunIntegrityError("Invalid input fingerprint ledger")
    raw_declarations = {
        "corpus": manifest.scope.snapshot.corpus_sha256,
        "splits": manifest.split_manifest_sha256,
        "index_manifest": manifest.retrieval.index_manifest_sha256,
        "source_repo": canonical_json_sha256(manifest.code.model_dump(mode="json")),
    }
    for role, declared in raw_declarations.items():
        if hashes[role] != declared:
            raise RunIntegrityError("Stored input fingerprint differs from its manifest declaration")
    for role in ("generator_prompt", "critic_prompt", "router_prompt"):
        declared = getattr(manifest, role + "_sha256")
        if (declared is None and role in hashes) or hashes.get(role) != declared:
            raise RunIntegrityError("Stored prompt fingerprint or presence differs from its manifest declaration")
    if manifest.split == "final":
        if not {"final_freeze", "calibration_artifact"} <= hashes.keys():
            raise RunIntegrityError("Final inputs require freeze and calibration artifact fingerprints")
    elif "calibration_artifact" in hashes:
        raise RunIntegrityError("Non-final inputs cannot declare a gated calibration artifact")
    # Outside final, an optional freeze is only an opaque, hashed input. Its
    # presence does not establish calibration review or acceptance.
    if not isinstance(record["questions"], list):
        raise RunIntegrityError("Invalid runtime question whitelist")
    questions = tuple(_checked(RuntimeQuestion, row, "runtime question") for row in record["questions"])
    if tuple(question.question_id for question in questions) != manifest.question_ids:
        raise RunIntegrityError("Runtime selection differs from manifest question order")
    if question_set_sha256(questions) != manifest.question_set_sha256:
        raise RunIntegrityError("Runtime question text differs from the manifest fingerprint")
    if record["manifest_settings_sha256"] != scientific_settings_sha256(manifest):
        raise RunIntegrityError("Verified settings differ from the manifest")
    return {question.question_id: question.question for question in questions}


class RunDirectory:
    """One immutable arm/repetition; create/resume hold an exclusive writer lock.

    Use a context manager or close(). A crash leaves .writer.lock for explicit
    review; resume never removes it automatically. Readers may inspect atomic
    completed cases while a writer is active. Cases are indexed/digested in one
    immutable envelope, so no separately mutable global index can drift.
    """

    def __init__(self, path, manifest, inputs):
        self.path = Path(path)
        self._manifest = manifest
        self._inputs = inputs
        self._lock_token = None

    @property
    def manifest(self) -> ExperimentManifest:
        return self._manifest

    @classmethod
    def create(cls, path: str | Path, manifest: ExperimentManifest,
               verified_inputs: VerifiedInputs) -> RunDirectory:
        manifest = _checked(ExperimentManifest, manifest, "experiment manifest")
        verified, inputs = _verified_record(manifest, verified_inputs)
        path = _output_location(path, verified.files.source_repo)
        path.mkdir(parents=True, mode=0o700)
        run = cls(path, manifest, inputs)
        try:
            run._acquire_lock()
            (path / "cases").mkdir(mode=0o700)
            manifest_payload, inputs_payload = _bytes(manifest.model_dump(mode="json")), _bytes(inputs)
            _write_new(path / "manifest.json", manifest_payload)
            _write_new(path / "inputs.json", inputs_payload)
            _write_new(path / "integrity.json", _bytes({"schema_version": SCHEMA_VERSION,
                "manifest_sha256": _sha(manifest_payload), "inputs_sha256": _sha(inputs_payload)}))
            _sync_directory(path)
        except BaseException:
            run.close()
            raise
        return run

    @classmethod
    def load(cls, path: str | Path) -> RunDirectory:
        path = Path(path)
        manifest, inputs = cls._headers(path)
        run = cls(path, manifest, inputs)
        run.load_cases()
        return run

    @classmethod
    def resume(cls, path: str | Path, requested_manifest: ExperimentManifest,
               verified_inputs: VerifiedInputs) -> RunDirectory:
        path = Path(path)
        if (path / ".writer.lock").exists() or (path / ".writer.lock").is_symlink():
            raise WriterLockError("Writer lock exists; close the active writer or review the crashed run before manually removing .writer.lock")
        run = cls.load(path)
        requested = _checked(ExperimentManifest, requested_manifest, "resume manifest")
        validate_resume_compatibility(run.manifest, requested)
        verified, inputs = _verified_record(requested, verified_inputs)
        _output_location(path, verified.files.source_repo)
        if inputs != run._inputs:
            raise RunIntegrityError("Resume changes verified input fingerprints or runtime selection")
        run._acquire_lock()
        try:
            run.load_cases()
        except BaseException:
            run.close()
            raise
        return run

    @staticmethod
    def _headers(path):
        if path.is_symlink() or not path.is_dir() or (path / "cases").is_symlink() or not (path / "cases").is_dir():
            raise RunIntegrityError("Run directory or cases directory is missing or unsafe")
        names = {"manifest.json", "inputs.json", "integrity.json", "cases", ".writer.lock"}
        active = _regular(path / ".writer.lock")
        for entry in path.iterdir():
            if entry.name not in names and not (active and entry.name.startswith(".tmp-") and _regular(entry)):
                raise RunIntegrityError("Unexpected or uncommitted file in runtime directory")
            if entry.name == ".writer.lock" and not _regular(entry):
                raise RunIntegrityError("Invalid writer lock file")
        raw_manifest, manifest_sha = _read(path / "manifest.json")
        inputs, inputs_sha = _read(path / "inputs.json")
        integrity, _ = _read(path / "integrity.json")
        if integrity != {"schema_version": SCHEMA_VERSION, "manifest_sha256": manifest_sha, "inputs_sha256": inputs_sha}:
            raise RunIntegrityError("Immutable manifest/input digest ledger differs")
        manifest = _checked(ExperimentManifest, raw_manifest, "stored manifest")
        _check_inputs(inputs, manifest)
        return manifest, inputs

    def _acquire_lock(self):
        token = uuid4().hex
        try:
            descriptor = os.open(self.path / ".writer.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise WriterLockError("Writer lock exists; review the active/crashed writer before manually removing it") from None
        self._lock_token = token
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_bytes({"token": token, "pid": os.getpid()}))
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(self.path)

    def close(self) -> None:
        if self._lock_token is None:
            return
        lock, _ = _read(self.path / ".writer.lock")
        if not isinstance(lock, dict) or lock.get("token") != self._lock_token:
            raise WriterLockError("Writer lock changed; inspect it before recovery")
        (self.path / ".writer.lock").unlink()
        self._lock_token = None
        _sync_directory(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()

    def load_cases(self, require_complete: bool = False) -> tuple[RuntimeCaseExport, ...]:
        manifest, inputs = self._headers(self.path)
        if manifest != self._manifest or inputs != self._inputs:
            raise RunIntegrityError("Stored immutable run metadata changed")
        questions = _check_inputs(inputs, manifest)
        active = _regular(self.path / ".writer.lock")
        cases = []
        for path in sorted((self.path / "cases").iterdir()):
            if active and path.name.startswith(".tmp-") and _regular(path):
                continue
            if not _CASE_NAME.fullmatch(path.name):
                raise RunIntegrityError("Unexpected or partial case file")
            envelope, _ = _read(path)
            if not isinstance(envelope, dict) or set(envelope) != {"schema_version", "ordinal", "question_id", "run_id", "sha256", "case"}:
                raise RunIntegrityError("Invalid case digest/index envelope")
            case = _checked(RuntimeCaseExport, envelope["case"], "runtime case")
            if case.question_id not in questions or case.question != questions[case.question_id]:
                raise RunIntegrityError("Case text differs from the verified runtime selection")
            ordinal = manifest.question_ids.index(case.question_id) + 1
            if (type(envelope["ordinal"]) is not int or envelope["ordinal"] != ordinal
                    or path.name != f"case_{ordinal:06d}.json" or envelope["schema_version"] != SCHEMA_VERSION
                    or envelope["question_id"] != case.question_id or envelope["run_id"] != case.run_id
                    or envelope["sha256"] != runtime_case_digest(case)):
                raise RunIntegrityError("Case digest or ordinal index differs")
            cases.append(case)
        validate_runtime_exports(manifest, tuple(cases), require_complete=require_complete)
        return tuple(cases)

    @property
    def pending_question_ids(self) -> tuple[str, ...]:
        done = {case.question_id for case in self.load_cases()}
        return tuple(qid for qid in self.manifest.question_ids if qid not in done)

    def append_case(self, case: RuntimeCaseExport) -> Path:
        if self._lock_token is None:
            raise WriterLockError("Appending requires an open create/resume writer")
        lock, _ = _read(self.path / ".writer.lock")
        if not isinstance(lock, dict) or lock.get("token") != self._lock_token:
            raise WriterLockError("Writer lock changed; append denied")
        case = _checked(RuntimeCaseExport, case, "runtime case")
        cases = self.load_cases()
        questions = _check_inputs(self._inputs, self.manifest)
        if case.question_id not in questions or case.question != questions[case.question_id]:
            raise RunIntegrityError("Case text differs from the verified runtime selection")
        validate_runtime_exports(self.manifest, (*cases, case))
        ordinal = self.manifest.question_ids.index(case.question_id) + 1
        path = self.path / "cases" / f"case_{ordinal:06d}.json"
        _write_new(path, _bytes({"schema_version": SCHEMA_VERSION, "ordinal": ordinal,
            "question_id": case.question_id, "run_id": case.run_id, "sha256": runtime_case_digest(case),
            "case": case.model_dump(mode="json")}))
        return path


def _run(value):
    return value if isinstance(value, RunDirectory) else RunDirectory.load(value)


def validate_pair(fixed: RunDirectory | str | Path, adaptive: RunDirectory | str | Path,
                  *, require_complete: bool = True) -> None:
    fixed, adaptive = _run(fixed), _run(adaptive)
    validate_paired_arms(fixed.manifest, adaptive.manifest,
        fixed.load_cases(require_complete=require_complete), adaptive.load_cases(require_complete=require_complete),
        require_complete=require_complete)


def export_scoring_inputs(runtime_dir: RunDirectory | str | Path, reference_file: str | Path,
                          output_dir: str | Path) -> tuple[ScoringInput, ...]:
    """Join explicit scoring-only JSONL after a completed arm; never run a judge."""
    run = _run(runtime_dir)
    cases = run.load_cases(require_complete=True)
    output = Path(output_dir).resolve()
    if output == run.path.resolve() or run.path.resolve() in output.parents:
        raise ExperimentIOError("Scoring output must be separate from the runtime directory")
    from src.evaluation.experiment_inputs import RuntimeQuestion

    digest, rows, seen = hashlib.sha256(), [], set()
    allowed = {"question_id", "question", "reference_answer", "answer_facts", "expected_doc_ids"}
    with Path(reference_file).open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            row = _json(line)
            if (not isinstance(row, dict) or not {"question_id", "question"} <= row.keys()
                    or not row.keys() <= allowed or not row.keys() & {"reference_answer", "answer_facts", "expected_doc_ids"}):
                raise RunIntegrityError("Reference rows require explicit question/reference fields; embedded hashes are forbidden")
            question = _checked(RuntimeQuestion, {key: row[key] for key in ("question_id", "question")}, "reference question")
            if question.question_id in seen:
                raise RunIntegrityError("Duplicate reference question IDs")
            seen.add(question.question_id)
            rows.append((question, row))
    references = {}
    for question, row in rows:
        references[question.question_id] = _checked(ScoringReference, {
            "question_id": question.question_id, "question_sha256": _sha(question.question.encode("utf-8")),
            "reference_set_sha256": digest.hexdigest(),
            **{key: row[key] for key in ("reference_answer", "answer_facts", "expected_doc_ids") if key in row}}, "scoring reference")
    scoring = []
    for case in cases:
        if case.question_id not in references:
            raise RunIntegrityError("Missing scoring reference for a completed runtime case")
        scoring.append(_checked(ScoringInput, {"runtime_case": case.model_dump(mode="json"),
            "runtime_case_sha256": runtime_case_digest(case),
            "reference": references[case.question_id].model_dump(mode="json")}, "scoring join"))
    _, manifest_sha = _read(run.path / "manifest.json")
    output = _output_location(output)
    output.mkdir(parents=True, mode=0o700)
    _write_new(output / "scoring_inputs.jsonl", b"".join(_bytes(item.model_dump(mode="json")) for item in scoring))
    _write_new(output / "manifest.json", _bytes({"schema_version": SCHEMA_VERSION,
        "runtime_manifest_id": run.manifest.manifest_id, "runtime_manifest_sha256": manifest_sha,
        "reference_set_sha256": digest.hexdigest(), "cases": len(scoring)}))
    return tuple(scoring)


def to_answer_evaluation_input(scoring: ScoringInput) -> AnswerEvaluationInput:
    """Optional passage-only adapter; preserves the old evaluator unchanged."""
    scoring = _checked(ScoringInput, scoring, "scoring input")
    case = scoring.runtime_case
    if any(isinstance(item, AggregateEvidence) for item in (*case.response.sources, *case.generation_evidence)):
        raise ExperimentIOError("Aggregate evidence requires the native scoring handoff")
    if case.error is not None or case.response.trace.stop_reason == "clarification":
        raise ExperimentIOError("Failed generation and clarification require the native scoring handoff")
    if scoring.reference.reference_answer is None:
        raise ExperimentIOError("The legacy evaluator requires a reference answer")
    from src.evaluation.answer_metrics import AnswerEvaluationInput, EvidenceItem

    return AnswerEvaluationInput(question_id=case.question_id, question=case.question,
        generated_answer=case.response.answer.text, reference_answer=scoring.reference.reference_answer,
        evidence=[EvidenceItem(citation_id=item.citation_id, text=item.chunk.text) for item in case.generation_evidence],
        citation_ids=list(case.response.answer.citation_ids))


def _input_files(path):
    from src.evaluation.experiment_inputs import InputFiles

    mapping, _ = _read(path)
    names = {field.name for field in fields(InputFiles)}
    if not isinstance(mapping, dict) or not mapping.keys() <= names:
        raise ExperimentIOError("Input mapping contains unknown fields")
    values = {}
    for key, value in mapping.items():
        if value is None:
            values[key] = None
        elif isinstance(value, str):
            values[key] = (path.parent / value).resolve()
        else:
            raise ExperimentIOError("Input mapping values must be file paths")
    try:
        return InputFiles(**values)
    except TypeError:
        raise ExperimentIOError("Input mapping is missing required paths") from None


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Verify files, selection and declared configuration without generation")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--inputs", type=Path, required=True)
    validate = commands.add_parser("validate", help="Read-only stored integrity/case checks")
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--require-complete", action="store_true")
    pair = commands.add_parser("pair", help="Validate explicitly ordered fixed/adaptive arms")
    pair.add_argument("--fixed", type=Path, required=True)
    pair.add_argument("--adaptive", type=Path, required=True)
    pair.add_argument("--allow-partial", action="store_true")
    scoring = commands.add_parser("export-scoring", help="Join references into a new, separate handoff directory")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--references", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command == "preflight":
            from src.evaluation.experiment_inputs import verify_inputs
            raw, _ = _read(args.manifest)
            manifest = _checked(ExperimentManifest, raw, "experiment manifest")
            verified = verify_inputs(manifest, _input_files(args.inputs))
            answer = {"valid": True, "manifest_id": manifest.manifest_id,
                      "question_ids": [question.question_id for question in verified.questions]}
        elif args.command == "validate":
            run = RunDirectory.load(args.run)
            answer = {"valid": True, "cases": len(run.load_cases(require_complete=args.require_complete)),
                      "pending_question_ids": run.pending_question_ids}
        elif args.command == "pair":
            validate_pair(args.fixed, args.adaptive, require_complete=not args.allow_partial)
            answer = {"valid": True}
        else:
            answer = {"exported": len(export_scoring_inputs(args.run, args.references, args.output)),
                      "judge_calls": 0}
    except (OSError, ValueError, TypeError) as error:
        command.error(str(error) if isinstance(error, ExperimentIOError) else "Experiment input verification failed")
    print(json.dumps(answer, allow_nan=False))


if __name__ == "__main__":
    main()

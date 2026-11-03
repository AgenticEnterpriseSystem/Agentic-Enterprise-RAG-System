"""Offline input verification for immutable experiment exports.

Verification reads files and Git provenance, never models, indexes or databases.
Scope/config/final-freeze files are producer/operator attestations; reading them
does not establish authorization, inspect a provider, or prove CAL completion.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field

from src.contracts import Contract, Identifier, ScopeBinding
from src.evaluation.experiment_contracts import (
    CodeProvenance, ExperimentManifest, GenerationBudgets, ModelProvenance,
    RetrievalProvenance,
)
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import catalog_fingerprint

_SPLITS = frozenset({"development", "calibration", "final"})


class RuntimeQuestion(Contract):
    question_id: Identifier
    question: Annotated[str, Field(min_length=1, max_length=8000, pattern=r"\S")]


class _RuntimeConfig(Contract):
    generation_model: ModelProvenance
    retrieval: RetrievalProvenance
    budgets: GenerationBudgets
    temperature: Annotated[float, Field(strict=True, ge=0)]
    workflow: Literal["fixed", "adaptive"]
    max_rounds: Annotated[int, Field(strict=True, ge=1)]


@dataclass(frozen=True)
class InputFiles:
    questions: Path
    splits: Path
    corpus: Path
    catalog: Path
    index_manifest: Path
    retrieval_config: Path
    comparison_config: Path
    scope: Path
    runtime_config: Path
    generator_prompt: Path
    source_repo: Path
    critic_prompt: Path | None = None
    router_prompt: Path | None = None
    final_freeze: Path | None = None

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None and item.name in {"critic_prompt", "router_prompt", "final_freeze"}:
                continue
            if not isinstance(value, Path):
                raise TypeError("Input files must use pathlib.Path values")
            object.__setattr__(self, item.name, value.resolve())


@dataclass(frozen=True)
class VerifiedInputs:
    questions: tuple[RuntimeQuestion, ...]
    hashes: Mapping[str, str]
    files: InputFiles
    manifest_settings_sha256: str

    def __post_init__(self):
        questions = tuple(RuntimeQuestion.model_validate(q.model_dump()) for q in self.questions)
        object.__setattr__(self, "questions", questions)
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))

    def recheck(self, manifest: ExperimentManifest) -> "VerifiedInputs":
        current = verify_inputs(manifest, self.files)
        if (current.hashes != self.hashes or current.questions != self.questions
                or current.manifest_settings_sha256 != self.manifest_settings_sha256):
            raise ValueError("Verified inputs changed since the original preflight")
        return current


def _json_value(value: object) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Canonical JSON requires finite JSON values and string object keys")


def canonical_json_sha256(value: object) -> str:
    _json_value(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(_value):
    raise ValueError("Non-finite JSON constants are not accepted")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _json_loads(raw: bytes):
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                           object_pairs_hook=_unique_object)
        _json_value(value)  # Reject numeric overflow (for example 1e999), too.
        return value
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("Input is not valid strict UTF-8 JSON") from None


def _read_json(path: Path):
    raw = path.read_bytes()
    return _json_loads(raw), hashlib.sha256(raw).hexdigest()


def _questions(path: Path) -> tuple[dict[str, RuntimeQuestion], str]:
    if path.suffix.lower() == ".jsonl":
        digest = hashlib.sha256()
        rows = []
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                if raw.strip():
                    rows.append(_json_loads(raw))
        raw_hash = digest.hexdigest()
    else:
        rows, raw_hash = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Questions must be a nonempty JSON array or JSONL records")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Question records must be JSON objects")
        # Only these fields can ever be forwarded to runtime code.
        question = RuntimeQuestion.model_validate({
            "question_id": row.get("question_id"), "question": row.get("question")})
        if question.question_id in result:
            raise ValueError("Duplicate source question IDs")
        result[question.question_id] = question
    return result, raw_hash


def _split_membership(value, questions: Mapping[str, RuntimeQuestion]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not isinstance(value.get("splits"), dict):
        raise ValueError("Split file must contain all three frozen splits")
    declared = value["splits"]
    if set(declared) != _SPLITS:
        raise ValueError("Split file must contain exactly development, calibration and final")
    result, seen = {}, set()
    for name in sorted(_SPLITS):
        ids = declared[name]
        if (not isinstance(ids, list) or any(not isinstance(qid, str) or not qid.strip() for qid in ids)):
            raise ValueError("Split IDs must be nonblank strings")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Frozen splits must be unique and disjoint")
        if not set(ids) <= questions.keys():
            raise ValueError("Split references questions missing from the question file")
        seen.update(ids)
        result[name] = tuple(ids)
    if seen != questions.keys():
        raise ValueError("Source question IDs are missing from the frozen splits")
    return result


def _select(questions, membership, split, question_ids, count, allow_final):
    if split not in _SPLITS:
        raise ValueError("Unknown frozen split")
    if split == "final" and allow_final is not True:
        raise ValueError("Final selection requires its explicit frozen-configuration gate")
    if question_ids is not None and count is not None:
        raise ValueError("Choose explicit question IDs or count, not both")
    available = set(membership[split])
    if question_ids is not None:
        if isinstance(question_ids, (str, bytes)):
            raise ValueError("Question IDs must be a sequence")
        selected = tuple(question_ids)
        if (not selected or any(not isinstance(qid, str) for qid in selected)
                or len(selected) != len(set(selected)) or not set(selected) <= available):
            raise ValueError("Explicit question IDs must be distinct members of the selected split")
    else:
        if count is not None and (type(count) is not int or not 1 <= count <= len(available)):
            raise ValueError("Count must be a positive integer within the selected split")
        selected = tuple(sorted(available))[:count]
    if not selected:
        raise ValueError("The selected split contains no runtime questions")
    return tuple(questions[qid] for qid in selected)


def select_runtime_questions(question_file: Path, split_file: Path, split: str, *,
                             question_ids: Sequence[str] | None = None, count: int | None = None,
                             allow_final: bool = False) -> tuple[RuntimeQuestion, ...]:
    questions, _ = _questions(question_file)
    split_value, _ = _read_json(split_file)
    membership = _split_membership(split_value, questions)
    return _select(questions, membership, split, question_ids, count, allow_final)


def question_set_sha256(questions: Sequence[RuntimeQuestion]) -> str:
    checked = tuple(RuntimeQuestion.model_validate(q.model_dump()) for q in questions)
    if len({q.question_id for q in checked}) != len(checked):
        raise ValueError("Duplicate selected question IDs")
    return canonical_json_sha256([q.model_dump(mode="json") for q in sorted(checked, key=lambda q: q.question_id)])


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
            "-c", "core.quotePath=true", "-C", str(repo), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("Source directory must be a readable Git checkout with a committed HEAD") from None


def capture_code_provenance(repo: Path) -> CodeProvenance:
    root = Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel").rstrip(b"\n"))).resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    common = ("--binary", "--no-color", "--no-ext-diff", "--no-textconv", "--ignore-submodules=none")
    staged = _git(root, "diff", "--cached", *common, commit, "--")
    unstaged = _git(root, "diff", *common, "--")
    untracked = []
    for name in sorted(_git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not name:
            continue
        relative = os.fsdecode(name)
        path = root / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Untracked source provenance requires ordinary files")
        untracked.append([relative, stat.S_IMODE(info.st_mode), file_sha256(path)])
    if _git(root, "rev-parse", "HEAD").decode("ascii").strip() != commit:
        raise ValueError("Source commit changed during provenance capture")
    if not staged and not unstaged and not untracked:
        return CodeProvenance(commit=commit)
    digest = canonical_json_sha256({"format": "rag-code-diff-v1",
        "staged_sha256": hashlib.sha256(staged).hexdigest(),
        "unstaged_sha256": hashlib.sha256(unstaged).hexdigest(), "untracked": untracked})
    return CodeProvenance(commit=commit, dirty=True, diff_sha256=digest)


def scientific_settings_sha256(manifest: ExperimentManifest) -> str:
    checked = ExperimentManifest.model_validate(manifest.model_dump())
    return canonical_json_sha256(checked.model_dump(mode="json", exclude={
        "experiment_id", "manifest_id", "created_at", "split", "split_manifest_sha256",
        "question_set_sha256", "question_ids", "repetition"}))


def _catalog_entries(path: Path, digest):
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if raw.strip():
                yield CatalogEntry.model_validate(_json_loads(raw))


def _assert_hash(actual: str, expected: str, role: str) -> None:
    if actual != expected:
        raise ValueError(f"Actual {role} input does not match the manifest")


def _verify_index_identity(value, snapshot) -> None:
    """Reject explicit known contradictions without inventing absent metadata."""
    if not isinstance(value, dict):
        return
    if "complete" in value and value["complete"] is not True:
        raise ValueError("Index manifest is declared incomplete")
    for key in ("snapshot_id", "corpus_sha256", "catalog_sha256"):
        if key in value and value[key] != getattr(snapshot, key):
            raise ValueError("Index metadata contradicts the declared snapshot")
    for key in ("documents", "document_count"):
        if key in value and (type(value[key]) is not int or value[key] != snapshot.document_count):
            raise ValueError("Index metadata contradicts the declared document count")
    if "snapshot" in value:
        bound = value["snapshot"]
        if isinstance(bound, str):
            if bound != snapshot.snapshot_id:
                raise ValueError("Index metadata contradicts the declared snapshot")
        elif isinstance(bound, dict):
            _verify_index_identity(bound, snapshot)
        else:
            raise ValueError("Index metadata has an invalid snapshot binding")
    for key in ("bm25", "dense", "stats"):
        if isinstance(value.get(key), dict):
            _verify_index_identity(value[key], snapshot)


def _final_gate(manifest, path: Path, hashes: dict[str, str]):
    freeze, hashes["final_freeze"] = _read_json(path)
    if not isinstance(freeze, dict) or freeze.get("accepted") is not True:
        raise ValueError("Final freeze requires explicit operator acceptance")
    settings = freeze.get("settings_sha256")
    if not isinstance(settings, dict) or settings.get(manifest.workflow) != scientific_settings_sha256(manifest):
        raise ValueError("Final freeze does not match the scientific settings")
    relative = freeze.get("calibration_artifact")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Final freeze requires a calibration artifact")
    artifact = (path.parent / relative).resolve()
    if not artifact.is_relative_to(path.parent.resolve()):
        raise ValueError("Calibration artifact must stay within the freeze directory")
    actual = file_sha256(artifact)
    if actual != freeze.get("calibration_sha256"):
        raise ValueError("Calibration artifact hash does not match the freeze")
    hashes["calibration_artifact"] = actual


def verify_inputs(manifest: ExperimentManifest, files: InputFiles) -> VerifiedInputs:
    manifest = ExperimentManifest.model_validate(manifest.model_dump())
    if not isinstance(files, InputFiles):
        raise TypeError("Verification requires InputFiles")
    hashes = {}
    questions, hashes["questions"] = _questions(files.questions)
    split_value, hashes["splits"] = _read_json(files.splits)
    _assert_hash(hashes["splits"], manifest.split_manifest_sha256, "split")
    membership = _split_membership(split_value, questions)
    selected = _select(questions, membership, manifest.split, manifest.question_ids, None,
                       manifest.split != "final" or files.final_freeze is not None)
    _assert_hash(question_set_sha256(selected), manifest.question_set_sha256, "selected question")
    hashes["corpus"] = file_sha256(files.corpus)
    _assert_hash(hashes["corpus"], manifest.scope.snapshot.corpus_sha256, "corpus")
    catalog_digest = hashlib.sha256()
    fingerprint = catalog_fingerprint(_catalog_entries(files.catalog, catalog_digest),
                                      expected_count=manifest.scope.snapshot.document_count)
    hashes["catalog"] = catalog_digest.hexdigest()
    _assert_hash(fingerprint, manifest.scope.snapshot.catalog_sha256, "catalog")
    index, hashes["index_manifest"] = _read_json(files.index_manifest)
    _assert_hash(hashes["index_manifest"], manifest.retrieval.index_manifest_sha256, "index manifest")
    _verify_index_identity(index, manifest.scope.snapshot)
    retrieval, hashes["retrieval_config"] = _read_json(files.retrieval_config)
    _assert_hash(canonical_json_sha256(retrieval), manifest.retrieval.config_sha256, "retrieval config")
    comparison, hashes["comparison_config"] = _read_json(files.comparison_config)
    _assert_hash(canonical_json_sha256(comparison), manifest.comparison_config_sha256, "comparison config")
    scope, hashes["scope"] = _read_json(files.scope)
    if not isinstance(scope, dict) or ScopeBinding.model_validate(scope.get("binding")) != manifest.scope:
        raise ValueError("Scope input binding differs from the manifest")
    _assert_hash(canonical_json_sha256(scope), manifest.scope_fingerprint_sha256, "scope")
    runtime, hashes["runtime_config"] = _read_json(files.runtime_config)
    actual = _RuntimeConfig.model_validate(runtime)
    expected = _RuntimeConfig(**{name: getattr(manifest, name) for name in _RuntimeConfig.model_fields})
    if actual != expected:
        raise ValueError("Effective runtime configuration differs from the manifest attestation")
    for role in ("generator_prompt", "critic_prompt", "router_prompt"):
        path, expected_hash = getattr(files, role), getattr(manifest, role + "_sha256")
        if path is None:
            if expected_hash is not None:
                raise ValueError(f"Missing declared {role} file")
            continue
        if expected_hash is None:
            raise ValueError(f"Undeclared {role} file")
        raw = path.read_bytes()
        raw.decode("utf-8")
        hashes[role] = hashlib.sha256(raw).hexdigest()
        _assert_hash(hashes[role], expected_hash, role)
    code = capture_code_provenance(files.source_repo)
    if code != manifest.code:
        raise ValueError("Actual source provenance differs from the manifest")
    hashes["source_repo"] = canonical_json_sha256(code.model_dump(mode="json"))
    if manifest.split == "final":
        if files.final_freeze is None:
            raise ValueError("Final verification requires an accepted freeze artifact")
        _final_gate(manifest, files.final_freeze, hashes)
    elif files.final_freeze is not None:
        _, hashes["final_freeze"] = _read_json(files.final_freeze)
    return VerifiedInputs(selected, hashes, files, scientific_settings_sha256(manifest))

"""Approval-gated, stage-journaled RC2 evaluation of immutable projections.

No provider is constructed here. The caller supplies a reviewed factory, invoked
only after durable case/stage admission. Unknown or failed attempts are retained;
there is deliberately no resume, retry, historical-label reuse or acceptance API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Annotated, Callable, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from src.agents.budget import CallBudget, RunBudgetExceeded, call_budget
from src.contracts import Contract, Identifier, Sha256
from src.evaluation.evaluator_bridge import (
    BRIDGE_VERSION, _candidate_sources, _projection_payloads, _validate_audit_schema, project_batch,
)
from src.evaluation.evaluator_candidate import (
    AUDIT_SYSTEM, SCORE_SYSTEM, _auditor_prompt, _scorer_prompt, build_record, evaluate_case,
)
from src.evaluation.experiment_contracts import ExperimentManifest, ScoringInput
from src.evaluation.experiment_io import _checked, _json, _output_location, _read, _sha, _write_new
from src.evaluation.verified_judge import make_case, validate_scores
from src.generation.generator import ProviderError

ROOT = Path(__file__).resolve().parents[2]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Money = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
FORMAT = "rag-guarded-evaluator-plan-v1"
EVALUATOR = "guarded-rc2-v1"
OUTPUT_PROTOCOLS = Literal["text_json_v1", "native_schema_v1"]
DEFAULT_MAX_RESPONSE_BYTES = 262144


class EvaluatorRunError(ValueError):
    """Content-free admission, integrity or execution boundary failure."""


def _need(value, message):
    if not value:
        raise EvaluatorRunError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _amount(value):
    _need(type(value) in (str, int, float, Decimal), "Invalid monetary value")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise EvaluatorRunError("Invalid monetary value") from None
    _need(result.is_finite() and result >= 0, "Nonfinite or negative monetary value")
    return result


class JudgeSettings(Contract):
    provider: Literal["anthropic"] = "anthropic"
    model_id: Identifier
    revision: Identifier | None = None
    input_usd_per_million: Money
    output_usd_per_million: Money
    max_output_tokens: PositiveInt
    max_tokens: PositiveInt
    timeout_seconds: Annotated[float, Field(gt=0, le=300)]
    score_cost_cap_usd: Money
    audit_cost_cap_usd: Money
    case_cost_cap_usd: Money
    total_cost_cap_usd: Money

    @model_validator(mode="after")
    def bounded(self):
        if (self.max_output_tokens > self.max_tokens
                or self.score_cost_cap_usd + self.audit_cost_cap_usd > self.case_cost_cap_usd
                or self.case_cost_cap_usd > self.total_cost_cap_usd):
            raise ValueError("Stage/output/case caps exceed their parent allowance")
        return self


class CaseIdentity(Contract):
    manifest_id: Identifier
    run_id: Identifier
    question_id: Identifier
    question_sha256: Sha256
    runtime_case_sha256: Sha256


class ProjectionInput(Contract):
    directory: Identifier
    manifest_id: Identifier
    projection_manifest_sha256: Sha256
    files: dict[str, Sha256]


class EvaluatorPlan(Contract):
    format: Literal["rag-guarded-evaluator-plan-v1"] = FORMAT
    evaluator: Literal["guarded-rc2-v1"] = EVALUATOR
    execution_context: Literal["live", "fixture"] = "live"
    output_protocol: OUTPUT_PROTOCOLS = "native_schema_v1"
    stop_on_invalid_response: Annotated[bool, Field(strict=True)] = True
    max_response_bytes: Annotated[int, Field(strict=True, ge=1, le=1048576)] = DEFAULT_MAX_RESPONSE_BYTES
    output_directory: Identifier
    settings: JudgeSettings
    bundles: tuple[ProjectionInput, ...] = Field(min_length=1)
    cases: tuple[CaseIdentity, ...] = Field(min_length=1)
    source_files: dict[str, Sha256]
    evaluator_sources: dict
    max_attempts_per_stage: Literal[1] = 1
    sdk_max_retries: Literal[0] = 0
    audit_mode: Literal["all"] = "all"

    @property
    def judge(self):
        """Convenience alias; the immutable serialized field is settings."""
        return self.settings

    @model_validator(mode="after")
    def unique_and_bounded(self):
        manifests = [bundle.manifest_id for bundle in self.bundles]
        keys = [(case.manifest_id, case.question_id) for case in self.cases]
        if (len(manifests) != len(set(manifests)) or len(keys) != len(set(keys))
                or len({case.run_id for case in self.cases}) != len(self.cases)
                or not {case.manifest_id for case in self.cases} <= set(manifests)
                or len(self.cases) * self.settings.case_cost_cap_usd > self.settings.total_cost_cap_usd):
            raise ValueError("Duplicate/unbound selection or total reservation cap exceeded")
        if self.execution_context == "live" and (
                self.output_protocol != "native_schema_v1" or not self.stop_on_invalid_response):
            raise ValueError("Live evaluator plans require native schema and stop on invalid response")
        return self


class StageAdmission(Contract):
    plan_sha256: Sha256
    identity: CaseIdentity
    stage: Literal["score", "citation_audit"]
    judge: JudgeSettings
    max_cost_usd: Money
    execution_context: Literal["live", "fixture"]
    output_protocol: OUTPUT_PROTOCOLS


@dataclass(frozen=True)
class _Bundle:
    declaration: ProjectionInput
    ledger: tuple[dict, ...]
    answers: dict
    references: dict
    metadata: dict


def _regular_bytes(path):
    path = Path(path)
    _need(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "Ordinary immutable file required")
    return path.read_bytes()


def _retained_response_bytes(path, maximum):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, "Retained response exceeds its byte bound")
        raw = stream.read(maximum + 1)
    _need(len(raw) <= maximum, "Retained response exceeds its byte bound")
    return raw


def _required_sources(context):
    result = {str(Path(__file__).resolve()), str(Path(__file__).with_name("evaluator_bridge.py").resolve()),
        str(ROOT / "src/generation/generator.py"), str(ROOT / "src/agents/budget.py")}
    result.update(str(ROOT / relative) for relative in _candidate_sources()["files"])
    if context == "live": result.add(str(ROOT / "src/evaluation/run_evaluator.py"))
    return result


def verify_projection_bundle(directory) -> _Bundle:
    directory = Path(directory).absolute()
    _need(not directory.is_symlink(), "Projection directory cannot be a symlink")
    metadata, manifest_sha = _read(directory / "manifest.json")
    required = {"answers.json", "questions.json", "case_ledger.jsonl", "scoring_inputs.jsonl", "runtime_manifest.json"}
    _need(metadata.get("schema_version") == "1" and metadata.get("bridge_version") == BRIDGE_VERSION
        and set(metadata.get("files", {})) == required, "Unsupported projection bundle")
    for name, expected in metadata["files"].items():
        _need(_sha(_regular_bytes(directory / name)) == expected, "Projection bytes changed")
    manifest = _checked(ExperimentManifest, _json(_regular_bytes(directory / "runtime_manifest.json")), "manifest")
    packets = tuple(_checked(ScoringInput, _json(line), "scoring input") for line in
        _regular_bytes(directory / "scoring_inputs.jsonl").splitlines() if line.strip())
    _need(type(metadata.get("allow_partial")) is bool, "Projection completeness must be explicit")
    ledger = project_batch(manifest, packets, require_complete=not metadata["allow_partial"])
    for name, payload in _projection_payloads(ledger).items():
        _need(_regular_bytes(directory / name) == payload, "Projection differs from canonical packets")
    _need(metadata.get("manifest_id") == manifest.manifest_id
        and metadata.get("experiment_id") == manifest.experiment_id
        and metadata.get("workflow") == manifest.workflow and metadata.get("repetition") == manifest.repetition
        and metadata.get("runtime_manifest_sha256") == metadata["files"]["runtime_manifest.json"]
        and metadata.get("scoring_input_sha256") == metadata["files"]["scoring_inputs.jsonl"]
        and metadata.get("selected_cases") == len(ledger) and metadata.get("committed_cases") == len(packets)
        and metadata.get("eligible_cases") == sum(row["disposition"] == "eligible" for row in ledger)
        and metadata.get("disposition_counts") == dict(Counter(row["disposition"] for row in ledger))
        and metadata.get("reference_set_sha256") == (packets[0].reference.reference_set_sha256 if packets else None)
        and metadata.get("evaluator") == _candidate_sources(), "Projection provenance differs")
    eligible = [row for row in ledger if row["disposition"] == "eligible"]
    return _Bundle(ProjectionInput(directory=str(directory), manifest_id=manifest.manifest_id,
        projection_manifest_sha256=manifest_sha, files=metadata["files"]), ledger,
        {row["identity"]["question_id"]: row["answer"] for row in eligible},
        {row["identity"]["question_id"]: row["reference"] for row in eligible}, metadata)


def evaluator_plan_sha256(plan):
    plan = _checked(EvaluatorPlan, plan, "evaluator plan")
    return _sha(_bytes(plan.model_dump(mode="json")))


def prepare_evaluator_plan(bundle_dirs, output_dir, *, settings, selected_question_ids=None,
                           execution_context="live", extra_source_files=None, output_protocol=None,
                           stop_on_invalid_response=None, max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES):
    settings = _checked(JudgeSettings, settings, "judge settings")
    bundles = tuple(verify_projection_bundle(path) for path in bundle_dirs)
    selection = selected_question_ids
    _need(selection is None or isinstance(selection, dict)
        and set(selection) <= {bundle.declaration.manifest_id for bundle in bundles}, "Unknown selected manifest")
    cases = []
    for bundle in bundles:
        mid = bundle.declaration.manifest_id
        requested = list(bundle.answers) if selection is None else selection.get(mid, [])
        _need(isinstance(requested, (list, tuple)) and len(requested) == len(set(requested))
            and set(requested) <= set(bundle.answers), "Unsupported, native, failed or duplicate scoring selection")
        cases.extend(CaseIdentity(**row["identity"]) for row in bundle.ledger
            if row["disposition"] == "eligible" and row["identity"]["question_id"] in requested)
    sources = _required_sources(execution_context)
    evaluator = _candidate_sources()
    sources.update(str(ROOT / relative) for relative in evaluator["files"])
    sources.update(str(Path(path).resolve()) for path in (extra_source_files or {}).values())
    return EvaluatorPlan(execution_context=execution_context, output_directory=str(Path(output_dir).absolute()),
        output_protocol=("native_schema_v1" if execution_context == "live" else "text_json_v1")
            if output_protocol is None else output_protocol,
        stop_on_invalid_response=(execution_context == "live" if stop_on_invalid_response is None else stop_on_invalid_response),
        max_response_bytes=max_response_bytes,
        settings=settings, bundles=tuple(bundle.declaration for bundle in bundles), cases=tuple(cases),
        source_files={path: _sha(_regular_bytes(path)) for path in sorted(sources)}, evaluator_sources=evaluator)


def _recheck(plan):
    _need(_required_sources(plan.execution_context) <= set(plan.source_files), "Required runner/provider source pins are missing")
    for path, expected in plan.source_files.items():
        _need(_sha(_regular_bytes(path)) == expected, "Reviewed evaluator source changed")
    _need(plan.evaluator_sources == _candidate_sources(), "Reviewed evaluator/prompt source changed")
    bundles = {item.manifest_id: verify_projection_bundle(item.directory) for item in plan.bundles}
    for expected in plan.bundles:
        _need(bundles[expected.manifest_id].declaration == expected, "Planned projection identity changed")
    for identity in plan.cases:
        rows = bundles[identity.manifest_id].ledger
        _need(any(row["disposition"] == "eligible" and row["identity"] == identity.model_dump(mode="json")
            for row in rows), "Selected case no longer matches its exact canonical identity")
    return bundles


def _approval(plan, filename, expected_sha):
    approval, actual = _read(Path(filename))
    _need(actual == expected_sha and approval.get("format") == "rag-evaluator-approval-v1"
        and approval.get("approved") is True and approval.get("approval_source") == "explicit_user_affirmative_reply"
        and isinstance(approval.get("user_reply"), str) and approval["user_reply"].strip()
        and approval.get("plan_sha256") == evaluator_plan_sha256(plan)
        and approval.get("provider") == plan.settings.provider and approval.get("model_id") == plan.settings.model_id
        and approval.get("execution_context") == plan.execution_context
        and type(approval.get("maximum_selected_cases")) is int
        and approval["maximum_selected_cases"] == len(plan.cases)
        and _amount(approval.get("maximum_total_cost_usd")) == plan.settings.total_cost_cap_usd
        and approval.get("automatic_retry_allowed") is False
        and type(approval.get("max_attempts_per_stage")) is int and approval["max_attempts_per_stage"] == 1
        and type(approval.get("sdk_max_retries")) is int and approval["sdk_max_retries"] == 0
        and approval.get("generation_runs_allowed") is False and approval.get("adaptive_runs_allowed") is False
        and approval.get("scoring_allowed") is True and approval.get("citation_audit_allowed") is True,
        "Exact separate evaluator approval is required")
    egress = approval.get("egress", {})
    _need(set(egress) == {"destination", "questions", "generated_answers", "permitted_fictional_corpus_excerpts",
            "reference_answers_for_scoring"} and egress.get("destination") == "Anthropic API"
        and all(egress.get(key) is True for key in set(egress) - {"destination"}), "Exact evaluator egress approval is required")
    return _regular_bytes(filename)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _new_output_directory(output):
    missing, parent = [], output
    while not parent.exists():
        missing.append(parent); parent = parent.parent
    _need(parent.is_dir(), "Evaluator output parent must be a directory")
    for path in reversed(missing):
        path.mkdir(mode=0o700)
        _fsync_directory(path.parent)


def _state_write(path, value):
    temporary = path.parent / (".journal-" + uuid4().hex)
    try:
        _write_new(temporary, _bytes(value) + b"\n")
        os.replace(temporary, path); _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _usage(value, *, exception=False):
    supplied = getattr(value, "usage", None)
    if supplied is None:
        return {"input_tokens": None, "output_tokens": None, "api_calls": None,
            "estimated_cost_usd": None, "token_counts_complete": False}
    raw = supplied.model_dump(mode="json") if hasattr(supplied, "model_dump") else (
        supplied if isinstance(supplied, dict) else vars(supplied))
    result = {key: raw.get(key) for key in ("input_tokens", "output_tokens", "api_calls", "estimated_cost_usd")}
    for key in ("input_tokens", "output_tokens", "api_calls"):
        _need(result[key] is None or type(result[key]) is int and result[key] >= 0, "Invalid provider usage counts")
    if result["estimated_cost_usd"] is not None: _amount(result["estimated_cost_usd"])
    result["token_counts_complete"] = not exception and all(result[key] is not None
        for key in ("input_tokens", "output_tokens"))
    return result


def _sum_usage(values):
    result = {}
    for key in ("input_tokens", "output_tokens", "api_calls"):
        result[key] = sum(value[key] for value in values) if all(value[key] is not None for value in values) else None
    costs = [value["estimated_cost_usd"] for value in values]
    result["estimated_cost_usd"] = float(sum((_amount(value) for value in costs), Decimal(0))) if all(
        value is not None for value in costs) else None
    result["token_counts_complete"] = all(value["token_counts_complete"] for value in values)
    return result


def _incomplete_usage(usage):
    return (any(usage[key] is None for key in (
        "input_tokens", "output_tokens", "api_calls", "estimated_cost_usd"))
        or not usage["token_counts_complete"])


def _response_payload(raw, *, stage, complete, truncated, protocol):
    """Classify retained bytes without rendering or executing provider text."""
    if not complete:
        return None, "response_too_large"
    if truncated:
        return None, "response_truncated"
    try:
        parsed = _json(raw.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return None, "invalid_json"
    try:
        if stage == "score":
            validate_scores(parsed)
            if protocol == "native_schema_v1":
                dimensions = {"correctness", "completeness", "citation_support", "abstention_limits"}
                _need(set(parsed) == dimensions | {"reasons"} and set(parsed["reasons"]) == dimensions,
                    "Native score response has unexpected fields")
        else:
            _validate_audit_schema(parsed)
        _bytes(parsed)  # Reject unpaired Unicode escapes before result serialization.
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
        return None, "invalid_score_schema" if stage == "score" else "invalid_audit_schema"
    return parsed, None


def _diagnostics(outcome):
    return {key: outcome[key] for key in ("response_sha256", "raw_response", "response_truncated",
        "validation_failure_code")}


class _Stages:
    def __init__(self, plan, identity, ordinal, output, journal, persist, recheck, factory, case, score_prompt,
                 audit_prompt, raw_artifacts):
        self.plan, self.identity, self.ordinal, self.output = plan, identity, ordinal, output
        self.journal, self.persist, self.recheck, self.factory = journal, persist, recheck, factory
        self.raw_artifacts = raw_artifacts
        self.expected = (("score", SCORE_SYSTEM, _scorer_prompt(case, score_prompt)),
                         ("citation_audit", AUDIT_SYSTEM, _auditor_prompt(case, audit_prompt)))
        self.outcomes, self.validated, self.fatal = {}, {}, None

    def complete(self, *, system, prompt):
        _need(self.fatal is None, "An earlier fatal stage forbids further dispatch")
        stage = self.expected[len(self.outcomes)][0] if len(self.outcomes) < 2 else None
        outcome = {"stage": stage, "provider_complete_called": False, "usage": {
            "input_tokens": 0, "output_tokens": 0, "api_calls": 0, "estimated_cost_usd": 0,
            "token_counts_complete": True}, "error": None, "response_sha256": None,
            "raw_response": None, "response_truncated": None, "validation_failure_code": None}
        provider = None
        try:
            _need(stage is not None and (stage, system, prompt) == self.expected[len(self.outcomes)],
                "Unexpected stage, retry or changed evaluator payload")
            try: self.recheck()
            except BaseException:
                self.fatal = "InputIntegrityError"
                raise EvaluatorRunError("Evaluator inputs changed before stage admission") from None
            cap = self.plan.settings.score_cost_cap_usd if stage == "score" else self.plan.settings.audit_cost_cap_usd
            admission = StageAdmission(plan_sha256=evaluator_plan_sha256(self.plan), identity=self.identity,
                stage=stage, judge=self.plan.settings, max_cost_usd=cap, execution_context=self.plan.execution_context,
                output_protocol=self.plan.output_protocol)
            intent = {"identity": self.identity.model_dump(mode="json"), "stage": stage, "state": "intent",
                "reservation_usd": str(cap), "suballocation_of_case_reservation": True,
                "system_sha256": _sha(system.encode()), "prompt_sha256": _sha(prompt.encode()),
                "output_protocol": self.plan.output_protocol, "max_response_bytes": self.plan.max_response_bytes}
            name = f"stages/case_{self.ordinal:06d}_{stage}"
            _write_new(self.output / (name + ".intent.json"), _bytes(intent) + b"\n")
            self.journal["cases"][-1]["stages"][stage] = intent
            self.persist(); self.recheck()
            try: provider = self.factory(admission)
            except BaseException:
                self.fatal = "ProviderFactoryError"
                raise
            self.recheck()
            outcome["provider_complete_called"] = True
            outcome["usage"] = {"input_tokens": None, "output_tokens": None, "api_calls": None,
                "estimated_cost_usd": None, "token_counts_complete": False}
            try:
                with call_budget(CallBudget(self.plan.settings.max_tokens, self.plan.settings.timeout_seconds, float(cap))):
                    completion = provider.complete(system=system, prompt=prompt)
            except BaseException as error:
                outcome["usage"] = _usage(error, exception=True)
                raise
            outcome["usage"] = _usage(completion)
            _need(isinstance(completion.text, str), "Evaluator response must be text")
            raw = completion.text.encode("utf-8")
            truncated = getattr(completion, "truncated", False)
            _need(type(truncated) is bool, "Evaluator response truncation must be explicit")
            retained = raw[:self.plan.max_response_bytes]
            raw_name = name + ".response.txt"
            _write_new(self.output / raw_name, retained)
            raw_hash = _sha(retained)
            self.raw_artifacts[raw_name] = raw_hash
            outcome.update(response_sha256=_sha(raw), response_truncated=truncated,
                raw_response={"path": raw_name, "sha256": raw_hash, "size_bytes": len(retained),
                    "total_size_bytes": len(raw), "complete": len(raw) <= self.plan.max_response_bytes})
            usage = outcome["usage"]
            _need(usage["estimated_cost_usd"] is None or _amount(usage["estimated_cost_usd"]) <= cap,
                "Provider exceeded its admitted stage cost")
            _need(usage["output_tokens"] is None or usage["output_tokens"] <= self.plan.settings.max_output_tokens,
                "Provider exceeded its admitted output bound")
            _need(usage["api_calls"] is None or usage["api_calls"] <= 2,
                "Provider reported unexpected retry/API-call count")
            self.recheck()
            parsed, failure = _response_payload(retained, stage=stage,
                complete=outcome["raw_response"]["complete"], truncated=truncated, protocol=self.plan.output_protocol)
            outcome["validation_failure_code"] = failure
            if failure is not None:
                if self.plan.stop_on_invalid_response:
                    self.fatal = "InvalidEvaluatorResponse"
                raise ValueError("Invalid evaluator response")
            self.validated[stage] = parsed
            if self.plan.stop_on_invalid_response and _incomplete_usage(outcome["usage"]):
                # Keep a usable grade/raw response, but do not admit its audit
                # or another case while the attempted stage's usage is unknown.
                self.fatal = "IncompleteEvaluatorUsage"
            return completion
        except BaseException as error:
            outcome["error"] = type(error).__name__
            if isinstance(error, EvaluatorRunError) or not isinstance(error, (ValueError, ProviderError, RunBudgetExceeded)):
                self.fatal = type(error).__name__
            elif self.plan.stop_on_invalid_response:
                if isinstance(error, (ProviderError, RunBudgetExceeded)):
                    self.fatal = type(error).__name__
                elif self.fatal is None and outcome["provider_complete_called"] and _incomplete_usage(outcome["usage"]):
                    self.fatal = "IncompleteEvaluatorUsage"
            raise
        finally:
            if provider is not None and callable(getattr(provider, "close", None)):
                try: provider.close()
                except BaseException:
                    self.fatal = "ProviderCleanupError"
            if stage is not None:
                self.outcomes[stage] = outcome
                if "name" in locals():
                    _write_new(self.output / (name + ".outcome.json"), _bytes(outcome) + b"\n")
                    self.journal["cases"][-1]["stages"][stage]["state"] = "finished"
                    self.persist()


def run_evaluator_plan(plan, *, approval_file, approval_sha256, provider_factory: Callable[[StageAdmission], object]):
    """Run each selected score/audit stage at most once; never resume an output."""
    plan = _checked(EvaluatorPlan, plan, "evaluator plan")
    approval_bytes = _approval(plan, approval_file, approval_sha256)
    bundles = _recheck(plan)
    output = _output_location(Path(plan.output_directory))
    _need(not output.exists() and not output.is_symlink(), "Evaluator output already exists; automatic replay is forbidden")
    _need(all(not Path(bundle.directory).resolve().is_relative_to(output.resolve())
        and not output.resolve().is_relative_to(Path(bundle.directory).resolve()) for bundle in plan.bundles),
        "Evaluator output must be separate from immutable projections")
    _new_output_directory(output)
    lock = output / ".writer.lock"
    _write_new(lock, _bytes({"pid": os.getpid(), "plan_sha256": evaluator_plan_sha256(plan)}) + b"\n")
    _write_new(output / "plan.json", _bytes(plan.model_dump(mode="json")) + b"\n")
    _write_new(output / "approval.json", approval_bytes)
    for name in ("stages", "cases"): (output / name).mkdir(mode=0o700)
    journal = {"format": "rag-evaluator-attempt-journal-v1", "plan_sha256": evaluator_plan_sha256(plan),
        "case_reservations_counted_once": True, "stage_reservations_are_suballocations": True,
        "reserved_usd": "0", "cases": []}
    journal_sha, raw_artifacts = None, {}
    def persist():
        nonlocal journal_sha
        if journal_sha is not None:
            _need(_sha(_regular_bytes(output / "journal.json")) == journal_sha, "Durable evaluator journal changed")
        _state_write(output / "journal.json", journal)
        journal_sha = _sha(_regular_bytes(output / "journal.json"))
    def unchanged():
        _approval(plan, approval_file, approval_sha256); _recheck(plan)
        for name, expected in raw_artifacts.items():
            _need(_sha(_retained_response_bytes(output / name, plan.max_response_bytes)) == expected,
                "Retained evaluator response changed")
        if journal_sha is not None:
            _need(_sha(_regular_bytes(output / "journal.json")) == journal_sha, "Durable evaluator journal changed")
    records, fatal = {}, None
    try:
        persist()
        score_prompt = _regular_bytes(ROOT / plan.evaluator_sources["score_prompt"]).decode("utf-8")
        audit_prompt = _regular_bytes(ROOT / plan.evaluator_sources["audit_prompt"]).decode("utf-8")
        for ordinal, identity in enumerate(plan.cases, 1):
            unchanged()
            reservation = _amount(journal["reserved_usd"]) + plan.settings.case_cost_cap_usd
            _need(reservation <= plan.settings.total_cost_cap_usd, "Global evaluator reservation exhausted")
            journal["reserved_usd"] = str(reservation)
            journal["cases"].append({"identity": identity.model_dump(mode="json"), "state": "intent",
                "reservation_usd": str(plan.settings.case_cost_cap_usd), "stages": {}})
            persist()
            bundle = bundles[identity.manifest_id]
            case = make_case(bundle.answers[identity.question_id], bundle.references[identity.question_id])
            stages = _Stages(plan, identity, ordinal, output, journal, persist, unchanged, provider_factory,
                case, score_prompt, audit_prompt, raw_artifacts)
            try:
                result = evaluate_case(case, stages, score_prompt, audit_prompt, max_attempts=1, existing=None)
            except Exception as error:
                # Pinned candidate review code can itself fail after completions.
                # Keep stage accounting and stop; never spend again to repair it.
                stages.fatal = "CandidateEvaluationError"
                score = stages.validated.get("score")
                result = build_record(case, scores=score, audit=None, audit_required=True,
                    score_error=type(error).__name__ if score is None else None, audit_error="CandidateEvaluationError")
            result["stage_usage"] = {name: item["usage"] for name, item in stages.outcomes.items()}
            result["stage_diagnostics"] = {name: _diagnostics(item) for name, item in stages.outcomes.items()}
            result["usage"] = _sum_usage(list(result["stage_usage"].values()))
            result["execution_identity"] = identity.model_dump(mode="json")
            result["provenance"].update(mode=plan.execution_context, execution_context=plan.execution_context,
                execution_runner=EVALUATOR,
                output_protocol=plan.output_protocol,
                score_source=("rc2_prompt_" + plan.execution_context if stages.outcomes.get("score", {}).get(
                    "provider_complete_called") else "not_run"),
                audit_source=("v5_cited_only_audit_" + plan.execution_context if stages.outcomes.get("citation_audit", {}).get(
                    "provider_complete_called") else "not_run"),
                failed_attempt_usage_retained=True, stage_reservations_are_suballocations=True)
            path = output / f"cases/case_{ordinal:06d}.json"
            _write_new(path, _bytes(result) + b"\n")
            records[(identity.manifest_id, identity.question_id)] = result
            journal["cases"][-1].update(state="committed", result_path=str(path.relative_to(output)),
                result_sha256=_sha(_regular_bytes(path)))
            persist()
            if stages.fatal:
                fatal = stages.fatal; break
        unchanged()
        return _finish(plan, output, bundles, journal, records, approval_sha256, fatal)
    finally:
        lock.unlink(missing_ok=True); _fsync_directory(output)


def _finish(plan, output, bundles, journal, records, approval_sha, fatal):
    selected = {(case.manifest_id, case.question_id) for case in plan.cases}
    coverage, bundle_files = [], []
    for index, spec in enumerate(plan.bundles, 1):
        bundle = bundles[spec.manifest_id]
        results = [records[(case.manifest_id, case.question_id)] for case in plan.cases
            if case.manifest_id == spec.manifest_id and (case.manifest_id, case.question_id) in records]
        result_name = f"bundle_{index:06d}.results.json"
        _write_new(output / result_name, _bytes(results) + b"\n")
        bundle_files.append((spec, result_name))
        for row in bundle.ledger:
            key = (spec.manifest_id, row["identity"]["question_id"])
            result = records.get(key)
            coverage.append({"identity": row["identity"], "runtime_disposition": row["disposition"],
                "selected_for_judging": key in selected, "judge_disposition": (
                    "unsupported" if row["disposition"] != "eligible" else "not_selected" if key not in selected else
                    "not_dispatched" if result is None else "score_error" if result.get("error") else
                    "audit_error" if result.get("audit_error") else "scored_and_audited_unverified")})
    _write_new(output / "coverage.jsonl", b"".join(_bytes(row) + b"\n" for row in coverage))
    artifacts = {str(path.relative_to(output)): _sha(_retained_response_bytes(path, plan.max_response_bytes)
            if path.name.endswith(".response.txt") else _regular_bytes(path)) for folder in ("stages", "cases")
        for path in sorted((output / folder).iterdir())}
    artifacts.update({name: _sha(_regular_bytes(output / name)) for name in (
        "plan.json", "approval.json", "journal.json", "coverage.jsonl", *(name for _spec, name in bundle_files))})
    receipt = {"format": "rag-guarded-evaluator-execution-v1", "evaluator": EVALUATOR,
        "plan_sha256": evaluator_plan_sha256(plan), "approval_sha256": approval_sha,
        "execution_context": plan.execution_context, "output_protocol": plan.output_protocol,
        "stop_on_invalid_response": plan.stop_on_invalid_response, "max_response_bytes": plan.max_response_bytes,
        "selected_case_identities": [case.model_dump(mode="json") for case in plan.cases],
        "selected_cases": len(plan.cases), "committed_results": len(records), "fatal_error": fatal,
        "reserved_usd": journal["reserved_usd"], "logical_provider_calls": sum(
            read_outcome["provider_complete_called"] for path in (output / "stages").glob("*.outcome.json")
            for read_outcome in [_json(_regular_bytes(path))]),
        "coverage_counts": dict(Counter(row["judge_disposition"] for row in coverage)),
        "human_verified": False, "automatic_replay_allowed": False, "artifacts": artifacts}
    all_usage = [usage for result in records.values() for usage in result["stage_usage"].values()]
    receipt["reported_usage"] = _sum_usage(all_usage)
    receipt["known_reported_cost_decimal"] = str(sum((_amount(value["estimated_cost_usd"])
        for value in all_usage if value["estimated_cost_usd"] is not None), Decimal(0)))
    receipt["unknown_cost_stages"] = sum(value["estimated_cost_usd"] is None for value in all_usage)
    diagnostics = [value for record in records.values() for value in record["stage_diagnostics"].values()]
    receipt["raw_response_artifacts"] = sum(value["raw_response"] is not None for value in diagnostics)
    receipt["validation_failure_counts"] = dict(Counter(value["validation_failure_code"] for value in diagnostics
        if value["validation_failure_code"] is not None))
    _write_new(output / "receipt.json", _bytes(receipt) + b"\n")
    refs = {name: {"path": name, "sha256": _sha(_regular_bytes(output / name))}
        for name in ("plan.json", "approval.json", "journal.json", "receipt.json")}
    for index, (spec, _results) in enumerate(bundle_files, 1):
        source = plan.evaluator_sources; files = source["files"]
        hashes = {"answers": spec.files["answers.json"], "questions": spec.files["questions.json"],
            "score_prompt": files[source["score_prompt"]]["sha256"], "audit_prompt": files[source["audit_prompt"]]["sha256"],
            "candidate_code": files["src/evaluation/evaluator_candidate.py"]["sha256"],
            "verified_code": files["src/evaluation/verified_judge.py"]["sha256"],
            "v6_audit_code": files["src/evaluation/verified_judge_v6.py"]["sha256"],
            "runner_code": plan.source_files[str(Path(__file__).resolve())]}
        manifest = {"evaluator": EVALUATOR, "model": plan.settings.model_id, "audit_mode": "all",
            "selected_ids": [case.question_id for case in plan.cases if case.manifest_id == spec.manifest_id],
            "sha256": hashes, "execution": {"format": "rag-guarded-evaluator-execution-v1",
                "plan_sha256": evaluator_plan_sha256(plan), "projection_manifest_sha256": spec.projection_manifest_sha256,
                "manifest_id": spec.manifest_id, "references": refs,
                "selected_case_identities": [case.model_dump(mode="json") for case in plan.cases if case.manifest_id == spec.manifest_id]}}
        _write_new(output / f"bundle_{index:06d}.manifest.json", _bytes(manifest) + b"\n")
    return receipt


def validate_guarded_result_manifest(bundle_dir, results_file, evaluator_manifest):
    """Pure exact plan/case/stage provenance verification for the bridge.

    This checks recorded execution and input identity, not model judgment quality
    or human agreement. All execution references must stay within the manifest's
    directory. It never constructs a provider, writes files, or calls the bridge.
    """
    manifest_path = Path(evaluator_manifest).absolute()
    root = manifest_path.parent.resolve()
    _need(not (root / ".writer.lock").exists(), "Evaluator writer is still active or uncertain")
    manifest, _ = _read(manifest_path)
    _need(manifest.get("evaluator") == EVALUATOR and manifest.get("audit_mode") == "all",
        "Unsupported guarded evaluator manifest")
    execution = manifest.get("execution", {})
    refs = execution.get("references", {})
    _need(execution.get("format") == "rag-guarded-evaluator-execution-v1"
        and set(refs) == {"plan.json", "approval.json", "journal.json", "receipt.json"},
        "Guarded execution references are incomplete")

    def contained(name):
        _need(isinstance(name, str) and not Path(name).is_absolute()
            and all(part not in {"", ".", ".."} for part in name.split("/")), "Unsafe execution reference")
        path = root / name
        _need(path.resolve().is_relative_to(root) and not path.is_symlink(), "Execution reference escapes its directory")
        return path

    values = {}
    for name, reference in refs.items():
        _need(reference.get("path") == name, "Unexpected guarded execution reference")
        value, actual = _read(contained(reference["path"]))
        _need(actual == reference["sha256"], "Guarded execution reference changed")
        values[name] = value
    plan = _checked(EvaluatorPlan, values["plan.json"], "evaluator plan")
    plan_hash = evaluator_plan_sha256(plan)
    _need(Path(plan.output_directory).resolve() == root and execution.get("plan_sha256") == plan_hash,
        "Guarded execution belongs to another output/plan")
    bundles = _recheck(plan)
    bundle = verify_projection_bundle(bundle_dir)
    _need(bundle.declaration in plan.bundles and execution.get("manifest_id") == bundle.declaration.manifest_id
        and execution.get("projection_manifest_sha256") == bundle.declaration.projection_manifest_sha256,
        "Guarded results belong to another projection")
    selected = [case for case in plan.cases if case.manifest_id == bundle.declaration.manifest_id]
    _need(execution.get("selected_case_identities") == [case.model_dump(mode="json") for case in selected]
        and manifest.get("selected_ids") == [case.question_id for case in selected]
        and manifest.get("model") == plan.settings.model_id
        and manifest.get("sha256", {}).get("runner_code") == plan.source_files[str(Path(__file__).resolve())],
        "Guarded result selection/model/runner differs")
    receipt, journal = values["receipt.json"], values["journal.json"]
    _need(receipt.get("format") == "rag-guarded-evaluator-execution-v1"
        and receipt.get("evaluator") == EVALUATOR and receipt.get("plan_sha256") == plan_hash
        and receipt.get("execution_context") == plan.execution_context
        and receipt.get("output_protocol") == plan.output_protocol
        and receipt.get("stop_on_invalid_response") is plan.stop_on_invalid_response
        and receipt.get("max_response_bytes") == plan.max_response_bytes
        and receipt.get("selected_case_identities") == [case.model_dump(mode="json") for case in plan.cases]
        and receipt.get("selected_cases") == len(plan.cases)
        and receipt.get("human_verified") is False and receipt.get("automatic_replay_allowed") is False,
        "Guarded execution receipt differs")
    _approval(plan, contained("approval.json"), receipt["approval_sha256"])
    artifacts = receipt.get("artifacts", {})
    _need(type(artifacts) is dict and {"plan.json", "approval.json", "journal.json", "coverage.jsonl"} <= artifacts.keys(),
        "Guarded execution artifact inventory is incomplete")
    for name, expected in artifacts.items():
        raw = (_retained_response_bytes(contained(name), plan.max_response_bytes)
            if name.endswith(".response.txt") else _regular_bytes(contained(name)))
        _need(_sha(raw) == expected, "Recorded execution artifact changed")
    _need(journal.get("format") == "rag-evaluator-attempt-journal-v1"
        and journal.get("plan_sha256") == plan_hash
        and journal.get("case_reservations_counted_once") is True
        and journal.get("stage_reservations_are_suballocations") is True
        and isinstance(journal.get("cases"), list)
        and len(journal["cases"]) == receipt.get("committed_results") <= len(plan.cases),
        "Guarded execution journal differs")
    all_records, known_usage, logical_calls, diagnostics, invalid_responses, policy_stops = {}, [], 0, [], [], []
    score_template = _regular_bytes(ROOT / plan.evaluator_sources["score_prompt"]).decode()
    audit_template = _regular_bytes(ROOT / plan.evaluator_sources["audit_prompt"]).decode()
    for ordinal, item in enumerate(journal["cases"], 1):
        identity = plan.cases[ordinal - 1]
        _need(item.get("identity") == identity.model_dump(mode="json") and item.get("state") == "committed"
            and _amount(item.get("reservation_usd")) == plan.settings.case_cost_cap_usd
            and item.get("result_path") == f"cases/case_{ordinal:06d}.json", "Invalid or repeated journal case")
        result, actual = _read(contained(item["result_path"]))
        _need(actual == item["result_sha256"] == artifacts.get(item["result_path"])
            and result.get("execution_identity") == identity.model_dump(mode="json"), "Case result identity changed")
        original = bundles[identity.manifest_id]
        case = make_case(original.answers[identity.question_id], original.references[identity.question_id])
        stage_usage, stage_calls, stage_diagnostics = {}, {}, {}
        for stage, intent in item["stages"].items():
            _need(stage in {"score", "citation_audit"}, "Unknown journal stage")
            base = f"stages/case_{ordinal:06d}_{stage}"
            for suffix in (".intent.json", ".outcome.json"):
                _need(base + suffix in artifacts, "Stage lacks a pinned durable artifact")
            original_intent, _ = _read(contained(base + ".intent.json"))
            outcome, _ = _read(contained(base + ".outcome.json"))
            cap = plan.settings.score_cost_cap_usd if stage == "score" else plan.settings.audit_cost_cap_usd
            system, prompt = ((SCORE_SYSTEM, _scorer_prompt(case, score_template)) if stage == "score" else
                (AUDIT_SYSTEM, _auditor_prompt(case, audit_template)))
            expected_intent = {"identity": identity.model_dump(mode="json"), "stage": stage, "state": "intent",
                "reservation_usd": str(cap), "suballocation_of_case_reservation": True,
                "system_sha256": _sha(system.encode()), "prompt_sha256": _sha(prompt.encode()),
                "output_protocol": plan.output_protocol, "max_response_bytes": plan.max_response_bytes}
            _need(original_intent == expected_intent and intent == {**expected_intent, "state": "finished"}
                and outcome.get("stage") == stage and type(outcome.get("provider_complete_called")) is bool
                and {"response_sha256", "raw_response", "response_truncated", "validation_failure_code"} <= outcome.keys(),
                "Stage order, payload or reservation differs from its durable admission")
            usage = outcome["usage"]
            _need(set(usage) == {"input_tokens", "output_tokens", "api_calls", "estimated_cost_usd", "token_counts_complete"}
                and type(usage["token_counts_complete"]) is bool
                and all(usage[key] is None or type(usage[key]) is int and usage[key] >= 0
                    for key in ("input_tokens", "output_tokens", "api_calls"))
                and (usage["estimated_cost_usd"] is None or _amount(usage["estimated_cost_usd"]) <= cap)
                and (usage["output_tokens"] is None or usage["output_tokens"] <= plan.settings.max_output_tokens)
                and (usage["api_calls"] is None or usage["api_calls"] <= 2),
                "Stage usage or spending exceeds its admitted bounds")
            if outcome["provider_complete_called"] and (
                    outcome.get("error") in {"ProviderError", "RunBudgetExceeded"} or _incomplete_usage(usage)):
                policy_stops.append((ordinal, stage))
            raw = outcome.get("raw_response")
            if raw is not None:
                _need(type(raw) is dict and set(raw) == {"path", "sha256", "size_bytes", "total_size_bytes", "complete"}
                    and raw["path"] == base + ".response.txt" and raw["sha256"] == artifacts.get(raw["path"])
                    and type(raw["complete"]) is bool
                    and type(raw["size_bytes"]) is int and type(raw["total_size_bytes"]) is int
                    and 0 <= raw["size_bytes"] <= plan.max_response_bytes
                    and raw["size_bytes"] == min(raw["total_size_bytes"], plan.max_response_bytes)
                    and raw["complete"] == (raw["total_size_bytes"] <= plan.max_response_bytes)
                    and type(outcome.get("response_truncated")) is bool
                    and outcome["provider_complete_called"] is True,
                    "Retained raw response metadata differs from its stage")
                raw_bytes = _retained_response_bytes(contained(raw["path"]), plan.max_response_bytes)
                response_hash = outcome.get("response_sha256")
                _need(len(raw_bytes) == raw["size_bytes"] and _sha(raw_bytes) == raw["sha256"]
                    and isinstance(response_hash, str) and len(response_hash) == 64
                    and all(character in "0123456789abcdef" for character in response_hash)
                    and (not raw["complete"] or response_hash == raw["sha256"]), "Retained raw response bytes changed")
                parsed, failure = _response_payload(raw_bytes, stage=stage, complete=raw["complete"],
                    truncated=outcome["response_truncated"], protocol=plan.output_protocol)
                _need(outcome.get("validation_failure_code") == failure, "Raw response validation classification changed")
                if failure is not None:
                    _need(outcome.get("error") == "ValueError", "Invalid raw response lacks a retained stage failure")
                    invalid_responses.append((ordinal, stage))
                    _need((result.get("scores") is None and result.get("raw_scores") is None
                        and isinstance(result.get("error"), str) and bool(result["error"].strip())) if stage == "score"
                        else (result.get("audit") is None and isinstance(result.get("audit_error"), str)
                            and bool(result["audit_error"].strip())),
                        "Invalid raw response was promoted into an evaluation result")
                elif stage == "score":
                    _need(result.get("scores") == result.get("raw_scores") == parsed,
                        "Scoring result differs from its retained response")
                else:
                    _need(result.get("audit") == parsed or result.get("audit") is None
                        and result.get("audit_error") == "CandidateEvaluationError",
                        "Citation audit differs from its retained response")
            else:
                _need(outcome.get("response_sha256") is None and outcome.get("response_truncated") is None
                    and outcome.get("validation_failure_code") is None and isinstance(outcome.get("error"), str),
                    "A completed response lacks retained raw diagnostics")
                _need((result.get("scores") is None and result.get("raw_scores") is None
                    and isinstance(result.get("error"), str) and bool(result["error"].strip())) if stage == "score"
                    else (result.get("audit") is None and isinstance(result.get("audit_error"), str)
                        and bool(result["audit_error"].strip())),
                    "A rawless failed stage cannot contain an evaluation grade")
            stage_diagnostics[stage] = _diagnostics(outcome)
            diagnostics.append(stage_diagnostics[stage])
            stage_usage[stage] = outcome["usage"]
            stage_calls[stage] = outcome["provider_complete_called"]
            known_usage.append(outcome["usage"]); logical_calls += outcome["provider_complete_called"]
        _need(set(item["stages"]) in ({"score"}, {"score", "citation_audit"})
            and result.get("stage_usage") == stage_usage and result.get("usage") == _sum_usage(list(stage_usage.values()))
            and result.get("stage_diagnostics") == stage_diagnostics
            and result.get("question_id") == identity.question_id and result.get("generation_status") == case.status,
            "Case stage/usage coverage differs")
        provenance = result.get("provenance", {})
        _need(provenance.get("mode") == provenance.get("execution_context") == plan.execution_context
            and provenance.get("execution_runner") == EVALUATOR
            and provenance.get("output_protocol") == plan.output_protocol
            and provenance.get("score_source") == ("rc2_prompt_" + plan.execution_context if stage_calls.get("score") else "not_run")
            and provenance.get("audit_source") == ("v5_cited_only_audit_" + plan.execution_context if stage_calls.get("citation_audit") else "not_run")
            and provenance.get("audit_is_human_verified") is False
            and provenance.get("score_overrides_applied") is False,
            "Case result misstates execution context, prompt source or human verification")
        all_records[(identity.manifest_id, identity.question_id)] = result
    if plan.stop_on_invalid_response and invalid_responses:
        ordinal, stage = invalid_responses[0]
        _need(len(invalid_responses) == 1 and ordinal == len(journal["cases"])
            and (stage != "score" or set(journal["cases"][-1]["stages"]) == {"score"})
            and receipt.get("fatal_error") in {"InvalidEvaluatorResponse", "ProviderCleanupError"},
            "Execution continued after its first invalid evaluator response")
    if plan.stop_on_invalid_response and policy_stops:
        # Stored JSON sorts stage keys; their dictionary order is not dispatch
        # order. There must be just one stopping boundary, at the final case,
        # with no audit admitted after a stopping score stage.
        _need(len(policy_stops) == 1 and policy_stops[0][0] == len(journal["cases"])
            and (policy_stops[0][1] != "score" or set(journal["cases"][-1]["stages"]) == {"score"})
            and isinstance(receipt.get("fatal_error"), str) and bool(receipt["fatal_error"]),
            "Execution continued after a provider failure or incomplete usage")
    reserved = len(journal["cases"]) * plan.settings.case_cost_cap_usd
    _need(_amount(journal["reserved_usd"]) == _amount(receipt["reserved_usd"]) == reserved <= plan.settings.total_cost_cap_usd
        and receipt.get("logical_provider_calls") == logical_calls
        and receipt.get("reported_usage") == _sum_usage(known_usage)
        and receipt.get("known_reported_cost_decimal") == str(sum((_amount(value["estimated_cost_usd"])
            for value in known_usage if value["estimated_cost_usd"] is not None), Decimal(0)))
        and receipt.get("unknown_cost_stages") == sum(value["estimated_cost_usd"] is None for value in known_usage)
        and receipt.get("raw_response_artifacts") == sum(value["raw_response"] is not None for value in diagnostics)
        and receipt.get("validation_failure_counts") == dict(Counter(value["validation_failure_code"] for value in diagnostics
            if value["validation_failure_code"] is not None))
        and (len(all_records) == len(plan.cases) or isinstance(receipt.get("fatal_error"), str)
            and bool(receipt["fatal_error"])), "Global reservation or reported usage differs")
    selected_keys = {(case.manifest_id, case.question_id) for case in plan.cases}
    coverage = []
    for spec in plan.bundles:
        for row in bundles[spec.manifest_id].ledger:
            key = (spec.manifest_id, row["identity"]["question_id"])
            result = all_records.get(key)
            coverage.append({"identity": row["identity"], "runtime_disposition": row["disposition"],
                "selected_for_judging": key in selected_keys, "judge_disposition": (
                    "unsupported" if row["disposition"] != "eligible" else "not_selected" if key not in selected_keys else
                    "not_dispatched" if result is None else "score_error" if result.get("error") else
                    "audit_error" if result.get("audit_error") else "scored_and_audited_unverified")})
    _need(_regular_bytes(contained("coverage.jsonl")) == b"".join(_bytes(row) + b"\n" for row in coverage)
        and receipt.get("coverage_counts") == dict(Counter(row["judge_disposition"] for row in coverage)),
        "Unsupported, failed or unattempted case coverage was changed")
    result_path = Path(results_file).absolute()
    _need(result_path.parent.resolve() == root and result_path.name in artifacts,
        "Results file is not part of this execution")
    result_rows, result_sha = _read(result_path)
    expected_rows = [all_records[(case.manifest_id, case.question_id)] for case in selected
        if (case.manifest_id, case.question_id) in all_records]
    _need(result_rows == expected_rows and result_sha == artifacts[result_path.name], "Candidate results differ from exact case records")
    return {"plan_sha256": plan_hash, "execution_context": plan.execution_context,
        "selected_case_identities": [case.model_dump(mode="json") for case in selected],
        "committed_results": len(expected_rows), "human_verified": False,
        "source_and_journal_verified": True}

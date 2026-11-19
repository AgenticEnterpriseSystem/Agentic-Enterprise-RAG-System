"""Bounded CAL-01 adaptive capture production; no model/runtime construction.

The caller owns the runtime and its resource/session lifecycle. Preparation is
non-generating. Execution requires a separate exact approval, fresh outputs and
durable per-question reservations. Unknown outcomes are never replayed. Private
critic inputs come only from the issuing runtime's current-authority export.
This producer creates neither human labels nor an accepted configuration freeze.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from src.contracts import QueryRequest, ScopeBinding
from src.evaluation.critic_calibration import (
    AuthorizationReceipt, CalibrationSelection, CriticInputCapture,
    capture_artifact_sha256, import_critic_episodes, recheck_selection,
)
from src.evaluation.experiment_contracts import ExperimentManifest, runtime_case_digest
from src.evaluation.experiment_inputs import (
    ExperimentAdmission, InputFiles, VerifiedInputs, canonical_json_sha256,
    capture_code_provenance, file_sha256, question_set_sha256,
    select_runtime_questions, verify_inputs,
)
from src.evaluation.experiment_io import RunDirectory, _output_location
from src.evaluation.fixed_runs import (
    _PROFILE_FIELDS, _amount, _bytes, _check_generation, _json, _resolved_identity,
    _sync, _write_new,
)

ROOT = Path(__file__).resolve().parents[2]


class CalibrationRunError(ValueError):
    """A fixed, nonsecret preparation/admission/capture failure."""


def _need(value, message):
    if not value:
        raise CalibrationRunError(message)


def _producer_sources():
    return {str(path): file_sha256(path) for path in (
        Path(__file__).resolve(), Path(__file__).with_name("critic_calibration.py").resolve())}


def _missing_ancestry(path):
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    return missing


def _sync_new_ancestry(missing):
    # Persist each new directory inode and, crucially, the entry naming it in
    # its parent. Fsyncing only journal files/the leaf directory is insufficient.
    for path in reversed(missing):
        _sync(path)
        _sync(path.parent)


def _durable_mkdir(path):
    missing = _missing_ancestry(path)
    path.mkdir(parents=True, mode=0o700)
    _sync_new_ancestry(missing)


@dataclass(frozen=True)
class CalibrationRun:
    manifest: ExperimentManifest
    inputs: VerifiedInputs
    selection: CalibrationSelection
    directory: Path
    profile_sha256: str
    bitmap_sha256: str
    producer_sources: dict

    def __post_init__(self):
        _need(type(self.inputs) is VerifiedInputs, "Calibration requires verified actual inputs")
        manifest = ExperimentManifest.model_validate(self.manifest.model_dump(mode="json"))
        selection = CalibrationSelection.model_validate(self.selection.model_dump(mode="json"))
        _need(manifest.workflow == "adaptive" and manifest.split == "calibration"
              and manifest.retrieval.retriever_id == "store-hybrid"
              and manifest.question_ids == tuple(q.question_id for q in selection.questions),
              "Calibration requires the exact selected adaptive hybrid arm")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "directory", Path(self.directory).absolute())
        object.__setattr__(self, "producer_sources", MappingProxyType(dict(self.producer_sources)))


def _body(question):
    # Filters are frozen in the admitted actual profile; this protocol evaluates
    # its full configured accessible corpus and supplies no browser scope claims.
    return QueryRequest(question=question.question, mode="agentic")


def _bitmap(actual, identity, binding):
    context = actual.resolver.resolve(_resolved_identity(identity))
    actual.resolver.assert_current(context); context.assert_bound()
    digest = getattr(context.scope, "bitmap_sha256", None)
    _need(context.binding == binding and isinstance(digest, str) and len(digest) == 64
          and all(c in "0123456789abcdef" for c in digest),
          "Current authorization must provide the exact scoped bitmap identity")
    return digest


def _profile_identity(profile):
    return canonical_json_sha256(profile)


def prepare_calibration_run(actual, identity, selection, *, sources, prepared_dir, run_dir, comparison):
    """Write verified adaptive inputs without calling query/provider methods."""
    _need(isinstance(sources, dict) and set(sources) == {"questions", "splits", "corpus", "catalog", "source_repo"}
          and all(isinstance(path, Path) for path in sources.values()), "Five exact source roles are required")
    selection = CalibrationSelection.model_validate(selection.model_dump(mode="json"))
    recheck_selection(selection, sources["questions"], sources["splits"])
    questions = select_runtime_questions(sources["questions"], sources["splits"], "calibration",
        question_ids=tuple(q.question_id for q in selection.questions))
    _need(all(q.question == q.question.strip() for q in questions), "Runtime question text must be exact and trimmed")
    _need(isinstance(comparison, dict), "Frozen comparison settings are required")
    _bytes(comparison)
    profile = actual.experiment_profile(_body(questions[0]), _resolved_identity(identity))
    _need(profile["workflow"] == "adaptive" and profile["retrieval"]["retriever_id"] == "store-hybrid"
          and profile["critic_prompt_sha256"] is not None, "Actual native adaptive hybrid settings are required")
    binding = ScopeBinding.model_validate(profile["scope"])
    bitmap = _bitmap(actual, identity, binding)
    prepared = _output_location(prepared_dir, sources["source_repo"])
    directory = _output_location(run_dir, sources["source_repo"])
    _need(not os.path.lexists(prepared) and not os.path.lexists(directory), "Preparation and runtime outputs must be new")
    _need(not prepared.resolve().is_relative_to(directory.resolve())
          and not directory.resolve().is_relative_to(prepared.resolve()), "Prepared and runtime directories must be separate")
    _durable_mkdir(prepared)
    from src.agents.search_critic import SYSTEM_PROMPT as CRITIC_PROMPT
    from src.agents.routing import ROUTER_PROMPT
    from src.generation.generator import SYSTEM_PROMPT as GENERATOR_PROMPT
    scope = {"binding": binding.model_dump(mode="json"), "origin": "operator",
        "purpose": "CAL-01 adaptive critic input capture", "effective_bitmap_sha256": bitmap}
    values = {"index-manifest.json": profile["index_manifest"], "retrieval-config.json": profile["retrieval_config"],
        "comparison-config.json": comparison, "scope.json": scope,
        "runtime-config.json": {key: profile[key] for key in (
            "generation_model", "retrieval", "budgets", "temperature", "workflow", "max_rounds")},
        "selection.json": selection.model_dump(mode="json"), "profile.json": profile}
    for name, value in values.items(): _write_new(prepared / name, _bytes(value))
    for name, prompt in (("critic.txt", CRITIC_PROMPT), ("generator.txt", GENERATOR_PROMPT)):
        _write_new(prepared / name, prompt.encode("utf-8"))
    router = prepared / "router.txt" if profile["router_prompt_sha256"] is not None else None
    if router is not None: _write_new(router, ROUTER_PROMPT.encode("utf-8"))
    experiment_id = comparison.get("experiment_id", "cal-" + canonical_json_sha256(comparison)[:20])
    manifest = ExperimentManifest(experiment_id=experiment_id, manifest_id=experiment_id + "-adaptive-hybrid-r1",
        created_at=datetime.now(timezone.utc), code=capture_code_provenance(sources["source_repo"]), split="calibration",
        split_manifest_sha256=selection.split_file_sha256, question_set_sha256=question_set_sha256(questions),
        question_ids=tuple(q.question_id for q in questions), scope=binding,
        scope_fingerprint_sha256=canonical_json_sha256(scope), comparison_config_sha256=canonical_json_sha256(comparison),
        **{key: profile[key] for key in _PROFILE_FIELDS})
    files = InputFiles(**sources, index_manifest=prepared / "index-manifest.json",
        retrieval_config=prepared / "retrieval-config.json", comparison_config=prepared / "comparison-config.json",
        scope=prepared / "scope.json", runtime_config=prepared / "runtime-config.json",
        generator_prompt=prepared / "generator.txt", critic_prompt=prepared / "critic.txt", router_prompt=router)
    verified = verify_inputs(manifest, files)
    for name, digest in profile["source_files"].items():
        _need(file_sha256(files.source_repo / name) == digest, "Loaded runtime source differs from the frozen checkout")
    run = CalibrationRun(manifest, verified, selection, directory, _profile_identity(profile), bitmap, _producer_sources())
    _check_generation(run, actual, profile)
    _write_new(prepared / "manifest.json", _bytes(manifest.model_dump(mode="json")))
    _write_new(prepared / "input-files.json", _bytes({name: str(path) if path is not None else None for name, path in asdict(files).items()}))
    _write_new(prepared / "producer.json", _bytes({"format": "rag-calibration-producer-v1",
        "profile_sha256": run.profile_sha256, "bitmap_sha256": run.bitmap_sha256,
        "producer_sources": dict(run.producer_sources), "canonical_directory": str(run.directory)}))
    return run


def load_calibration_run(prepared_dir):
    """Rehydrate exact file preparation in a later reviewed worker; no runtime."""
    prepared = Path(prepared_dir)
    manifest = ExperimentManifest.model_validate(_json(prepared / "manifest.json"))
    paths = _json(prepared / "input-files.json")
    files = InputFiles(**{name: Path(value) if value is not None else None for name, value in paths.items()})
    meta = _json(prepared / "producer.json")
    _need(meta.get("format") == "rag-calibration-producer-v1", "Unsupported calibration preparation")
    run = CalibrationRun(manifest, verify_inputs(manifest, files),
        CalibrationSelection.model_validate(_json(prepared / "selection.json")), Path(meta["canonical_directory"]),
        meta["profile_sha256"], meta["bitmap_sha256"], meta["producer_sources"])
    _check_run(run)
    return run


def _check_run(run):
    _need(type(run) is CalibrationRun and dict(run.producer_sources) == _producer_sources(),
          "Prepared calibration producer source changed")
    run.inputs.recheck(run.manifest)
    recheck_selection(run.selection, run.inputs.files.questions, run.inputs.files.splits)
    _need(_profile_identity(_json(run.inputs.files.index_manifest.parent / "profile.json")) == run.profile_sha256,
          "Frozen loaded runtime profile changed")
    _need(_json(run.inputs.files.index_manifest.parent / "selection.json") == run.selection.model_dump(mode="json"),
          "Frozen calibration selection changed")
    _output_location(run.directory, run.inputs.files.source_repo)


def _runtime_matches(run, actual, identity):
    _check_run(run)
    profile = actual.experiment_profile(_body(run.inputs.questions[0]), _resolved_identity(identity))
    _need(_profile_identity(profile) == run.profile_sha256, "Current runtime/source/scope differs from the prepared calibration")
    _need(_bitmap(actual, identity, run.manifest.scope) == run.bitmap_sha256, "Current effective document permissions changed")
    _check_generation(run, actual, profile)
    return profile


def calibration_plan(run, max_total_cost_usd, output_dir):
    """Metadata for root review/approval; this is not an approval receipt."""
    _check_run(run)
    cap = _amount(max_total_cost_usd)
    per_case = _amount(run.manifest.budgets.max_run_cost_usd)
    free = run.manifest.generation_model.provider == "fixture"
    _need((free and cap == per_case == 0) or (not free and run.manifest.generation_model.provider == "anthropic"
          and 0 < per_case and per_case * len(run.manifest.question_ids) <= cap), "Exact conservative calibration case/total caps required")
    generation = _json(run.inputs.files.comparison_config).get("generation")
    _need(free or isinstance(generation, dict) and _amount(generation["budget_usd"]) == cap,
          "Managed provider budget must equal the reviewed total cap")
    output = _output_location(output_dir, run.inputs.files.source_repo)
    _need(not output.resolve().is_relative_to(run.directory.resolve())
          and not run.directory.resolve().is_relative_to(output.resolve()), "Capture journal must be outside the canonical runtime directory")
    return {"format": "rag-critic-calibration-run-plan-v1", "manifest": run.manifest.model_dump(mode="json"),
        "selection": run.selection.model_dump(mode="json"), "input_hashes": dict(run.inputs.hashes),
        "input_files": {name: str(path) if path is not None else None for name, path in asdict(run.inputs.files).items()},
        "canonical_directory": str(run.directory), "output_directory": str(output),
        "profile_sha256": run.profile_sha256, "effective_bitmap_sha256": run.bitmap_sha256,
        "producer_sources": dict(run.producer_sources), "maximum_selected_cases": len(run.manifest.question_ids),
        "maximum_case_cost_usd": str(per_case), "maximum_total_cost_usd": str(cap),
        "execution_context": "fixture" if free else "live", "automatic_retry_allowed": False,
        "human_labels_created": False, "calibration_accepted": False}


def calibration_plan_sha256(plan):
    return canonical_json_sha256(plan)


def _approval(plan, path, expected_sha):
    _need(file_sha256(path) == expected_sha, "Exact separate calibration approval bytes required")
    value = _json(path)
    model = plan["manifest"]["generation_model"]
    _need(value.get("format") == "rag-critic-calibration-approval-v1" and value.get("approved") is True
          and value.get("approval_source") == "explicit_user_affirmative_reply"
          and isinstance(value.get("user_reply"), str) and value["user_reply"].strip()
          and value.get("root_execution_review_complete") is True
          and value.get("plan_sha256") == calibration_plan_sha256(plan)
          and type(value.get("maximum_selected_cases")) is int
          and value["maximum_selected_cases"] == plan["maximum_selected_cases"]
          and value.get("question_ids") == plan["manifest"]["question_ids"]
          and value.get("generation_model") == model
          and value.get("critic_prompt_sha256") == plan["manifest"]["critic_prompt_sha256"]
          and type(value.get("maximum_rounds")) is int
          and value.get("maximum_rounds") == plan["manifest"]["max_rounds"]
          and _amount(value.get("maximum_case_cost_usd")) == _amount(plan["maximum_case_cost_usd"])
          and _amount(value.get("maximum_total_cost_usd")) == _amount(plan["maximum_total_cost_usd"])
          and value.get("execution_context") == plan["execution_context"]
          and value.get("calibration_runs_allowed") is True
          and all(value.get(name) is False for name in ("automatic_retry_allowed", "final_runs_allowed", "judging_allowed"))
          and type(value.get("sdk_max_retries")) is int and value["sdk_max_retries"] == 0,
          "Exact affirmative calibration scope, model and budget approval required")
    _need(value.get("egress") == {"destination": "Anthropic API" if plan["execution_context"] == "live" else "fixture",
          "question": True, "retrieved_permitted_fictional_corpus_excerpts": True}, "Exact calibration evidence-transfer approval required")
    return value


def _serialize_records(records, case):
    result, captures, seen = [], {}, set()
    expected_ordinal = 1
    for record in records:
        _need(set(record) == {"request_evidence", "selected_evidence", "payload_sha256", "provider_call_planned",
              "trace_step_index", "critic_call_ordinal", "disposition"}
              and type(record["trace_step_index"]) is int and record["trace_step_index"] >= 0
              and type(record["critic_call_ordinal"]) is int and record["critic_call_ordinal"] == expected_ordinal,
              "Issuing runtime critic export has an invalid index/ordinal identity")
        expected_ordinal += 1
        index = record["trace_step_index"]
        _need(index not in seen and record["disposition"] in {"captured", "input_unavailable"}, "Repeated or unknown critic capture")
        seen.add(index)
        raw = {name: record[name] for name in ("request_evidence", "selected_evidence", "payload_sha256", "provider_call_planned")}
        if record["disposition"] == "captured":
            value = CriticInputCapture.model_validate({**raw, "request_evidence": [e.model_dump(mode="json") for e in raw["request_evidence"]],
                "selected_evidence": [e.model_dump(mode="json") for e in raw["selected_evidence"]]})
            captures[index] = value
            raw = value.model_dump(mode="json")
        else:
            _need(all(value is None for value in raw.values()), "Unavailable inputs must not invent an empty capture")
        result.append({**record, **raw})
    expected = [index for index, step in enumerate(case.response.trace.steps) if step.tool.split(":", 1)[0] == "critic"]
    _need([record["trace_step_index"] for record in result] == expected,
          "Private capture identities must cover the actual ordered critic trace events")
    return result, captures


def run_calibration(run, *, open_runtime, approval_file, approval_sha256, max_total_cost_usd,
                    output_dir, allow_paid=False):
    """One execution only. A caller must explicitly prepare/review another run after uncertainty."""
    plan = calibration_plan(run, max_total_cost_usd, output_dir)
    approval_file = Path(approval_file)
    approval = _approval(plan, approval_file, approval_sha256)
    live = plan["execution_context"] == "live"
    _need(type(allow_paid) is bool and (allow_paid if live else not allow_paid), "Explicit paid/fixture execution mode required")
    output = Path(plan["output_directory"])
    _need(not os.path.lexists(output) and not os.path.lexists(run.directory), "Existing outputs/intents prohibit automatic retry or resume")
    _durable_mkdir(output)
    _durable_mkdir(output / "captures")
    _write_new(output / "plan.json", _bytes(plan)); _write_new(output / "approval.json", _bytes(approval))
    state = {"format": "rag-critic-calibration-attempts-v1", "plan_sha256": calibration_plan_sha256(plan),
        "reserved_usd": "0", "attempts": [], "automatic_retry_allowed": False}
    state_sha = None
    def persist():
        nonlocal state_sha
        if state_sha is not None:
            _need(file_sha256(output / "journal.json") == state_sha, "Calibration attempt journal changed")
        temporary = output / (".state-" + uuid4().hex)
        _write_new(temporary, _bytes(state)); os.replace(temporary, output / "journal.json"); _sync(output)
        state_sha = file_sha256(output / "journal.json")
    def unchanged():
        _check_run(run); _approval(plan, approval_file, approval_sha256)
        _need(file_sha256(output / "journal.json") == state_sha, "Calibration attempt journal changed")
    summary = {"passed": False, "canonical_cases": 0, "critic_episodes": 0, "cases_without_critic_decision": 0,
        "human_labels_created": False, "calibration_accepted": False, "automatic_retry_allowed": False,
        "execution_context": plan["execution_context"], "actual_invoice_verified": False}
    stage = "runtime_admission"
    try:
        persist()
        with ExitStack() as stack:
            actual, identity = stack.enter_context(open_runtime(run))
            _runtime_matches(run, actual, identity)
            if live:
                from src.generation.generator import AnthropicProvider
                _need(type(actual.provider) is AnthropicProvider and actual.provider.config.allow_paid
                      and actual.provider.reserved_usd == 0, "A fresh guarded paid provider is required")
            missing_canonical = _missing_ancestry(run.directory)
            directory = stack.enter_context(RunDirectory.create(run.directory, run.manifest, run.inputs))
            _sync_new_ancestry(missing_canonical)
            for ordinal, question in enumerate(run.inputs.questions, 1):
                unchanged(); _runtime_matches(run, actual, identity)
                reservation = _amount(plan["maximum_case_cost_usd"])
                total = _amount(state["reserved_usd"]) + reservation
                _need(total <= _amount(plan["maximum_total_cost_usd"]), "Conservative calibration reservation exhausted")
                item = {"question_id": question.question_id, "state": "intent", "reservation_usd": str(reservation)}
                state["attempts"].append(item); state["reserved_usd"] = str(total); persist()
                stage = "query_dispatch"
                capture = actual.query_with_capture(_body(question), _resolved_identity(identity),
                    experiment=ExperimentAdmission(run.manifest, run.inputs, question.question_id))
                item.update(state="captured", run_id=capture.run_id); persist()
                stage = "canonical_export"
                case = actual.export_capture(capture, manifest=run.manifest, verified_inputs=run.inputs,
                    question_id=question.question_id, directory=directory)
                item.update(state="runtime_committed", case_sha256=runtime_case_digest(case),
                    reported_cost_usd=case.response.trace.usage.estimated_cost_usd, api_calls=case.response.trace.usage.api_calls)
                persist(); summary["canonical_cases"] += 1
                _need(live or case.response.trace.usage.api_calls == 0 and case.response.trace.usage.estimated_cost_usd in (None, 0),
                      "Fixture calibration reported external model activity")
                stage = "private_critic_export"
                raw_records = actual.export_critic_inputs(capture)
                records, captures = _serialize_records(raw_records, case)
                _need(_bitmap(actual, identity, run.manifest.scope) == run.bitmap_sha256, "Current document bitmap changed at critic export")
                receipt = AuthorizationReceipt(producer="AuthenticatedRuntime", scope=run.manifest.scope,
                    effective_bitmap_sha256=run.bitmap_sha256, capture_artifact_sha256=capture_artifact_sha256(captures),
                    checked_at=datetime.now(timezone.utc))
                artifact = {"format": "rag-issued-critic-inputs-v1", "runtime_case_sha256": runtime_case_digest(case),
                    "authorization": receipt.model_dump(mode="json"), "records": records}
                path = output / "captures" / f"case_{ordinal:06d}.inputs.json"
                _write_new(path, _bytes(artifact))
                stage = "critic_episode_import"
                episodes = import_critic_episodes(run.selection, run.manifest, case, captures,
                    authorization_receipt=receipt, critic_model=run.manifest.generation_model)
                episode_path = output / "captures" / f"case_{ordinal:06d}.episodes.json"
                _write_new(episode_path, _bytes([episode.model_dump(mode="json") for episode in episodes]))
                item.update(state="committed", critic_inputs_sha256=file_sha256(path),
                    episodes_sha256=file_sha256(episode_path), critic_episodes=len(episodes),
                    disposition="native_critic_episodes" if episodes else "no_successful_critic_decision")
                persist(); summary["critic_episodes"] += len(episodes)
                summary["cases_without_critic_decision"] += int(not episodes)
                unchanged(); _runtime_matches(run, actual, identity)
            directory.load_cases(require_complete=True)
        summary["passed"] = True
    except BaseException as error:
        summary.update(failure_type=type(error).__name__, failure_stage=stage,
            unresolved_or_partial_attempts_require_review=True)
    finally:
        summary.update(reserved_cost_usd=state["reserved_usd"], committed_attempts=sum(
            item["state"] == "committed" for item in state["attempts"]), attempt_count=len(state["attempts"]))
        known = [item.get("reported_cost_usd") for item in state["attempts"]]
        summary["known_reported_cost_usd"] = str(sum((_amount(value) for value in known if value is not None), Decimal(0)))
        summary["spend_may_be_unknown"] = any(item["state"] in {"intent", "captured"} or item.get("reported_cost_usd") is None for item in state["attempts"])
        _write_new(output / "summary.json", _bytes(summary))
    return {"directory": str(output), **summary}

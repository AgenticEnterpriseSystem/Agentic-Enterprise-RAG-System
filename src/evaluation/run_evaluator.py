"""Prepare a pinned RC2 evaluation plan or execute an explicitly approved one.

Preparation never constructs a provider. Execution uses the core's durable
case/stage admission before reading credentials or constructing an SDK client.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import Field, StrictBool
from src.contracts import Contract


ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://api.anthropic.com"
MAX_METADATA_BYTES = 4 * 1024 * 1024


class _ScoreReasons(Contract):
    correctness: str = Field(min_length=1)
    completeness: str = Field(min_length=1)
    citation_support: str = Field(min_length=1)
    abstention_limits: str = Field(min_length=1)


class ScorePayload(Contract):
    """Transport schema for the unchanged RC2 grading rubric."""

    correctness: Literal[0, 1, 2]
    completeness: Literal[0, 1, 2] | None
    citation_support: Literal[0, 1, 2] | None
    abstention_limits: Literal[0, 1, 2]
    reasons: _ScoreReasons


class _AuditClaim(Contract):
    answer_quote: str = Field(min_length=1)
    central: StrictBool
    support: Literal["direct", "entailed", "related", "none"]
    citation_id: str | None
    evidence_quote: str | None
    target_relation: Literal["same_target", "general_rule", "different_case", "ambiguous"]


class CitationAuditPayload(Contract):
    response_type: Literal["answer", "partial_answer", "refusal"]
    claims: tuple[_AuditClaim, ...] = Field(max_length=30)
    coverage_confident: StrictBool
    explanation: str = Field(min_length=1)


class EvaluatorCLIError(ValueError):
    """A fixed diagnostic which contains no question, evidence or credential."""


def _need(condition, message):
    if not condition:
        raise EvaluatorCLIError(message)


def _money(raw):
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError):
        raise argparse.ArgumentTypeError("Expected a finite positive decimal") from None
    if not value.is_finite() or value <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive decimal")
    return value


def _read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _need(key not in value, "Duplicate metadata field")
            value[key] = item
        return value

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_METADATA_BYTES,
              "Expected bounded ordinary metadata")
        data = stream.read(MAX_METADATA_BYTES + 1)
    _need(len(data) <= MAX_METADATA_BYTES, "Metadata exceeds its bound")
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          EvaluatorCLIError("Nonfinite metadata")))


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class _OwnedStageProvider:
    """Close both owned HTTP resources without changing the provider protocol."""

    def __init__(self, provider, client, http=None, response_schema=None):
        self.provider = provider
        self.client = client
        self.http = http
        self.response_schema = response_schema

    def complete(self, system, prompt):
        # The execution core retains raw text and usage before closing us.
        # Closing here could discard a successful completion if cleanup fails.
        if self.response_schema is not None:
            return self.provider.complete_structured(system, prompt, self.response_schema)
        return self.provider.complete(system, prompt)

    def close(self):
        resources = (self.client, self.http)
        self.client, self.http = None, None
        first_error = None
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error


def anthropic_stage_provider(admission):
    """Called by the core only after the exact approved stage intent is durable.

    The SDK uses one explicit endpoint and credential. Environment proxies,
    redirected requests, endpoint/profile overrides and SDK retries cannot
    silently change the reviewed evidence destination or spending scope.
    """
    settings = admission.judge
    _need(settings.provider == "anthropic", "Unsupported judge provider")
    _need(admission.output_protocol == "native_schema_v1",
          "Live judging requires the reviewed native JSON protocol")
    _need(admission.stage in ("score", "citation_audit"), "Unknown judge stage")
    response_schema = ScorePayload if admission.stage == "score" else CitationAuditPayload
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    _need(bool(key.strip()) and "\n" not in key and "\r" not in key,
          "An existing Anthropic API key is required")
    _need(not os.environ.get("ANTHROPIC_CUSTOM_HEADERS"),
          "Custom Anthropic headers are outside this execution plan")
    # Import and instantiate the SDK only inside the admitted provider factory.
    import anthropic
    import httpx
    from src.generation.generator import AnthropicConfig, AnthropicProvider

    config = AnthropicConfig(model=settings.model_id, allow_paid=True,
        budget_usd=float(admission.max_cost_usd),
        input_usd_per_million=float(settings.input_usd_per_million),
        output_usd_per_million=float(settings.output_usd_per_million),
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds)
    http = httpx.Client(trust_env=False, follow_redirects=False,
                        timeout=settings.timeout_seconds)
    try:
        client = anthropic.Anthropic(api_key=key, auth_token=None, base_url=API_BASE,
            max_retries=0, timeout=settings.timeout_seconds, http_client=http)
        # The SDK emits an Authorization header for an empty string, and None
        # in its constructor can inherit ANTHROPIC_AUTH_TOKEN. Clear the owned
        # client's token explicitly so only the reviewed API key authenticates.
        client.auth_token = None
    except BaseException:
        http.close()
        raise
    return _OwnedStageProvider(AnthropicProvider(config, client=client), client, http, response_schema)


def prepare(args):
    from src.evaluation.evaluator_runs import (
        JudgeSettings, evaluator_plan_sha256, prepare_evaluator_plan,
    )

    selection = _read(args.selection) if args.selection else None
    settings = JudgeSettings(provider="anthropic", model_id=args.model,
        input_usd_per_million=args.input_price,
        output_usd_per_million=args.output_price,
        max_output_tokens=args.max_output_tokens, max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        score_cost_cap_usd=args.score_cost_cap,
        audit_cost_cap_usd=args.audit_cost_cap,
        case_cost_cap_usd=args.case_cost_cap,
        total_cost_cap_usd=args.total_cost_cap)
    plan = prepare_evaluator_plan(args.bundle, args.run_output,
        settings=settings, selected_question_ids=selection,
        execution_context="live", output_protocol="native_schema_v1",
        stop_on_invalid_response=True, max_response_bytes=262144,
        extra_source_files={
            "cli": Path(__file__),
            "anthropic_provider": ROOT / "src/generation/generator.py",
            "provider_budget": ROOT / "src/agents/budget.py",
            "result_binding": ROOT / "src/evaluation/evaluator_bridge.py",
        })
    _write_new(args.plan, plan.model_dump(mode="json"))
    return {"plan": str(args.plan), "plan_sha256": evaluator_plan_sha256(plan),
            "paid_authorized": False, "provider_calls": 0}


def run(args):
    _need(args.allow_paid is True, "Execution requires explicit --allow-paid")
    from src.evaluation.evaluator_runs import (
        EvaluatorPlan, evaluator_plan_sha256, run_evaluator_plan,
    )

    plan = EvaluatorPlan.model_validate(_read(args.plan))
    _need(evaluator_plan_sha256(plan) == args.plan_sha256,
          "The exact reviewed plan is required")
    _need(plan.execution_context == "live", "The live CLI requires a live plan")
    _need(args.max_total_cost_usd == plan.judge.total_cost_cap_usd,
          "The execution cap must equal the reviewed plan")
    return run_evaluator_plan(plan, approval_file=args.approval,
        approval_sha256=args.approval_sha256,
        provider_factory=anthropic_stage_provider)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("prepare", help="File-only plan; no model call")
    draft.add_argument("--bundle", type=Path, action="append", required=True)
    draft.add_argument("--selection", type=Path)
    draft.add_argument("--plan", type=Path, required=True)
    draft.add_argument("--run-output", type=Path, required=True)
    draft.add_argument("--model", required=True)
    draft.add_argument("--input-price", type=_money, required=True)
    draft.add_argument("--output-price", type=_money, required=True)
    draft.add_argument("--max-output-tokens", type=int, default=4096)
    draft.add_argument("--max-tokens", type=int, default=32000)
    draft.add_argument("--timeout-seconds", type=float, default=60)
    for name in ("score", "audit", "case", "total"):
        draft.add_argument("--" + name + "-cost-cap", type=_money, required=True)
    execute = commands.add_parser("run", help="One explicitly approved evaluation")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--approval", type=Path, required=True)
    execute.add_argument("--approval-sha256", required=True)
    execute.add_argument("--allow-paid", action="store_true")
    execute.add_argument("--max-total-cost-usd", type=_money, required=True)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        value = prepare(arguments) if arguments.command == "prepare" else run(arguments)
    except (ValueError, OSError, KeyError, TypeError):
        raise SystemExit("Evaluator CLI validation failed; no automatic retry. "
                         "Keep existing plans and execution journals.") from None
    print(json.dumps(value, sort_keys=True, default=str))
    return value


if __name__ == "__main__":
    main()

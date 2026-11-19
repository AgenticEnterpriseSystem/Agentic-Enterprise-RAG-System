"""Offline boundary tests for the paid evaluator CLI; no SDK request is made."""

from decimal import Decimal
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from src.evaluation import run_evaluator as cli


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("CLI tests cannot access the network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def fake_core(monkeypatch, *, plan=None, run=None, prepare=None):
    plan = plan or SimpleNamespace(execution_context="live",
        judge=SimpleNamespace(total_cost_cap_usd=Decimal("0.30")))
    module = SimpleNamespace(
        EvaluatorPlan=SimpleNamespace(model_validate=lambda value: plan),
        evaluator_plan_sha256=lambda value: "a" * 64,
        run_evaluator_plan=run or (lambda *a, **k: pytest.fail("Unexpected run")),
        JudgeSettings=lambda **value: SimpleNamespace(**value),
        prepare_evaluator_plan=prepare or (lambda *a, **k: pytest.fail("Unexpected prepare")),
    )
    monkeypatch.setitem(sys.modules, "src.evaluation.evaluator_runs", module)
    return module


def run_args(tmp_path, **overrides):
    path = tmp_path / "plan.json"; path.write_text("{}")
    value = dict(allow_paid=True, plan=path, plan_sha256="a" * 64,
        approval=tmp_path / "SYNTHETIC-APPROVAL-NOT-WRITTEN.json",
        approval_sha256="b" * 64, max_total_cost_usd=Decimal("0.30"))
    value.update(overrides)
    return SimpleNamespace(**value)


def test_missing_paid_flag_fails_before_plan_or_core_access(tmp_path):
    args = run_args(tmp_path, allow_paid=False, plan=tmp_path / "missing.json")
    with pytest.raises(cli.EvaluatorCLIError, match="allow-paid"):
        cli.run(args)


@pytest.mark.parametrize("change,match", [
    ({"plan_sha256": "c" * 64}, "exact reviewed plan"),
    ({"max_total_cost_usd": Decimal("0.31")}, "execution cap"),
    ({"max_total_cost_usd": Decimal("0.29")}, "execution cap"),
])
def test_reviewed_hash_and_exact_cap_precede_runner(tmp_path, monkeypatch, change, match):
    fake_core(monkeypatch)
    with pytest.raises(cli.EvaluatorCLIError, match=match):
        cli.run(run_args(tmp_path, **change))


def test_fixture_plan_cannot_use_live_cli(tmp_path, monkeypatch):
    fake_core(monkeypatch, plan=SimpleNamespace(execution_context="fixture"))
    with pytest.raises(cli.EvaluatorCLIError, match="live plan"):
        cli.run(run_args(tmp_path))


def test_runner_gets_exact_approval_and_deferred_provider(tmp_path, monkeypatch):
    calls = []
    fake_core(monkeypatch, run=lambda *args, **kw: calls.append((args, kw)) or {"passed": True})
    args = run_args(tmp_path)
    assert cli.run(args) == {"passed": True}
    assert calls[0][1] == {"approval_file": args.approval,
        "approval_sha256": args.approval_sha256,
        "provider_factory": cli.anthropic_stage_provider}
    assert not args.approval.exists()


def test_prepare_writes_only_plan_and_pins_live_adapter_sources(tmp_path, monkeypatch):
    calls = []
    plan = SimpleNamespace(model_dump=lambda **kw: {"SYNTHETIC_PLAN": True})
    fake_core(monkeypatch, prepare=lambda *args, **kw: calls.append((args, kw)) or plan)
    selection = tmp_path / "selection.json"; selection.write_text('{"manifest-1":["q1"]}')
    args = SimpleNamespace(selection=selection, model="test-model", input_price=Decimal("1"),
        output_price=Decimal("5"), max_output_tokens=4096, max_tokens=32000,
        timeout_seconds=60, score_cost_cap=Decimal(".05"), audit_cost_cap=Decimal(".05"),
        case_cost_cap=Decimal(".10"), total_cost_cap=Decimal(".30"),
        bundle=[tmp_path / "bundle"], run_output=tmp_path / "NOT_EXECUTED",
        plan=tmp_path / "private" / "plan.json")
    result = cli.prepare(args)
    assert result["paid_authorized"] is False and result["provider_calls"] == 0
    assert json.loads(args.plan.read_text()) == {"SYNTHETIC_PLAN": True}
    assert not args.run_output.exists()
    assert calls[0][1]["selected_question_ids"] == {"manifest-1": ["q1"]}
    assert calls[0][1]["output_protocol"] == "native_schema_v1"
    assert calls[0][1]["stop_on_invalid_response"] is True
    pins = calls[0][1]["extra_source_files"]
    assert set(pins) == {"cli", "anthropic_provider", "provider_budget", "result_binding"}
    assert all(Path(p).is_file() for p in pins.values())


def test_new_metadata_write_cannot_overwrite_history(tmp_path):
    path = tmp_path / "existing.json"; path.write_text("original")
    with pytest.raises(FileExistsError):
        cli._write_new(path, {"replacement": True})
    assert path.read_text() == "original"


@pytest.mark.parametrize("raw", ["0", "-1", "NaN", "Infinity", "-Infinity", "true"])
def test_invalid_cli_money_rejected(raw):
    with pytest.raises(Exception, match="finite positive decimal"):
        cli._money(raw)


def test_duplicate_or_nonfinite_metadata_rejected(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(cli.EvaluatorCLIError, match="Duplicate"):
        cli._read(path)
    path.write_text('{"a":NaN}')
    with pytest.raises(cli.EvaluatorCLIError, match="Nonfinite"):
        cli._read(path)


def stage():
    return SimpleNamespace(max_cost_usd=Decimal(".05"), stage="score", output_protocol="native_schema_v1",
        judge=SimpleNamespace(provider="anthropic", model_id="test-model",
            timeout_seconds=30, input_usd_per_million=Decimal("1"),
            output_usd_per_million=Decimal("5"), max_output_tokens=4096))


def test_missing_key_precedes_sdk_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(cli.EvaluatorCLIError, match="API key"):
        cli.anthropic_stage_provider(stage())


def test_custom_headers_fail_closed_without_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SYNTHETIC_NOT_A_REAL_KEY")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "UNREVIEWED_HEADER")
    with pytest.raises(cli.EvaluatorCLIError, match="Custom Anthropic headers"):
        cli.anthropic_stage_provider(stage())


def test_live_factory_uses_fixed_endpoint_no_proxies_redirects_or_retries(monkeypatch):
    import anthropic
    import httpx
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SYNTHETIC_NOT_A_REAL_KEY")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unexpected.invalid")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    calls = {}
    http = SimpleNamespace(close=lambda: calls.setdefault("http_closed", True))
    client = SimpleNamespace(close=lambda: calls.setdefault("client_closed", True))
    monkeypatch.setattr(httpx, "Client", lambda **kw: calls.update(http=kw) or http)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: calls.update(sdk=kw) or client)
    provider = cli.anthropic_stage_provider(stage())
    assert calls["http"]["trust_env"] is False
    assert calls["http"]["follow_redirects"] is False
    assert calls["sdk"]["base_url"] == "https://api.anthropic.com"
    assert calls["sdk"]["auth_token"] is None
    assert client.auth_token is None
    assert calls["sdk"]["max_retries"] == 0
    assert provider.provider.client is client
    assert provider.provider.config.budget_usd == .05
    assert provider.response_schema is cli.ScorePayload
    provider.close(); provider.close()
    assert calls["client_closed"] is True
    assert calls["http_closed"] is True


def test_real_sdk_uses_only_explicit_api_key_without_blank_or_ambient_bearer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SYNTHETIC_OFFLINE_API_KEY")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "SYNTHETIC_AMBIENT_TOKEN")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    owned = cli.anthropic_stage_provider(stage())
    try:
        assert owned.client.auth_headers == {"X-Api-Key": "SYNTHETIC_OFFLINE_API_KEY"}
        assert owned.client.auth_token is None
    finally:
        owned.close()


def test_stage_provider_leaves_cleanup_to_core_after_returning_completion():
    closed = []
    client = SimpleNamespace(close=lambda: closed.append(True))
    wrapped = cli._OwnedStageProvider(SimpleNamespace(complete=lambda *a: "result"), client)
    assert wrapped.complete("system", "prompt") == "result"
    assert closed == []
    wrapped.close()
    def broken(*args):
        raise RuntimeError("fixture")
    wrapped = cli._OwnedStageProvider(SimpleNamespace(complete=broken), client)
    with pytest.raises(RuntimeError):
        wrapped.complete("system", "prompt")
    assert closed == [True]
    wrapped.close()
    assert closed == [True, True]


def test_cleanup_closes_transport_even_if_sdk_close_raises():
    closed = []
    def broken():
        closed.append("sdk")
        raise ValueError("scripted SDK cleanup failure")
    wrapped = cli._OwnedStageProvider(None, SimpleNamespace(close=broken),
        SimpleNamespace(close=lambda: closed.append("http")))
    with pytest.raises(ValueError, match="SDK cleanup"):
        wrapped.close()
    wrapped.close()
    assert closed == ["sdk", "http"]


def test_native_stage_uses_structured_completion_without_changing_prompt():
    seen = []
    provider = SimpleNamespace(complete=lambda *a: pytest.fail("Unstructured live call"),
        complete_structured=lambda *a: seen.append(a) or "structured-result")
    wrapped = cli._OwnedStageProvider(provider, None, response_schema=cli.ScorePayload)
    assert wrapped.complete("UNCHANGED_SYSTEM", "UNCHANGED_RC2_PROMPT") == "structured-result"
    assert seen == [("UNCHANGED_SYSTEM", "UNCHANGED_RC2_PROMPT", cli.ScorePayload)]


def test_native_transport_schemas_match_pinned_parser_contract():
    from src.evaluation.verified_judge import validate_scores
    from src.evaluation.evaluator_bridge import _validate_audit_schema
    value = {"correctness": 0, "completeness": 0, "citation_support": None,
        "abstention_limits": 2, "reasons": {k: "fixture" for k in
        ["correctness", "completeness", "citation_support", "abstention_limits"]}}
    parsed = cli.ScorePayload.model_validate(value).model_dump(mode="json")
    assert validate_scores(parsed) == value
    audit = cli.CitationAuditPayload(response_type="refusal", claims=(),
        coverage_confident=True, explanation="fixture").model_dump(mode="json")
    _validate_audit_schema(audit)


def test_main_diagnostic_does_not_echo_private_metadata(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bad.json"; path.write_text('{"PRIVATE_SENTINEL":NaN}')
    fake_core(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--plan", str(path), "--plan-sha256", "a"*64,
                  "--approval", str(tmp_path/"never-created.json"),
                  "--approval-sha256", "b"*64, "--allow-paid", "--max-total-cost-usd", ".30"])
    assert "PRIVATE_SENTINEL" not in str(error.value)
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out

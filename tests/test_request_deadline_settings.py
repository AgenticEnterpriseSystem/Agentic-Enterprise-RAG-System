"""The HTTP envelope may outlast policy work without expanding agent budgets."""
import pytest

from src.api.runtime import WebSettings


def test_omitted_http_deadline_preserves_existing_callers():
    settings = WebSettings(max_seconds=.08)
    assert settings.request_timeout_seconds is None
    assert settings.effective_request_timeout_seconds == .08


def test_explicit_http_envelope_is_separate_from_agent_and_spending_limits():
    settings = WebSettings(request_timeout_seconds=180, budget_usd=.5)
    assert settings.effective_request_timeout_seconds == 180
    assert settings.max_seconds == 120 and settings.max_tokens == 16000
    assert settings.question_budget_usd == .1 and settings.budget_usd == .5
    assert settings.critic_max_output_tokens == 2048 and settings.model_max_output_tokens == 1024


@pytest.mark.parametrize('value', [True, False, 0, -1, 119.99, 360.01, float('nan'), float('inf')])
def test_explicit_deadline_rejects_invalid_short_or_unbounded_values(value):
    with pytest.raises(ValueError):
        WebSettings(max_seconds=120, request_timeout_seconds=value)


def test_explicit_deadline_must_cover_actual_custom_agent_budget():
    with pytest.raises(ValueError, match='agent time allowance'):
        WebSettings(max_seconds=240, request_timeout_seconds=180)
    assert WebSettings(max_seconds=240, request_timeout_seconds=240).effective_request_timeout_seconds == 240
    assert WebSettings(max_seconds=300, request_timeout_seconds=360).effective_request_timeout_seconds == 360


def test_environment_explicit_or_empty_deadline_uses_same_validation(monkeypatch):
    monkeypatch.setenv('RAG_MAX_SECONDS', '120')
    monkeypatch.setenv('RAG_REQUEST_TIMEOUT_SECONDS', '180')
    assert WebSettings.from_env().effective_request_timeout_seconds == 180
    monkeypatch.setenv('RAG_REQUEST_TIMEOUT_SECONDS', '')
    legacy = WebSettings.from_env()
    assert legacy.request_timeout_seconds is None and legacy.effective_request_timeout_seconds == 120
    monkeypatch.setenv('RAG_REQUEST_TIMEOUT_SECONDS', '90')
    with pytest.raises(ValueError):
        WebSettings.from_env()


def test_settings_copy_revalidates_http_and_agent_deadlines_together():
    value = WebSettings(request_timeout_seconds=180).model_dump()
    value['max_seconds'] = 240
    with pytest.raises(ValueError):
        WebSettings.model_validate(value)

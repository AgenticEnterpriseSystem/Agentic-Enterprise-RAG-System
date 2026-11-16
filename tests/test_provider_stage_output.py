"""Stage ceilings change output capacity, never run/cost/authority budgets."""
from types import SimpleNamespace
import pytest

from src.agents.budget import (CallBudget, RunBudgetExceeded, call_budget,
                               current_output_token_limit, output_token_limit)
from src.generation.generator import AnthropicConfig, AnthropicProvider
from src.observability import Recorder, observation_context, span


class Messages:
    def __init__(self, *, input_tokens=100, stop_reason='end_turn'):
        self.input_tokens, self.stop_reason = input_tokens, stop_reason
        self.options = []

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=self.input_tokens)

    def create(self, **kwargs):
        self.options.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='{}')],
            usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=10),
            stop_reason=self.stop_reason)


def provider(messages):
    return AnthropicProvider(AnthropicConfig(model='synthetic', allow_paid=True,
        budget_usd=.5, input_usd_per_million=1, output_usd_per_million=5),
        client=SimpleNamespace(messages=messages))


def test_stage_override_restores_answer_default_and_nested_context():
    messages = Messages(); adapter = provider(messages)
    assert current_output_token_limit() is None
    with output_token_limit(2048):
        adapter.complete('system', 'critic')
        with output_token_limit(512):
            adapter.complete('system', 'router')
        assert current_output_token_limit() == 2048
    adapter.complete('system', 'answer')
    assert [item['max_tokens'] for item in messages.options] == [2048, 512, 1024]
    assert current_output_token_limit() is None


@pytest.mark.parametrize('value', [0, -1, 8193, True, 2048.0, '2048'])
def test_invalid_stage_cap_fails_before_dispatch(value):
    messages = Messages(); adapter = provider(messages)
    with pytest.raises(ValueError), output_token_limit(value):
        adapter.complete('system', 'question')
    assert messages.options == []


def test_large_stage_cap_is_clamped_to_real_remaining_run_tokens():
    messages = Messages(input_tokens=300); adapter = provider(messages)
    allowance = CallBudget(remaining_tokens=1800, remaining_seconds=10, remaining_cost_usd=.1)
    with call_budget(allowance), output_token_limit(2048):
        adapter.complete('system', 'critic')
    assert messages.options[0]['max_tokens'] == 1500
    assert allowance.remaining_tokens == 1490


def test_stage_override_cannot_exceed_run_cost_budget():
    messages = Messages(); adapter = provider(messages)
    allowance = CallBudget(remaining_tokens=16000, remaining_seconds=10, remaining_cost_usd=.001)
    with pytest.raises(RunBudgetExceeded), call_budget(allowance), output_token_limit(2048):
        adapter.complete('system', 'critic')
    assert not messages.options and adapter.reserved_usd == 0


@pytest.mark.parametrize('reason,expected', [('end_turn', 'end_turn'), ('max_tokens', 'max_tokens'),
    ('refusal', 'refusal'), ('PRIVATE output/query/token', 'other')])
def test_provider_retains_only_categorical_stop_reasons(reason, expected):
    result = provider(Messages(stop_reason=reason)).complete('system', 'prompt')
    assert result.stop_reason == expected
    assert result.truncated == (reason != 'end_turn')


def test_span_retains_safe_failure_reason_but_never_exception_prose():
    class Rejected(Exception):
        code = 'truncated_output'
        provider_stop_reason = 'max_tokens'
    recorder = Recorder(log=False); observation = recorder.start_request()
    with pytest.raises(Rejected), observation_context(observation), span('critic'):
        raise Rejected('PRIVATE query/document/credentials')
    event = recorder.get(observation.request_id)['stages'][-1]
    assert event['error_code'] == 'truncated_output'
    assert event['provider_stop_reason'] == 'max_tokens'
    assert 'PRIVATE' not in str(event)


def test_observer_exception_properties_cannot_replace_the_original_error():
    class Original(Exception):
        @property
        def code(self):
            raise RuntimeError('Broken diagnostic property')
        @property
        def provider_stop_reason(self):
            raise RuntimeError('Broken diagnostic property')
    error = Original('private original message')
    recorder = Recorder(log=False); observation = recorder.start_request()
    with pytest.raises(Original) as caught, observation_context(observation), span('critic'):
        raise error
    assert caught.value is error
    event = recorder.get(observation.request_id)['stages'][-1]
    assert event['error_type'] == 'Original'
    assert 'private original' not in str(event)


def test_failed_warning_handler_does_not_lose_critic_error_usage(monkeypatch):
    import src.agents.search_critic as critic
    from src.contracts import Usage
    monkeypatch.setattr(critic.logging.getLogger(critic.__name__), 'warning',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('broken log sink')))
    usage = Usage(api_calls=2, output_tokens=1024, estimated_cost_usd=.01)
    error = critic._decision_error(usage, 'truncated_output', 'max_tokens')
    assert isinstance(error, critic.AgentError) and error.usage == usage
    assert error.code == 'truncated_output' and error.provider_stop_reason == 'max_tokens'


def test_stage_override_uses_actual_larger_reservation_at_process_cap():
    from src.generation.generator import ProviderError
    messages = Messages()
    adapter = AnthropicProvider(AnthropicConfig(model='synthetic', allow_paid=True,
        budget_usd=.008, input_usd_per_million=1, output_usd_per_million=5),
        client=SimpleNamespace(messages=messages))
    with pytest.raises(ProviderError), output_token_limit(2048):
        adapter.complete('system', 'critic')
    assert messages.options == [] and adapter.reserved_usd == 0
    adapter.complete('system', 'answer')
    assert messages.options[0]['max_tokens'] == 1024


@pytest.mark.parametrize('reason', ['refusal', 'tool_use', 'pause_turn', 'stop_sequence'])
def test_nonfinal_provider_stop_is_not_mislabeled_as_token_cutoff(reason):
    from src.agents.search_critic import AgentError, structured_completion
    from src.contracts import EvidenceAssessment, Usage
    from src.generation.generator import Completion
    class Stopped:
        def complete(self, *_args):
            return Completion('{}', Usage(api_calls=2, estimated_cost_usd=.01),
                              truncated=True, stop_reason=reason)
    with pytest.raises(AgentError) as raised:
        structured_completion(Stopped(), 'system', {}, EvidenceAssessment)
    assert raised.value.code == 'provider_error'
    assert raised.value.provider_stop_reason == reason


def test_database_failure_metadata_is_allowlisted_without_sql_or_parameters():
    from src.observability import emit
    recorder = Recorder(log=False); observation = recorder.start_request()
    with observation_context(observation):
        emit('policy.grants', 'error', database_sqlstate='25P03', database_phase='materialize',
             sql='PRIVATE SQL', parameters='PRIVATE password')
        emit('policy.grants', 'error', database_sqlstate='PRIVATE', database_phase='PRIVATE')
    rows = recorder.get(observation.request_id)['stages']
    assert rows[-2]['database_sqlstate'] == '25P03' and rows[-2]['database_phase'] == 'materialize'
    assert 'database_sqlstate' not in rows[-1] and 'database_phase' not in rows[-1]
    assert 'PRIVATE' not in str(rows)

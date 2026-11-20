"""Offline critic output-limit, schema and safe-diagnostic regressions.

The budget-aware provider double models a1536-output-token native checklist;
its counts are synthetic test usage, not a claim about any actual tokenizer.
"""
import json
import socket
from types import SimpleNamespace

import pytest

from src.access.contracts import AccessDenied
from src.agents.budget import current_output_token_limit, output_token_limit
from src.agents.search_critic import AgentError, EvidenceCritic
from src.contracts import (Chunk, CriticDecision, Evidence, GenerationRequest,
                           Usage, UserContext)
from src.generation.generator import Completion, ProviderError


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs): raise AssertionError('No HTTP, model or Docker execution in critic tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def request():
    text = 'Authorized passage supplies the requested policy facts.'
    cid = f'a:0:{len(text)}'
    evidence = Evidence(citation_id=cid, chunk=Chunk(chunk_id=cid, doc_id='a', source_type='confluence',
        title='Policy', text=text, start_char=0, end_char=len(text)), score=1, retrieval_method='hybrid')
    return GenerationRequest(question='What are the requested policy facts?',
        user=UserContext(user_id='UNIT', allowed_doc_ids=frozenset({'a'})), evidence=(evidence,))


def assessment(supplied, count=1):
    cid = supplied.evidence[0].citation_id
    return {'facts': [{'fact': f'Requested fact{i+1}',
        'claims': [{'citation_id': cid, 'statement': f'The requested value{i+1} is stated.',
                    'scope_reason': 'Same requested policy and purpose.', 'answers_requested_fact': True}],
        'status': 'supported', 'citation_ids': [cid]} for i in range(count)],
        'suggested_queries': [], 'suggested_sources': [], 'read_more_doc_ids': [], 'clarification_question': None}


class NativeProvider:
    supports_structured_output = True
    def __init__(self, output, *, usage=None, truncated=False, stop_reason='end_turn', failure=None):
        self.output, self.usage = output, usage or Usage(api_calls=2, input_tokens=40, output_tokens=20, estimated_cost_usd=.00014)
        self.truncated, self.stop_reason, self.failure, self.calls = truncated, stop_reason, failure, []
    def complete_structured(self, system, prompt, schema):
        self.calls.append({'system': system, 'payload': json.loads(prompt), 'schema': schema,
                           'limit': current_output_token_limit()})
        if self.failure is not None: raise self.failure
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return SimpleNamespace(text=text, usage=self.usage, truncated=self.truncated, stop_reason=self.stop_reason)


class BudgetAwareChecklist(NativeProvider):
    """One deterministic1536-token response; truncates under the old1024 limit."""
    def complete_structured(self, system, prompt, schema):
        limit = current_output_token_limit() or 1024
        self.truncated = limit < 1536
        self.stop_reason = 'max_tokens' if self.truncated else 'end_turn'
        self.usage = Usage(api_calls=2, input_tokens=100, output_tokens=min(limit, 1536),
                           estimated_cost_usd=(100 + min(limit, 1536) * 5) / 1_000_000)
        original = self.output
        if self.truncated: self.output = json.dumps(original)[:100]
        try: return super().complete_structured(system, prompt, schema)
        finally: self.output = original


def test_native_checklist_over_1024_output_tokens_completes_at_2048_without_schema_relaxation():
    supplied = request(); provider = BudgetAwareChecklist(assessment(supplied, count=10))
    decision, usage = EvidenceCritic(provider).assess(supplied)
    assert decision.sufficient and len(decision.facts) == 10 and usage.output_tokens == 1536
    assert len(provider.calls) == 1 and provider.calls[0]['limit'] == 2048
    assert provider.calls[0]['schema']['properties']['facts']['maxItems'] == 10
    assert provider.calls[0]['schema']['$defs']['FactAssessment']['properties']['claims']['maxItems'] == 2
    assert all(fact.status == 'supported' and fact.claims and fact.citation_ids == (supplied.evidence[0].citation_id,)
               for fact in decision.facts)
    assert provider.calls[0]['payload']['evidence'][0]['citation_id'] == supplied.evidence[0].citation_id
    assert current_output_token_limit() is None


def test_old_1024_limit_truncates_same_checklist_fail_closed_once_and_preserves_usage():
    supplied = request(); provider = BudgetAwareChecklist(assessment(supplied, count=10))
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider, max_output_tokens=1024).assess(supplied)
    assert raised.value.code == 'truncated_output' and raised.value.provider_stop_reason == 'max_tokens'
    assert raised.value.usage.output_tokens == 1024 and raised.value.usage.api_calls == 2
    assert len(provider.calls) == 1 and current_output_token_limit() is None


@pytest.mark.parametrize('limit', [True, False, 0, -1, 1.5, '2048', 8193, None])
def test_critic_rejects_nonpositive_nonstrict_or_excessive_output_limit(limit):
    with pytest.raises(ValueError, match='output token limit'): EvidenceCritic(None, max_output_tokens=limit)


@pytest.mark.parametrize('truncated,reason', [(True, 'max_tokens'), (False, 'max_tokens'),
    (True, 'end_turn'), (False, 'model_context_window_exceeded')])
def test_explicit_truncation_or_provider_limit_is_never_approved_even_with_valid_json(truncated, reason):
    supplied = request(); provider = NativeProvider(assessment(supplied), truncated=truncated, stop_reason=reason)
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(supplied)
    assert raised.value.code == 'truncated_output' and raised.value.provider_stop_reason == reason
    assert raised.value.usage == provider.usage and len(provider.calls) == 1


@pytest.mark.parametrize('output,expected', [
    ('{"facts": [', 'invalid_json'), ('not JSON MODEL_PRIVATE_OUTPUT', 'invalid_json'),
    ({'facts': [], 'MODEL_PRIVATE_FIELD': 'MODEL_PRIVATE_OUTPUT'}, 'invalid_schema')])
def test_safe_json_schema_failure_categories_no_raw_model_text_or_paths_in_logs(output, expected, caplog):
    provider = NativeProvider(output)
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(request())
    assert raised.value.code == expected and raised.value.usage == provider.usage
    assert raised.value.provider_stop_reason == 'end_turn' and len(provider.calls) == 1
    assert 'MODEL_PRIVATE' not in caplog.text and 'MODEL_PRIVATE' not in str(raised.value)
    assert raised.value.__suppress_context__ is True
    assert current_output_token_limit() is None


def test_provider_failure_has_safe_category_usage_and_no_retry():
    usage = Usage(api_calls=1, input_tokens=27, estimated_cost_usd=None)
    provider = NativeProvider({}, failure=ProviderError(retryable=True, usage=usage))
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(request())
    assert raised.value.code == 'provider_error' and raised.value.usage == usage
    assert raised.value.provider_stop_reason is None and len(provider.calls) == 1
    assert current_output_token_limit() is None


def test_invalid_provider_stop_metadata_is_not_logged_or_exposed(caplog):
    provider = NativeProvider('not json', stop_reason='MODEL_PRIVATE_STOP_REASON')
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(request())
    assert raised.value.provider_stop_reason is None
    assert 'MODEL_PRIVATE' not in caplog.text


@pytest.mark.parametrize('field', ['fact_citation', 'claim_citation', 'read_more'])
def test_references_outside_actual_selected_context_fail_closed_with_category(field):
    supplied = request(); result = assessment(supplied)
    if field == 'fact_citation': result['facts'][0]['citation_ids'] = ['b:0:5']
    if field == 'claim_citation': result['facts'][0]['claims'][0]['citation_id'] = 'b:0:5'
    if field == 'read_more': result['read_more_doc_ids'] = ['b']
    provider = NativeProvider(result)
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(supplied)
    assert raised.value.code == 'invalid_citation' and raised.value.usage == provider.usage
    assert len(provider.calls) == 1


def test_native_supported_fact_still_requires_claims():
    supplied = request(); result = assessment(supplied); result['facts'][0]['claims'] = []
    provider = NativeProvider(result)
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(supplied)
    assert raised.value.code == 'missing_claims' and len(provider.calls) == 1


def test_unrelated_claim_still_cannot_make_a_fact_sufficient():
    supplied = request(); result = assessment(supplied)
    result['facts'][0]['claims'][0]['answers_requested_fact'] = False
    decision, _ = EvidenceCritic(NativeProvider(result)).assess(supplied)
    assert not decision.sufficient and decision.facts[0].status == 'missing'
    assert decision.missing_facts == ('Requested fact1',)


def test_expanded_output_budget_does_not_accept_an_eleventh_fact_or_third_claim():
    supplied = request()
    too_many = assessment(supplied, count=11)
    too_many_claims = assessment(supplied); too_many_claims['facts'][0]['claims'] *= 3
    for result in (too_many, too_many_claims):
        provider = NativeProvider(result)
        with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(supplied)
        assert raised.value.code == 'invalid_schema' and len(provider.calls) == 1


def test_empty_evidence_still_avoids_provider():
    supplied = request().model_copy(update={'evidence': ()}); provider = NativeProvider({})
    decision, usage = EvidenceCritic(provider).assess(supplied)
    assert not decision.sufficient and usage.api_calls == 0 and not provider.calls
    assert current_output_token_limit() is None


def test_context_limit_restored_on_revocation_without_swallowing_access_error():
    error = AccessDenied('Synthetic current authority changed')
    provider = NativeProvider({}, failure=error)
    with output_token_limit(777):
        with pytest.raises(AccessDenied) as raised: EvidenceCritic(provider).assess(request())
        assert raised.value is error and current_output_token_limit() == 777
    assert current_output_token_limit() is None and len(provider.calls) == 1


def test_legacy_complete_only_provider_protocol_remains_supported():
    supplied = request()
    legacy = CriticDecision(sufficient=True, confidence=1, reason='Recorded policy fact answers the question.')
    class Provider:
        def __init__(self): self.calls = 0
        def complete(self, system, prompt):
            self.calls += 1
            assert current_output_token_limit() == 2048
            return Completion(legacy.model_dump_json(), Usage(estimated_cost_usd=0))
    provider = Provider(); decision, _ = EvidenceCritic(provider).assess(supplied)
    assert decision == legacy and provider.calls == 1 and current_output_token_limit() is None


def test_question_length_has_explicit_nonprovider_category():
    supplied = request().model_copy(update={'question': 'q' * 8001}); provider = NativeProvider({})
    with pytest.raises(AgentError) as raised: EvidenceCritic(provider).assess(supplied)
    assert raised.value.code == 'invalid_input' and not provider.calls


@pytest.mark.parametrize('claims,expected_status,sufficient', [
    # Live Haiku replies quoted the evidence but left the redundant citation list
    # empty; the decision used to be rejected as invalid_schema and end the run.
    ([True], 'supported', True),
    # Claims that do not answer the fact are not support; same rule the critic
    # applies after validation, so this stops searching only when evidence exists.
    ([False], 'missing', False),
])
def test_quoted_fact_without_citation_ids_is_judged_from_its_claims(claims, expected_status, sufficient):
    supplied = request(); output = assessment(supplied)
    fact = output['facts'][0]
    fact['citation_ids'] = []
    fact['claims'] = [dict(fact['claims'][0], answers_requested_fact=answers) for answers in claims]
    decision, _ = EvidenceCritic(NativeProvider(output)).assess(supplied)
    assert decision.facts[0].status == expected_status and decision.sufficient is sufficient
    assert decision.facts[0].citation_ids == ((supplied.evidence[0].citation_id,) if sufficient else ())


def test_supported_fact_with_neither_claims_nor_citations_is_still_rejected():
    supplied = request(); output = assessment(supplied)
    output['facts'][0].update(claims=[], citation_ids=[])
    with pytest.raises(AgentError) as error:
        EvidenceCritic(NativeProvider(output)).assess(supplied)
    assert error.value.code == 'invalid_schema'

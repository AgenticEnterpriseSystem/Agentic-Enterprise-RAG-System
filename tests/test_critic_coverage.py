"""Offline tests for the CAL-01 fixed-checklist critic and coverage harness.

Scripted providers exercise the interface and scoring; they do not measure
real model accuracy.
"""
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.search_critic import (
    AgentError, EvidenceCritic, REQUESTED_FACTS_ADDENDUM, SYSTEM_PROMPT,
)
from src.contracts import CriticDecision, FactAssessment, Usage
from src.evaluation import critic_coverage
from src.evaluation.critic_coverage import (
    CoverageCase, aggregate, classify, load_cases, main, run,
)

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs): raise AssertionError('No network or model calls in coverage tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


class ScriptedNative:
    """Native structured provider whose answer is computed from the payload."""
    supports_structured_output = True

    def __init__(self, script, usage=None):
        self.script, self.calls = script, []
        self.usage = usage or Usage(api_calls=1, input_tokens=10, output_tokens=5, estimated_cost_usd=0.001)

    def complete_structured(self, system, prompt, schema):
        payload = json.loads(prompt)
        self.calls.append({'system': system, 'payload': payload, 'schema': schema})
        result = self.script(payload)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=json.dumps(result), usage=self.usage, truncated=False, stop_reason='end_turn')


def assessment(payload, statuses, clarification=None):
    cid = payload['evidence'][0]['citation_id']
    def fact(label, status):
        claims = [] if status == 'missing' else [{'citation_id': cid, 'statement': f'{label} stated.',
                  'scope_reason': 'Same entity.', 'answers_requested_fact': True}]
        return {'fact': label, 'claims': claims, 'status': status,
                'citation_ids': [] if status == 'missing' else [cid]}
    return {'facts': [fact(label, status) for label, status in statuses.items()],
            'suggested_queries': [], 'suggested_sources': [], 'read_more_doc_ids': [],
            'clarification_question': clarification}


def case(*facts, case_id='c1'):
    return CoverageCase(id=case_id, question='Retention and payer?', evidence='Retention is 60 days.',
                        requested_facts=tuple({'label': label, 'answered': answered, 'match_terms': terms}
                                              for label, answered, terms in facts))


RETENTION = ('Retention', True, ('retention',))
PAYER_MISSING = ('Payer', False, ('pay',))
PAYER_PRESENT = ('Payer', True, ('pay',))


def test_fixture_extends_development_cases_without_changing_them():
    original = {c['id']: c for c in json.loads((FIXTURES / 'critic_development_cases.json').read_text())['cases']}
    cases = load_cases(FIXTURES / 'critic_coverage_cases.json')
    assert len(cases) == 20
    assert sum(c.expected_sufficient for c in cases) == 11
    for c in cases:
        if c.id in original:
            assert (c.question, c.evidence) == (original[c.id]['question'], original[c.id]['evidence'])
            assert c.expected_sufficient == original[c.id]['expected_sufficient']
    assert set(original) <= {c.id for c in cases}


def test_without_checklist_the_deployed_request_is_unchanged():
    provider = ScriptedNative(lambda p: assessment(p, {'Retention': 'supported'}))
    EvidenceCritic(provider).assess(case(RETENTION).request())
    call = provider.calls[0]
    assert call['system'] == SYSTEM_PROMPT
    assert 'requested_facts' not in call['payload']
    assert 'enum' not in call['schema']['$defs']['FactAssessment']['properties']['fact']


def test_checklist_is_sent_and_constrains_the_schema():
    provider = ScriptedNative(lambda p: assessment(p, {'Retention': 'supported', 'Payer': 'supported'}))
    EvidenceCritic(provider).assess(case(RETENTION, PAYER_PRESENT).request(), requested_facts=('Retention', 'Payer'))
    call = provider.calls[0]
    assert call['system'] == SYSTEM_PROMPT + REQUESTED_FACTS_ADDENDUM
    assert call['payload']['requested_facts'] == ['Retention', 'Payer']
    facts_schema = call['schema']['properties']['facts']
    assert (facts_schema['minItems'], facts_schema['maxItems']) == (2, 2)
    assert call['schema']['$defs']['FactAssessment']['properties']['fact']['enum'] == ['Retention', 'Payer']


@pytest.mark.parametrize('requested,returned,sufficient,missing', [
    # Baseline: the critic's own list decides, so an omitted fact approves.
    (None, {'Retention': 'supported'}, True, ()),
    (('Retention', 'Payer'), {'Retention': 'supported', 'Payer': 'supported'}, True, ()),
    (('Retention', 'Payer'), {'Retention': 'supported'}, False, ('Payer',)),
    (('Retention', 'Payer'), {'Retention': 'supported', 'Payer': 'missing'}, False, ('Payer',)),
    (('Retention', 'Payer'), {'Retention': 'supported', 'payer': 'supported'}, False, ('Payer',)),
    (('Retention',), {'Retention': 'supported', 'Approval date': 'missing'}, True, ()),
])
def test_checklist_decides_sufficiency(requested, returned, sufficient, missing):
    provider = ScriptedNative(lambda p: assessment(p, returned))
    decision, _ = EvidenceCritic(provider).assess(case(RETENTION).request(), requested_facts=requested)
    assert decision.sufficient is sufficient
    assert decision.missing_facts == missing
    if not sufficient:
        assert decision.reason.startswith('Unresolved requested facts')


@pytest.mark.parametrize('requested', [(), ('A', 'A'), ('A', ' '), tuple(f'F{i}' for i in range(11))])
def test_invalid_checklist_is_rejected_before_any_call(requested):
    provider = ScriptedNative(lambda p: pytest.fail('model called'))
    with pytest.raises(AgentError) as error:
        EvidenceCritic(provider).assess(case(RETENTION).request(), requested_facts=requested)
    assert error.value.code == 'invalid_input' and not provider.calls


def decision(facts, clarification=None):
    gaps = tuple(label for label, status in facts.items() if status != 'supported')
    return CriticDecision(sufficient=not gaps and not clarification, confidence=0, reason='scripted',
                          missing_facts=gaps, clarification_question=clarification,
                          facts=tuple(FactAssessment(fact=label, status=status, citation_ids=('x',) if status
                                      in ('supported', 'conflicting') else ()) for label, status in facts.items()))


@pytest.mark.parametrize('facts,requested,condition,action,causes', [
    # Premature approvals (a requested fact is absent from the evidence).
    ({'ProofPoint retention': 'supported'}, (RETENTION, PAYER_MISSING), 'own', 'approve', ('planning_omission',)),
    ({'Retention': 'supported', 'Billing payer': 'supported'}, (RETENTION, PAYER_MISSING), 'own', 'approve',
     ('critic_judgment',)),
    ({'Retention': 'supported', 'Payer': 'supported'}, (RETENTION, PAYER_MISSING), 'given', 'approve',
     ('critic_judgment',)),
    # Unnecessary continuations (every requested fact is present).
    ({'Retention': 'supported', 'Payer': 'missing'}, (RETENTION, PAYER_PRESENT), 'own', 'continue',
     ('critic_judgment',)),
    ({'Retention': 'supported', 'Approval date': 'missing'}, (RETENTION,), 'own', 'continue', ('over_asking',)),
    ({'Retention': 'supported'}, (RETENTION,), 'own', 'clarify', ('critic_judgment',)),
    # Correct decisions carry no cause.
    ({'Retention': 'supported'}, (RETENTION,), 'own', 'approve', ()),
    ({'Retention': 'supported', 'Payer': 'missing'}, (RETENTION, PAYER_MISSING), 'given', 'continue', ()),
])
def test_classify_attributes_both_error_directions(facts, requested, condition, action, causes):
    clarification = 'Which service?' if action == 'clarify' else None
    outcome = classify(case(*requested), condition, decision(facts, clarification))
    assert (outcome.action, outcome.causes) == (action, causes)


def test_given_checklist_omission_is_noncompliance():
    gap = CriticDecision(sufficient=False, confidence=0, reason='scripted', missing_facts=('Payer',),
                         facts=(FactAssessment(fact='Retention', status='supported', citation_ids=('x',)),))
    outcome = classify(case(RETENTION, PAYER_PRESENT), 'given', gap)
    assert outcome.causes == ('checklist_noncompliance',)


def test_run_scores_both_conditions_and_saves_only_counts():
    # The scripted critic forgets the payer when choosing its own checklist.
    def script(payload):
        if 'requested_facts' in payload:
            return assessment(payload, {label: 'supported' if label == 'Retention' else 'missing'
                                        for label in payload['requested_facts']})
        return assessment(payload, {'Retention': 'supported'})
    cases = (case(RETENTION, PAYER_MISSING, case_id='gap'), case(RETENTION, case_id='full'))
    outcomes, stopped = run(EvidenceCritic(ScriptedNative(script)), cases)
    assert not stopped and len(outcomes) == 4
    own, given = aggregate(outcomes, 'own'), aggregate(outcomes, 'given')
    assert (own['premature_approvals'], own['missing_fact_cases']) == (1, 1)
    assert own['premature_approval_causes'] == {'planning_omission': 1}
    assert (given['premature_approvals'], given['unnecessary_continuations']) == (0, 0)
    assert given['cost_usd'] == 0.002
    dumped = json.dumps(critic_coverage.report(outcomes, stopped, cases, ('own', 'given'), {}))
    assert 'Retention is 60 days' not in dumped and 'Retention and payer?' not in dumped


def test_run_stops_at_first_provider_failure_and_keeps_errors_out_of_rates():
    from src.generation.generator import ProviderError
    provider = ScriptedNative(lambda p: ProviderError(usage=Usage(api_calls=1, estimated_cost_usd=0)))
    cases = (case(RETENTION, case_id='a'), case(RETENTION, case_id='b'))
    outcomes, stopped = run(EvidenceCritic(provider), cases)
    assert stopped and len(outcomes) == 1
    summary = aggregate(outcomes, 'own')
    assert summary['errors'] == {'provider_error': 1}
    assert (summary['completed'], summary['complete_cases'], summary['unnecessary_continuations']) == (0, 0, 0)


def test_unknown_cost_is_not_reported_as_zero():
    provider = ScriptedNative(lambda p: assessment(p, {'Retention': 'supported'}),
                              usage=Usage(api_calls=1, estimated_cost_usd=None))
    outcomes, _ = run(EvidenceCritic(provider), (case(RETENTION),), ('own',))
    assert aggregate(outcomes, 'own')['cost_usd'] is None


def test_cli_dry_run_makes_no_model_calls(capsys, monkeypatch):
    monkeypatch.setattr('src.generation.generator.AnthropicProvider',
                        lambda *a, **k: pytest.fail('provider constructed'))
    assert main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == {'ready': True, 'paid_calls': False, 'cases': 20, 'conditions': ['own', 'given'],
                    'planned_model_calls': 40, 'missing_fact_cases': 9, 'complete_cases': 11}


@pytest.mark.parametrize('extra', [[], ['--model', 'm'], ['--model', 'm', '--budget-usd', '1',
                                                            '--input-price', '1', '--output-price', '1']])
def test_cli_live_requires_explicit_model_budget_prices_and_output(extra):
    with pytest.raises(SystemExit) as error:
        main(['--live', *extra])
    assert error.value.code == 2


def test_cli_refuses_to_overwrite_results(tmp_path):
    existing = tmp_path / 'result.json'
    existing.write_text('{}')
    with pytest.raises(SystemExit):
        main(['--live', '--model', 'm', '--budget-usd', '1', '--input-price', '1', '--output-price', '1',
              '--output', str(existing)])
    assert existing.read_text() == '{}'

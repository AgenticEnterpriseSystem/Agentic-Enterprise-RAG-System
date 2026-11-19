"""Private critic input receipts with scripted providers and no services."""

from copy import deepcopy
import hashlib
import json
import socket
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.access.contracts import AccessDenied, EvidenceValidationError, StaleContext, guarded_dispatch
from src.agents.budget import RunBudgetExceeded
from src.agents.search_critic import EvidenceCritic
from src.api.runtime import AuthenticatedRuntime, RuntimeCapture, WebSettings, _ScopedRetriever
from src.contracts import (Answer, Chunk, CriticDecision, Evidence, GenerationRequest,
                           QueryRequest, QueryResponse, SearchRequest, Trace, TraceStep, Usage, UserContext)
from src.generation.generator import Completion, GenerationConfig, evidence_payload
from tests.test_native_evidence_contracts import CanonicalRecords, Registry, aggregate, boundary, record
from tests.test_scoped_agent import passage, scope


ZERO = Usage(estimated_cost_usd=0)


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('Critic capture checks must not contact a service')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    # Storage DTO/persistence behavior has its own PostgreSQL suite. These pure
    # observer tests use only an in-memory lifecycle and do not import its DB SDK.
    storage = ModuleType('src.storage.runs')
    storage.RunStepMetadata = lambda **values: SimpleNamespace(**values)
    monkeypatch.setitem(sys.modules, 'src.storage.runs', storage)


class Provider:
    supports_structured_output = False

    def __init__(self, *, followup=False, after=None):
        self.calls, self.critic_calls = [], 0
        self.followup, self.after = followup, after

    def complete(self, system, prompt):
        payload = json.loads(prompt)
        self.calls.append(payload)
        if system.startswith('Check whether the supplied passages'):
            self.critic_calls += 1
            more = self.followup and self.critic_calls == 1
            output = CriticDecision(sufficient=not more, confidence=.95, reason='Scripted fixture.',
                missing_facts=('Missing fixture detail',) if more else (),
                suggested_queries=('followup',) if more else ())
        else:
            output = Answer(status='answered', text='Fixture facts.',
                            citation_ids=tuple(item['citation_id'] for item in payload['evidence']))
        if self.after:
            self.after()
        return Completion(output.model_dump_json(), ZERO)


def legacy_request(text='Policy needle details. ' * 30):
    evidence = Evidence(citation_id='doc', score=1, retrieval_method='bm25',
        chunk=Chunk(chunk_id='doc', doc_id='doc', source_type='confluence', text=text,
                    start_char=0, end_char=len(text)))
    return GenerationRequest(question='needle details?', user=UserContext(user_id='fixture',
        allowed_doc_ids=frozenset({'doc'})), evidence=(evidence,))


def test_observer_gets_cropped_input_and_cannot_mutate_wire_or_request():
    request = legacy_request(); provider = Provider(); seen = []
    def observe(original, selected, payload):
        seen.append((original, selected, deepcopy(payload)))
        payload['question'] = 'ATTACK'
        payload['evidence'][0]['text'] = 'ATTACK'
        object.__setattr__(original[0].chunk, 'text', 'ATTACK')
        object.__setattr__(selected[0].chunk, 'text', 'ATTACK')
    EvidenceCritic(provider, GenerationConfig(max_context_chars=64), on_input=observe).assess(request)
    original, selected, payload = seen[0]
    assert len(request.evidence[0].chunk.text) > 64 and 'ATTACK' not in request.evidence[0].chunk.text
    assert provider.calls == [payload] and provider.calls[0]['question'] == request.question
    selected_payload = payload['evidence'][0]
    assert len(selected_payload['text']) <= 64 and selected_payload['citation_id'] != 'doc'
    _doc, start, end = selected_payload['citation_id'].split(':')
    assert int(start) > 0 and int(end) - int(start) == len(selected_payload['text'])


def test_empty_selection_is_observed_without_a_provider_call():
    request = legacy_request().model_copy(update={'evidence': ()})
    provider = Provider(); seen = []
    decision, usage = EvidenceCritic(provider, on_input=lambda *values: seen.append(values)).assess(request)
    assert not decision.sufficient and usage.api_calls == 0 and provider.calls == []
    assert seen == [((), (), {'question': request.question, 'evidence': []})]


@pytest.mark.parametrize('behavior', ['raise', 'return'])
def test_observer_failure_stops_before_provider(behavior):
    provider = Provider()
    def observe(*_values):
        if behavior == 'raise':
            raise RuntimeError('Private observer failed')
        return False
    with pytest.raises((RuntimeError, ValueError)):
        EvidenceCritic(provider, on_input=observe).assess(legacy_request())
    assert provider.calls == []


def test_revocation_during_observer_stops_before_provider(scope):
    provider = Provider()
    def observe(*_values):
        scope.guard.error = StaleContext('Revoked synthetic scope')
    request = GenerationRequest(question='policy', user=scope.user, evidence=(passage(scope),))
    with pytest.raises(StaleContext), guarded_dispatch(scope.context, scope.guard):
        EvidenceCritic(provider, on_input=observe).assess(request)
    assert provider.calls == []


def test_revocation_after_provider_prevents_a_decision(scope):
    provider = Provider(after=lambda: setattr(scope.guard, 'error', StaleContext('Revoked synthetic scope')))
    request = GenerationRequest(question='policy', user=scope.user, evidence=(passage(scope),))
    with pytest.raises(StaleContext), guarded_dispatch(scope.context, scope.guard):
        EvidenceCritic(provider).assess(request)
    assert len(provider.calls) == 1


def scripted_runtime(scope, *, followup=False, max_context_chars=24000):
    """Exercise the real request/capture path with an in-memory policy/store double.

    No database-backed runtime is constructed; these fixtures claim no actual
    corpus, model quality or persistence acceptance.
    """
    actual = object.__new__(AuthenticatedRuntime)
    actual.settings = WebSettings(mode='fixture', auth_mode='local', max_rounds=2,
                                  max_seconds=20, max_context_chars=max_context_chars)
    actual.snapshot, actual.resolver = scope.context.binding.snapshot, scope.guard
    scope.guard.resolve = lambda _identity: scope.context
    actual._source_types = frozenset({'confluence'})
    source = passage(scope, text='Policy fixture details. ' * 12)
    actual._catalog = SimpleNamespace(positions={'allowed': 0}, entries=(SimpleNamespace(
        revision_id=source.revision_id, source_type=source.chunk.source_type,
        timestamp=source.chunk.timestamp, metadata_origin=source.chunk.metadata_origin,
        text_length=len(source.chunk.text)),))
    class Retriever:
        calls = 0
        def search(self, _context, _query, **_kwargs):
            self.calls += 1
            start = 0 if self.calls == 1 else 100
            text = source.chunk.text[start:start+80]
            key = f'allowed:{start}:{start+len(text)}'
            return (source.model_copy(update={'citation_id': key, 'chunk': source.chunk.model_copy(update={
                'chunk_id': key, 'text': text, 'start_char': start, 'end_char': start+len(text)})}),)
    actual.retriever, actual.provider, actual.router = Retriever(), Provider(followup=followup), None
    actual.jira_tool = SimpleNamespace(validate_records=lambda *_args, **_kwargs: None)
    actual.runs = SimpleNamespace(begin=lambda *_a, **_k: None, append=lambda *_a, **_k: None,
        finish=lambda *_a, **_k: None, get_metadata=lambda *_a, **_k: None)
    actual._source_files = lambda: {}
    actual._assert_loaded_source = lambda: None
    def unavailable(_body):
        raise EvidenceValidationError('Fixture has no experiment attestation')
    actual._scientific_profile = unavailable
    return actual


def test_runtime_captures_each_critic_input_and_actual_trace_index_without_http_fields(scope):
    actual = scripted_runtime(scope, followup=True)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    inputs = actual.export_critic_inputs(captured)
    steps = captured.response.trace.steps
    assert len(inputs) == 2 and tuple(r['critic_call_ordinal'] for r in inputs) == (1, 2)
    assert all(steps[r['trace_step_index']].tool == 'critic' for r in inputs)
    assert all(r['disposition'] == 'captured' and r['provider_call_planned'] is True for r in inputs)
    for row in inputs:
        expected = {'question': 'policy', 'evidence': [evidence_payload(e) for e in row['selected_evidence']]}
        assert row['payload_sha256'] == hashlib.sha256(json.dumps(expected, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert tuple(e.citation_id for e in row['request_evidence']) == steps[row['trace_step_index']].evidence_ids
    assert 'critic_inputs' not in captured.response.model_dump()
    assert all(key not in json.dumps(actual.provider.calls) for key in ('opaque-row-id', 'server-scope', 'credentials'))


def test_capture_views_are_copies_and_current_export_requires_issuer_and_authority(scope):
    actual = scripted_runtime(scope)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    first = captured.critic_inputs
    first[0]['payload_sha256'] = 'f' * 64
    object.__setattr__(first[0]['selected_evidence'][0].chunk, 'text', 'ATTACK')
    assert actual.export_critic_inputs(captured)[0]['payload_sha256'] != 'f' * 64
    assert 'ATTACK' not in captured.critic_inputs[0]['selected_evidence'][0].chunk.text
    with pytest.raises(AccessDenied):
        actual.export_critic_inputs(object.__new__(RuntimeCapture))
    with pytest.raises(AccessDenied):
        scripted_runtime(scope).export_critic_inputs(captured)
    scope.guard.error = StaleContext('Revoked synthetic scope')
    with pytest.raises(StaleContext):
        actual.export_critic_inputs(captured)


def test_baseline_capture_does_not_invent_critic_inputs(scope):
    actual = scripted_runtime(scope)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='baseline'), scope.context.identity)
    assert captured.response.answer.status == 'answered'
    assert actual.export_critic_inputs(captured) == () and actual.provider.critic_calls == 0
    assert captured.generation_evidence


def test_empty_runtime_critic_capture_is_distinct_from_unavailable_input(scope, monkeypatch):
    # A tiny driver exercises the runtime observer when no context was selected;
    # the production agent normally stops earlier on an entirely empty search.
    import src.api.runtime as runtime_module
    actual = scripted_runtime(scope)
    class EmptyDriver:
        def __init__(self, _bridge, critic, _generator, **_kwargs):
            self.critic = critic
        def run(self, request, *, run_id, **_kwargs):
            decision, usage = self.critic.assess(GenerationRequest(question=request.query, user=request.user))
            return QueryResponse(snapshot=scope.context.binding.snapshot,
                answer=Answer(status='insufficient_evidence', text='No fixture evidence.'),
                trace=Trace(run_id=run_id, stop_reason='round_limit', steps=(TraceStep(
                    round_number=1, tool='critic', query=request.query, critic=decision,
                    elapsed_ms=0, usage=usage),)))
    monkeypatch.setattr(runtime_module, 'SearchAgent', EmptyDriver)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    inputs = actual.export_critic_inputs(captured)
    assert len(inputs) == 1 and inputs[0]['disposition'] == 'captured'
    assert inputs[0]['request_evidence'] == inputs[0]['selected_evidence'] == ()
    assert inputs[0]['provider_call_planned'] is False and inputs[0]['payload_sha256']
    assert actual.provider.calls == []


def test_preobserver_error_preserves_error_response_but_never_invents_capture(scope, monkeypatch, caplog):
    actual = scripted_runtime(scope)
    def failed(*_args):
        raise ValueError('PRIVATE_OBSERVER_SECRET')
    monkeypatch.setattr(actual, '_critic_input', failed)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    inputs = actual.export_critic_inputs(captured)
    assert captured.response.answer.status == 'error' and actual.provider.calls == []
    assert len(inputs) == 1 and inputs[0]['disposition'] == 'input_unavailable'
    assert all(inputs[0][key] is None for key in (
        'request_evidence', 'selected_evidence', 'payload_sha256', 'provider_call_planned'))
    assert 'PRIVATE_OBSERVER_SECRET' not in captured.response.model_dump_json() + caplog.text


def test_critic_budget_stop_keeps_observed_input_without_claiming_dispatch(scope):
    actual = scripted_runtime(scope)
    def denied(_system, _prompt):
        raise RunBudgetExceeded(usage=Usage(api_calls=0, estimated_cost_usd=0))
    actual.provider.complete = denied
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    row, = actual.export_critic_inputs(captured)
    assert captured.response.trace.stop_reason == 'budget'
    assert captured.response.trace.steps[row['trace_step_index']].tool == 'critic:budget'
    assert row['disposition'] == 'captured' and row['provider_call_planned'] is True
    assert captured.response.trace.usage.api_calls == 0


def test_current_export_rechecks_canonical_passage_offsets(scope):
    actual = scripted_runtime(scope)
    captured = actual.query_with_capture(QueryRequest(question='policy', mode='agentic'), scope.context.identity)
    actual._catalog.entries[0].text_length = 4
    with pytest.raises(EvidenceValidationError, match='canonical document metadata'):
        actual.export_critic_inputs(captured)


def test_runtime_native_capture_uses_authoritative_registry_and_record_validator(boundary):
    actual = object.__new__(AuthenticatedRuntime)
    actual.resolver = boundary.guard
    native, count = record(boundary), aggregate(boundary)
    canonical = CanonicalRecords(native)
    actual.jira_tool = SimpleNamespace(validate_records=lambda _context, values, **_kwargs: canonical(values))
    request = SearchRequest(query='native', user=boundary.user)
    bridge = _ScopedRetriever(None, boundary.context, boundary.guard, request,
                              SimpleNamespace(positions={}, entries=()))
    selected = (native, count)
    payload = {'question': 'native', 'evidence': [evidence_payload(e) for e in selected]}
    result = actual._critic_input(boundary.context, bridge, Registry(count), 'native', selected, selected, payload)
    assert result['selected_evidence'] == selected and canonical.calls
    empty = actual._critic_input(boundary.context, bridge, Registry(count), 'native', selected, (),
                                 {'question': 'native', 'evidence': []})
    assert empty['request_evidence'] == selected and empty['selected_evidence'] == ()
    assert empty['provider_call_planned'] is False
    changed = native.model_copy(update={'status': 'DENIED'})
    with pytest.raises(AccessDenied):
        actual._critic_input(boundary.context, bridge, Registry(count), 'native', (changed,), (changed,),
            {'question': 'native', 'evidence': [evidence_payload(changed)]})
    with pytest.raises(EvidenceValidationError):
        actual._critic_input(boundary.context, bridge, Registry(count), 'native', selected, selected,
                             {**payload, 'question': 'Different question'})

"""Managed document ceilings and exact model contexts, with no live providers."""
from datetime import datetime, timezone
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from src.access.contracts import AccessError, guarded_dispatch
from src.agents.context import (ContextBudgetError, document_ids, select_shared_context)
from src.agents.routing import Route, ToolCall, ToolSpec
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic
from src.contracts import (Answer, Chunk, CriticDecision, Evidence, EvidenceClaim, FactAssessment, GenerationRequest,
    SearchRequest, ToolResult, Trace, Usage, UserContext)
from src.generation.generator import (Completion, FrozenContextError, GenerationConfig,
    GroundedGenerator, evidence_chars, evidence_fingerprint, evidence_payload, freeze_context)
from tests.test_native_evidence_contracts import CanonicalRecords, Registry, aggregate, boundary, passage, record


@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    def denied(*_args, **_kwargs): raise AssertionError('No services, Docker or model clients in document-cap tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(subprocess, 'run', denied)


def hit(doc, text='Policy contains this fact.', start=0):
    cid = f'{doc}:{start}:{start+len(text)}'
    return Evidence(citation_id=cid, chunk=Chunk(chunk_id=cid, doc_id=doc, source_type='confluence',
        title='Policy', text=text, start_char=start, end_char=start+len(text)), score=1, retrieval_method='hybrid')


def request(items, **changes):
    return GenerationRequest(question='What policy fact is recorded?', user=UserContext(user_id='UNIT',
        allowed_doc_ids=frozenset(item.chunk.doc_id for item in items)), evidence=tuple(items), **changes)


@pytest.mark.parametrize('limit', [5, 8, 10])
def test_global_ceiling_counts_documents_once_and_preserves_rank_and_provenance(limit):
    items = [hit('d0', start=0), hit('d0', start=100)] + [hit(f'd{i}') for i in range(1, 16)]
    pool = request(items)
    selected, stats = select_shared_context(pool, 24000, {item.citation_id for item in items}, document_limit=limit)
    assert document_ids(selected) == tuple(f'd{i}' for i in range(limit))
    assert len(selected) == limit + 1  # Two exact windows of d0, not two documents.
    assert selected[:2] == tuple(items[:2])
    assert stats.pool_documents == 16 and stats.pool_passages == 17
    assert stats.selected_documents == limit and stats.selected_passages == limit + 1
    assert stats.exclusion_reasons == {'document_limit': 16 - limit}
    assert stats.evidence_sha256 == evidence_fingerprint(selected)


def test_character_budget_and_crop_are_visible_without_padding_to_document_quota():
    items = [hit('a', 'policy ' * 700), hit('b', 'other ' * 700)]
    selected, stats = select_shared_context(request(items), 500, {item.citation_id for item in items}, document_limit=8)
    assert document_ids(selected) == ('a',) and stats.selected_chars == 500
    assert stats.cropped_passages == 1 and stats.excluded_documents == stats.excluded_passages == 1
    assert stats.exclusion_reasons == {'char_budget': 1, 'cropped': 1}
    part = selected[0]
    assert part.chunk.text == items[0].chunk.text[part.chunk.start_char:part.chunk.end_char]
    assert part.citation_id == f'a:{part.chunk.start_char}:{part.chunk.end_char}'


def test_supported_initial_text_survives_newer_ranked_documents_and_read_more_windows():
    initial = hit('initial', 'policy fact '*70)
    expanded = hit('initial', initial.chunk.text + ' more policy detail '*90)
    newer = [hit(f'new{i}', 'policy detail '*20) for i in range(8)]
    pool = request([*newer, expanded, initial])
    selected, stats = select_shared_context(pool, 5000, {initial.citation_id}, (initial,), document_limit=2, round_number=2)
    assert initial in selected and selected[0] == initial
    assert len(document_ids(selected)) <= 2 and stats.round_number == 2
    for item in selected:
        parent = next(source for source in pool.evidence if source.chunk.doc_id == item.chunk.doc_id
            and source.chunk.start_char <= item.chunk.start_char < item.chunk.end_char <= source.chunk.end_char)
        offset = item.chunk.start_char-parent.chunk.start_char
        assert item.chunk.text == parent.chunk.text[offset:offset+len(item.chunk.text)]


def test_supported_document_overflow_stops_instead_of_dropping_prior_facts():
    items = [hit('a'), hit('b')]
    with pytest.raises(ContextBudgetError) as error:
        select_shared_context(request(items), 24000, {item.citation_id for item in items}, tuple(items), document_limit=1)
    assert error.value.reason == 'retained_document_limit'


def test_global_cap_keeps_useful_followup_group_when_many_short_initial_passages_fit():
    initial = tuple(hit(f'old{i}', 'policy old fact') for i in range(8))
    followup = tuple(hit(f'new{i}', 'policy new missing detail') for i in range(8))
    selected, stats = select_shared_context(request((*followup, *initial)), 11000,
        {item.citation_id for item in initial}, document_limit=8, round_number=2)
    docs = document_ids(selected)
    assert docs == tuple(f'old{i}' for i in range(4)) + tuple(f'new{i}' for i in range(4))
    assert stats.pool_documents == 16 and stats.selected_documents == 8


def test_native_record_and_passage_share_one_document_aggregate_count_is_not_a_document_quota(boundary):
    native, text = record(boundary), passage(boundary)
    count = aggregate(boundary).model_copy(update={'count': 12345})
    with guarded_dispatch(boundary.context, boundary.guard, aggregates=Registry(count), record_validator=CanonicalRecords(native)):
        supplied = GenerationRequest(question='policy', user=boundary.user, evidence=(text, native, count))
        selected, stats = select_shared_context(supplied, 24000, (text.citation_id,), document_limit=1)
        assert text in selected and native in selected and count in selected
        assert stats.selected_documents == 1 and stats.selected_native_items == 2
        assert count.count == 12345 and stats.selected_chars == sum(evidence_chars(item) for item in selected)


class Provider:
    supports_structured_output = True
    def __init__(self): self.calls = []
    def complete_structured(self, system, prompt, schema):
        payload = json.loads(prompt); self.calls.append(payload)
        cid = payload['evidence'][0]['citation_id']
        if 'status' in schema['properties']:
            value = {'status': 'answered', 'text': 'The recorded policy supplies this fact.', 'citation_ids': [cid]}
        else:
            value = {'facts': [{'fact': 'policy fact', 'status': 'supported', 'citation_ids': [cid],
                'claims': [{'citation_id': cid, 'statement': 'The fact is supplied.', 'scope_reason': 'Requested policy fact',
                    'answers_requested_fact': True}]}]}
        return Completion(json.dumps(value), Usage(estimated_cost_usd=0))


def test_actual_native_critic_and_generator_payloads_preserve_exact_frozen_ordered_input():
    items = tuple(hit(f'd{i}') for i in range(12)); supplied = request(items)
    selected, _stats = select_shared_context(supplied, 24000, {item.citation_id for item in items}, document_limit=8)
    frozen = freeze_context(supplied.model_copy(update={'evidence': selected}))
    provider = Provider(); decision, _ = EvidenceCritic(provider).assess(frozen)
    answer, _ = GroundedGenerator(provider).generate(frozen.model_copy(update={'assessment': decision}))
    assert answer.status == 'answered' and provider.calls[0]['evidence'] == provider.calls[1]['evidence']
    assert len({item['doc_id'] for item in provider.calls[1]['evidence']}) == 8
    assert frozen.frozen_evidence_sha256 == evidence_fingerprint(selected)


@pytest.mark.parametrize('consumer', [EvidenceCritic, GroundedGenerator])
@pytest.mark.parametrize('change', ['different_evidence', 'different_order', 'lower_stage_budget'])
def test_frozen_context_mismatch_fails_before_any_provider_call(consumer, change):
    items = (hit('a'), hit('b'))
    frozen = freeze_context(request(items)); config = GenerationConfig()
    if change == 'different_evidence': frozen = frozen.model_copy(update={'evidence': (items[0],)})
    elif change == 'different_order': frozen = frozen.model_copy(update={'evidence': tuple(reversed(items))})
    else: config = GenerationConfig(max_context_chars=1)
    provider = Provider(); instance = consumer(provider, config)
    with pytest.raises(FrozenContextError):
        (instance.assess if consumer is EvidenceCritic else instance.generate)(frozen)
    assert provider.calls == []


def test_multiple_text_queries_get_eight_each_but_share_global_eight_and_leave_tool_limit_five():
    items = tuple(hit(f'd{i}') for i in range(16)); user = request(items).user
    class Router:
        def route(self, req, feedback, tools):
            return Route(reason='Search', text_queries=('first', 'second'),
                tool_calls=(ToolCall(tool='jira', operation='filter'),)), Usage(estimated_cost_usd=0)
    class Search:
        def __init__(self): self.requests = []
        def search(self, req):
            self.requests.append(req); return items[:8] if req.query == 'first' else items[8:]
    class Tool:
        request = None
        def execute(self, req):
            self.request = req; return ToolResult(coverage='verified_records')
    search, tool, provider = Search(), Tool(), Provider()
    agent = SearchAgent(search, EvidenceCritic(provider), GroundedGenerator(provider), router=Router(),
        tools={'jira': ToolSpec(tool, 'jira', 'UNIT native tool', ('filter',), ())},
        config=AgentConfig(max_rounds=2, max_tokens=16000, evidence_doc_limit=8))
    result = agent.run(SearchRequest(query='policy', user=user, top_k=5))
    assert result.answer.status == 'answered' and [req.top_k for req in search.requests] == [8, 8]
    assert tool.request.limit == 5
    stats = result.trace.context_selection[0]
    assert stats.pool_documents == 16 and stats.selected_documents == 8
    assert len({item.chunk.doc_id for item in result.sources}) == 8
    assert provider.calls[0]['evidence'] == provider.calls[1]['evidence']


def test_actual_read_more_keeps_supported_exact_text_and_shared_one_document_context():
    initial = hit('a', 'policy first fact. ' * 20)
    expanded = hit('a', initial.chunk.text + 'policy second fact. ' * 20)
    user = request((initial,)).user
    class Search:
        def __init__(self): self.calls = 0; self.anchors = []
        def search(self, req): self.calls += 1; return (initial,)
        def read_more(self, req, anchors, *, max_chars):
            self.anchors.append(anchors); assert max_chars == 3000
            return (expanded,)
    class Critic:
        def __init__(self): self.requests = []
        def assess(self, req):
            self.requests.append(req)
            facts = [FactAssessment(fact='first', status='supported', citation_ids=(initial.citation_id,),
                claims=(EvidenceClaim(citation_id=initial.citation_id, statement='First policy fact',
                    scope_reason='Requested fact', answers_requested_fact=True),))]
            first = len(self.requests) == 1
            facts.append(FactAssessment(fact='second', status='missing' if first else 'supported',
                citation_ids=() if first else (req.evidence[-1].citation_id,),
                claims=() if first else (EvidenceClaim(citation_id=req.evidence[-1].citation_id,
                    statement='Second policy fact', scope_reason='Requested fact', answers_requested_fact=True),)))
            return CriticDecision(facts=tuple(facts), sufficient=not first, confidence=0, reason='UNIT checklist',
                missing_facts=('second',) if first else (), read_more_doc_ids=('a',) if first else ()), Usage(estimated_cost_usd=0)
    search, critic, provider = Search(), Critic(), Provider()
    result = SearchAgent(search, critic, GroundedGenerator(provider),
        config=AgentConfig(max_rounds=2, max_tokens=16000, evidence_doc_limit=1)).run(SearchRequest(query='policy', user=user))
    assert result.answer.status == 'answered' and search.calls == 1 and len(search.anchors) == 1
    assert any(step.tool == 'read_more' for step in result.trace.steps)
    assert initial in critic.requests[1].evidence
    assert provider.calls[0]['evidence'] == [evidence_payload(item) for item in critic.requests[1].evidence]
    assert all(stats.selected_documents == 1 for stats in result.trace.context_selection)
    assert len(critic.requests[1].evidence) == 2  # Exact supported old window plus nonoverlapping new text.


def test_managed_document_limit_never_admits_unauthorized_passage(boundary):
    denied = hit('hidden')
    # Snapshot provenance is valid; permission is deliberately absent.
    from src.contracts import PassageEvidence
    denied = PassageEvidence(**denied.model_dump(), snapshot_id=boundary.snapshot.snapshot_id, revision_id='d' * 64)
    search = SimpleNamespace(search=lambda req: (denied,))
    provider = Provider()
    with pytest.raises(AccessError):
        SearchAgent(search, EvidenceCritic(provider), GroundedGenerator(provider),
            config=AgentConfig(evidence_doc_limit=8)).run(SearchRequest(query='policy', user=boundary.user),
                context=boundary.context, guard=boundary.guard)
    assert provider.calls == []


def test_selection_observations_contain_counts_not_query_evidence_or_fingerprint():
    from src.observability import Recorder, observation_context
    items = tuple(hit(f'PRIVATE-DOC-{i}', 'PRIVATE-EVIDENCE') for i in range(16)); user = request(items).user
    class Router:
        def route(self, req, feedback, tools): return Route(reason='PRIVATE-REASON', text_queries=('one', 'two')), Usage(estimated_cost_usd=0)
    search = SimpleNamespace(search=lambda req: items[:8] if req.query == 'one' else items[8:])
    recorder = Recorder(log=False); observation = recorder.start_request()
    with observation_context(observation):
        result = SearchAgent(search, EvidenceCritic(Provider()), GroundedGenerator(Provider()), router=Router(),
            config=AgentConfig(evidence_doc_limit=8)).run(SearchRequest(query='PRIVATE-QUESTION', user=user, top_k=5), single_pass=True)
    recorded = recorder.get(observation.request_id)
    events = [event for event in recorded['stages'] if event['stage'] == 'context_select' and event['event'] == 'end']
    assert events[0]['selected_documents'] == 8 and events[0]['pool_documents'] == 16
    text = json.dumps(recorded)
    assert all(value not in text for value in ('PRIVATE-DOC-', 'PRIVATE-EVIDENCE', 'PRIVATE-QUESTION', 'PRIVATE-REASON',
        result.trace.context_selection[0].evidence_sha256))


def test_legacy_none_keeps_wire_and_existing_callers_compatible():
    assert AgentConfig().evidence_doc_limit is None
    trace = Trace(run_id='UNIT', stop_reason='single_pass')
    assert 'context_selection' not in trace.model_dump(mode='json')
    assert 'frozen_evidence_sha256' not in request((hit('a'),)).model_dump(mode='json')


@pytest.mark.parametrize('value', [True, False, 0, 21, 8.0, '8'])
def test_generic_document_ceiling_rejects_non_strict_or_out_of_range_values(value):
    with pytest.raises(ValueError): AgentConfig(evidence_doc_limit=value)

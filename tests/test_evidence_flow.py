"""Behavioral regressions for bounded evidence reuse, expansion and assessment.

Scripted model replies exercise contracts and orchestration, not judge accuracy.
Real development answers are reviewed separately in evaluation/agent_development.
"""
import json
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from src.agents.context import ContextBudgetError, select_agent_context
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import AgentError, EvidenceCritic
from src.contracts import (Answer, Chunk, CriticDecision, Evidence, FactAssessment,
                           GenerationRequest, SearchRequest, Usage, UserContext)
from src.generation.generator import Completion, GroundedGenerator
from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, document_text
from src.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeProvider


ZERO = Usage(estimated_cost_usd=0)
USER = UserContext(user_id='test', allowed_doc_ids=frozenset({'ticket', 'other'}))


def hit(doc, text, start=0):
    key = f'{doc}:{start}:{start + len(text)}'
    return Evidence(citation_id=key, chunk=Chunk(chunk_id=key, doc_id=doc,
        source_type='jira', title='Nightly evaluation jobs', text=text,
        start_char=start, end_char=start + len(text)), score=.5, retrieval_method='hybrid')


def context(*hits):
    return GenerationRequest(question='Why did the nightly evaluation jobs fail after the policy change?',
                             user=USER, evidence=hits)


def reply(**changes):
    return Completion(CriticDecision(sufficient=False, confidence=.95,
        reason='Check requested facts', **changes).model_dump_json(), ZERO)


def test_fact_checklist_resolves_inconsistent_boolean_and_missing_summary():
    item = hit('ticket', 'Escalate if p95 exceeds 1800 ms for 8 minutes.')
    facts = tuple(FactAssessment(fact=fact, status='supported', citation_ids=(item.citation_id,))
                  for fact in ('Threshold: 1800 ms', 'Duration: 8 minutes'))
    provider = FakeProvider([reply(facts=facts, missing_facts=('Find signed approval',))])
    decision, _ = EvidenceCritic(provider).assess(context(item))
    assert decision.sufficient and decision.missing_facts == ()
    assert decision.facts == facts


@pytest.mark.parametrize('kind', ['citation', 'document'])
def test_critic_rejects_references_outside_its_supplied_evidence(kind):
    item = hit('ticket', 'Job failed after a policy update.')
    changes = ({'facts': (FactAssessment(fact='Cause', status='supported', citation_ids=('invented',)),)}
               if kind == 'citation' else {'read_more_doc_ids': ('unseen-document',)})
    with pytest.raises(AgentError, match='outside the supplied context'):
        EvidenceCritic(FakeProvider([reply(**changes)])).assess(context(item))


def test_supported_passage_survives_keyword_heavy_followup_without_recropping():
    supported = hit('ticket', 'The policy affected service accounts. ' * 20, start=300)
    followup = hit('other', 'nightly evaluation jobs policy change ' * 100)
    selected = select_agent_context(context(supported, followup), 1800,
                                    {supported.citation_id}, retained=(supported,))
    assert selected[0] == supported
    assert any(e.chunk.doc_id == 'other' for e in selected)
    assert sum(len(e.chunk.text) for e in selected) <= 1800
    assert len({e.citation_id for e in selected}) == len(selected)
    with pytest.raises(ContextBudgetError):
        select_agent_context(context(supported), 10, {supported.citation_id}, retained=(supported,))


def test_retained_slice_does_not_duplicate_overlapping_parent_text():
    text = 'The nightly evaluation jobs fail after a policy change. ' * 40
    parent = hit('ticket', text)
    keep = hit('ticket', text[400:800], start=400)
    selected = select_agent_context(context(parent), 1400, {parent.citation_id}, retained=(keep,))
    assert selected[0] == keep
    for item in selected[1:]:
        assert item.chunk.end_char <= 400 or item.chunk.start_char >= 800
        assert item.chunk.text == text[item.chunk.start_char:item.chunk.end_char]


@pytest.fixture
def document_store():
    content = ('Nightly evaluation jobs failed. ' + 'Investigation notes. ' * 80
               + 'The policy mistakenly enabled device trust for service accounts, revoking refresh tokens. '
               + 'Resolution notes. ' * 50)
    table = pa.table({'doc_id': ['ticket'], 'source_type': ['jira'],
                      'title': ['Nightly evaluation jobs'], 'content': [content]})
    store = CorpusBM25(None, ['ticket'], BM25Config(), corpus=table)
    full_text = document_text('Nightly evaluation jobs', content)
    return store, full_text, hit('ticket', full_text[:600])


def test_read_more_reaches_ticket_cause_with_exact_offsets_and_parent_provenance(document_store):
    store, text, initial = document_store
    request = SearchRequest(query='Why did jobs fail?', user=USER, source_types=('jira',))
    expanded = HybridRetriever(store, store).read_more(request, (initial,))
    assert len(expanded) == 1 and 'mistakenly enabled device trust' in expanded[0].chunk.text
    item = expanded[0]
    assert item.chunk.text == text[item.chunk.start_char:item.chunk.end_char]
    assert item.citation_id == f'ticket:{item.chunk.start_char}:{item.chunk.end_char}'
    assert item.score == initial.score and item.retrieval_method == 'hybrid'
    assert len(item.chunk.text) <= 3000


def test_read_more_enforces_authorization_sources_dates_and_anchor_integrity(document_store):
    store, _, initial = document_store
    base = SearchRequest(query='Why?', user=USER)
    with pytest.raises(ValueError, match='authorized'):
        store.read_more(base.model_copy(update={'user': UserContext()}), (initial,))
    assert store.read_more(base.model_copy(update={'source_types': ('slack',)}), (initial,)) == ()
    assert store.read_more(base.model_copy(update={'after': datetime.now(timezone.utc)}), (initial,)) == ()
    tampered = initial.model_copy(update={'chunk': initial.chunk.model_copy(update={'text': 'x' * 600})})
    with pytest.raises(ValueError, match='canonical'):
        store.read_more(base, (tampered,))
    with pytest.raises(ValueError, match='bounds'):
        store.read_more(base, (initial,), max_chars=10000)


def test_agent_reads_matching_document_before_broad_search_and_generates_same_context(document_store):
    store, _, initial = document_store
    request = SearchRequest(query=context().question, user=USER)
    expanded = store.read_more(request, (initial,))[0]
    class Search:
        searches = 0
        reads = 0
        def search(self, request):
            self.searches += 1
            return (initial,)
        def read_more(self, request, anchors, **kwargs):
            self.reads += 1
            return (expanded,)
    search = Search()
    # The final citation uses the initial hit to keep this interface test
    # independent of the context selector's exact window boundaries.
    provider = FakeProvider([
        reply(facts=(FactAssessment(fact='Failure cause', status='missing'),),
              read_more_doc_ids=('ticket',), suggested_queries=('policy update cause',)),
        Completion(CriticDecision(sufficient=True, confidence=.95,
            reason='Cause supplied').model_dump_json(), ZERO),
        Completion(Answer(status='answered', text='The policy revoked service account tokens.',
                          citation_ids=(initial.citation_id,)).model_dump_json(), ZERO),
    ])
    result = SearchAgent(search, EvidenceCritic(provider), GroundedGenerator(provider),
                         config=AgentConfig(max_rounds=2)).run(request)
    assert search.searches == 1 and search.reads == 1
    assert result.trace.stop_reason == 'sufficient'
    assessed = json.loads(provider.calls[1][1])['evidence']
    generated = json.loads(provider.calls[2][1])['evidence']
    assert assessed == generated
    assert any('mistakenly enabled device trust' in item['text'] for item in assessed)
    assert [s.tool for s in result.trace.steps].count('read_more') == 1


def test_ambiguous_incidents_ask_one_clarification_without_another_model_call():
    a, b = hit('ticket', 'March incident: cache eviction.'), hit('other', 'June incident: rate limiter.')
    provider = FakeProvider([reply(facts=(FactAssessment(fact='Which incident?', status='ambiguous',
        citation_ids=(a.citation_id, b.citation_id)),), read_more_doc_ids=('ticket', 'other'),
        clarification_question='Which incident date do you mean?')])
    class Search:
        def search(self, request):
            return (a, b)
        def read_more(self, *args, **kwargs):
            pytest.fail('Reading one candidate must not erase an already established incident ambiguity')
    result = SearchAgent(Search(), EvidenceCritic(provider), GroundedGenerator(provider)).run(
        SearchRequest(query='What caused the latency spike?', user=USER))
    assert result.trace.stop_reason == 'clarification'
    assert result.answer.text == 'Which incident date do you mean?'
    assert result.answer.status == 'insufficient_evidence'
    assert len(provider.calls) == 1


def test_complete_fact_checklist_does_not_research_based_on_uncalibrated_confidence():
    item = hit('ticket', 'Verified staging rollback recovery target is under 9 minutes.')
    class Search:
        def search(self, request):
            return (item,)
    assessment = CriticDecision(sufficient=True, confidence=.4,
        reason='Explicit target in the runner notes.',
        clarification_question='Are these notes official?', facts=(FactAssessment(
            fact='Recovery target: under 9 minutes', status='supported', citation_ids=(item.citation_id,)),))
    provider = FakeProvider([Completion(assessment.model_dump_json(), ZERO),
        Completion(Answer(status='answered', text='Under 9 minutes.',
                          citation_ids=(item.citation_id,)).model_dump_json(), ZERO)])
    result = SearchAgent(Search(), EvidenceCritic(provider), GroundedGenerator(provider)).run(
        SearchRequest(query='What recovery target do the runner notes specify?', user=USER))
    assert result.trace.stop_reason == 'sufficient'
    assert len(provider.calls) == 2


def test_conflicting_fact_checklist_cannot_be_overridden_by_sufficient_boolean():
    a, b = hit('ticket', 'Current setting: 15 days.'), hit('other', 'Same setting: 25 days.')
    assessment = CriticDecision(sufficient=True, confidence=.99, reason='Records disagree.',
        facts=(FactAssessment(fact='Current retention', status='conflicting',
                              citation_ids=(a.citation_id, b.citation_id)),))
    decision, _ = EvidenceCritic(FakeProvider([Completion(assessment.model_dump_json(),ZERO)])).assess(context(a,b))
    assert not decision.sufficient and decision.missing_facts == ('Current retention',)


def test_native_critic_schema_allows_only_supplied_passages_and_matched_documents():
    item = hit('ticket', 'The configured recovery target is under 9 minutes.')
    class StructuredProvider:
        calls = []
        def complete_structured(self, system, prompt, schema):
            self.calls.append(schema)
            return reply(facts=(FactAssessment(fact='Target: under 9 minutes', status='supported',
                                              citation_ids=(item.citation_id,)),))
    provider = StructuredProvider()
    decision, _ = EvidenceCritic(provider).assess(context(item))
    schema = provider.calls[0]
    assert schema['$defs']['FactAssessment']['properties']['citation_ids']['items']['enum'] == [item.citation_id]
    assert schema['properties']['read_more_doc_ids']['items']['enum'] == ['ticket']
    assert 'facts' in schema['required'] and schema['properties']['facts']['minItems'] == 1
    assert decision.sufficient

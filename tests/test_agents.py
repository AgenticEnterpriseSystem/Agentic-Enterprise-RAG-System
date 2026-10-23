import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from src.agents.pilot import (main, run_pilot, run_comparison, select_questions,
                              development_sample, prepare_review, timing_breakdown)
from src.agents.routing import ModelRouter, Route, ToolCall, ToolSpec
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import AgentError, EvidenceCritic, SYSTEM_PROMPT
from src.contracts import Answer, Chunk, CriticDecision, Evidence, GenerationRequest, SearchRequest, ToolResult, Usage, UserContext
from src.generation.generator import Completion, GroundedGenerator, ProviderError
from src.retrieval.corpus_bm25 import CorpusBM25
from src.retrieval.corpus_dense import CorpusDense
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from tests.fakes import FakeProvider
from src.agents.pilot import run_interactive, RecordingProvider


# Critic, routing, and search loop

ZERO = Usage(estimated_cost_usd=0)
USER = UserContext(user_id="jane", allowed_doc_ids=frozenset({"a", "b"}))


def test_interactive_retrieval_only_records_real_evidence_without_model(tmp_path):
    search = Search([(evidence(),)])
    search.doc_ids = ['a']
    entries = iter(['', 'Upload limit?', '/quit'])
    lines = []
    summary = run_interactive(search, None, AgentConfig(), tmp_path/'session',
                              read=lambda _: next(entries), emit=lines.append)
    case = json.loads((tmp_path/'session/case_01.json').read_text())
    assert summary['completed'] == 1 and summary['usage']['api_calls'] == 0
    assert case['model_calls'] == [] and case['mode'] == 'retrieval-only'
    assert case['debug_steps'][0]['new_evidence_ids'] == ['a']
    assert case['response']['answer']['status'] == 'insufficient_evidence'
    assert any('no critic' in line for line in lines)
    assert any('Upload limit is 10 MB.' in line for line in lines)


def test_interactive_live_scripted_adaptation_records_why_and_citations(tmp_path):
    search = Search([(evidence('a'),), (evidence('b'),)])
    search.doc_ids = ['a', 'b']
    provider = FakeProvider([
        Completion(decision(False, missing_facts=('Current limit',),
                            suggested_queries=('current upload policy',)).model_dump_json(), ZERO),
        Completion(decision(True).model_dump_json(), ZERO),
        Completion(Answer(status='answered', text='10 MB', citation_ids=('b',)).model_dump_json(), ZERO),
    ])
    entries = iter(['Upload limit?', '/quit'])
    lines = []
    summary = run_interactive(search, provider, AgentConfig(max_rounds=2), tmp_path/'session',
                              live=True, read=lambda _: next(entries), emit=lines.append)
    case = json.loads((tmp_path/'session/case_01.json').read_text())
    assert summary['stops'] == {'sufficient': 1}
    assert [r.query for r in search.requests] == ['Upload limit?', 'current upload policy']
    assert len(case['model_calls']) == 3
    assert case['response']['answer']['citation_ids'] == ['b']
    assert any('Missing: Current limit' in line for line in lines)
    assert any('Round 2' in line for line in lines)
    assert 'reference' not in case


def test_interactive_repeated_results_and_eof(tmp_path):
    search = Search([(evidence(),), (evidence(),)])
    search.doc_ids = ['a']
    provider = FakeProvider([Completion(decision(False,
        suggested_queries=('another search',)).model_dump_json(), ZERO),
        Completion(Answer(status='insufficient_evidence', text='Evidence remains incomplete.',
                          citation_ids=('a',)).model_dump_json(), ZERO)])
    def read(_):
        if search.requests:
            raise EOFError
        return 'Upload limit?'
    summary = run_interactive(search, provider, AgentConfig(max_rounds=2), tmp_path/'session',
                              live=True, read=read, emit=lambda _: None)
    case = json.loads((tmp_path/'session/case_01.json').read_text())
    searches = [s for s in case['debug_steps'] if 'new_evidence_ids' in s]
    assert [s['new_evidence_ids'] for s in searches] == [['a'], []]
    assert summary['stops'] == {'no_new_evidence': 1}
    assert len(provider.calls) == 2


def test_interactive_denied_evidence_not_saved_or_printed(tmp_path):
    search = Search([(evidence('secret', text='secret content'),)])
    search.doc_ids = ['a']
    lines = []
    summary = run_interactive(search, None, AgentConfig(), tmp_path/'session',
                              read=lambda _: 'q', emit=lines.append)
    # Backend validation failures must not retain denied text in recording logs.
    assert summary['stops'] == {'error': 1}
    assert not any('secret content' in line for line in lines)
    assert 'secret content' not in (tmp_path/'session/case_01.json').read_text()


def test_recording_provider_preserves_budget_denial():
    from src.agents.budget import RunBudgetExceeded
    class Denied:
        def complete(self, system, prompt):
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))
    provider = RecordingProvider(Denied())
    with pytest.raises(RunBudgetExceeded):
        provider.complete('system', '{}')
    assert provider.calls[0]['error'] == 'RunBudgetExceeded'
    assert provider.calls[0]['usage']['api_calls'] == 1


def evidence(doc="a", text="Upload limit is 10 MB.", source="docs"):
    return Evidence(citation_id=doc, chunk=Chunk(chunk_id=doc, doc_id=doc,
                    source_type=source, text=text, start_char=0, end_char=len(text)),
                    score=1, retrieval_method="bm25")


def decision(enough=False, **kwargs):
    return CriticDecision(sufficient=enough, confidence=0.95 if enough else 0.2,
                          reason="Enough evidence" if enough else "Limit missing", **kwargs)


class Search:
    def __init__(self, hits):
        self.doc_ids: list[str] = []
        self.hits = iter(hits)
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return next(self.hits)


class Critic:
    def __init__(self, decisions, usage=ZERO):
        self.decisions, self.usage = iter(decisions), usage
        self.requests = []

    def assess(self, request):
        self.requests.append(request)
        return next(self.decisions), self.usage


class Generator:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return Answer(status="answered", text="10 MB", citation_ids=(request.evidence[-1].citation_id,)), ZERO


class Router:
    def __init__(self, routes):
        self.routes = iter(routes)
        self.feedback = []

    def route(self, request, feedback, tools):
        self.feedback.append(feedback)
        return next(self.routes), ZERO


class Tool:
    def __init__(self, result):
        self.result, self.requests = result, []

    def execute(self, request):
        self.requests.append(request)
        return self.result


def request(**kwargs):
    return SearchRequest(query="What is the upload limit?", user=USER, **kwargs)


@pytest.mark.parametrize("text,expected,missing", [
    ("Upload limit is 10 MB.", True, ()),
    ("Files can be uploaded.", False, ("Upload limit",)),
    ("Limit is 10 MB. Another policy says 20 MB.", False, ("Resolve conflict",)),
    ("The old policy in 2019 had a 10 MB limit.", False, ("Current policy",)),
])
def test_critic_preserves_labeled_decisions(text, expected, missing):
    # Scripted fixtures test the interface; they do not measure real model accuracy.
    expected_decision = decision(expected, missing_facts=missing)
    provider = FakeProvider([Completion(expected_decision.model_dump_json(), ZERO)])
    result, _ = EvidenceCritic(provider).assess(GenerationRequest(
        question="What is the current upload limit?", user=USER, evidence=(evidence(text=text),)))
    assert result == expected_decision
    payload = json.loads(provider.calls[0][1])
    assert payload['evidence'][0]['timestamp'] is None
    assert "Unknown dates" in SYSTEM_PROMPT
    assert "reference" not in payload


def test_empty_critic_does_not_call_model():
    provider = FakeProvider([])
    result, usage = EvidenceCritic(provider).assess(GenerationRequest(question="Limit?", user=USER))
    assert not result.sufficient and usage.api_calls == 0 and not provider.calls


@pytest.mark.parametrize("response", [
    Completion("bad json", ZERO), Completion(decision(True).model_dump_json(), ZERO, truncated=True),
    ProviderError(usage=Usage(api_calls=1)),
])
def test_critic_failure_is_explicit(response):
    with pytest.raises(AgentError):
        EvidenceCritic(FakeProvider([response])).assess(GenerationRequest(
            question="Limit?", user=USER, evidence=(evidence(),)))


def test_critic_cannot_approve_missing_facts():
    provider = FakeProvider([Completion(decision(True, missing_facts=("Date",)).model_dump_json(), ZERO)])
    result, _ = EvidenceCritic(provider).assess(GenerationRequest(question="Limit?", user=USER, evidence=(evidence(),)))
    assert not result.sufficient


def test_followup_accumulates_evidence_and_preserves_scope():
    search = Search([(evidence(text="Uploads supported"),), (evidence("b"),)])
    critic = Critic([decision(suggested_queries=("maximum upload size",)), decision(True)])
    generator = Generator()
    result = SearchAgent(search, critic, generator).run(request(source_types=("docs",)))
    assert result.answer.status == "answered" and result.trace.stop_reason == "sufficient"
    assert [r.query for r in search.requests] == ["What is the upload limit?", "maximum upload size"]
    assert all(r.user == USER and r.source_types == ("docs",) for r in search.requests)
    assert len(generator.requests[0].evidence) == 2
    assert len([s for s in result.trace.steps if s.critic]) == 2


@pytest.mark.parametrize("hits,rounds,stop", [
    ([()], 3, "no_new_evidence"),
    ([(evidence(),), (evidence(),)], 3, "no_new_evidence"),
    ([(evidence(),)], 1, "round_limit"),
])
def test_stop_conditions_never_generate_guessed_answer(hits, rounds, stop):
    generator = Generator()
    result = SearchAgent(Search(hits), Critic([decision()] * 3), generator,
                         config=AgentConfig(max_rounds=rounds)).run(request())
    assert result.trace.stop_reason == stop
    assert result.answer.status == "insufficient_evidence"
    assert bool(generator.requests) == any(hits)


@pytest.mark.parametrize("usage,config", [
    (Usage(input_tokens=10, estimated_cost_usd=0), AgentConfig(max_tokens=10)),
    (Usage(api_calls=1), AgentConfig()),
    (Usage(estimated_cost_usd=0.2), AgentConfig(max_cost_usd=0.1)),
])
def test_usage_budget_stops_before_generation(usage, config):
    generator = Generator()
    result = SearchAgent(Search([(evidence(),)]), Critic([decision(True)], usage), generator,
                         config=config).run(request())
    assert result.trace.stop_reason == "budget" and not generator.requests
    assert result.trace.usage == usage


def test_elapsed_budget_stops_before_call():
    ticks = iter([0, 2, 2])
    result = SearchAgent(Search([]), Critic([]), Generator(), config=AgentConfig(max_seconds=1),
                         clock=lambda: next(ticks)).run(request())
    assert result.trace.stop_reason == "budget"


@pytest.mark.parametrize("hit,scope", [(evidence("secret"), ()), (evidence(source="gmail"), ("docs",))])
def test_unauthorized_evidence_never_reaches_critic(hit, scope):
    critic = Critic([])
    result = SearchAgent(Search([(hit,)]), critic, Generator()).run(request(source_types=scope))
    assert result.trace.stop_reason == "error" and not critic.requests and not result.sources


def test_low_confidence_only_generates_limited_answer():
    generator = Generator()
    result = SearchAgent(Search([(evidence(),)]), Critic([decision(True).model_copy(update={"confidence": 0.2})]),
                         generator, config=AgentConfig(max_rounds=1)).run(request())
    assert result.trace.stop_reason == "round_limit"
    assert result.answer.status == "insufficient_evidence"
    assert generator.requests[0].search_limitation


@pytest.mark.parametrize("mixed,operation", [(False, "lookup"), (False, "count"), (True, "lookup")])
def test_structured_and_mixed_routes(mixed, operation):
    tool = Tool(ToolResult(evidence=(evidence("b", "Count: 2", "hubspot"),), count=2 if operation == "count" else None))
    router = Router([Route(reason="CRM lookup", text_queries=("policy",) if mixed else (),
                           tool_calls=(ToolCall(tool="crm", operation=operation, filters={"status": "open"}),))])
    search, generator = Search([(evidence(),)]), Generator()
    result = SearchAgent(search, Critic([decision(True)]), generator, router=router,
                         tools={"crm": ToolSpec(tool, "hubspot", "CRM records", ("lookup", "count"), ("status",))}).run(request())
    assert result.answer.status == "answered"
    assert bool(search.requests) == mixed
    assert tool.requests[0].user == USER
    assert tool.requests[0].operation == operation
    assert any(s.tool == "crm" for s in result.trace.steps)


def test_model_router_sends_capabilities_and_feedback_without_identity():
    route = Route(reason="Need ticket", tool_calls=(ToolCall(tool="tickets", operation="lookup", filters={"id": "T-42"}),))
    provider = FakeProvider([Completion(route.model_dump_json(), ZERO)])
    feedback = decision(missing_facts=("Ticket owner",))
    result, _ = ModelRouter(provider).route(request(), feedback, {
        "tickets": ToolSpec(Tool(ToolResult()), "jira", "Ticket lookup", ("lookup",), ("id",))})
    payload = json.loads(provider.calls[0][1])
    assert result == route and payload['feedback']['missing_facts'] == ['Ticket owner']
    assert payload['tools']['tickets']['filter_fields'] == ['id']
    assert 'user' not in payload


@pytest.mark.parametrize("call", [ToolCall(tool="missing", operation="lookup"),
    ToolCall(tool="crm", operation="count"), ToolCall(tool="crm", operation="lookup", filters={"invented": True})])
def test_unsupported_tools_fail_before_execution(call):
    tool = Tool(ToolResult())
    result = SearchAgent(Search([]), Critic([]), Generator(),
                         router=Router([Route(reason="Lookup", tool_calls=(call,))]),
                         tools={"crm": ToolSpec(tool, "hubspot", "CRM", ("lookup",), ("id",))}).run(request())
    assert result.trace.stop_reason == "error" and not tool.requests


def test_critic_feedback_can_switch_from_text_to_tool():
    feedback = decision(missing_facts=("Ticket status",))
    router = Router([Route(reason="Text", text_queries=("ticket",)),
                     Route(reason="Need structured status", tool_calls=(ToolCall(tool="tickets", operation="lookup", filters={"id": "T-42"}),))])
    tool = Tool(ToolResult(evidence=(evidence("b", source="jira"),)))
    result = SearchAgent(Search([(evidence(),)]), Critic([feedback, decision(True)]), Generator(), router=router,
        tools={"tickets": ToolSpec(tool, "jira", "Tickets", ("lookup",), ("id",))}).run(request())
    assert result.answer.status == "answered" and router.feedback == [None, feedback]


def test_count_without_citable_evidence_fails():
    tool = Tool(ToolResult(count=20))
    result = SearchAgent(Search([]), Critic([]), Generator(),
        router=Router([Route(reason="Count", tool_calls=(ToolCall(tool="crm", operation="count"),))]),
        tools={"crm": ToolSpec(tool, "hubspot", "CRM", ("count",), ())}).run(request())
    assert result.trace.stop_reason == "error"


def test_same_chunk_with_new_citation_is_not_new_evidence():
    original = evidence()
    alias = original.model_copy(update={"citation_id": "alias"})
    result = SearchAgent(Search([(original,), (alias,)]), Critic([decision()]), Generator()).run(request())
    assert result.trace.stop_reason == "no_new_evidence"
    assert len(result.sources) == 1


def test_unknown_date_is_rejected_for_dated_search():
    from datetime import datetime, timezone
    critic = Critic([])
    result = SearchAgent(Search([(evidence(),)]), critic, Generator()).run(
        request(after=datetime(2025, 1, 1, tzinfo=timezone.utc)))
    assert result.trace.stop_reason == "error" and not critic.requests


def test_router_cannot_expand_requested_sources():
    search = Search([])
    result = SearchAgent(search, Critic([]), Generator(), router=Router([
        Route(reason="Wrong source", text_queries=("limit",), source_types=("gmail",))
    ])).run(request(source_types=("docs",)))
    assert result.trace.stop_reason == "error" and not search.requests


def test_provider_error_usage_is_in_trace():
    critic = EvidenceCritic(FakeProvider([ProviderError(usage=Usage(api_calls=1, input_tokens=7))]))
    result = SearchAgent(Search([(evidence(),)]), critic, Generator()).run(request())
    assert result.trace.stop_reason == "error"
    assert result.trace.usage.api_calls == 1 and result.trace.usage.input_tokens == 7
    assert result.trace.usage.estimated_cost_usd is None


def test_new_evidence_gets_context_space_on_followup():
    critic = Critic([decision(), decision(True)])
    agent = SearchAgent(Search([(evidence(text="x" * 20),), (evidence("b", text="Limit: 10 MB"),)]),
                        critic, Generator(), config=AgentConfig(max_context_chars=20))
    result = agent.run(request())
    assert result.answer.status == "answered"
    assert [e.chunk.doc_id for e in critic.requests[-1].evidence] == ['b']


def test_unknown_generator_citation_is_rejected():
    class BadGenerator:
        def generate(self, request):
            return Answer(status="answered", text="10 MB", citation_ids=("invented",)), ZERO
    result = SearchAgent(Search([(evidence(),)]), Critic([decision(True)]), BadGenerator()).run(request())
    assert result.trace.stop_reason == "error" and result.answer.status == "error"


# Development pilot

def questions():
    return [dict(question_id=f'q{i}', question='What is the limit?', question_type=t,
                 expected_doc_ids=['a'], answer='GOLD MUST NOT REACH MODEL')
            for i, t in enumerate(['basic', 'basic', 'conflicting', 'multi'])]


def test_selection_is_stable_and_covers_categories():
    chosen = select_questions(questions(), 3)
    assert chosen == select_questions(list(reversed(questions())), 3)
    assert {q['question_type'] for q in chosen} == {'basic', 'conflicting', 'multi'}
    with pytest.raises(ValueError):
        select_questions(questions(), 9)


def test_explicit_development_sample_rejects_unknown_and_duplicate_ids():
    assert [q['question_id'] for q in development_sample(questions(), 1, ['q2', 'q0'])] == ['q2', 'q0']
    for ids in (['final-only-id'], ['q0', 'q0'], []):
        with pytest.raises(ValueError):
            development_sample(questions(), 1, ids)


def test_prepared_packet_keeps_references_out_of_retrieval_and_preserves_outputs(tmp_path):
    search = Search([(evidence(),)])
    search.doc_ids = ['a']
    output = tmp_path / 'packet'
    report = prepare_review(questions()[:1], search, output)
    case = json.loads((output/'case_01.json').read_text())
    assert report['answers_generated'] == 0 and not report['paid_calls']
    assert case['response'] is None and case['model_calls'] == []
    assert case['search']['evidence'][0]['chunk']['text'] == 'Upload limit is 10 MB.'
    assert 'GOLD MUST NOT' not in json.dumps(case)
    assert json.loads((output/'review.json').read_text())[0]['reference']['answer'] == 'GOLD MUST NOT REACH MODEL'
    with pytest.raises(FileExistsError):
        prepare_review(questions()[:1], search, output)


def test_saved_prompt_is_used_and_timing_breakdown_keeps_failures(tmp_path):
    from src.contracts import Trace, TraceStep
    provider = FakeProvider([
        Completion(decision(True).model_dump_json(), ZERO),
        Completion(Answer(status='answered', text='10 MB', citation_ids=('a',)).model_dump_json(), ZERO),
    ])
    search = Search([(evidence(),)])
    search.doc_ids = ['a']
    run_pilot(questions()[:1], search, provider, AgentConfig(), tmp_path/'run',
              critic_prompt='Saved pre-change critic instructions')
    assert provider.calls[0][0] == 'Saved pre-change critic instructions'
    assert provider.calls[1][0] != provider.calls[0][0]
    assert all('GOLD MUST NOT' not in prompt for _, prompt in provider.calls)
    trace = Trace(run_id='timing', stop_reason='error', steps=(
        TraceStep(round_number=1, query='limit?', tool='text_search', elapsed_ms=4),
        TraceStep(round_number=1, query='limit?', tool='critic:error', elapsed_ms=12),
        TraceStep(round_number=1, query='limit?', tool='generate', elapsed_ms=7)))
    assert timing_breakdown(trace) == dict(search=4, critic=12, generate=7, other=0)


def test_single_pass_skips_critic_and_preserves_permission_checks():
    search, critic, generator = Search([(evidence(),)]), Critic([]), Generator()
    response = SearchAgent(search, critic, generator).run(
        SearchRequest(query='limit?', user=USER), single_pass=True)
    assert response.trace.stop_reason == 'single_pass'
    assert len(search.requests) == 1 and critic.requests == []
    assert len(generator.requests) == 1
    denied = SearchAgent(Search([(evidence('secret'),)]), Critic([]), Generator()).run(
        SearchRequest(query='limit?', user=USER), single_pass=True)
    assert denied.answer.status == 'error'


def test_comparison_records_followup_and_identical_questions_without_gold_leak(tmp_path):
    provider = FakeProvider([
        Completion(Answer(status='answered', text='first', citation_ids=('a',)).model_dump_json(), ZERO),
        Completion(decision(False, missing_facts=('current limit',),
                            suggested_queries=('current limit',)).model_dump_json(), ZERO),
        Completion(decision(True).model_dump_json(), ZERO),
        Completion(Answer(status='answered', text='updated', citation_ids=('b',)).model_dump_json(), ZERO),
    ])
    search = Search([(evidence('a'),), (evidence('a'),), (evidence('b'),)])
    search.doc_ids = ['a', 'b']
    output = tmp_path / 'comparison'
    report = run_comparison(questions()[:1], search, provider, AgentConfig(max_rounds=2), output)
    assert report['paired_cases'] == 1
    pair = report['pairs'][0]
    assert pair['baseline']['searches'] == 1
    assert pair['agent']['searches'] == 2
    assert pair['agent']['new_followup_evidence_ids'] == ['b']
    assert [r.query for r in search.requests] == ['What is the limit?', 'What is the limit?', 'current limit']
    assert all('GOLD MUST NOT' not in prompt for _, prompt in provider.calls)
    assert (output/'baseline/review.json').exists() and (output/'agent/review.json').exists()
    with pytest.raises(FileExistsError):
        run_comparison(questions()[:1], search, provider, AgentConfig(), output)


def test_comparison_stops_before_second_arm_on_error(tmp_path):
    provider = FakeProvider([ProviderError(usage=Usage(api_calls=1))])
    search = Search([(evidence(),)])
    search.doc_ids = ['a']
    report = run_comparison(questions()[:1], search, provider, AgentConfig(), tmp_path/'run')
    assert report['paired_cases'] == 0 and 'agent' not in report['summaries']


def test_pilot_records_calls_evidence_reviews_and_no_gold_leak(tmp_path):
    usage = Usage(api_calls=1, input_tokens=50, output_tokens=20, estimated_cost_usd=0.001)
    provider = FakeProvider([
        Completion(CriticDecision(sufficient=True, confidence=0.99, reason='Supported').model_dump_json(), usage),
        Completion(json.dumps({'status': 'answered', 'text': '10 MB', 'citation_ids': ['a']}), usage),
    ])
    retriever = Search([(evidence(),)])
    retriever.doc_ids = ['a']
    output = tmp_path / 'run'
    summary = run_pilot(questions()[:1], retriever, provider, AgentConfig(max_cost_usd=0.1), output)
    assert summary['usage']['estimated_cost_usd'] == 0.002
    case = json.loads((output / 'case_01.json').read_text())
    assert len(case['model_calls']) == 2
    assert case['searches'][0]['evidence'][0]['chunk']['text'] == 'Upload limit is 10 MB.'
    assert all('GOLD MUST NOT' not in prompt for _, prompt in provider.calls)
    review = json.loads((output / 'review.json').read_text())
    assert review[0]['reference']['answer'] == 'GOLD MUST NOT REACH MODEL'
    assert review[0]['critic_correct'] is None
    with pytest.raises(FileExistsError):
        run_pilot(questions()[:1], retriever, provider, AgentConfig(), output)


def test_pilot_stops_after_provider_failure_and_saves_usage(tmp_path):
    provider = FakeProvider([ProviderError(usage=Usage(api_calls=1))])
    retriever = Search([(evidence(),)])
    retriever.doc_ids = ['a']
    summary = run_pilot(questions(), retriever, provider, AgentConfig(), tmp_path/'run')
    assert summary['completed'] == 1 and summary['requested'] == 4
    assert summary['usage']['api_calls'] == 1
    assert summary['usage']['estimated_cost_usd'] is None


def test_preflight_uses_real_small_index_without_model_calls(tmp_path, monkeypatch, capsys):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.retrieval.corpus_bm25 import CorpusBM25
    corpus = tmp_path/'documents.parquet'
    pq.write_table(pa.table({'doc_id': ['a'], 'title': ['Upload'],
                            'content': ['Upload limit is 10 MB.'], 'source_type': ['docs']}), corpus)
    index = tmp_path/'index'
    CorpusBM25.build(corpus).save(index)
    qpath, splits = tmp_path/'questions.jsonl', tmp_path/'splits.json'
    qpath.write_text(json.dumps(questions()[0]) + '\n')
    splits.write_text(json.dumps({'splits': {'development': ['q0'], 'final': ['q3']}}))
    monkeypatch.setattr('sys.argv', ['pilot', '--count', '1', '--corpus', str(corpus),
                                   '--index', str(index), '--questions', str(qpath), '--splits', str(splits)])
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    main()
    assert '"paid_calls": false' in capsys.readouterr().out
    output = tmp_path/'prepared'
    monkeypatch.setattr('sys.argv', ['pilot', '--prepare', '--question-ids', 'q0',
        '--corpus', str(corpus), '--index', str(index), '--questions', str(qpath),
        '--splits', str(splits), '--output', str(output)])
    main()
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['question_ids'] == ['q0'] and not manifest['paid_calls']
    assert manifest['critic_prompt_sha256']
    assert (output/'review.json').exists()


def test_pilot_rejects_empty_sample_before_creating_outputs(tmp_path):
    output = tmp_path / "empty-pilot"
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="at least one question"):
        run_pilot([], Search([]), provider, AgentConfig(), output)
    assert not output.exists() and not provider.calls


def test_context_budget_preserves_search_ranking():
    best = evidence("a", text="Best match")
    second = evidence("b", text="Less useful")
    critic = Critic([decision(True)])
    result = SearchAgent(Search([(best, second)]), critic, Generator(),
                         config=AgentConfig(max_context_chars=len(second.chunk.text))).run(request())
    assert result.answer.status == "answered"
    assert critic.requests[0].evidence == (best,)


def test_conflicting_chunk_identity_in_one_batch_is_rejected_atomically():
    first = evidence()
    conflicting = evidence("b", text="Different document").model_copy(
        update={"chunk": evidence("b", text="Different document").chunk.model_copy(
            update={"chunk_id": first.chunk.chunk_id})})
    critic = Critic([])
    result = SearchAgent(Search([(first, conflicting)]), critic, Generator()).run(request())
    assert result.trace.stop_reason == "error"
    assert not result.sources and not critic.requests


@pytest.mark.parametrize("bad_query", ["   ", "x" * 8001])
def test_entire_text_plan_is_validated_before_any_search(bad_query):
    search = Search([(evidence(),)])
    result = SearchAgent(search, Critic([]), Generator(), router=Router([
        Route(reason="Followup", text_queries=("valid query", bad_query))
    ])).run(request())
    assert result.trace.stop_reason == "error" and not search.requests


def test_text_router_ignores_out_of_scope_and_blank_critic_suggestions():
    search = Search([(evidence(),), (evidence("b"),)])
    critic = Critic([decision(suggested_queries=(" ", "upload size"),
                              suggested_sources=("gmail", "docs")), decision(True)])
    result = SearchAgent(search, critic, Generator()).run(request(source_types=("docs",)))
    assert result.answer.status == "answered"
    assert search.requests[-1].query == "upload size"
    assert search.requests[-1].source_types == ("docs",)


def test_positive_cost_budget_stops_at_exact_limit():
    generator = Generator()
    usage = Usage(api_calls=1, estimated_cost_usd=0.1)
    result = SearchAgent(Search([(evidence(),)]), Critic([decision(True)], usage), generator,
                         config=AgentConfig(max_cost_usd=0.1)).run(request())
    assert result.trace.stop_reason == "budget" and not generator.requests


def test_generator_abstention_is_not_traced_as_sufficient():
    class AbstainingGenerator:
        def generate(self, request):
            return Answer(status="insufficient_evidence", text="Conflicting policies."), ZERO
    result = SearchAgent(Search([(evidence(),)]), Critic([decision(True)]),
                         AbstainingGenerator()).run(request())
    assert result.answer.status == "insufficient_evidence"
    assert result.trace.stop_reason == "no_new_evidence"


def test_critic_revalidates_copied_authorization_before_model_call():
    provider = FakeProvider([])
    context = GenerationRequest(question="Limit?", user=USER).model_copy(
        update={"evidence": (evidence("secret"),)})
    with pytest.raises(ValueError, match="authorized"):
        EvidenceCritic(provider).assess(context)
    assert not provider.calls


def test_model_router_receives_date_constraints():
    from datetime import datetime, timezone
    after = datetime(2025, 1, 1, tzinfo=timezone.utc)
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    provider = FakeProvider([Completion(Route(reason="Dated search", text_queries=("policy",)).model_dump_json(), ZERO)])
    ModelRouter(provider).route(request(after=after, before=before), None, {})
    payload = json.loads(provider.calls[0][1])
    assert payload['after'] == after.isoformat() and payload['before'] == before.isoformat()


# Cooperating providers receive remaining allowances without changing interfaces.
def test_model_phases_receive_remaining_run_budget_and_scope_is_reset():
    from src.agents.budget import current_call_budget
    observed = []
    clock = [0.0]

    class BudgetRouter:
        def route(self, request, feedback, tools):
            budget = current_call_budget()
            observed.append((budget.remaining_tokens, budget.remaining_seconds, budget.remaining_cost_usd))
            clock[0] = 1
            return Route(reason="Search", text_queries=(request.query,)), Usage(
                input_tokens=2, output_tokens=1, estimated_cost_usd=0.01)

    class BudgetCritic:
        def assess(self, context):
            budget = current_call_budget()
            observed.append((budget.remaining_tokens, budget.remaining_seconds, budget.remaining_cost_usd))
            clock[0] = 3
            return decision(True), Usage(input_tokens=4, output_tokens=2, estimated_cost_usd=0.02)

    class BudgetGenerator(Generator):
        def generate(self, context):
            budget = current_call_budget()
            observed.append((budget.remaining_tokens, budget.remaining_seconds, budget.remaining_cost_usd))
            return super().generate(context)

    result = SearchAgent(Search([(evidence(),)]), BudgetCritic(), BudgetGenerator(),
                         router=BudgetRouter(), clock=lambda: clock[0],
                         config=AgentConfig(max_tokens=20, max_seconds=10, max_cost_usd=0.1)).run(request())
    assert result.trace.stop_reason == "sufficient"
    assert [item[:2] for item in observed] == [(20, 10), (17, 9), (11, 7)]
    assert [item[2] for item in observed] == pytest.approx([0.1, 0.09, 0.07])
    assert current_call_budget() is None


def test_provider_preflight_denial_is_budget_stop_and_preserves_usage():
    from src.agents.budget import RunBudgetExceeded, current_call_budget

    class DeniedCritic:
        def assess(self, context):
            assert current_call_budget().remaining_tokens == 10
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))

    generator = Generator()
    result = SearchAgent(Search([(evidence(),)]), DeniedCritic(), generator,
                         config=AgentConfig(max_tokens=10)).run(request())
    assert result.trace.stop_reason == "budget" and not generator.requests
    assert result.trace.usage.api_calls == 1
    assert result.trace.steps[-1].tool == "critic:budget"
    assert current_call_budget() is None


def test_slow_synchronous_retriever_stops_before_next_model_operation():
    clock = [0.0]

    class SlowSearch(Search):
        def search(self, request):
            clock[0] = 2
            return super().search(request)

    critic = Critic([])
    result = SearchAgent(SlowSearch([(evidence(),)]), critic, Generator(),
                         clock=lambda: clock[0], config=AgentConfig(max_seconds=1)).run(request())
    assert result.trace.stop_reason == "budget" and not critic.requests
    # A synchronous search cannot be forcibly interrupted; elapsed time is honest.
    assert result.trace.total_latency_ms == 2000


def test_call_budget_debits_multiple_attempts_and_restores_outer_scope():
    from src.agents.budget import CallBudget, call_budget, current_call_budget
    outer = CallBudget(100, 10, 1)
    inner = CallBudget(10, 2, 0.1)
    with call_budget(outer):
        with call_budget(inner):
            inner.consume(tokens=3, seconds=0.5, cost_usd=0.03)
            inner.consume(tokens=20, seconds=3, cost_usd=1)
            assert current_call_budget() == CallBudget(0, 0, 0)
        assert current_call_budget() is outer
    assert current_call_budget() is None


# Retrieval-to-agent integration (offline)



def test_hybrid_agent_scopes_reranker_critic_and_answer(tmp_path):
    corpus = pa.table({
        "doc_id": ["public", "denied", "other-source"],
        "source_type": ["docs", "docs", "email"],
        "title": ["", "", ""],
        "content": ["Upload limit is 10 MB.", "Secret upload limit is 100 MB.",
                    "Email upload limit is 20 MB."],
    })
    path = tmp_path / "corpus.parquet"
    pq.write_table(corpus, path)
    bm25 = CorpusBM25.build(path)
    bm25.corpus = corpus

    class Embeddings:
        def encode(self, texts, query=False):
            assert query
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    dense = CorpusDense(np.asarray([[0.8, 0.6], [1., 0.], [0.9, 0.4]], dtype=np.float32),
                        np.asarray([0, 1, 2]), corpus.column("doc_id").to_pylist(),
                        Embeddings(), {}, corpus=corpus,
                        spans=np.asarray([(0, len(text)) for text in corpus.column("content").to_pylist()]))
    scored = []

    def score(pairs):
        scored.extend(pairs)
        return [1.] * len(pairs)

    hybrid = HybridRetriever(bm25, dense, reranker=CrossEncoderReranker(
        "offline-test", "a" * 40, scorer=score))
    citation = "public:0:22"
    zero = Usage(estimated_cost_usd=0)
    provider = FakeProvider([
        Completion('{"sufficient":true,"confidence":0.95,"reason":"Supported"}', zero),
        Completion('{"status":"answered","text":"10 MB","citation_ids":["' + citation + '"]}', zero),
    ])
    request = SearchRequest(query="upload limit", source_types=("docs",),
                            user=UserContext(user_id="test", allowed_doc_ids=frozenset({"public", "other-source"})))
    response = SearchAgent(hybrid, EvidenceCritic(provider), GroundedGenerator(provider)).run(request)
    assert response.answer.status == "answered"
    assert response.answer.citation_ids == (citation,)
    assert {item.chunk.doc_id for item in response.sources} == {"public"}
    assert scored == [("upload limit", "Upload limit is 10 MB.")]
    assert all("Secret upload" not in prompt and "Email upload" not in prompt
               for _, prompt in provider.calls)
    assert response.trace.stop_reason == "sufficient"


@pytest.mark.parametrize("scope", [(), ("docs",)])
def test_descriptive_critic_sources_do_not_filter_out_followup_evidence(scope):
    search = Search([(evidence(),), (evidence("b"),), (evidence("b"),)])
    critic = Critic([decision(suggested_queries=("current upload policy",),
                              suggested_sources=("signed agreement",)), decision(True)])
    result = SearchAgent(search, critic, Generator()).run(request(source_types=scope))
    assert result.answer.status == "answered"
    assert all(r.user == USER and r.source_types == scope for r in search.requests)
    assert search.requests[1].query == "current upload policy"
    assert "b" in {e.citation_id for e in result.sources}


def test_router_preserves_valid_platform_and_uses_description_as_query():
    from src.agents.routing import TextRouter
    route, _ = TextRouter().route(request(), decision(
        suggested_sources=("gmail", "signed agreement")), {})
    assert route.source_types == ("gmail",)
    assert "signed agreement" in route.text_queries[0]


def test_router_out_of_scope_platform_cannot_replace_restricted_scope():
    from src.agents.routing import TextRouter
    route, _ = TextRouter().route(request(source_types=("docs",)), decision(
        suggested_queries=("specific policy",), suggested_sources=("gmail",)), {})
    assert route.source_types == ("docs",)
    assert route.text_queries == ("specific policy",)

@pytest.mark.parametrize('text', ['Retention is 90 days; billing party unknown.',
                                  'Policy A says 10 MB; policy B says 20 MB.'])
def test_partial_generation_gets_evidence_and_search_limitation(text):
    provider = FakeProvider([Completion(Answer(status='insufficient_evidence',
        text='Only the supplied facts are known; the remaining fact is unresolved.',
        citation_ids=('a',)).model_dump_json(), ZERO)])
    result = SearchAgent(Search([(evidence(text=text),)]), Critic([decision()]),
        GroundedGenerator(provider), config=AgentConfig(max_rounds=1)).run(request())
    payload = json.loads(provider.calls[0][1])
    assert payload['question'] == request().query
    assert payload['search_limitation'] and 'Do not resolve contradictions' in payload['response_requirement']
    assert payload['evidence'][0]['text'] == text
    assert result.answer.status == 'insufficient_evidence'
    assert result.trace.stop_reason == 'round_limit'
    assert result.answer.citation_ids == ('a',)
    assert result.trace.steps[-1].tool == 'generate'


def test_partial_generation_rejects_unknown_citation():
    class InvalidGenerator:
        def generate(self, context):
            return Answer(status='answered', text='Unsupported', citation_ids=('unknown',)), ZERO
    result = SearchAgent(Search([(evidence(),)]), Critic([decision()]),
        InvalidGenerator(), config=AgentConfig(max_rounds=1)).run(request())
    assert result.trace.stop_reason == 'error'


def test_partial_generation_cannot_bypass_token_budget():
    generator = Generator()
    critic = Critic([decision()], Usage(input_tokens=10, estimated_cost_usd=0))
    result = SearchAgent(Search([(evidence(),)]), critic, generator,
        config=AgentConfig(max_rounds=1, max_tokens=10)).run(request())
    assert result.trace.stop_reason == 'budget' and not generator.requests


def test_partial_context_shrinks_with_remaining_tokens_and_keeps_original_facts():
    first = evidence(text='A' * 3000)
    second = evidence('b', text='B' * 3000)
    generator = Generator()
    critic = Critic([decision(suggested_queries=('followup',)), decision()],
                    Usage(input_tokens=6000, estimated_cost_usd=0))
    result = SearchAgent(Search([(first,), (second,)]), critic, generator,
        config=AgentConfig(max_rounds=2, max_tokens=16000)).run(request())
    assert result.answer.status == 'insufficient_evidence'
    assert generator.requests[0].evidence[0] == first
    assert sum(len(e.chunk.text) for e in generator.requests[0].evidence) == 5000
    assert generator.requests[0].question == request().query


def test_last_critic_leaves_context_room_for_answer():
    first = evidence(text='A' * 6000)
    second = evidence('b', text='B' * 6000)
    critic = Critic([decision(suggested_queries=('followup',)), decision()],
                    Usage(input_tokens=3000, estimated_cost_usd=0))
    generator = Generator()
    result = SearchAgent(Search([(first,), (second,)]), critic, generator,
        config=AgentConfig(max_rounds=2, max_tokens=10000)).run(request())
    assert sum(len(e.chunk.text) for e in critic.requests[-1].evidence) == 4000
    assert result.answer.status == 'insufficient_evidence'
    assert generator.requests


def test_sufficient_followup_generation_preserves_initial_and_followup_passages():
    first = evidence(text='Upload limits and policy. ' * 150)
    followup = evidence('b', text='Upload limit exception and policy details. ' * 120)
    generator = Generator()
    result = SearchAgent(Search([(first,), (followup,)]),
        Critic([decision(suggested_queries=('upload exception',)), decision(True)]), generator,
        config=AgentConfig(max_rounds=2, max_context_chars=3000)).run(request())
    assert result.trace.stop_reason == 'sufficient'
    assert {e.chunk.doc_id for e in generator.requests[0].evidence} == {'a','b'}
    assert sum(len(e.chunk.text) for e in generator.requests[0].evidence) <= 3000

"""Typed native pipeline behavior with small fixtures and no provider calls."""

import hashlib
import json

import pytest

from src.access.contracts import AccessDenied, StaleContext, guarded_dispatch
from src.agents.context import ContextBudgetError, select_agent_context
from src.agents.routing import Route, ToolCall, ToolSpec
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import AgentError, EvidenceCritic
from src.contracts import (
    Answer, CriticDecision, FactAssessment, GenerationRequest, QueryResponse,
    SearchRequest, ToolResult, Usage,
)
from src.evaluation.experiment_contracts import (
    RuntimeCaseExport, ScoringInput, ScoringReference, runtime_case_digest,
    validate_runtime_exports,
)
from src.evaluation.experiment_io import ExperimentIOError, to_answer_evaluation_input
from src.generation.generator import (
    Completion, GenerationConfig, GroundedGenerator, evidence_chars, evidence_payload, select_context,
)
from tests.test_evaluation import _experiment_manifest
from tests.test_native_evidence_contracts import (
    CanonicalRecords, Registry, aggregate, boundary, passage, record,
)


ZERO = Usage(estimated_cost_usd=0)
CARDINALITY = "Counts canonical Jira records in this pinned benchmark; native keys may repeat."


class Provider:
    def __init__(self, output, *, before=None):
        self.output, self.before, self.calls = output, before, []
    def complete(self, system, prompt):
        if self.before:
            self.before()
        self.calls.append((system, json.loads(prompt)))
        return Completion(self.output.model_dump_json(), ZERO)


class Never:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            pytest.fail("Unexpected call: " + name)
        return fail


class Search:
    def __init__(self, *hits):
        self.hits, self.calls = hits, []
    def search(self, request):
        self.calls.append(request)
        return self.hits


class Router:
    def __init__(self, route):
        self.route_value, self.calls = route, 0
    def route(self, request, feedback, tools):
        self.calls += 1
        return self.route_value, ZERO


class Tool:
    def __init__(self, result, *, before=None):
        self.result, self.before, self.calls = result, before, []
    def execute(self, request):
        self.calls.append(request)
        if self.before:
            self.before()
        return self.result


def request(boundary, **changes):
    return SearchRequest(query="What do the accessible Jira records show?", user=boundary.user,
                         source_types=("jira",), **changes)


def native_run(boundary, result, *, generator=None, critic=None, retrieval_only=False,
               mixed=False, single_pass=True, config=None, operation=None, **changes):
    operation = operation or ("count" if result.count is not None else "lookup")
    router = Router(Route(reason="Use the registered Jira tool.",
        text_queries=("policy",) if mixed else (), tool_calls=(ToolCall(tool="jira", operation=operation),)))
    tool = Tool(result)
    search = Search(passage(boundary))
    native = tuple(result.records)
    artifacts = tuple(item for item in result.evidence if item.kind == "aggregate")
    agent = SearchAgent(search, critic or Never(), generator or Never(), router=router,
        tools={"jira": ToolSpec(tool, "jira", "Jira source records", ("lookup", "filter", "count"), ())},
        config=config or AgentConfig(max_rounds=1))
    response = agent.run(request(boundary), context=boundary.context, guard=boundary.guard,
        aggregates=Registry(artifacts[0]) if artifacts else None,
        record_validator=CanonicalRecords(*native), run_id="owned-runtime-run",
        retrieval_only=retrieval_only, single_pass=single_pass, **changes)
    return response, search, tool, router


@pytest.mark.parametrize("kind", ["record", "aggregate"])
def test_native_context_is_atomic_and_full_serialization_is_charged(boundary, kind):
    source = record(boundary) if kind == "record" else aggregate(boundary)
    with guarded_dispatch(boundary.context, boundary.guard, aggregates=Registry(aggregate(boundary)),
                          record_validator=CanonicalRecords(record(boundary))):
        supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
        size = evidence_chars(source)
        assert size == len(json.dumps(evidence_payload(source), ensure_ascii=False,
                                      separators=(',', ':'), allow_nan=False))
        assert select_context(supplied, size) == [source]
        assert select_context(supplied, size - 1) == []
        assert select_agent_context(supplied, size, (), (source,)) == [source]
        with pytest.raises(ContextBudgetError):
            select_agent_context(supplied, size - 1, (), (source,))


def test_mixed_followup_selection_retains_native_exactly_and_only_crops_passages(boundary):
    native, text = record(boundary), passage(boundary)
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(native)):
        supplied = GenerationRequest(question="Text", user=boundary.user, evidence=(text, native))
        selected = select_agent_context(supplied, evidence_chars(native) + 2, (text.citation_id,), (native,))
        assert native in selected and next(item for item in selected if item.kind == "record") == native
        assert sum(evidence_chars(item) for item in selected) <= evidence_chars(native) + 2
        assert all(not hasattr(item, "chunk") or len(item.chunk.text) <= 2 for item in selected)


@pytest.mark.parametrize("kind", ["record", "aggregate"])
def test_generator_captures_exact_selected_native_input_before_dispatch(boundary, kind):
    source = record(boundary) if kind == "record" else aggregate(boundary)
    events, captured = [], []
    provider = Provider(Answer(status="answered", text="Supported.", citation_ids=(source.citation_id,)),
                        before=lambda: events.append("provider"))
    def capture(selected, payload):
        events.append("capture")
        captured.append((selected, json.loads(json.dumps(payload))))
        payload["question"] = "Callback cannot change dispatched input"
    with guarded_dispatch(boundary.context, boundary.guard, aggregates=Registry(aggregate(boundary)),
                          record_validator=CanonicalRecords(record(boundary))):
        supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
        answer, _ = GroundedGenerator(provider, on_input=capture).generate(supplied)
    assert events == ["capture", "provider"]
    assert captured[0][0] == (source,) and captured[0][1] == provider.calls[0][1]
    prompt_source = provider.calls[0][1]["evidence"][0]
    assert prompt_source["kind"] == kind and "text" not in prompt_source and "chunk" not in prompt_source
    assert prompt_source["scope"] == boundary.binding.model_dump(mode="json")
    assert answer.citation_ids == (source.citation_id,)
    if kind == "aggregate":
        assert prompt_source["count"] == 0 and prompt_source["coverage"] == "verified_records"
        assert prompt_source["lineage_sha256"] == source.lineage_sha256
    else:
        assert prompt_source["field_provenance"] == [item.model_dump() for item in source.field_provenance]


def test_capture_uses_post_selection_cropped_passages_without_reconstructing_response(boundary):
    original = passage(boundary)
    text = "Text " * 40
    source = original.model_copy(update={"citation_id": "ticket:0:200", "chunk":
        original.chunk.model_copy(update={"chunk_id": "ticket:0:200", "text": text, "end_char": len(text)})})
    captured = []
    provider = Provider(Answer(status="insufficient_evidence", text="Small context."))
    with guarded_dispatch(boundary.context, boundary.guard):
        supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
        GroundedGenerator(provider, GenerationConfig(max_context_chars=40),
            on_input=lambda selected, payload: captured.append((selected, payload))).generate(supplied)
    assert len(captured[0][0][0].chunk.text) == 40
    assert captured[0][0][0].citation_id != source.citation_id
    assert captured[0][1] == provider.calls[0][1]


def test_revocation_inside_capture_stops_provider(boundary):
    source = record(boundary)
    provider = Provider(Answer(status="answered", text="Supported.", citation_ids=(source.citation_id,)))
    def revoke(*_):
        boundary.guard.revoked = True
    with pytest.raises(StaleContext):
        with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
            supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
            GroundedGenerator(provider, on_input=revoke).generate(supplied)
    assert provider.calls == []


def test_native_record_source_is_revalidated_after_provider_work(boundary):
    source = record(boundary)
    canonical = CanonicalRecords(source)
    provider = Provider(Answer(status="answered", text="Supported.", citation_ids=(source.citation_id,)),
                        before=lambda: canonical.expected.clear())
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=canonical):
        supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
        with pytest.raises(AccessDenied, match="authoritative"):
            GroundedGenerator(provider).generate(supplied)
    assert len(provider.calls) == 1


def test_failed_input_capture_cannot_be_reported_as_success_or_dispatch_provider(boundary):
    source = record(boundary)
    provider = Provider(Answer(status="answered", text="Supported.", citation_ids=(source.citation_id,)))
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        supplied = GenerationRequest(question="Question", user=boundary.user, evidence=(source,))
        with pytest.raises(ValueError, match="capture must return None"):
            GroundedGenerator(provider, on_input=lambda *_: False).generate(supplied)
    assert provider.calls == []


def test_native_critic_payload_and_citation_validation_only_allow_passage_read_more(boundary):
    source = record(boundary)
    verdict = CriticDecision(sufficient=True, confidence=1, reason="Native field answers the question.",
        facts=(FactAssessment(fact="Status", status="supported", citation_ids=(source.citation_id,)),))
    provider = Provider(verdict)
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        supplied = GenerationRequest(question="Status?", user=boundary.user, evidence=(source,))
        result, _ = EvidenceCritic(provider).assess(supplied)
        assert result.sufficient and provider.calls[0][1]["evidence"][0]["kind"] == "record"
        bad = Provider(verdict.model_copy(update={"read_more_doc_ids": (source.doc_id,)}))
        with pytest.raises(AgentError, match="outside"):
            EvidenceCritic(bad).assess(supplied)


@pytest.mark.parametrize("mixed", [False, True])
def test_zero_count_and_mixed_paths_use_authoritative_aggregate_and_owned_run(boundary, mixed):
    zero = aggregate(boundary)
    provider = Provider(Answer(status="answered", text="Zero accessible matching source records.",
                               citation_ids=(zero.citation_id,)))
    result = ToolResult(evidence=(zero,), count=0, coverage="verified_records", limitation=CARDINALITY)
    response, search, tool, _ = native_run(boundary, result, generator=GroundedGenerator(provider), mixed=mixed)
    assert response.answer.status == "answered" and response.trace.run_id == "owned-runtime-run"
    assert bool(search.calls) is mixed and len(tool.calls) == 1
    step = next(item for item in response.trace.steps if item.tool == "jira")
    assert json.loads(step.query)["result"] == {"count": 0, "coverage": "verified_records",
        "limitation": CARDINALITY, "truncated": False, "ambiguous": False}
    assert provider.calls[0][1]["evidence"][-1]["count"] == 0


def test_lookup_record_response_is_version_three_and_retrieval_only_skips_models(boundary):
    source = record(boundary)
    result = ToolResult(records=(source,), coverage="verified_records", limitation=CARDINALITY)
    response, _, _, router = native_run(boundary, result, retrieval_only=True)
    assert router.calls == 1 and response.sources == (source,) and response.contract_version == "3"
    assert response.answer.citation_ids == () and response.trace.stop_reason == "single_pass"
    assert not any(step.tool in {"critic", "generate"} for step in response.trace.steps)


@pytest.mark.parametrize("kind", ["record", "aggregate"])
def test_adaptive_native_path_critiques_and_generates_typed_evidence(boundary, kind):
    source = record(boundary) if kind == "record" else aggregate(boundary)
    tool_result = (ToolResult(records=(source,), coverage="verified_records", limitation=CARDINALITY)
        if kind == "record" else ToolResult(evidence=(source,), count=0,
                                           coverage="verified_records", limitation=CARDINALITY))
    decision = CriticDecision(sufficient=True, confidence=1, reason="Requested value is supported.",
        facts=(FactAssessment(fact="Requested value", status="supported", citation_ids=(source.citation_id,)),))
    critic_provider = Provider(decision)
    generator_provider = Provider(Answer(status="answered", text="Supported value.", citation_ids=(source.citation_id,)))
    response, _, _, _ = native_run(boundary, tool_result, single_pass=False,
        critic=EvidenceCritic(critic_provider), generator=GroundedGenerator(generator_provider))
    assert response.trace.stop_reason == "sufficient" and response.answer.status == "answered"
    assert [step.tool for step in response.trace.steps] == ["route", "jira", "critic", "generate"]
    assert critic_provider.calls[0][1]["evidence"][0]["kind"] == kind
    assert generator_provider.calls[0][1]["evidence"][0]["kind"] == kind


@pytest.mark.parametrize("flag,expected", [("ambiguous", "clarification"),
    ("truncated", "no_new_evidence"), ("unavailable", "no_new_evidence"),
    ("unsupported_limitation", "no_new_evidence")])
def test_incomplete_native_results_keep_flags_and_stop_before_generation(boundary, flag, expected):
    source = record(boundary)
    kwargs = {"records": (source,), "coverage": "verified_records", "limitation": CARDINALITY}
    if flag in {"ambiguous", "truncated"}:
        kwargs[flag] = True
    elif flag == "unavailable":
        kwargs = {"coverage": "unknown", "limitation": "Verified field coverage is unavailable."}
    else:
        kwargs["limitation"] = "Only part of this field is authoritative."
    response, _, _, _ = native_run(boundary, ToolResult(**kwargs))
    assert response.trace.stop_reason == expected and response.answer.status == "insufficient_evidence"
    metadata = json.loads(next(step.query for step in response.trace.steps if step.tool == "jira"))["result"]
    assert metadata["limitation"] == kwargs["limitation"]
    if flag in {"ambiguous", "truncated"}:
        assert metadata[flag] is True and response.sources == (source,)


def test_denied_record_validation_never_reaches_generator_or_returns_partial_evidence(boundary):
    source = record(boundary)
    result = ToolResult(records=(source,), coverage="verified_records")
    tool = Tool(result)
    route = Router(Route(reason="Lookup", tool_calls=(ToolCall(tool="jira", operation="lookup"),)))
    def deny(_):
        raise AccessDenied("Record is no longer authorized")
    agent = SearchAgent(Search(), Never(), Never(), router=route,
        tools={"jira": ToolSpec(tool, "jira", "Jira", ("lookup",), ())})
    with pytest.raises(AccessDenied):
        agent.run(request(boundary), context=boundary.context, guard=boundary.guard, record_validator=deny)


def test_native_export_scope_and_legacy_judge_adapter_do_not_fabricate_passages(boundary):
    source = record(boundary)
    response = QueryResponse(snapshot=boundary.snapshot,
        answer=Answer(status="answered", text="Open.", citation_ids=(source.citation_id,)), sources=(source,),
        trace={"run_id": "native-run", "stop_reason": "single_pass"})
    question = "What is the status?"
    case = RuntimeCaseExport(manifest_id="fixed", run_id="native-run", question_id="q1", question=question,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(), response=response, generation_evidence=(source,))
    manifest = _experiment_manifest(scope=boundary.binding,
        retrieval=_experiment_manifest().retrieval.model_copy(update={"snapshot": boundary.snapshot}))
    validate_runtime_exports(manifest, (case,))
    assert RuntimeCaseExport.model_validate_json(case.model_dump_json()) == case
    other = manifest.model_copy(update={"scope": boundary.binding.model_copy(update={"scope_id": "different"})})
    with pytest.raises(ValueError, match="permission scope"):
        validate_runtime_exports(other, (case,))
    scoring = ScoringInput(runtime_case=case, runtime_case_sha256=runtime_case_digest(case),
        reference=ScoringReference(question_id="q1", question_sha256=case.question_sha256,
            reference_set_sha256="a" * 64, reference_answer="Open."))
    with pytest.raises(ExperimentIOError, match="record"):
        to_answer_evaluation_input(scoring)

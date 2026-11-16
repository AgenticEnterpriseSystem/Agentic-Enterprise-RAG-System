"""Offline stage-cap/context-reserve simulations; no tokenizer or live-quality claims.

The deterministic oracle charges evidence twice and the critic assessment again
as generator input. It is deliberately explicit, not Anthropic token counting.
"""
import json
import math
import socket
import subprocess

import pytest

from src.agents.budget import current_call_budget, current_output_token_limit, RunBudgetExceeded
from src.agents.routing import Route
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, AgentError
from src.contracts import Chunk, Evidence, SearchRequest, Usage, UserContext
from src.generation.generator import Completion, GenerationConfig, GroundedGenerator
from src.observability import Recorder, observation_context


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("Stage reserve checks forbid network, services and models")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(subprocess, "run", denied)


def passage(doc, size):
    text = ("Documented policy facts and exceptions. " * (size // 30 + 1))[:size]
    cid = f"{doc}:0:{len(text)}"
    return Evidence(citation_id=cid, score=1, retrieval_method="hybrid", chunk=Chunk(
        chunk_id=cid, doc_id=doc, source_type="confluence", title="Synthetic policy", text=text,
        start_char=0, end_char=len(text)))


class Router:
    def __init__(self): self.limits = []
    def route(self, request, feedback, tools):
        self.limits.append(current_output_token_limit())
        # First route leaves14606 tokens, giving7558 context characters with
        # the explicit7048 reserve. No real routing/provider request is made.
        tokens = (1200, 194) if len(self.limits) == 1 else (80, 20)
        return Route(reason="Synthetic route", text_queries=("follow-up" if feedback else request.query,)), Usage(
            input_tokens=tokens[0], output_tokens=tokens[1], api_calls=0, estimated_cost_usd=0)


class Search:
    def __init__(self, rounds): self.rounds, self.requests = iter(rounds), []
    def search(self, request): self.requests.append(request); return next(self.rounds)


class MultipartProvider:
    supports_structured_output = True
    def __init__(self, *, two_rounds=False, first_input_override=None):
        self.calls, self.critics = [], 0
        self.two_rounds, self.first_input_override = two_rounds, first_input_override

    def complete_structured(self, _system, prompt, schema):
        payload = json.loads(prompt)
        answer = "status" in schema["properties"]
        if not answer: self.critics += 1
        first_partial = self.two_rounds and self.critics == 1 and not answer
        cid = payload["evidence"][0]["citation_id"]
        if answer:
            result = {"status": "answered", "text": "The supplied policy covers the requested details.", "citation_ids": [cid]}
        else:
            facts = []
            for i in range(10):
                supported = not first_partial or i == 0
                facts.append({"fact": f"Requested policy field {i}", "status": "supported" if supported else "missing",
                    "citation_ids": [cid] if supported else [], "claims": [{"citation_id": cid,
                        "statement": f"Field {i} has the documented value and a specified applicable policy exception.",
                        "scope_reason": "The same requested entity, purpose and documented policy scope.",
                        "answers_requested_fact": True}] if supported else []})
            result = {"facts": facts, "suggested_queries": ["follow-up"] if first_partial else []}
        raw = json.dumps(result)
        evidence_chars = sum(len(item["text"]) for item in payload["evidence"])
        assessment_tokens = math.ceil(len(json.dumps(payload.get("evidence_assessment", {}))) / 2) if answer else 0
        input_tokens = 900 + math.ceil(evidence_chars / 2) + assessment_tokens
        if not answer and self.critics == 1 and self.first_input_override is not None:
            input_tokens = self.first_input_override
        required_output = 250 if answer else math.ceil(len(raw) / 3)
        budget = current_call_budget(); limit = current_output_token_limit() or 1024
        assert budget is not None
        allowed_output = min(limit, budget.remaining_tokens - input_tokens)
        if allowed_output < 1:
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))
        output = min(required_output, allowed_output)
        cost = (input_tokens + output * 5) / 1_000_000
        if cost > budget.remaining_cost_usd:
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))
        self.calls.append({"kind": "generate" if answer else "critic", "payload": payload,
            "chars": evidence_chars, "input_tokens": input_tokens, "required_output": required_output,
            "output_tokens": output, "output_limit": limit, "remaining_tokens": budget.remaining_tokens,
            "assessment_tokens": assessment_tokens})
        budget.consume(tokens=input_tokens + output, cost_usd=cost)
        truncated = output < required_output
        return Completion(raw, Usage(input_tokens=input_tokens, output_tokens=output, api_calls=2,
            estimated_cost_usd=cost), truncated=truncated, stop_reason="max_tokens" if truncated else "end_turn")


def execute(rounds, *, critic_output=2048, generation_output=1024, provider=None, **config):
    pool = tuple(item for items in rounds for item in items)
    request = SearchRequest(query="Describe the policy and all ten requested limits and exceptions.",
        user=UserContext(user_id="synthetic", allowed_doc_ids=frozenset(item.chunk.doc_id for item in pool)), top_k=8)
    provider = provider or MultipartProvider()
    router, search = Router(), Search(rounds)
    agent = SearchAgent(search, EvidenceCritic(provider, max_output_tokens=critic_output),
        GroundedGenerator(provider, GenerationConfig(retries=0)), router=router,
        config=AgentConfig(max_tokens=16000, max_context_chars=24000, max_seconds=120, max_cost_usd=.10,
            evidence_doc_limit=8, critic_max_output_tokens=critic_output,
            generation_max_output_tokens=generation_output, **config))
    return agent.run(request), provider, router, search


def test_multipart_critic_over1024_completes_and_identical_frozen_context_funds_final_answer():
    items = tuple(passage(f"doc-{i}", 1206 if i == 0 else 1200) for i in range(8))
    response, provider, router, _ = execute([items])
    assert response.answer.status == "answered"
    critic, generator = provider.calls
    assert critic["required_output"] > 1024 and critic["output_tokens"] == critic["required_output"] <= 2048
    assert critic["output_limit"] == 2048 and generator["output_limit"] == 1024
    assert router.limits == [1024]
    assert critic["chars"] == generator["chars"] == 7558
    assert critic["payload"]["evidence"] == generator["payload"]["evidence"]
    assert generator["assessment_tokens"] > 0
    assert all(s.selected_documents <= 8 for s in response.trace.context_selection)
    assert response.trace.usage.input_tokens + response.trace.usage.output_tokens <= 16000
    assert response.trace.usage.estimated_cost_usd <= .10
    assert current_output_token_limit() is None and current_call_budget() is None


def test_two_rounds_keep_exact_supported_passage_then_generate_from_final_assessed_context():
    first = passage("initial-supported", 500)
    newer = tuple(passage(f"new-{i}", 500) for i in range(7))
    response, provider, router, search = execute([(first,), newer], provider=MultipartProvider(two_rounds=True), max_rounds=2)
    assert response.answer.status == "answered"
    assert [c["kind"] for c in provider.calls] == ["critic", "critic", "generate"]
    assert [r.query for r in search.requests] == ["Describe the policy and all ten requested limits and exceptions.", "follow-up"]
    final_critic, generator = provider.calls[1:]
    retained = next(item for item in final_critic["payload"]["evidence"] if item["citation_id"] == first.citation_id)
    assert retained["text"] == first.chunk.text
    assert final_critic["payload"]["evidence"] == generator["payload"]["evidence"]
    assert router.limits == [1024, 1024]
    assert len({item["doc_id"] for item in generator["payload"]["evidence"]}) <= 8
    assert all(s.selected_documents <= 8 for s in response.trace.context_selection)
    assert response.trace.usage.input_tokens + response.trace.usage.output_tokens <= 16000
    assert response.trace.usage.estimated_cost_usd <= .10


def test_supported_retained_text_that_cannot_fit_new_reserve_stops_without_unassessed_generation():
    provider = MultipartProvider(two_rounds=True, first_input_override=10000)
    first = passage("retained", 1200)
    response, provider, _router, _search = execute([(first,), (passage("new", 1200),)], provider=provider, max_rounds=2)
    assert response.trace.stop_reason == "budget" and response.answer.status == "insufficient_evidence"
    assert [c["kind"] for c in provider.calls] == ["critic"]
    assert response.trace.context_selection[-1].selected_documents == 0
    assert response.trace.context_selection[-1].exclusion_reasons
    assert response.trace.usage.input_tokens + response.trace.usage.output_tokens <= 16000


def test_explicit1024_critic_truncation_retains_usage_and_fixed_error_classification():
    recorder = Recorder(log=False); observed = recorder.start_request()
    items = tuple(passage(f"doc-{i}", 1200) for i in range(8))
    try:
        with observation_context(observed):
            response, provider, _router, _search = execute([items], critic_output=1024)
        assert response.answer.status == "error" and response.trace.stop_reason == "error"
        assert len(provider.calls) == 1 and provider.calls[0]["output_tokens"] == 1024
        detail = recorder.get(observed.request_id)
        errors = [s for s in detail["stages"] if s["stage"] == "critic" and s["event"] == "error"]
        assert errors and errors[-1]["error_code"] == "truncated_output"
        assert errors[-1]["provider_stop_reason"] == "max_tokens"
        assert response.trace.usage.output_tokens >= 1024
        assert current_output_token_limit() is None and current_call_budget() is None
    finally:
        recorder.close()


@pytest.mark.parametrize("field", ["critic_max_output_tokens", "generation_max_output_tokens"])
@pytest.mark.parametrize("value", [True, False, 0, -1, "2048", 1.5, 8193])
def test_stage_output_configuration_rejects_noninteger_or_outofrange_caps(field, value):
    with pytest.raises(ValueError):
        AgentConfig(**{field: value})

"""Offline adaptive context-reserve regressions with native scoped contracts.

The provider below enforces an explicit deterministic token-cost oracle. It is
not Anthropic's tokenizer and makes no claim about the cause of historical live
failures. No provider client, model, service or network request is constructed.
"""

import json
import math
import socket
import subprocess

import pytest

from src.access.contracts import AccessError, SessionExpired, assert_dispatch_current
from src.agents.budget import RunBudgetExceeded, current_call_budget
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic
from src.contracts import (Answer, Chunk, CriticDecision, EvidenceClaim, FactAssessment,
    PassageEvidence, SearchRequest, Usage)
from src.generation.generator import Completion, GenerationConfig, GroundedGenerator
from tests.test_scoped_agent import scope


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("Context-reserve regression forbids services, models and network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(subprocess, "run", denied)


def window(scope, ordinal=0, *, size=3000, doc_id="allowed"):
    text = (f"Policy window {ordinal} contains documented details. " * 100)[:size]
    start = ordinal * 3000; end = start + len(text)
    cid = f"{doc_id}:{start}:{end}"
    return PassageEvidence(citation_id=cid, snapshot_id=scope.context.binding.snapshot.snapshot_id,
        revision_id="synthetic-canonical-revision", retrieval_method="hybrid", score=1 / (ordinal + 1),
        chunk=Chunk(chunk_id=cid, doc_id=doc_id, source_type="confluence", title=f"Policy window {ordinal}",
            text=text, start_char=start, end_char=end))


class Search:
    def __init__(self, rounds):
        self.rounds, self.requests = iter(rounds), []

    def search(self, request):
        self.requests.append(request)
        return next(self.rounds)


class BudgetProvider:
    supports_structured_output = True

    def __init__(self, statuses=("supported",), *, on_generation=None):
        self.statuses, self.calls = iter(statuses), []
        self.on_generation = on_generation

    def complete_structured(self, system, prompt, schema):
        assert_dispatch_current()
        payload = json.loads(prompt)
        is_answer = "status" in schema["properties"]
        if is_answer and self.on_generation:
            self.on_generation(); assert_dispatch_current()
        allowance = current_call_budget()
        assert allowance is not None
        chars = sum(len(item["text"]) for item in payload["evidence"])
        # Explicit deterministic oracle, not a surrogate measurement of Haiku.
        # Both operations charge their input evidence; generation additionally
        # charges room for the native fact assessment passed to it.
        input_tokens = 1000 + math.ceil(chars / 2) + (1000 if is_answer else 0)
        limit = min(1024, allowance.remaining_tokens - input_tokens)
        call = {"kind": "answer" if is_answer else "critic", "payload": payload,
            "evidence_chars": chars, "input_tokens": input_tokens,
            "remaining_tokens": allowance.remaining_tokens, "admitted": False}
        self.calls.append(call)
        if limit < 1:
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))
        output_tokens = min(200, limit)
        cost = (input_tokens + output_tokens * 5) / 1_000_000
        if cost > allowance.remaining_cost_usd:
            raise RunBudgetExceeded(usage=Usage(api_calls=1, estimated_cost_usd=0))
        allowance.consume(tokens=input_tokens + output_tokens, cost_usd=cost)
        call["admitted"] = True
        cid = payload["evidence"][0]["citation_id"]
        if is_answer:
            result = {"status": "answered", "text": "The supplied policy documents the requested detail.", "citation_ids": [cid]}
        else:
            status = next(self.statuses)
            result = {"facts": [{"fact": "Requested policy detail", "status": status,
                "citation_ids": [cid] if status == "supported" else [],
                "claims": [{"citation_id": cid, "statement": "The policy documents this detail.",
                    "scope_reason": "The requested policy detail", "answers_requested_fact": True}]
                    if status == "supported" else []}],
                "suggested_queries": ["follow-up policy detail"] if status == "missing" else []}
        return Completion(json.dumps(result), Usage(input_tokens=input_tokens,
            output_tokens=output_tokens, api_calls=2, estimated_cost_usd=cost))


def scenario(scope, *, max_rounds=2, single_pass=False, agent_class=SearchAgent):
    provider = BudgetProvider()
    hits = tuple(window(scope, i) for i in range(8))
    search = Search([hits])
    agent = agent_class(search, EvidenceCritic(provider),
        GroundedGenerator(provider, GenerationConfig(retries=0)),
        config=AgentConfig(max_rounds=max_rounds, max_tokens=16000,
            max_context_chars=24000, max_seconds=120, max_cost_usd=.05))
    response = agent.run(SearchRequest(query="What policy detail is documented?", user=scope.user,
        source_types=("confluence",)), single_pass=single_pass, context=scope.context, guard=scope.guard)
    return response, provider, search


@pytest.mark.parametrize("max_rounds", [2, 3])
def test_large_early_sufficient_context_leaves_budget_for_native_answer(scope, max_rounds):
    response, provider, search = scenario(scope, max_rounds=max_rounds)
    assert response.answer.status == "answered" and response.trace.stop_reason == "sufficient"
    assert [step.tool for step in response.trace.steps] == ["route", "text_search", "critic", "generate"]
    assert all(step.round_number == 1 for step in response.trace.steps)
    assert [call["kind"] for call in provider.calls] == ["critic", "answer"]
    critic, answer = provider.calls
    assert critic["admitted"] and answer["admitted"]
    assert critic["evidence_chars"] == answer["evidence_chars"] == 11000
    assert critic["payload"]["evidence"] == answer["payload"]["evidence"]
    # Initial-only retrieval ranking is stable; the fourth window is bounded
    # before approval and its exact canonical offset is used in both prompts.
    assert [item["citation_id"] for item in critic["payload"]["evidence"]] == [
        "allowed:0:3000", "allowed:3000:6000", "allowed:6000:9000", "allowed:9000:11000"]
    assert set(response.answer.citation_ids) <= {item.citation_id for item in response.sources}
    assert response.snapshot == scope.context.binding.snapshot
    assert all(item.snapshot_id == response.snapshot.snapshot_id for item in response.sources)
    assert all(request.user is scope.user and request.source_types == ("confluence",) for request in search.requests)
    assert response.trace.usage.input_tokens + response.trace.usage.output_tokens <= 16000
    assert response.trace.usage.estimated_cost_usd <= .05


def test_last_round_keeps_existing_reserve_behavior(scope):
    response, provider, _search = scenario(scope, max_rounds=1)
    assert response.answer.status == "answered"
    assert [call["evidence_chars"] for call in provider.calls] == [11000, 11000]
    assert provider.calls[0]["payload"]["evidence"] == provider.calls[1]["payload"]["evidence"]


def test_single_pass_does_not_add_a_critic_or_shrink_baseline_context(scope):
    response, provider, _search = scenario(scope, single_pass=True)
    assert response.answer.status == "answered" and response.trace.stop_reason == "single_pass"
    assert [step.tool for step in response.trace.steps] == ["route", "text_search", "generate"]
    assert len(provider.calls) == 1 and provider.calls[0]["kind"] == "answer"
    assert provider.calls[0]["evidence_chars"] == 24000


def test_insufficient_first_critic_still_reformulates_and_researches(scope):
    provider = BudgetProvider(("missing", "supported"))
    search = Search([(window(scope, 0, size=500),), (window(scope, 1, size=500),)])
    response = SearchAgent(search, EvidenceCritic(provider), GroundedGenerator(provider),
        config=AgentConfig(max_rounds=3, max_tokens=16000, max_seconds=120, max_cost_usd=.05)).run(
            SearchRequest(query="policy", user=scope.user, source_types=("confluence",)),
            context=scope.context, guard=scope.guard)
    assert response.answer.status == "answered"
    assert [request.query for request in search.requests] == ["policy", "follow-up policy detail"]
    decisions = [step.critic for step in response.trace.steps if step.tool == "critic"]
    assert [decision.sufficient for decision in decisions] == [False, True]
    assert decisions[0].missing_facts == ("Requested policy detail",)
    assert [call["kind"] for call in provider.calls] == ["critic", "critic", "answer"]
    assert provider.calls[1]["payload"]["evidence"] == provider.calls[2]["payload"]["evidence"]
    assert all(request.user is scope.user for request in search.requests)


def test_supported_passages_that_no_longer_fit_stop_safely_without_dropping_facts(scope):
    class Critic:
        calls = 0
        def assess(self, request):
            self.calls += 1
            cid = request.evidence[0].citation_id
            return CriticDecision(sufficient=False, confidence=0, reason="One fact still missing",
                facts=(FactAssessment(fact="Supported fact", status="supported", citation_ids=(cid,),
                    claims=(EvidenceClaim(citation_id=cid, statement="Supported detail", scope_reason="Requested fact",
                        answers_requested_fact=True),)), FactAssessment(fact="Missing fact", status="missing")),
                missing_facts=("Missing fact",), suggested_queries=("follow-up policy",)), Usage(
                    input_tokens=8500, output_tokens=100, estimated_cost_usd=0)
    class NeverGenerator:
        def generate(self, _request): pytest.fail("An unreviewed/dropped supported fact must not reach generation")
    critic = Critic()
    response = SearchAgent(Search([(window(scope),), (window(scope, 1),)]), critic, NeverGenerator(),
        config=AgentConfig(max_rounds=3, max_tokens=16000, max_seconds=120, max_cost_usd=.05)).run(
            SearchRequest(query="policy", user=scope.user), context=scope.context, guard=scope.guard)
    assert critic.calls == 1 and response.answer.status == "insufficient_evidence"
    assert response.trace.stop_reason == "budget"
    assert response.answer.text == "Supported evidence no longer fits the remaining context budget."


def test_denied_evidence_cannot_reach_new_context_selection_or_provider(scope):
    provider = BudgetProvider()
    agent = SearchAgent(Search([(window(scope, doc_id="executive"),)]), EvidenceCritic(provider), GroundedGenerator(provider),
        config=AgentConfig(max_rounds=2, max_tokens=16000, max_cost_usd=.05))
    with pytest.raises(AccessError):
        agent.run(SearchRequest(query="policy", user=scope.user), context=scope.context, guard=scope.guard)
    assert provider.calls == []


def test_revocation_between_critic_and_generation_still_aborts(scope):
    def revoke(): scope.guard.error = SessionExpired("Synthetic expiry")
    provider = BudgetProvider(on_generation=revoke)
    agent = SearchAgent(Search([tuple(window(scope, i) for i in range(8))]), EvidenceCritic(provider), GroundedGenerator(provider),
        config=AgentConfig(max_rounds=2, max_tokens=16000, max_cost_usd=.05))
    with pytest.raises(SessionExpired):
        agent.run(SearchRequest(query="policy", user=scope.user), context=scope.context, guard=scope.guard)
    assert len(provider.calls) == 1 and provider.calls[0]["kind"] == "critic"

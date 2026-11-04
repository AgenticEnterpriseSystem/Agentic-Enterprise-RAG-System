"""Scoped agent/dispatch boundaries with small metadata and scripted providers."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.access.contracts import (
    AccessDenied, AccessError, LegacySetScope, RequestContext, ResolvedIdentity,
    ScopedUserContext, SessionExpired, StaleContext, assert_dispatch_current,
    guarded_dispatch,
)
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import AgentError, EvidenceCritic
from src.contracts import (
    Answer, Chunk, CriticDecision, Evidence, EvidenceClaim, FactAssessment,
    GenerationRequest, PassageEvidence, ScopeBinding, SearchRequest, SnapshotRef,
    Usage, UserContext,
)
from src.generation.generator import AnthropicConfig, AnthropicProvider, GroundedGenerator


@pytest.fixture
def scope():
    snapshot = SnapshotRef(snapshot_id="scoped-agent", corpus_sha256="a" * 64,
                           catalog_sha256="b" * 64, document_count=2)
    identity = ResolvedIdentity(user_id="intern", session_id="opaque-row-id", session_version=1,
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc))
    permission = LegacySetScope(binding=ScopeBinding(snapshot=snapshot, scope_id="server-scope",
        user_id=identity.user_id, session_version=1, policy_revision=2), allowed_doc_ids=frozenset({"allowed"}))
    context = RequestContext(identity, permission)
    class Guard:
        error = None
        calls = 0
        def assert_current(self, current):
            assert current is context
            self.calls += 1
            if self.error:
                raise self.error
    return SimpleNamespace(context=context, guard=Guard(), user=ScopedUserContext.from_context(context))


def passage(scope, *, doc_id="allowed", text="Policy facts. " * 20, snapshot_id=None):
    key = f"{doc_id}:0:{len(text)}"
    return PassageEvidence(citation_id=key, snapshot_id=snapshot_id or scope.context.binding.snapshot.snapshot_id,
        revision_id="canonical-revision", score=1, retrieval_method="hybrid",
        chunk=Chunk(chunk_id=key, doc_id=doc_id, source_type="confluence", text=text,
                    start_char=0, end_char=len(text)))


class Retriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = 0
    def search(self, request):
        self.calls += 1
        return self.hits


class Generator:
    def __init__(self):
        self.requests = []
    def generate(self, request):
        self.requests.append(request)
        return Answer(status="answered", text="Supported policy.",
                      citation_ids=(request.evidence[0].citation_id,)), Usage(estimated_cost_usd=0)


class NeverCritic:
    def assess(self, request):
        pytest.fail("Baseline must not invoke the critic")


def search(scope, *, user=None, **changes):
    return SearchRequest(query="policy", user=user or scope.user, source_types=("confluence",), **changes)


def test_bridge_stays_private_nested_and_copied_or_deserialized_users_deny(scope):
    scope.user.require_context(scope.context)
    assert scope.user.permits("allowed") and not scope.user.permits("executive")
    assert scope.user.allowed_doc_ids == frozenset()
    request = search(scope)
    generation = GenerationRequest(question="policy", user=request.user, evidence=(passage(scope),))
    assert request.user is scope.user and generation.user is scope.user
    assert request.model_copy(update={"query": "changed"}).user is scope.user
    dumped = request.model_dump_json(serialize_as_any=True)
    assert all(value not in dumped for value in ("opaque-row-id", "server-scope", "_seal", "bitmap", "catalog_sha256"))
    copies = [scope.user.model_copy(), scope.user.model_copy(update={"user_id": "executive",
              "allowed_doc_ids": frozenset({"executive"})}), deepcopy(scope.user),
              ScopedUserContext.model_validate(scope.user.model_dump()),
              UserContext.model_validate(scope.user.model_dump()),
              SearchRequest.model_validate(request.model_dump()).user]
    for copied in copies:
        assert not copied.permits("allowed") and not copied.permits("executive")
        with pytest.raises(AccessDenied):
            ScopedUserContext.require_context(copied, scope.context)


def test_dispatch_context_resets_on_failure_and_does_not_authorize_a_scoped_user_outside_it(scope):
    assert_dispatch_current()  # Legacy unscoped behavior.
    with pytest.raises(RuntimeError):
        with guarded_dispatch(scope.context, scope.guard):
            assert_dispatch_current(scope.user)
            raise RuntimeError("scripted failure")
    assert_dispatch_current()
    with pytest.raises(AccessDenied):
        assert_dispatch_current(scope.user)
    with guarded_dispatch(scope.context, scope.guard):
        with pytest.raises(AccessDenied):
            assert_dispatch_current(UserContext(user_id="intern", allowed_doc_ids=frozenset({"allowed"})))


def test_scoped_baseline_keeps_snapshot_tags_and_cropped_trace_citations(scope):
    generator = Generator()
    agent = SearchAgent(Retriever((passage(scope),)), NeverCritic(), generator,
                        config=AgentConfig(max_context_chars=40))
    response = agent.run(search(scope), single_pass=True, context=scope.context, guard=scope.guard)
    assert response.snapshot == scope.context.binding.snapshot and response.contract_version == "2"
    assert isinstance(response.sources[0], PassageEvidence)
    assert len(response.sources[0].chunk.text) == 40
    assert generator.requests[0].user is scope.user
    ids = {value.citation_id for value in response.sources}
    assert set(response.trace.steps[1].evidence_ids) - ids  # Earlier full excerpt is a valid trace source.
    assert response.model_dump(mode="json")["sources"][0]["kind"] == "passage"


@pytest.mark.parametrize("bad", ["denied", "snapshot", "copied_user", "legacy_evidence"])
def test_untrusted_scope_or_evidence_never_reaches_critic_or_generator(scope, bad):
    generator = Generator()
    hit = passage(scope, doc_id="executive") if bad == "denied" else passage(scope,
        snapshot_id="another-snapshot" if bad == "snapshot" else None)
    if bad == "legacy_evidence":
        hit = Evidence.model_validate(hit.model_dump(exclude={"kind", "snapshot_id", "revision_id"}))
    user = scope.user.model_copy() if bad == "copied_user" else scope.user
    agent = SearchAgent(Retriever((hit,)), NeverCritic(), generator)
    with pytest.raises(AccessError):
        agent.run(search(scope, user=user), single_pass=True, context=scope.context, guard=scope.guard)
    assert not generator.requests


@pytest.mark.parametrize("failure", ["return", "raise"])
def test_revocation_during_retrieval_propagates_instead_of_returning_partial_sources(scope, failure):
    class Revoking(Retriever):
        def search(self, request):
            scope.guard.error = SessionExpired("Session is unavailable")
            if failure == "raise":
                raise AgentError("sensitive adapter failure")
            return self.hits
    generator = Generator()
    agent = SearchAgent(Revoking((passage(scope),)), NeverCritic(), generator)
    with pytest.raises(SessionExpired):
        agent.run(search(scope), single_pass=True, context=scope.context, guard=scope.guard)
    assert not generator.requests


@pytest.mark.parametrize("bad", ["fact", "claim", "read_more"])
def test_invalid_critic_citations_are_not_retained_in_error_trace_and_usage_is_kept(scope, bad):
    hit = passage(scope)
    leaked = "executive-secret-id"
    class Critic:
        def assess(self, request):
            decision = CriticDecision(sufficient=False, confidence=0, reason="Scripted decision",
                read_more_doc_ids=(leaked,) if bad == "read_more" else (),
                facts=(FactAssessment(fact="Policy", status="supported",
                    citation_ids=(leaked if bad == "fact" else request.evidence[0].citation_id,),
                    claims=(EvidenceClaim(citation_id=leaked, statement="Bad claim", scope_reason="Unbound",
                                          answers_requested_fact=True),) if bad == "claim" else ()),))
            return decision, Usage(api_calls=1, estimated_cost_usd=.01)
    generator = Generator()
    agent = SearchAgent(Retriever((hit,)), Critic(), generator, config=AgentConfig(max_cost_usd=1))
    response = agent.run(search(scope), context=scope.context, guard=scope.guard)
    assert response.answer.status == "error" and not generator.requests
    assert leaked not in response.model_dump_json()
    assert response.trace.usage.api_calls == 1 and response.trace.usage.estimated_cost_usd == .01


@pytest.mark.parametrize("consumer", [GroundedGenerator, EvidenceCritic])
def test_access_error_from_injected_provider_is_not_converted_to_a_model_error(scope, consumer):
    class Provider:
        def complete(self, *args):
            raise StaleContext("Policy changed")
    request = GenerationRequest(question="policy", user=scope.user, evidence=(passage(scope),))
    with pytest.raises(StaleContext):
        with guarded_dispatch(scope.context, scope.guard):
            operation = consumer(Provider())
            (operation.generate if consumer is GroundedGenerator else operation.assess)(request)


def fake_sdk(monkeypatch):
    sdk = ModuleType("anthropic")
    class Error(Exception):
        pass
    sdk.APIError = sdk.APITimeoutError = sdk.APIConnectionError = sdk.AnthropicError = Error
    sdk.NOT_GIVEN = sdk.omit = object()
    types = ModuleType("anthropic.types")
    types.MessageParam = types.OutputConfigParam = dict
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.setitem(sys.modules, "anthropic.types", types)


@pytest.mark.parametrize("revoke_at", ["count", "create"])
def test_real_provider_sdk_boundaries_abort_revocation_and_keep_reservations(scope, monkeypatch, revoke_at):
    fake_sdk(monkeypatch)
    calls = []
    class Messages:
        def count_tokens(self, **kwargs):
            calls.append("count")
            if revoke_at == "count":
                scope.guard.error = SessionExpired("Session revoked during count")
            return SimpleNamespace(input_tokens=20)
        def create(self, **kwargs):
            calls.append("create")
            scope.guard.error = SessionExpired("Session revoked during create")
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=20, output_tokens=5),
                content=[SimpleNamespace(type="text", text="discarded")], stop_reason="end_turn")
    provider = AnthropicProvider(AnthropicConfig(model="scripted", allow_paid=True, budget_usd=1,
        input_usd_per_million=1, output_usd_per_million=1, max_output_tokens=64),
        client=SimpleNamespace(messages=Messages()))
    with pytest.raises(SessionExpired):
        with guarded_dispatch(scope.context, scope.guard):
            provider.complete("system", "authorized prompt")
    assert calls == (["count"] if revoke_at == "count" else ["count", "create"])
    assert (provider.reserved_usd == 0) if revoke_at == "count" else (provider.reserved_usd > 0)
    assert_dispatch_current()  # The failed dispatch did not leave authority installed.


def test_legacy_agent_wire_and_provider_behavior_remain_unscoped(scope):
    value = passage(scope)
    legacy = Evidence.model_validate(value.model_dump(exclude={"kind", "snapshot_id", "revision_id"}))
    user = UserContext(user_id="fixture", allowed_doc_ids=frozenset({"allowed"}))
    generator = Generator()
    response = SearchAgent(Retriever((legacy,)), NeverCritic(), generator).run(
        SearchRequest(query="policy", user=user), single_pass=True)
    assert response.snapshot is None and "snapshot" not in response.model_dump(mode="json")
    assert response.answer.status == "answered" and len(generator.requests) == 1

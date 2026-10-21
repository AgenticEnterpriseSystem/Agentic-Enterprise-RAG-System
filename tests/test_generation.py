import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from src.config import Settings
from src.contracts import Answer, Chunk, GenerationRequest, QueryRequest, QueryResponse, UserContext, Evidence, Usage
from types import SimpleNamespace
from src.generation.generator import AnthropicConfig, AnthropicProvider, Completion, GenerationConfig, GroundedGenerator, ProviderError
from tests.fakes import FakeProvider


# Shared contracts

FIXTURE = json.loads((Path(__file__).parent / "fixtures/contracts.json").read_text())


def test_default_and_unknown_identity_deny_access():
    assert not UserContext().permits("demo-1")
    assert not UserContext(allowed_doc_ids=frozenset({"demo-1"})).permits("demo-1")
    assert not UserContext(user_id="known", role="admin").permits("demo-1")


def test_client_cannot_supply_authorization():
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"question": "hello", "allowed_doc_ids": ["demo-1"]})


def test_generation_rejects_denied_evidence():
    response = QueryResponse.model_validate(FIXTURE["response"])
    with pytest.raises(ValidationError):
        GenerationRequest(
            question="limit?", user=UserContext(), evidence=response.sources
        )
    request = GenerationRequest(
        question="limit?",
        user=UserContext.model_validate(FIXTURE["user"]),
        evidence=response.sources,
    )
    assert request.evidence == response.sources


def test_unknown_citation_rejected():
    data = json.loads(json.dumps(FIXTURE["response"]))
    data["answer"]["citation_ids"] = ["invented"]
    with pytest.raises(ValidationError):
        QueryResponse.model_validate(data)


def test_invalid_offsets_and_nonfinite_budget():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="x",
            doc_id="d",
            source_type="slack",
            text="abc",
            start_char=5,
            end_char=7,
        )
    with pytest.raises(ValidationError):
        Settings(max_cost_usd=float("nan"))


def test_abstention_needs_no_citation_but_answer_does():
    Answer(status="insufficient_evidence", text="No supporting evidence found.")
    with pytest.raises(ValidationError):
        Answer(status="answered", text="I guessed.")


# Answer generation

def request():
    user = UserContext(user_id="demo", allowed_doc_ids=frozenset({"d"}))
    text = "Upload limit is 10 MB."
    chunk = Chunk(
        chunk_id="d:0:22",
        doc_id="d",
        source_type="docs",
        title="Limit",
        text=text,
        start_char=0,
        end_char=len(text),
    )
    evidence = (
        Evidence(
            citation_id=chunk.chunk_id, chunk=chunk, score=1.0, retrieval_method="bm25"
        ),
    )
    return GenerationRequest(
        question="What is the upload limit?", user=user, evidence=evidence
    )


def completion(req, **changes):
    output = {
        "status": "answered",
        "text": "The upload limit is 10 MB.",
        "citation_ids": [req.evidence[0].citation_id],
    }
    output.update(changes)
    return Completion(
        json.dumps(output),
        Usage(input_tokens=20, output_tokens=10, api_calls=1, estimated_cost_usd=0.001),
    )


def test_retrieval_to_generation_uses_only_supplied_evidence():
    req = request()
    provider = FakeProvider([completion(req)])
    answer, usage = GroundedGenerator(provider).generate(req)
    assert answer.status == "answered" and usage.api_calls == 1
    payload = json.loads(provider.calls[0][1])
    assert payload["evidence"][0]["text"] == "Upload limit is 10 MB."
    assert set(payload) == {"question", "evidence"}


def test_empty_or_over_budget_context_never_calls_provider():
    req = request()
    provider = FakeProvider([])
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(max_context_chars=1)
    ).generate(req)
    assert answer.status == "insufficient_evidence" and usage.api_calls == 0
    empty = GenerationRequest(question="Unknown?", user=UserContext())
    assert (
        GroundedGenerator(provider).generate(empty)[0].status == "insufficient_evidence"
    )
    assert not provider.calls


@pytest.mark.parametrize("kind", ["invented", "malformed", "truncated"])
def test_invalid_responses_are_not_answers(kind):
    req = request()
    value = {
        "invented": completion(req, citation_ids=["made-up"]),
        "malformed": Completion("not JSON", Usage(api_calls=1)),
        "truncated": Completion(
            completion(req).text, Usage(api_calls=1), truncated=True
        ),
    }[kind]
    answer, usage = GroundedGenerator(FakeProvider([value])).generate(req)
    assert answer.status == "error" and usage.api_calls == 1


def test_retries_bounded_and_usage_retained():
    req = request()
    provider = FakeProvider([ProviderError(retryable=True), completion(req)])
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(retry_delay_seconds=0)
    ).generate(req)
    assert answer.status == "answered" and usage.api_calls == 2
    assert usage.estimated_cost_usd is None  # Failed call's billing is unknown.
    provider = FakeProvider(
        [ProviderError(retryable=True), ProviderError(retryable=True)]
    )
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(retry_delay_seconds=0)
    ).generate(req)
    assert answer.status == "error" and usage.api_calls == 2


def test_paid_calls_disabled_by_default():
    provider = AnthropicProvider(
        AnthropicConfig(model="explicit-model"), client=object()
    )
    with pytest.raises(ProviderError) as result:
        provider.complete("system", "question")
    assert result.value.usage.api_calls == 0


class Messages:
    def __init__(self):
        self.created = 0

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=100)

    def create(self, **kwargs):
        self.created += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
            stop_reason="end_turn",
        )


def test_anthropic_budget_reservation_and_sdk_mapping():
    messages = Messages()
    config = AnthropicConfig(
        model="explicit-model",
        allow_paid=True,
        budget_usd=0.0003,
        input_usd_per_million=1,
        output_usd_per_million=2,
        max_output_tokens=100,
    )
    provider = AnthropicProvider(config, client=SimpleNamespace(messages=messages))
    result = provider.complete("system", "prompt")
    assert (
        result.usage.api_calls == 2
        and result.usage.estimated_cost_usd == pytest.approx(0.00012)
    )
    with pytest.raises(ProviderError):
        provider.complete("system", "prompt")
    assert messages.created == 1


def test_citation_to_evidence_excluded_by_context_budget_is_rejected():
    req = request()
    other = req.evidence[0].model_copy(
        update={
            "citation_id": "excluded",
            "chunk": req.evidence[0].chunk.model_copy(update={"chunk_id": "other"}),
        }
    )
    request_with_two = GenerationRequest(
        question=req.question, user=req.user, evidence=(*req.evidence, other)
    )
    provider = FakeProvider([completion(req, citation_ids=["excluded"])])
    answer, _ = GroundedGenerator(
        provider, GenerationConfig(max_context_chars=len(req.evidence[0].chunk.text))
    ).generate(request_with_two)
    assert answer.status == "error"
    assert len(json.loads(provider.calls[0][1])["evidence"]) == 1


def test_authentication_error_is_not_retried():
    provider = FakeProvider([ProviderError(retryable=False)])
    answer, usage = GroundedGenerator(provider).generate(request())
    assert answer.status == "error" and usage.api_calls == 1
    assert len(provider.calls) == 1


def test_provider_run_budget_caps_output_and_timeout():
    from src.agents.budget import CallBudget, call_budget

    class RecordingMessages(Messages):
        def create(self, **kwargs):
            self.arguments = kwargs
            return super().create(**kwargs)

    messages = RecordingMessages()
    provider = AnthropicProvider(AnthropicConfig(model="offline", allow_paid=True,
        budget_usd=1, input_usd_per_million=1, output_usd_per_million=2,
        max_output_tokens=100), client=SimpleNamespace(messages=messages))
    budget = CallBudget(remaining_tokens=120, remaining_seconds=5, remaining_cost_usd=1)
    with call_budget(budget):
        result = provider.complete("system", "prompt")
    assert messages.arguments["max_tokens"] == 20
    assert 0 < messages.arguments["timeout"] <= 5
    assert budget.remaining_tokens == 10
    assert budget.remaining_cost_usd == pytest.approx(1 - result.usage.estimated_cost_usd)


@pytest.mark.parametrize("tokens,seconds,cost", [(100, 5, 1), (1000, 0, 1), (1000, 5, 0.00001)])
def test_provider_run_budget_denies_generation(tokens, seconds, cost):
    from src.agents.budget import CallBudget, RunBudgetExceeded, call_budget
    messages = Messages()
    provider = AnthropicProvider(AnthropicConfig(model="offline", allow_paid=True,
        budget_usd=1, input_usd_per_million=1, output_usd_per_million=2),
        client=SimpleNamespace(messages=messages))
    with call_budget(CallBudget(tokens, seconds, cost)), pytest.raises(RunBudgetExceeded):
        provider.complete("system", "prompt")
    assert messages.created == 0


def test_budget_denial_preserves_prior_retry_usage():
    from src.agents.budget import RunBudgetExceeded
    provider = FakeProvider([ProviderError(retryable=True,
        usage=Usage(api_calls=1, input_tokens=7, estimated_cost_usd=0.001))])
    original = provider.complete

    def complete(system, prompt):
        if provider.calls:
            raise RunBudgetExceeded()
        return original(system, prompt)

    provider.complete = complete
    with pytest.raises(RunBudgetExceeded) as caught:
        GroundedGenerator(provider, GenerationConfig(retry_delay_seconds=0)).generate(request())
    assert caught.value.usage.api_calls == 1
    assert caught.value.usage.input_tokens == 7
    assert caught.value.usage.estimated_cost_usd == 0.001


def test_provider_deadline_expiry_prevents_token_count(monkeypatch):
    from src.agents.budget import CallBudget, RunBudgetExceeded, call_budget
    messages = Messages()
    provider = AnthropicProvider(AnthropicConfig(model="offline", allow_paid=True,
        budget_usd=1, input_usd_per_million=1, output_usd_per_million=2),
        client=SimpleNamespace(messages=messages))
    budget = CallBudget(1000, 5, 1)
    monkeypatch.setattr("src.agents.budget.time.monotonic", lambda: budget.deadline + 1)
    with call_budget(budget), pytest.raises(RunBudgetExceeded) as caught:
        provider.complete("system", "prompt")
    assert caught.value.usage.api_calls == 0


def test_large_sole_evidence_is_sliced_instead_of_dropped():
    from src.generation.generator import select_context
    req = request()
    text = "irrelevant filler " * 1500 + " Upload limit is 10 MB. " + "filler " * 500
    chunk = Chunk(chunk_id="long", doc_id="d", source_type="docs", text=text,
                  start_char=30, end_char=30 + len(text))
    evidence = Evidence(citation_id="long", chunk=chunk, score=1, retrieval_method="bm25")
    req = req.model_copy(update={"evidence": (evidence,)})
    selected = select_context(req, 24000)
    assert len(selected) == 1 and len(selected[0].chunk.text) == 3000
    assert "Upload limit is 10 MB." in selected[0].chunk.text
    start, end = selected[0].chunk.start_char - 30, selected[0].chunk.end_char - 30
    assert selected[0].chunk.text == text[start:end]
    provider = FakeProvider([completion(req, citation_ids=[selected[0].citation_id])])
    answer, _ = GroundedGenerator(provider).generate(req)
    assert answer.status == "answered"
    # Applying the context selector again for the critic/generator preserves IDs.
    context = req.model_copy(update={"evidence": tuple(selected)})
    assert select_context(context, 24000) == selected


def test_context_passages_respect_total_budget_and_preserve_short_ids():
    from src.generation.generator import select_context
    req = request()
    selected = select_context(req, 24000)
    assert selected == list(req.evidence)
    long = req.evidence[0].model_copy(update={
        "citation_id": "long", "chunk": Chunk(chunk_id="long", doc_id="d", source_type="docs",
            text="upload " * 1000, start_char=0, end_char=7000)})
    selected = select_context(req.model_copy(update={"evidence": (long, req.evidence[0])}), 100)
    assert sum(len(item.chunk.text) for item in selected) <= 100
    assert selected[0].citation_id != "long"


@pytest.mark.parametrize("wrap", [lambda s: s, lambda s: "```json\n" + s + "\n```", lambda s: "```\n" + s + "\n```"])
def test_json_response_wrapper_preserves_schema_checks(wrap):
    from src.generation.generator import json_response_text
    from src.contracts import CriticDecision
    raw = json.dumps({"status": "answered", "text": "Supported.", "citation_ids": ["source-1"]})
    assert Answer.model_validate_json(json_response_text(wrap(raw))).status == "answered"
    with pytest.raises(ValidationError):
        Answer.model_validate_json(json_response_text(wrap('{"status":"made-up"}')))
    critic = json.dumps({"sufficient": False, "confidence": 0, "reason": "missing"})
    assert not CriticDecision.model_validate_json(json_response_text(wrap(critic))).sufficient


@pytest.mark.parametrize("text", ['Before\n```json\n{}\n```', '```json\n{}\n```\nAfter', '```json\n{}\n```\n```json\n{}\n```'])
def test_json_wrapper_rejects_extra_content(text):
    from src.generation.generator import json_response_text
    with pytest.raises(ValidationError):
        Answer.model_validate_json(json_response_text(text))

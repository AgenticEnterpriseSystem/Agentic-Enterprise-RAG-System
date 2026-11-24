"""Experiment options on hira/testing: excerpt size, read-more size/off, clarification off,
the matching-source critic rule. Each default keeps main's behaviour."""
import pytest

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import MATCHING_SOURCE_RULE, SYSTEM_PROMPT, EvidenceCritic
from src.contracts import Answer, Chunk, Evidence, SearchRequest, UserContext
from src.retrieval.hybrid import HybridConfig, HybridRetriever
from tests.test_agents import ZERO, Critic, decision

USER = UserContext(user_id="jane", allowed_doc_ids=frozenset({"a", "b"}))
DOC = "".join(f"Sentence {i} about the gateway. " for i in range(800))  # ~24,000 characters


def hit(doc="a", start=0, end=None, text=None):
    text = text if text is not None else DOC[start:end]
    end = start + len(text)
    key = f"{doc}:{start}:{end}"
    return Evidence(citation_id=key, chunk=Chunk(chunk_id=key, doc_id=doc, source_type="docs",
                    text=text, start_char=start, end_char=end), score=1, retrieval_method="hybrid")


class Ranked:
    """One retriever of a hybrid pair: fixed results, and the whole document by ID."""
    doc_ids = ["a", "b"]

    def __init__(self, items):
        self.items = items

    def search(self, request):
        return self.items

    def full_text(self, doc_id):
        return DOC


def test_passage_chars_recuts_returned_results_around_the_match():
    match = hit(start=12000, end=12500)
    plain = HybridRetriever(Ranked((match,)), Ranked(()), config=HybridConfig())
    wide = HybridRetriever(Ranked((match,)), Ranked(()), config=HybridConfig(passage_chars=6000))
    request = SearchRequest(query="gateway", user=USER, top_k=5)
    assert plain.search(request)[0].chunk.text == match.chunk.text
    (result,) = wide.search(request)
    chunk = result.chunk
    assert chunk.end_char - chunk.start_char == 6000 == len(chunk.text)
    assert chunk.start_char <= 12000 and chunk.end_char >= 12500
    assert chunk.text == DOC[chunk.start_char:chunk.end_char]
    assert result.citation_id == f"a:{chunk.start_char}:{chunk.end_char}"


def test_passage_chars_keeps_an_excerpt_whose_offsets_are_not_the_whole_document():
    odd = hit(start=0, text="Not the indexed document text.")
    wide = HybridRetriever(Ranked((odd,)), Ranked(()), config=HybridConfig(passage_chars=6000))
    assert wide.search(SearchRequest(query="gateway", user=USER))[0].chunk == odd.chunk


class Expanding:
    """Search results plus a read-more that records its size."""
    doc_ids = ["a"]

    def __init__(self):
        self.read_sizes, self.searches = [], 0

    def search(self, request):
        self.searches += 1
        return (hit(start=0, end=3000),) if self.searches == 1 else (hit("b", text="Other."),)

    def read_more(self, request, anchors, *, max_chars):
        self.read_sizes.append(max_chars)
        return (hit(start=2800, end=2800 + max_chars),)


class Generator:
    def generate(self, request):
        return Answer(status="answered", text="Answer.", citation_ids=(request.evidence[0].citation_id,)), ZERO


def run(config, decisions):
    retriever = Expanding()
    agent = SearchAgent(retriever, Critic(decisions), Generator(), config=config)
    response = agent.run(SearchRequest(query="What is the gateway limit?", user=USER))
    return response, retriever


@pytest.mark.parametrize("chars,expected", [(3000, [3000]), (6000, [6000]), (0, [])])
def test_read_more_size_and_off_switch(chars, expected):
    wants_more = decision(False, missing_facts=("Limit",), read_more_doc_ids=("a",),
                          suggested_queries=("gateway limit value",))
    response, retriever = run(AgentConfig(max_rounds=2, read_more_chars=chars), [wants_more, decision(True)])
    assert retriever.read_sizes == expected
    assert retriever.searches == (1 if expected else 2)  # Off: the follow-up is a new search.
    assert response.answer.status == "answered"


def test_clarification_off_keeps_searching_instead_of_asking():
    ambiguous = decision(False, missing_facts=("Which incident",), clarification_question="Which incident?",
                         suggested_queries=("gateway incident March",))
    asked, _ = run(AgentConfig(max_rounds=2), [ambiguous])
    assert asked.trace.stop_reason == "clarification"
    searched, retriever = run(AgentConfig(max_rounds=2, clarification=False), [ambiguous, decision(True)])
    assert searched.trace.stop_reason == "sufficient" and retriever.searches == 2


def test_matching_source_rule_is_appended_and_keeps_the_native_checklist():
    plain = EvidenceCritic(object())
    strict = EvidenceCritic(object(), matching_source=True)
    assert plain.system_prompt == SYSTEM_PROMPT
    assert strict.system_prompt == SYSTEM_PROMPT + MATCHING_SOURCE_RULE
    assert plain.native_prompt and strict.native_prompt
    assert not EvidenceCritic(object(), system_prompt="Custom.", matching_source=True).native_prompt


def test_web_settings_read_the_experiment_options(monkeypatch):
    from src.api.runtime import WebSettings
    for name, value in {"PASSAGE_CHARS": "6000", "READ_MORE_CHARS": "0", "CLARIFICATION": "false",
                        "MATCHING_SOURCE": "true"}.items():
        monkeypatch.setenv("RAG_" + name, value)
    settings = WebSettings.from_env()
    assert (settings.passage_chars, settings.read_more_chars, settings.clarification,
            settings.matching_source) == (6000, 0, False, True)
    assert settings.hybrid_config().passage_chars == 6000
    defaults = WebSettings()
    assert (defaults.passage_chars, defaults.read_more_chars, defaults.clarification,
            defaults.matching_source) == (None, 3000, True, False)

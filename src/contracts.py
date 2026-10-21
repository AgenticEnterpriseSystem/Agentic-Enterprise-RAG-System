"""Version 1 boundary schemas; authorization must also be enforced by adapters."""

from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"\S")]
Nonnegative = Annotated[float, Field(ge=0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Document(Contract):
    doc_id: Identifier
    source_type: Identifier
    title: str
    content: str
    timestamp: AwareDatetime | None = None
    # These are enrichment fields, not claims about the upstream dataset.
    metadata_origin: Literal["upstream", "extracted", "synthetic"] = "upstream"


class Chunk(Contract):
    chunk_id: Identifier
    doc_id: Identifier
    source_type: Identifier
    title: str = ""
    metadata_origin: Literal["upstream", "extracted", "synthetic"] = "upstream"
    text: Annotated[str, Field(min_length=1)]
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    timestamp: AwareDatetime | None = None

    @model_validator(mode="after")
    def offsets(self):
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("Offsets must describe the exact unmodified text slice")
        return self


class UserContext(Contract):
    """Server-resolved scope; empty/unknown identity denies everything."""

    user_id: Identifier | None = None
    role: str | None = None
    department: str | None = None
    project_ids: tuple[str, ...] = ()
    allowed_doc_ids: frozenset[str] = frozenset()

    def permits(self, doc_id: str) -> bool:
        return self.user_id is not None and doc_id in self.allowed_doc_ids


class QueryRequest(Contract):
    """Public API input: clients cannot supply identity or permission claims."""

    question: Identifier
    mode: Literal["baseline", "agentic"] = "baseline"


class SearchRequest(Contract):
    query: Identifier
    user: UserContext
    top_k: int = Field(default=10, ge=1, le=100)
    source_types: tuple[str, ...] = ()
    after: AwareDatetime | None = None
    before: AwareDatetime | None = None

    @model_validator(mode="after")
    def dates(self):
        if self.after and self.before and self.after > self.before:
            raise ValueError("after must not exceed before")
        return self


class Evidence(Contract):
    citation_id: Identifier
    chunk: Chunk
    score: float
    retrieval_method: Literal["bm25", "dense", "hybrid", "structured"]


def validate_evidence(user: UserContext, evidence: tuple[Evidence, ...]) -> None:
    """Call before passing results to rerankers, critics, generators or clients."""
    if any(not user.permits(item.chunk.doc_id) for item in evidence):
        raise ValueError("Evidence outside authorized document scope")
    ids = [item.citation_id for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate citation IDs")


class CriticDecision(Contract):
    sufficient: bool
    confidence: float = Field(ge=0, le=1)
    reason: Identifier
    missing_facts: tuple[str, ...] = ()
    suggested_queries: tuple[str, ...] = ()
    suggested_sources: tuple[str, ...] = ()


class Usage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    api_calls: int = Field(default=0, ge=0)
    estimated_cost_usd: Nonnegative | None = None  # None means unknown, not free.


class TraceStep(Contract):
    round_number: int = Field(ge=1)
    query: Identifier
    tool: Identifier
    source_types: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    critic: CriticDecision | None = None
    elapsed_ms: Nonnegative
    usage: Usage = Field(default_factory=Usage)


class Trace(Contract):
    run_id: Identifier
    steps: tuple[TraceStep, ...] = ()
    total_latency_ms: Nonnegative = 0
    usage: Usage = Field(default_factory=Usage)
    stop_reason: Literal[
        "sufficient", "round_limit", "no_new_evidence", "budget", "error", "single_pass"
    ]


class Answer(Contract):
    status: Literal["answered", "insufficient_evidence", "error"]
    text: Identifier
    citation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def citations(self):
        if self.status == "answered" and not self.citation_ids:
            raise ValueError("Answered outputs need supporting citations")
        if len(set(self.citation_ids)) != len(self.citation_ids):
            raise ValueError("Duplicate answer citation IDs")
        return self


class QueryResponse(Contract):
    answer: Answer
    sources: tuple[Evidence, ...] = ()
    trace: Trace

    @model_validator(mode="after")
    def cited_sources_exist(self):
        ids = [item.citation_id for item in self.sources]
        if len(ids) != len(set(ids)) or not set(self.answer.citation_ids) <= set(ids):
            raise ValueError("Citations must resolve to unique supplied sources")
        return self


class GenerationRequest(Contract):
    search_limitation: str | None = None
    question: Identifier
    user: UserContext
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def authorized(self):
        validate_evidence(self.user, self.evidence)
        return self


class ToolRequest(Contract):
    user: UserContext
    source_type: Identifier
    operation: Literal["lookup", "filter", "count"]
    filters: dict[str, str | int | bool] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=100)


class ToolResult(Contract):
    evidence: tuple[Evidence, ...] = ()
    count: int | None = Field(default=None, ge=0)


class Retriever(Protocol):
    def search(self, request: SearchRequest) -> tuple[Evidence, ...]: ...


class StructuredTool(Protocol):
    def execute(self, request: ToolRequest) -> ToolResult: ...


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> tuple[Answer, Usage]: ...


class SearchCritic(Protocol):
    def assess(self, request: GenerationRequest) -> tuple[CriticDecision, Usage]: ...

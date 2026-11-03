"""Shared application contracts; authorization is enforced by server adapters.

Snapshot-bound evidence is explicitly enabled by supplying a response snapshot.
Existing requests and responses keep their wire representation until adoption.
"""

from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

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

    question: Annotated[str, Field(min_length=1, max_length=8000, pattern=r"\S")]
    mode: Literal["baseline", "agentic"] = "baseline"
    source_types: tuple[Annotated[str, Field(min_length=1, max_length=64)], ...] = Field(default=(), max_length=9)
    after: AwareDatetime | None = None
    before: AwareDatetime | None = None

    @model_validator(mode="after")
    def valid_dates(self):
        if self.after and self.before and self.after > self.before:
            raise ValueError("after must not exceed before")
        return self


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


class EvidenceClaim(Contract):
    citation_id: Identifier
    statement: Identifier
    scope_reason: Identifier
    answers_requested_fact: bool


class FactAssessment(Contract):
    fact: Identifier
    # Native structured output is generated in this order: inspect claims and
    # scope before assigning the verdict, rather than justifying it afterward.
    claims: tuple[EvidenceClaim, ...] = Field(default=(), max_length=2)
    status: Literal['supported', 'missing', 'conflicting', 'ambiguous']
    citation_ids: tuple[Identifier, ...] = Field(default=(), max_length=6)

    @model_validator(mode='after')
    def supported_has_citations(self):
        if self.status in ('supported', 'conflicting') and not self.citation_ids:
            raise ValueError('Supported or conflicting facts require evidence citations')
        return self


class EvidenceAssessment(Contract):
    """Native model output; overall search decisions are derived in code."""
    facts: tuple[FactAssessment, ...] = Field(min_length=1, max_length=10)
    suggested_queries: tuple[str, ...] = ()
    suggested_sources: tuple[str, ...] = ()
    read_more_doc_ids: tuple[Identifier, ...] = Field(default=(), max_length=2)
    clarification_question: Annotated[str, Field(min_length=1, max_length=500)] | None = None


class CriticDecision(EvidenceAssessment):
    sufficient: bool
    confidence: float = Field(ge=0, le=1)
    reason: Identifier
    missing_facts: tuple[str, ...] = ()
    facts: tuple[FactAssessment, ...] = Field(default=(), max_length=10)


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
        "sufficient", "round_limit", "no_new_evidence", "budget", "error", "single_pass", "clarification"
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


CONTRACT_VERSION = "2"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[int, Field(strict=True, ge=0)]


class SnapshotRef(Contract):
    snapshot_id: Identifier
    corpus_sha256: Sha256
    catalog_sha256: Sha256
    document_count: Annotated[int, Field(strict=True, ge=0)]


class ScopeBinding(Contract):
    """Internal binding, never accepted as authority from the browser/model."""

    scope_id: Identifier
    user_id: Identifier
    snapshot: SnapshotRef
    policy_revision: Revision
    session_version: Revision


class BoundFilter(Contract):
    field: Identifier
    value: StrictStr | StrictInt | StrictBool


class PassageEvidence(Evidence):
    kind: Literal["passage"] = "passage"
    snapshot_id: Identifier
    revision_id: Identifier


class AggregateEvidence(Contract):
    """A computed result, not a fabricated document or sampled-text quotation.

    Metadata alone does not prove authorization: the access contract requires an
    exact match with a trusted, owner/scope-bound stored artifact. The producing
    adapter must count the full authorized relation and record its lineage.
    """

    kind: Literal["aggregate"] = "aggregate"
    citation_id: Identifier
    artifact_id: Identifier
    scope: ScopeBinding
    template_id: Identifier
    template_version: Identifier
    filters: tuple[BoundFilter, ...] = Field(default=(), max_length=32)
    operation: Literal["count"] = "count"
    count: Annotated[int, Field(strict=True, ge=0)]
    lineage_sha256: Sha256
    coverage: Literal["verified_records", "synthetic_fixture"]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def unique_filters(self):
        names = [item.field for item in self.filters]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate aggregate filter fields")
        return self


GroundedEvidence = Annotated[
    PassageEvidence | AggregateEvidence, Field(discriminator="kind")
]


class QueryResponse(Contract):
    """Shared response; snapshot-bound evidence requires explicit provenance.

    Model validation checks structure, not session authorization. Adapters must
    run the access boundary before releasing snapshot-bound responses.
    """

    contract_version: Literal["2"] | None = Field(default=None, exclude_if=lambda value: value is None)
    snapshot: SnapshotRef | None = Field(default=None, exclude_if=lambda value: value is None)
    answer: Answer
    sources: tuple[GroundedEvidence | Evidence, ...] = ()
    trace: Trace

    @model_validator(mode="after")
    def cited_sources_exist(self):
        ids = [item.citation_id for item in self.sources]
        if len(ids) != len(set(ids)) or not set(self.answer.citation_ids) <= set(ids):
            raise ValueError("Citations must resolve to unique supplied sources")
        if self.snapshot is None:
            if self.contract_version is not None or any(
                isinstance(source, (PassageEvidence, AggregateEvidence)) for source in self.sources
            ):
                raise ValueError("Snapshot-bound evidence requires a response snapshot")
            return self
        object.__setattr__(self, "contract_version", CONTRACT_VERSION)
        for source in self.sources:
            if isinstance(source, PassageEvidence):
                if source.snapshot_id != self.snapshot.snapshot_id:
                    raise ValueError("Passage belongs to another snapshot")
            elif isinstance(source, AggregateEvidence):
                if source.scope.snapshot != self.snapshot:
                    raise ValueError("Aggregate belongs to another snapshot")
            else:
                raise ValueError("Snapshot-bound responses require tagged evidence")
        return self


class GenerationRequest(Contract):
    search_limitation: str | None = None
    assessment: CriticDecision | None = None
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

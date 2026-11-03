"""Versioned V2 data contracts; authorization remains a server-side operation.

The running V1 API deliberately keeps its existing schemas until APP/WIRE-01.
V2 passages carry a snapshot/revision; aggregates carry reproducible provenance
and must be resolved against an authoritative artifact registry before use.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StrictStr, model_validator

from src.contracts import Answer, Contract, Evidence, Identifier, Trace

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


class QueryResponseV2(Contract):
    """Structural validation only; call the access boundary before releasing it."""

    contract_version: Literal["2"] = CONTRACT_VERSION
    snapshot: SnapshotRef
    answer: Answer
    sources: tuple[GroundedEvidence, ...] = ()
    trace: Trace

    @model_validator(mode="after")
    def consistent_sources(self):
        ids = [source.citation_id for source in self.sources]
        if len(ids) != len(set(ids)) or not set(self.answer.citation_ids) <= set(ids):
            raise ValueError("Citations must resolve to unique supplied sources")
        for source in self.sources:
            if isinstance(source, PassageEvidence):
                if source.snapshot_id != self.snapshot.snapshot_id:
                    raise ValueError("Passage belongs to another snapshot")
            elif source.scope.snapshot != self.snapshot:
                raise ValueError("Aggregate belongs to another snapshot")
        return self

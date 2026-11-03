"""Immutable identity and scope boundaries for future authenticated adapters.

These are server-only contracts, not an authorization policy implementation.
The running V1 API stays unchanged. Future adapters must resolve policy from a
trusted store; role labels and browser-supplied metadata never grant authority.
"""

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from pydantic import AwareDatetime, model_validator

from src.contracts import Contract, Identifier
from src.contracts_v2 import AggregateEvidence, PassageEvidence, Revision, ScopeBinding, SnapshotRef


class AccessError(ValueError):
    """A content-free authorization boundary failure."""


class AccessDenied(AccessError):
    """The current principal cannot use the requested resource."""

    code = "scope_denied"


class SnapshotMismatch(AccessError):
    """Evidence or an index belongs to a different immutable dataset."""


class StaleContext(AccessError):
    """Policy/scope versions changed; resolve a fresh request context."""

    code = "stale_context"


class SessionExpired(StaleContext):
    """An expired/revoked session requires authentication again."""

    code = "session_expired"


class PolicyUnavailable(AccessError):
    """Authoritative access state is unavailable; sensitive work must stop."""

    code = "policy_unavailable"


class EvidenceValidationError(AccessError):
    """Evidence violates its structural or authoritative provenance contract."""


class ResolvedIdentity(Contract):
    """Trusted account/session metadata; session_id is a row ID, never a token.

    Metadata is descriptive. Authorization comes exclusively from the resolved
    scope. Expiry and current versions are checked by FreshnessGuard at every
    sensitive operation, rather than being assumed valid at construction time.
    """

    user_id: Identifier
    session_id: Identifier
    session_version: Revision
    expires_at: AwareDatetime
    role: Identifier | None = None
    department: Identifier | None = None
    project_ids: tuple[Identifier, ...] = ()
    group_ids: tuple[Identifier, ...] = ()


@runtime_checkable
class AuthorizationScope(Protocol):
    """An immutable, trusted decision snapshot with index-aligned masks.

    Implementations must never consult or modify caller-provided role claims.
    mask_for must reject a different SnapshotRef (including fingerprint changes)
    and preserve the exact order and length of doc_ids. Unknown IDs deny.
    """

    @property
    def binding(self) -> ScopeBinding: ...

    def permits(self, doc_id: str) -> bool: ...

    def mask_for(self, snapshot: SnapshotRef, doc_ids: Sequence[str]) -> Sequence[bool]: ...


def _checked_binding(binding: ScopeBinding) -> ScopeBinding:
    # Pydantic model_copy(update=...) deliberately skips validation. Revalidate
    # the serialized data instead of trusting an already-constructed instance.
    if not isinstance(binding, ScopeBinding):
        raise EvidenceValidationError("Invalid server scope binding")
    try:
        return ScopeBinding.model_validate(binding.model_dump(warnings=False))
    except ValueError:
        raise EvidenceValidationError("Invalid server scope binding") from None


@dataclass(frozen=True)
class RequestContext:
    """Per-request identity plus an immutable, owner/version-bound scope.

    Binding is captured at construction so a faulty scope adapter cannot swap
    snapshots or owners between stages. Arbitrary Protocol implementations are
    trusted server objects and must themselves honor the immutable-scope contract.
    Do not serialize this context into HTTP/model input.
    """

    identity: ResolvedIdentity
    scope: AuthorizationScope
    binding: ScopeBinding = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ResolvedIdentity):
            raise EvidenceValidationError("Invalid server-resolved identity")
        try:
            identity = ResolvedIdentity.model_validate(self.identity.model_dump(warnings=False))
        except ValueError:
            raise EvidenceValidationError("Invalid server-resolved identity") from None
        if not isinstance(self.scope, AuthorizationScope):
            raise EvidenceValidationError("Invalid authorization scope adapter")
        binding = _checked_binding(self.scope.binding)
        if (identity.user_id != binding.user_id
                or identity.session_version != binding.session_version):
            raise AccessDenied("Identity and authorization scope do not match")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "binding", binding)

    def assert_bound(self) -> None:
        if _checked_binding(self.scope.binding) != self.binding:
            raise StaleContext("Authorization scope changed during the request")
        if (self.identity.user_id != self.binding.user_id
                or self.identity.session_version != self.binding.session_version):
            raise AccessDenied("Identity and authorization scope do not match")


class FreshnessGuard(Protocol):
    """Check current session/account status, expiry, policy and artifact versions.

    Raise SessionExpired for expiry/revocation, disabled accounts or an invalidated
    session version; StaleContext for changed policy/scope/artifact versions;
    AccessDenied for a current identity lacking permission; PolicyUnavailable when
    authoritative state cannot be checked. These typed codes determine HTTP error
    semantics without inspecting exception messages. All failures are content-free.
    Never refresh or broaden context. Admitted provider input cannot be recalled;
    call again after long work and before releasing any retained output.
    """

    def assert_current(self, context: RequestContext) -> None: ...


class AggregateRegistry(Protocol):
    """Trusted, server-side canonical aggregate lookup, never caller metadata.

    Implementations retain the producing query's complete authorized-relation
    lineage (or a reproducible immutable snapshot/query reference), even for zero
    results. The consumer must compare the entire resolved artifact with output.
    """

    def resolve(self, artifact_id: str) -> AggregateEvidence | None: ...


class LegacySetScope(Contract):
    """Small-fixture/reference adapter for the existing allowed-document sets.

    Future large-corpus bitmap scopes implement the same Protocol. Construct this
    adapter only from trusted server policy, not from an HTTP/model payload.
    """

    binding: ScopeBinding
    allowed_doc_ids: frozenset[Identifier] = frozenset()

    @model_validator(mode="after")
    def validated_binding(self):
        object.__setattr__(self, "binding", _checked_binding(self.binding))
        if len(self.allowed_doc_ids) > self.binding.snapshot.document_count:
            raise ValueError("Scope contains more documents than its snapshot")
        return self

    def permits(self, doc_id: str) -> bool:
        return doc_id in self.allowed_doc_ids

    def mask_for(self, snapshot: SnapshotRef, doc_ids: Sequence[str]) -> tuple[bool, ...]:
        if snapshot != self.binding.snapshot:
            raise SnapshotMismatch("Permission mask belongs to another snapshot")
        return tuple(self.permits(doc_id) for doc_id in doc_ids)


def validate_grounded_evidence(
    context: RequestContext,
    sources: Sequence[PassageEvidence | AggregateEvidence],
    guard: FreshnessGuard,
    aggregates: AggregateRegistry | None = None,
) -> None:
    """Fail closed before using or releasing passage/aggregate evidence.

    This proves permission/snapshot binding and canonical aggregate provenance;
    passage text/revision authenticity remains the trusted DocumentStore's job.
    Valid citations do not prove factual support. Aggregate owner/version metadata
    cannot establish authority without the registry's exact canonical artifact.
    """

    guard.assert_current(context)
    context.assert_bound()
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, (PassageEvidence, AggregateEvidence)):
            raise EvidenceValidationError("Unsupported grounded evidence type")
        try:
            checked = type(source).model_validate(source.model_dump(warnings=False))
        except ValueError:
            raise EvidenceValidationError("Invalid grounded evidence") from None
        if checked.citation_id in seen:
            raise EvidenceValidationError("Duplicate grounded evidence citations")
        seen.add(checked.citation_id)
        if isinstance(checked, PassageEvidence):
            if checked.snapshot_id != context.binding.snapshot.snapshot_id:
                raise SnapshotMismatch("Passage belongs to another snapshot")
            if not context.scope.permits(checked.chunk.doc_id):
                raise AccessDenied("Passage is outside the authorized document scope")
            continue
        if checked.scope != context.binding:
            raise AccessDenied("Aggregate is outside the current owner or scope")
        if aggregates is None:
            raise AccessDenied("Aggregate requires authoritative artifact resolution")
        canonical = aggregates.resolve(checked.artifact_id)
        if not isinstance(canonical, AggregateEvidence):
            raise AccessDenied("Aggregate artifact is unavailable to this request")
        try:
            canonical = AggregateEvidence.model_validate(canonical.model_dump(warnings=False))
        except ValueError:
            raise EvidenceValidationError("Invalid authoritative aggregate artifact") from None
        if canonical.scope != context.binding or canonical != checked:
            raise AccessDenied("Aggregate differs from its authoritative artifact")
    # Registry resolution may perform I/O. Recheck revocation/expiry and binding
    # before callers use any validated output, including an empty source list.
    guard.assert_current(context)
    context.assert_bound()

"""Server-owned authorization contracts; no policy or login implementation."""

from src.access.contracts import (
    AccessDenied,
    AccessError,
    AggregateRegistry,
    AuthorizationScope,
    EvidenceValidationError,
    FreshnessGuard,
    LegacySetScope,
    PolicyUnavailable,
    RequestContext,
    ResolvedIdentity,
    SessionExpired,
    SnapshotMismatch,
    StaleContext,
    validate_grounded_evidence,
)

__all__ = [
    "AccessDenied", "AccessError", "AggregateRegistry", "AuthorizationScope",
    "EvidenceValidationError", "FreshnessGuard", "LegacySetScope", "PolicyUnavailable",
    "RequestContext", "ResolvedIdentity", "SessionExpired", "SnapshotMismatch",
    "StaleContext", "validate_grounded_evidence",
]

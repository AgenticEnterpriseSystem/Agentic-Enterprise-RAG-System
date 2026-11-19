"""Behavioral access boundaries; no login, database or policy implementation."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, PolicyUnavailable,
    RequestContext, ResolvedIdentity, SessionExpired, SnapshotMismatch,
    StaleContext, validate_grounded_evidence,
)
from src.auth.contracts import ACCESS_ERROR_STATUS, AccessErrorDetail, AccessErrorResponse
from src.contracts import AggregateEvidence, BoundFilter, Chunk, PassageEvidence, ScopeBinding, SnapshotRef


NOW = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
SNAPSHOT = SnapshotRef(snapshot_id="accepted", corpus_sha256="a" * 64,
                       catalog_sha256="b" * 64, document_count=3)


def binding(**changes):
    values = dict(scope_id="intern-scope", user_id="intern", snapshot=SNAPSHOT,
                  policy_revision=2, session_version=3)
    return ScopeBinding(**(values | changes))


def request_context(*, bound=None, allowed=("handbook",)):
    bound = bound or binding()
    identity = ResolvedIdentity(user_id=bound.user_id, session_id="session-row-id",
        session_version=bound.session_version, expires_at=NOW + timedelta(hours=1),
        role="intern", department="finance", project_ids=("atlas",), group_ids=("general",))
    return RequestContext(identity, LegacySetScope(binding=bound, allowed_doc_ids=allowed))


class Guard:
    def __init__(self):
        self.now = NOW
        self.revoked = False
        self.policy_revision = 2

    def assert_current(self, context):
        if (self.revoked or self.now >= context.identity.expires_at
                or context.binding.policy_revision != self.policy_revision):
            raise StaleContext("Request identity is no longer current")


def passage(doc="handbook", *, start=0, text="Allowed excerpt", snapshot_id="accepted"):
    key = f"{doc}:{start}:{start + len(text)}"
    return PassageEvidence(citation_id=key, chunk=Chunk(chunk_id=key, doc_id=doc,
        source_type="confluence", text=text, start_char=start, end_char=start + len(text)),
        score=1, retrieval_method="hybrid", snapshot_id=snapshot_id, revision_id="revision-1")


def aggregate(*, bound=None, count=0):
    return AggregateEvidence(citation_id="aggregate:artifact-1", artifact_id="artifact-1",
        scope=bound or binding(), template_id="authorized-ticket-count", template_version="1",
        filters=(BoundFilter(field="status", value="open"),), count=count,
        lineage_sha256="c" * 64, coverage="synthetic_fixture", created_at=NOW)


class Registry:
    def __init__(self, *artifacts):
        self.artifacts = {artifact.artifact_id: artifact for artifact in artifacts}

    def resolve(self, artifact_id):
        return self.artifacts.get(artifact_id)


def test_intern_metadata_does_not_grant_executive_document_or_adjacent_passage():
    context, guard = request_context(), Guard()
    validate_grounded_evidence(context, (passage(), passage(start=500, text="Adjacent handbook")), guard)
    for denied in (passage("executive-strategy"), passage("executive-strategy", start=500)):
        with pytest.raises(AccessDenied, match="authorized document scope"):
            validate_grounded_evidence(context, (denied,), guard)
    assert not context.scope.permits("executive-strategy")


def test_scope_mask_preserves_order_duplicates_and_denies_unknown_ids():
    context = request_context()
    assert context.scope.mask_for(SNAPSHOT, ["secret", "handbook", "unknown", "handbook"]) == (
        False, True, False, True)
    same_id_changed_fingerprint = SNAPSHOT.model_copy(update={"catalog_sha256": "d" * 64})
    with pytest.raises(SnapshotMismatch):
        context.scope.mask_for(same_id_changed_fingerprint, ["handbook"])


@pytest.mark.parametrize("change", [{"user_id": "executive"}, {"session_version": 4}])
def test_request_rejects_scope_owner_or_session_version_mismatch(change):
    context = request_context()
    scope = LegacySetScope(binding=binding(**change), allowed_doc_ids={"handbook"})
    with pytest.raises(AccessDenied):
        RequestContext(context.identity, scope)


def test_request_identity_and_nested_scope_models_are_frozen():
    context = request_context()
    with pytest.raises(FrozenInstanceError):
        context.scope = LegacySetScope(binding=binding(), allowed_doc_ids={"secret"})
    for model, field, replacement in (
        (context.identity, "role", "executive"),
        (context.scope, "allowed_doc_ids", frozenset({"secret"})),
        (context.scope.binding, "user_id", "executive"),
        (context.scope.binding.snapshot, "snapshot_id", "other"),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, replacement)
    assert isinstance(context.identity.group_ids, tuple)
    assert isinstance(context.scope.allowed_doc_ids, frozenset)


def test_unvalidated_copies_are_revalidated_at_server_boundaries():
    context = request_context()
    malformed_identity = context.identity.model_copy(update={"expires_at": "not-a-date"})
    with pytest.raises(EvidenceValidationError):
        RequestContext(malformed_identity, context.scope)
    malformed_source = passage().model_copy(update={"chunk": passage().chunk.model_copy(
        update={"end_char": 10000})})
    with pytest.raises(EvidenceValidationError):
        validate_grounded_evidence(context, (malformed_source,), Guard())


def test_mutable_scope_cannot_swap_binding_mid_request():
    class FaultyScope:
        binding = binding()
        def permits(self, doc_id):
            return True
        def mask_for(self, snapshot, doc_ids):
            return (True,) * len(doc_ids)
    scope = FaultyScope()
    context = RequestContext(request_context().identity, scope)
    scope.binding = binding(scope_id="changed")
    with pytest.raises(StaleContext, match="changed"):
        validate_grounded_evidence(context, (passage(),), Guard())


@pytest.mark.parametrize("state", ["revoked", "expired", "policy_changed"])
def test_revocation_and_expiry_prevent_even_empty_evidence_release(state):
    context, guard = request_context(), Guard()
    if state == "revoked":
        guard.revoked = True
    elif state == "expired":
        guard.now = context.identity.expires_at
    else:
        guard.policy_revision += 1
    with pytest.raises(StaleContext):
        validate_grounded_evidence(context, (), guard)


def test_cross_snapshot_passage_and_duplicate_citations_reject():
    context, guard = request_context(), Guard()
    with pytest.raises(SnapshotMismatch):
        validate_grounded_evidence(context, (passage(snapshot_id="old"),), guard)
    with pytest.raises(EvidenceValidationError, match="Duplicate"):
        validate_grounded_evidence(context, (passage(), passage()), guard)


def test_zero_aggregate_requires_exact_authoritative_artifact():
    context, source, guard = request_context(), aggregate(), Guard()
    validate_grounded_evidence(context, (source,), guard, Registry(source))
    with pytest.raises(AccessDenied, match="authoritative"):
        validate_grounded_evidence(context, (source,), guard)
    with pytest.raises(AccessDenied, match="unavailable"):
        validate_grounded_evidence(context, (source,), guard, Registry())


@pytest.mark.parametrize("change", [
    {"user_id": "executive"}, {"scope_id": "different"}, {"policy_revision": 1},
    {"session_version": 4}, {"snapshot": SNAPSHOT.model_copy(update={"snapshot_id": "other"})},
])
def test_canonical_aggregate_cannot_cross_owner_scope_snapshot_or_version(change):
    source = aggregate(bound=binding(**change))
    with pytest.raises(AccessDenied, match="owner or scope"):
        validate_grounded_evidence(request_context(), (source,), Guard(), Registry(source))


@pytest.mark.parametrize("change", [
    {"count": 9}, {"lineage_sha256": "d" * 64}, {"template_version": "2"},
    {"filters": (BoundFilter(field="status", value="closed"),)},
])
def test_model_copy_cannot_forge_canonical_count_query_or_lineage(change):
    source = aggregate()
    forged = source.model_copy(update=change)
    with pytest.raises(AccessDenied, match="authoritative artifact"):
        validate_grounded_evidence(request_context(), (forged,), Guard(), Registry(source))


def test_invalid_unvalidated_canonical_artifact_fails_closed():
    source = aggregate()
    invalid = source.model_copy(update={"count": True})
    with pytest.raises(EvidenceValidationError, match="authoritative"):
        validate_grounded_evidence(request_context(), (source,), Guard(), Registry(invalid))


def test_revocation_during_registry_lookup_prevents_valid_artifact_release():
    source, guard = aggregate(), Guard()
    class RevokingRegistry(Registry):
        def resolve(self, artifact_id):
            guard.revoked = True
            return super().resolve(artifact_id)
    with pytest.raises(StaleContext):
        validate_grounded_evidence(request_context(), (source,), guard, RevokingRegistry(source))


def test_registry_and_scope_failures_do_not_become_successful_empty_results():
    class UnavailableRegistry:
        def resolve(self, artifact_id):
            raise RuntimeError("policy store unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_grounded_evidence(request_context(), (aggregate(),), Guard(), UnavailableRegistry())
    with pytest.raises(ValidationError):
        ResolvedIdentity(user_id="intern", session_id="row", session_version=3,
                         expires_at=datetime(2026, 11, 2))


@pytest.mark.parametrize("error_type,status", [
    (SessionExpired, 401), (StaleContext, 409), (PolicyUnavailable, 503), (AccessDenied, 403),
])
def test_typed_guard_errors_keep_auth_http_semantics_without_message_matching(error_type, status):
    failure = error_type("Request cannot continue.")
    class FailingGuard:
        def assert_current(self, context):
            raise failure
    class NeverResolve:
        def resolve(self, artifact_id):
            pytest.fail("Invalid identity must stop before aggregate access")
    with pytest.raises(error_type) as caught:
        validate_grounded_evidence(request_context(), (aggregate(),), FailingGuard(), NeverResolve())
    assert caught.value is failure
    assert ACCESS_ERROR_STATUS[caught.value.code] == status
    # HTTP messages are generic, not built from identity, document, provider,
    # credential, or arbitrary internal exception details.
    response = AccessErrorResponse(detail=AccessErrorDetail(
        code=caught.value.code, message="Request cannot continue."))
    wire = response.model_dump_json()
    assert "session-row-id" not in wire and "intern" not in wire and "artifact-1" not in wire


def test_session_failure_remains_a_stale_context_but_requires_new_login():
    failure = SessionExpired("Authentication required.")
    assert isinstance(failure, StaleContext)
    assert ACCESS_ERROR_STATUS[failure.code] == 401
    assert ACCESS_ERROR_STATUS[StaleContext.code] == 409

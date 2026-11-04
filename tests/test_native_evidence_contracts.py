"""Native evidence contracts and live dispatch authority, without services."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, RequestContext,
    ResolvedIdentity, ScopedUserContext, StaleContext, guarded_dispatch,
    validate_dispatch_evidence, validate_grounded_evidence,
)
from src.contracts import (
    AggregateEvidence, Answer, Chunk, Evidence, GenerationRequest, GroundedEvidence,
    PassageEvidence, QueryResponse, RecordField, RecordSource, ScopeBinding,
    SnapshotRef, StructuredRecord, ToolResult, Trace, UserContext, validate_evidence,
)


@pytest.fixture
def boundary():
    now = datetime(2026, 11, 4, tzinfo=timezone.utc)
    snapshot = SnapshotRef(snapshot_id="snapshot", corpus_sha256="a" * 64,
        catalog_sha256="b" * 64, document_count=3)
    binding = ScopeBinding(scope_id="scope", user_id="intern", snapshot=snapshot,
        policy_revision=2, session_version=1)
    context = RequestContext(ResolvedIdentity(user_id="intern", session_id="private-session-row",
        session_version=1, expires_at=now + timedelta(hours=1)),
        LegacySetScope(binding=binding, allowed_doc_ids={"ticket", "other"}))

    class Guard:
        revoked = False
        def assert_current(self, supplied):
            assert supplied is context
            if self.revoked:
                raise StaleContext("Policy is no longer current")

    return SimpleNamespace(now=now, context=context, guard=Guard(),
        user=ScopedUserContext.from_context(context), binding=binding, snapshot=snapshot)


def record(boundary, *, doc="ticket", scope=None):
    return StructuredRecord(scope=scope or boundary.binding,
        citation_id="jira-record:" + "c" * 64 + ":" + doc, record_id=doc, doc_id=doc,
        revision_id="d" * 64, native_key="ATL-1", status="open", priority=None, project_id="atlas",
        source=RecordSource(feed_id="c" * 64, repository="pinned-repository",
            git_commit="e" * 40, git_blob="f" * 40,
            path="generated_data/sources/jira/ticket.json", raw_sha256="1" * 64,
            parser_version="1"),
        field_provenance=tuple(RecordField(field=name, json_pointer=pointer, state=state)
            for name, pointer, state in (("native_key", "/key", "verified"),
                ("status", "/status", "verified"), ("priority", "/priority", "missing"),
                ("project_id", "/project", "verified"))))


def aggregate(boundary):
    return AggregateEvidence(citation_id="aggregate:zero", artifact_id="zero", scope=boundary.binding,
        template_id="jira-count", template_version="1", count=0, lineage_sha256="2" * 64,
        coverage="verified_records", created_at=boundary.now)


def passage(boundary):
    chunk = Chunk(chunk_id="ticket:0:4", doc_id="ticket", source_type="jira",
                  text="Text", start_char=0, end_char=4)
    return PassageEvidence(citation_id=chunk.chunk_id, chunk=chunk, score=1,
        retrieval_method="hybrid", snapshot_id=boundary.snapshot.snapshot_id, revision_id="d" * 64)


class CanonicalRecords:
    def __init__(self, *values):
        self.expected = {value.record_id: value for value in values}
        self.calls = []
    def __call__(self, values):
        assert type(values) is tuple
        self.calls.append(values)
        if any(self.expected.get(value.record_id) != value for value in values):
            raise AccessDenied("Native record differs from authoritative data")


class Registry:
    def __init__(self, value):
        self.value = value
    def resolve(self, artifact_id):
        return self.value if artifact_id == self.value.artifact_id else None


def response(boundary, sources):
    return QueryResponse(snapshot=boundary.snapshot,
        answer=Answer(status="answered", text="Supported.",
                      citation_ids=tuple(item.citation_id for item in sources)),
        sources=tuple(sources), trace=Trace(run_id="run", stop_reason="single_pass"))


def test_three_native_tags_roundtrip_without_invented_chunks_and_keep_tool_abi(boundary):
    native = record(boundary)
    sources = (passage(boundary), aggregate(boundary), native)
    result = response(boundary, sources)
    copied = QueryResponse.model_validate_json(result.model_dump_json())
    assert tuple(item.kind for item in copied.sources) == ("passage", "aggregate", "record")
    assert copied == result and copied.contract_version == "3"
    for item in sources:
        assert TypeAdapter(GroundedEvidence).validate_json(item.model_dump_json()) == item
    producer = ToolResult(records=(native,), coverage="verified_records")
    assert producer.records == (native,) and producer.evidence == ()
    assert native.citation_id == "jira-record:" + "c" * 64 + ":ticket"
    assert "chunk" not in native.model_dump()


def test_record_requires_authoritative_callback_and_receives_revalidated_copies(boundary):
    source = record(boundary)
    with pytest.raises(AccessDenied, match="authoritative"):
        validate_grounded_evidence(boundary.context, (source,), boundary.guard)
    canonical = CanonicalRecords(source)
    validate_grounded_evidence(boundary.context, (source,), boundary.guard, record_validator=canonical)
    assert canonical.calls == [(source,)] and canonical.calls[0][0] is not source


def test_explicit_version_two_cannot_hide_a_record_but_missing_version_upgrades(boundary):
    value = response(boundary, (record(boundary),))
    payload = value.model_dump(mode="json")
    payload["contract_version"] = "2"
    with pytest.raises(ValidationError, match="version 3"):
        QueryResponse.model_validate(payload)
    payload.pop("contract_version")
    assert QueryResponse.model_validate(payload).contract_version == "3"
    assert QueryResponse.model_validate(value).contract_version == "3"


@pytest.mark.parametrize("kind", ["empty", "passage", "aggregate"])
def test_version_two_defaults_remain_and_explicit_version_three_is_not_downcast(boundary, kind):
    sources = {"empty": (), "passage": (passage(boundary),), "aggregate": (aggregate(boundary),)}[kind]
    if sources:
        value = response(boundary, sources)
    else:
        value = QueryResponse(snapshot=boundary.snapshot,
            answer=Answer(status="insufficient_evidence", text="No evidence."),
            trace=Trace(run_id="empty", stop_reason="single_pass"))
    assert value.contract_version == "2"
    payload = value.model_dump(mode="json")
    payload["contract_version"] = "3"
    parsed = QueryResponse.model_validate(payload)
    assert parsed.contract_version == "3"
    assert QueryResponse.model_validate_json(parsed.model_dump_json()).contract_version == "3"


@pytest.mark.parametrize("field,value", [
    ("scope_id", "another"), ("user_id", "executive"), ("policy_revision", 3),
    ("session_version", 2),
    ("snapshot", SnapshotRef(snapshot_id="other", corpus_sha256="a" * 64,
                             catalog_sha256="b" * 64, document_count=3)),
])
def test_record_cannot_cross_scope_owner_versions_or_snapshot_before_callback(boundary, field, value):
    source = record(boundary, scope=boundary.binding.model_copy(update={field: value}))
    callback = CanonicalRecords(source)
    with pytest.raises(AccessDenied):
        validate_grounded_evidence(boundary.context, (source,), boundary.guard, record_validator=callback)
    assert callback.calls == []


@pytest.mark.parametrize("change", [
    {"revision_id": "3" * 64}, {"status": "closed"},
    {"field_provenance": (RecordField(field="native_key", json_pointer="/key", state="verified"),
        RecordField(field="status", json_pointer="/status", state="verified"),
        RecordField(field="priority", json_pointer="/priority", state="invalid"),
        RecordField(field="project_id", json_pointer="/project", state="verified"))},
])
def test_validly_shaped_revision_fields_and_provenance_still_require_exact_source(boundary, change):
    source = record(boundary)
    with pytest.raises(AccessDenied, match="authoritative"):
        validate_grounded_evidence(boundary.context, (source.model_copy(update=change),), boundary.guard,
                                   record_validator=CanonicalRecords(source))


@pytest.mark.parametrize("change", [
    {"revision_id": "invalid"}, {"record_id": "forged"}, {"kind": "passage"},
    {"field_provenance": ()},
    {"source": {"path": "generated_data/sources/slack/ticket.json"}},
])
def test_unvalidated_record_copies_reject_before_authoritative_lookup(boundary, change):
    source = record(boundary).model_copy(update=change)
    callback = CanonicalRecords(source)
    with pytest.raises(EvidenceValidationError):
        validate_grounded_evidence(boundary.context, (source,), boundary.guard, record_validator=callback)
    assert callback.calls == []
    with pytest.raises(ValidationError):
        response(boundary, (source,))


def test_denied_record_and_duplicate_citations_do_not_reach_callback(boundary):
    source = record(boundary, doc="secret")
    callback = CanonicalRecords(source)
    with pytest.raises(AccessDenied):
        validate_grounded_evidence(boundary.context, (source,), boundary.guard, record_validator=callback)
    assert callback.calls == []
    source = record(boundary)
    with pytest.raises(EvidenceValidationError, match="Duplicate"):
        validate_grounded_evidence(boundary.context, (source, source), boundary.guard, record_validator=callback)


@pytest.mark.parametrize("result", [False, True, (), {"valid": True}])
def test_record_callback_is_validate_or_raise_not_a_truthiness_predicate(boundary, result):
    with pytest.raises(EvidenceValidationError, match="return None"):
        validate_grounded_evidence(boundary.context, (record(boundary),), boundary.guard,
                                   record_validator=lambda _: result)


def test_revocation_during_record_validation_prevents_release(boundary):
    source = record(boundary)
    def validate(values):
        assert values == (source,)
        boundary.guard.revoked = True
    with pytest.raises(StaleContext):
        validate_grounded_evidence(boundary.context, (source,), boundary.guard, record_validator=validate)


def test_revoking_dispatch_resets_private_authority_even_when_exit_guard_also_fails(boundary):
    source = record(boundary)
    def revoke(_):
        boundary.guard.revoked = True
    with pytest.raises(StaleContext):
        with guarded_dispatch(boundary.context, boundary.guard, record_validator=revoke):
            validate_evidence(boundary.user, (source,))
    with pytest.raises(AccessDenied, match="active dispatch"):
        validate_dispatch_evidence(boundary.user, (source,))


def test_invalid_inner_callback_does_not_replace_outer_dispatch_authority(boundary):
    source = record(boundary)
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        with pytest.raises(EvidenceValidationError):
            with guarded_dispatch(boundary.context, boundary.guard, record_validator=False):
                pytest.fail("Invalid callback cannot enter dispatch")
        validate_evidence(boundary.user, (source,))


def test_native_generation_checks_passage_snapshot_against_active_binding(boundary):
    source = record(boundary)
    wrong = passage(boundary).model_copy(update={"snapshot_id": "another"})
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        with pytest.raises(ValidationError, match="another snapshot"):
            GenerationRequest(question="Question", user=boundary.user, evidence=(wrong, source))


def test_record_callback_does_not_replace_zero_aggregate_registry(boundary):
    zero = aggregate(boundary)
    with pytest.raises(AccessDenied, match="authoritative"):
        validate_grounded_evidence(boundary.context, (zero,), boundary.guard,
                                   record_validator=lambda _: None)
    def never(_):
        pytest.fail("A zero aggregate must not call a record validator")
    validate_grounded_evidence(boundary.context, (zero,), boundary.guard, Registry(zero), record_validator=never)


def test_response_requires_snapshot_and_one_native_scope_but_is_not_authority(boundary):
    source = record(boundary)
    value = response(boundary, (source,))  # Structural DTOs are not registry access.
    with pytest.raises(ValidationError):
        QueryResponse(answer=value.answer, sources=value.sources, trace=value.trace)
    with pytest.raises(ValidationError, match="snapshot"):
        response(boundary, (source.model_copy(update={"scope": boundary.binding.model_copy(update={
            "snapshot": boundary.snapshot.model_copy(update={"catalog_sha256": "4" * 64})})}),))
    with pytest.raises(ValidationError, match="permission scopes"):
        response(boundary, (aggregate(boundary), source.model_copy(update={
            "scope": boundary.binding.model_copy(update={"scope_id": "another"})})))


@pytest.mark.parametrize("kind", ["record", "aggregate"])
def test_native_generation_requires_exact_sealed_user_and_active_authority(boundary, kind):
    source = record(boundary) if kind == "record" else aggregate(boundary)
    kwargs = {"record_validator": CanonicalRecords(source)} if kind == "record" else {"aggregates": Registry(source)}
    with pytest.raises(AccessDenied):
        validate_evidence(boundary.user, (source,))
    with guarded_dispatch(boundary.context, boundary.guard, **kwargs):
        request = GenerationRequest(question="What is supported?", user=boundary.user, evidence=(source,))
        assert request.user is boundary.user and request.evidence == (source,)
        wire = request.model_dump_json(serialize_as_any=True)
        assert all(secret not in wire for secret in ("_seal", "record_validator", "private-session-row", "allowed_positions"))
        for user in (UserContext(user_id="intern", allowed_doc_ids={"ticket"}),
                     boundary.user.model_copy(), UserContext.model_validate(boundary.user.model_dump())):
            with pytest.raises(AccessDenied):
                validate_evidence(user, (source,))
    with pytest.raises(AccessDenied):
        validate_dispatch_evidence(boundary.user, (source,))


def test_metadata_and_even_active_context_without_native_resolvers_cannot_authorize_generation(boundary):
    with guarded_dispatch(boundary.context, boundary.guard):
        for source in (record(boundary), aggregate(boundary)):
            with pytest.raises(ValidationError, match="authoritative"):
                GenerationRequest(question="Question", user=boundary.user, evidence=(source,))


@pytest.mark.parametrize("throw", [False, True])
def test_nested_dispatch_clears_validators_then_restores_them_even_after_exception(boundary, throw):
    source = record(boundary)
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        validate_evidence(boundary.user, (source,))
        try:
            with guarded_dispatch(boundary.context, boundary.guard):
                with pytest.raises(AccessDenied, match="authoritative"):
                    validate_evidence(boundary.user, (source,))
                if throw:
                    raise RuntimeError("nested failure")
        except RuntimeError:
            assert throw
        validate_evidence(boundary.user, (source,))
    with pytest.raises(AccessDenied):
        validate_evidence(boundary.user, (source,))


def test_legacy_passage_validation_remains_available_without_guard(boundary):
    source = passage(boundary)
    legacy = Evidence.model_validate(source.model_dump(exclude={"kind", "snapshot_id", "revision_id"}))
    user = UserContext(user_id="legacy", allowed_doc_ids={"ticket"})
    validate_evidence(user, (legacy,))
    assert GenerationRequest(question="Question", user=user, evidence=(legacy,)).evidence == (legacy,)
    with pytest.raises(ValueError):
        validate_evidence(UserContext(), (legacy,))


def test_native_path_rejects_legacy_passage_beside_native_evidence(boundary):
    source = record(boundary)
    legacy = Evidence.model_validate(passage(boundary).model_dump(exclude={"kind", "snapshot_id", "revision_id"}))
    with guarded_dispatch(boundary.context, boundary.guard, record_validator=CanonicalRecords(source)):
        with pytest.raises(EvidenceValidationError, match="Unsupported"):
            validate_evidence(boundary.user, (legacy, source))

"""V2 wire compatibility, provenance, strictness and public auth boundaries."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.auth.contracts import (
    ACCESS_ERROR_STATUS, AccessErrorDetail, CurrentUser, LoginRequest, SessionResponse,
)
from src.contracts import Evidence, QueryRequest, QueryResponse
from src.contracts_v2 import (
    AggregateEvidence, BoundFilter, PassageEvidence, QueryResponseV2, ScopeBinding, SnapshotRef,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/contracts_v2.json").read_text())


def payload():
    return copy.deepcopy(FIXTURE)


def test_mixed_passage_and_zero_count_round_trip_without_fake_document():
    result = QueryResponseV2.model_validate(FIXTURE)
    passage, aggregate = result.sources
    assert isinstance(passage, PassageEvidence)
    assert isinstance(aggregate, AggregateEvidence) and aggregate.count == 0
    assert not hasattr(aggregate, "chunk")
    assert QueryResponseV2.model_validate_json(result.model_dump_json()) == result
    wire = result.model_dump(mode="json")
    assert wire["sources"][1]["coverage"] == "synthetic_fixture"
    assert "chunk" not in wire["sources"][1]


def test_v2_passage_preserves_legacy_evidence_fields_and_offsets():
    source = QueryResponseV2.model_validate(FIXTURE).sources[0]
    legacy_fields = source.model_dump(exclude={"kind", "snapshot_id", "revision_id"})
    legacy = Evidence.model_validate(legacy_fields)
    assert legacy.citation_id == source.citation_id
    assert legacy.chunk == source.chunk
    assert source.chunk.end_char - source.chunk.start_char == len(source.chunk.text)


@pytest.mark.parametrize("change", ["citation", "duplicate", "passage_snapshot", "aggregate_snapshot", "kind"])
def test_response_rejects_inconsistent_provenance(change):
    data = payload()
    if change == "citation":
        data["answer"]["citation_ids"] = ["not-supplied"]
    elif change == "duplicate":
        data["sources"].append(data["sources"][0])
    elif change == "passage_snapshot":
        data["sources"][0]["snapshot_id"] = "another"
    elif change == "aggregate_snapshot":
        data["sources"][1]["scope"]["snapshot"]["catalog_sha256"] = "d" * 64
    else:
        data["sources"][1]["kind"] = "passage"
    with pytest.raises(ValidationError):
        QueryResponseV2.model_validate(data)


@pytest.mark.parametrize("count", [-1, True, 1.0, "1"])
def test_aggregate_count_is_a_strict_nonnegative_integer(count):
    data = payload()["sources"][1]
    data["count"] = count
    with pytest.raises(ValidationError):
        AggregateEvidence.model_validate(data)


def test_aggregate_rejects_duplicate_or_untyped_filters_and_unknown_coverage():
    data = payload()["sources"][1]
    data["filters"] *= 2
    with pytest.raises(ValidationError, match="Duplicate"):
        AggregateEvidence.model_validate(data)
    with pytest.raises(ValidationError):
        BoundFilter(field="status", value=["open"])
    with pytest.raises(ValidationError):
        BoundFilter(field="count", value=1.5)
    data = payload()["sources"][1]
    data["coverage"] = "all_company_tickets"
    with pytest.raises(ValidationError):
        AggregateEvidence.model_validate(data)


def test_epoch_hash_and_timezone_fields_reject_ambiguous_values():
    data = payload()["sources"][1]
    data["created_at"] = "2026-11-02T12:00:00"
    with pytest.raises(ValidationError):
        AggregateEvidence.model_validate(data)
    binding = payload()["sources"][1]["scope"]
    binding["policy_revision"] = True
    with pytest.raises(ValidationError):
        ScopeBinding.model_validate(binding)
    snapshot = payload()["snapshot"]
    snapshot["catalog_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        SnapshotRef.model_validate(snapshot)


def test_nested_v2_contracts_are_immutable():
    aggregate = QueryResponseV2.model_validate(FIXTURE).sources[1]
    with pytest.raises(ValidationError):
        aggregate.scope.policy_revision = 2
    with pytest.raises(ValidationError):
        aggregate.filters[0].value = "closed"
    with pytest.raises(TypeError):
        aggregate.filters[0] = BoundFilter(field="status", value="closed")


def test_v2_schema_is_explicitly_opt_in_and_legacy_requests_keep_rejecting_identity():
    schema = QueryResponseV2.model_json_schema()
    assert schema["properties"]["sources"]["items"]["discriminator"]["propertyName"] == "kind"
    with pytest.raises(ValidationError):
        QueryResponse.model_validate(FIXTURE)
    for field in ("user_id", "role", "allowed_doc_ids", "scope"):
        with pytest.raises(ValidationError):
            QueryRequest.model_validate({"question": "policy?", field: "executive"})


def test_auth_input_cannot_grant_roles_and_passwords_are_redacted():
    request = LoginRequest(login="intern", password="synthetic-secret")
    assert request.password.get_secret_value() == "synthetic-secret"
    assert "synthetic-secret" not in request.model_dump_json()
    assert "synthetic-secret" not in repr(request)
    for bad in ({"role": "executive"}, {"user_id": "ceo"}, {"allowed_doc_ids": ["exec"]}):
        with pytest.raises(ValidationError):
            LoginRequest(login="intern", password="synthetic-secret", **bad)
    with pytest.raises(ValidationError):
        LoginRequest(login="intern", password="")
    with pytest.raises(ValidationError):
        LoginRequest(login=" two users ", password="synthetic-secret")


def test_public_session_does_not_expose_token_digest_or_document_grants():
    user = CurrentUser(user_id="intern", display_name="Intern", project_ids=("atlas",))
    response = SessionResponse(user=user, csrf_token="csrf-for-js-memory", expires_at="2026-11-02T13:00:00Z")
    assert set(response.model_dump()) == {"user", "csrf_token", "expires_at"}
    with pytest.raises(ValidationError):
        CurrentUser(user_id="intern", display_name="Intern", allowed_doc_ids=("exec",))
    assert ACCESS_ERROR_STATUS["stale_context"] == 409
    assert ACCESS_ERROR_STATUS["policy_unavailable"] == 503
    assert AccessErrorDetail(code="scope_denied", message="Access denied").code == "scope_denied"

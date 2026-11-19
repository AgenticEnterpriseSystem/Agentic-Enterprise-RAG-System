"""Actual RLS, complete-relation counts and native lookup provenance."""

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from src.access.contracts import AccessDenied, EvidenceValidationError, SessionExpired, StaleContext, validate_grounded_evidence
from src.access.policy import PolicyResolver
from src.auth.accounts import disable_account
from src.contracts import AggregateEvidence, StructuredRecord, ToolRequest, UserContext
from src.db import schema
from tests.db_support import pg_urls, url_string
from tests.jira_support import env, ready, source


def execute(data, operation="count", filters=None, *, principal="intern", limit=10, context=None, guard=None, **kwargs):
    return data.tool.execute(ToolRequest(source_type="jira", operation=operation, filters=filters or {}, limit=limit,
        user=UserContext(user_id="untrusted-compatibility-ceo", role="ceo", allowed_doc_ids={"hidden", "unknown"})),
        context=context or data.contexts[principal], guard=guard or data.resolver, **kwargs)


def test_no_feed_is_unknown_even_when_ticket_table_is_empty(env):
    result = execute(env, filters={"status": "In Progress"})
    assert result.coverage == "unknown" and result.count is None and not result.evidence
    assert result.limitation and not result.records


def test_native_lookup_is_scoped_bound_and_never_fabricates_canonical_prose(ready):
    result = execute(ready, "lookup", {"doc_id": "one"})
    assert not result.evidence and len(result.records) == 1
    row = result.records[0]
    assert isinstance(row, StructuredRecord) and row.scope == ready.contexts["intern"].binding
    assert row.record_id == row.doc_id == "one" and row.native_key == "SUP-100"
    assert row.status == "In Progress" and row.project_id == "customer-support"
    assert row.source.git_commit == ready.source.inputs.commit
    assert row.source.feed_id == ready.source.inputs.feed_id
    assert "In Progress" not in ready.source.rows[0]["content"]
    assert {field.field for field in row.field_provenance} == {"native_key", "status", "priority", "project_id"}
    assert execute(ready, "lookup", {"doc_id": "hidden"}).records == ()
    assert execute(ready, "lookup", {"doc_id": "does-not-exist"}).records == ()


def test_duplicate_native_keys_are_ambiguous_only_within_the_visible_relation(ready):
    result = execute(ready, "lookup", {"native_key": "SUP-100"}, limit=1)
    assert result.ambiguous and result.truncated and len(result.records) == 1
    narrow = PolicyResolver(ready.database, ready.source.snapshot, doc_ids=("one",))
    context = narrow.resolve(ready.identities["intern"])
    result = execute(ready, "lookup", {"native_key": "SUP-100"}, context=context, guard=narrow)
    assert not result.ambiguous and not result.truncated and [row.doc_id for row in result.records] == ["one"]
    assert "hidden" not in result.model_dump_json()


def test_native_records_require_exact_authoritative_fields_and_provenance(ready):
    row = execute(ready, "lookup", {"doc_id": "one"}).records[0]
    context = ready.contexts["intern"]
    ready.tool.validate_records(context, (row,), guard=ready.resolver)
    forged_status = row.model_copy(update={"status": "Closed"})
    forged_source = row.model_copy(update={"source": row.source.model_copy(update={"raw_sha256": "f" * 64})})
    for forged in (forged_status, forged_source):
        with pytest.raises(AccessDenied):
            ready.tool.validate_records(context, (forged,), guard=ready.resolver)
    with pytest.raises(AccessDenied):
        ready.tool.validate_records(ready.contexts["executive"], (row,), guard=ready.resolver)
    with pytest.raises(AccessDenied):
        ready.tool.validate_records(context, (row, row), guard=ready.resolver)
    with pytest.raises(EvidenceValidationError):
        ready.tool.validate_records(context, (row.model_copy(update={"field_provenance": ()}),), guard=ready.resolver)
    with pytest.raises(EvidenceValidationError):
        ready.tool.validate_records(context, ({"status": "secret forged input"},), guard=ready.resolver)
    with pytest.raises(ValidationError):
        row.source.raw_sha256 = "f" * 64
    with pytest.raises(ValidationError):
        row.field_provenance[0].state = "missing"


def test_exact_filtered_count_uses_entire_authorized_relation_not_limit_or_grant_joins(ready):
    result = execute(ready, filters={"native_key": "SUP-100"}, limit=1)
    assert result.count == 2 and len(result.evidence) == 1
    artifact = result.evidence[0]
    assert isinstance(artifact, AggregateEvidence) and artifact.coverage == "synthetic_fixture"
    assert artifact.template_id == "jira-canonical-record-count"
    assert artifact.scope == ready.contexts["intern"].binding
    assert artifact.filters[0].field == "native_key" and artifact.filters[0].value == "SUP-100"
    assert execute(ready, filters={"status": "In Progress"}).count == 1
    assert execute(ready, filters={"status": "in progress"}).count == 0
    validate_grounded_evidence(ready.contexts["intern"], result.evidence, ready.resolver,
                              ready.tool.registry(ready.contexts["intern"], ready.resolver))


def test_zero_has_owned_reproducible_artifact_and_cannot_be_forged_or_read_by_other_user(ready):
    first = execute(ready, filters={"status": "Exact absent status"})
    second = execute(ready, filters={"status": "Exact absent status"})
    assert first.count == 0 and first.coverage == "synthetic_fixture"
    left, right = first.evidence[0], second.evidence[0]
    assert left.lineage_sha256 == right.lineage_sha256 and left.artifact_id != right.artifact_id
    registry = ready.tool.registry(ready.contexts["intern"], ready.resolver)
    assert registry.resolve(left.artifact_id) == left
    assert ready.tool.registry(ready.contexts["executive"], ready.resolver).resolve(left.artifact_id) is None
    with pytest.raises(AccessDenied):
        validate_grounded_evidence(ready.contexts["intern"], (left.model_copy(update={"count": 999}),), ready.resolver, registry)


def test_missing_authorized_field_is_unknown_not_zero_but_hidden_missing_fields_do_not_leak(ready):
    assert execute(ready, filters={"status": "In Progress"}).count == 1
    result = execute(ready, filters={"status": "In Progress"}, principal="executive")
    assert result.count is None and result.coverage == "unknown" and not result.evidence
    assert "missing_records" not in result.model_dump_json() and "expected_records" not in result.model_dump_json()
    unknown = execute(ready, "lookup", {"doc_id": "unknown"}, principal="executive").records[0]
    assert unknown.status is None
    assert next(field for field in unknown.field_provenance if field.field == "status").state == "missing"
    assert execute(ready, principal="executive").count == 4


@pytest.mark.parametrize("filters", [{"sql": "SELECT *"}, {"status": True}, {"priority": 1},
    {"status": ""}, {"status": "x\0y"}, {"after": "2020-01-01"}])
def test_unsupported_fields_and_values_reject_without_minting_an_artifact(ready, filters):
    with pytest.raises(ValueError):
        execute(ready, filters=filters)
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.aggregate_results)).scalar_one() == 0


def test_sql_injection_is_an_exact_parameter_value_and_cannot_widen_the_relation(ready):
    assert execute(ready, filters={"status": "' OR TRUE --"}).count == 0
    assert execute(ready, "filter", {"native_key": "' OR TRUE --"}).records == ()


def test_revoked_grants_or_disabled_session_stop_before_tool_output(ready):
    ready.repository.revoke(ready.manifest)
    with pytest.raises(StaleContext):
        execute(ready)
    disable_account("intern", url=url_string(ready.urls.migration))
    with pytest.raises(SessionExpired):
        execute(ready)


def test_caller_cannot_attach_an_aggregate_to_another_users_owned_run(ready):
    other = execute(ready, principal="executive").evidence[0]
    with ready.owner.connect() as connection:
        run_id = connection.execute(sa.select(schema.aggregate_results.c.run_id).where(
            schema.aggregate_results.c.artifact_id == other.artifact_id)).scalar_one()
    with pytest.raises(AccessDenied):
        execute(ready, run_id=run_id)


def test_operator_deny_all_is_a_valid_zero_over_proven_feed_without_hidden_counts(ready):
    resolver = PolicyResolver(ready.database, ready.source.snapshot, doc_ids=())
    context = resolver.resolve(ready.identities["intern"])
    result = execute(ready, filters={"status": "In Progress"}, context=context, guard=resolver)
    assert result.count == 0 and result.evidence[0].scope == context.binding
    assert not result.records and "expected_records" not in result.model_dump_json()

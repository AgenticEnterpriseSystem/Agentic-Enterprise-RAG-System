"""Native Jira lookup/filter/count over the complete RLS-authorized relation.

Counts mean canonical Jira source records in a pinned benchmark, not unique
live Jira issues. Duplicate native keys remain distinct records. This adapter
uses the existing ToolRequest/ToolResult; explicit context/guard arguments are
mandatory and the compatibility request.user field grants no authority.
Registration with SearchAgent and rendering native records remain WIRE-01.
"""

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from pydantic import ValidationError

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, PolicyUnavailable, RequestContext,
    SnapshotMismatch, StaleContext, validate_grounded_evidence,
)
from src.contracts import (
    AggregateEvidence, BoundFilter, RecordField, RecordSource, SnapshotRef,
    StructuredRecord, ToolRequest, ToolResult,
)
from src.data.structured import FIELDS, canonical
from src.db import schema
from src.db.runtime import Database, DatabaseRoleError
from src.storage.fingerprints import ValidatedCatalog

TEMPLATE_ID = "jira-canonical-record-count"
TEMPLATE_VERSION = "1"
CARDINALITY = "Counts canonical Jira records in this pinned benchmark; native keys may repeat."
_FILTERS = {"doc_id", "native_key", "status", "priority", "project_id"}
_FEED_FIELDS = ("feed_id", "snapshot_id", "source_type", "coverage", "state", "expected_records",
                "imported_records", "source_set_sha256", "corpus_sha256", "git_commit", "repository")


class JiraTool:
    def __init__(self, database: Database, snapshot: SnapshotRef, *, catalog: ValidatedCatalog | None = None):
        if not isinstance(database, Database) or not isinstance(snapshot, SnapshotRef):
            raise TypeError("Jira tools require the runtime Database and canonical snapshot")
        self.database = database
        self.snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        self._catalog = None
        self._jira_doc_ids = ()
        if catalog is not None:
            self._catalog = ValidatedCatalog.require(catalog, self.snapshot, issuer=database)
            # Reuse frozen ID references. No fields, bodies, permissions or
            # second full catalogue are materialized for this optimization.
            self._jira_doc_ids = tuple(doc_id for doc_id, source in
                zip(self._catalog.doc_ids, self._catalog.sources) if source == 'jira')

    def _current(self, context, guard):
        if not isinstance(context, RequestContext):
            raise EvidenceValidationError("Jira tools require a resolved request context")
        guard.assert_current(context)
        context.assert_bound()
        if context.binding.snapshot != self.snapshot:
            raise SnapshotMismatch("Jira request belongs to another snapshot")

    def _request(self, request):
        if not isinstance(request, ToolRequest):
            raise ValueError("A typed structured request is required")
        request = ToolRequest.model_validate(request.model_dump(warnings=False))
        if request.source_type != "jira":
            raise AccessDenied("This adapter does not serve the requested source")
        if not set(request.filters) <= _FILTERS or any(type(value) is not str or not value.strip()
                or len(value) > 512 or "\0" in value for value in request.filters.values()):
            raise ValueError("Unsupported Jira filter field or value")
        if request.operation == "lookup" and len(set(request.filters) & {"doc_id", "native_key"}) != 1:
            raise ValueError("Jira lookup requires one canonical document ID or native key")
        return request

    def _scope_predicate(self, context, guard):
        """Only narrow the SQL relation; live RLS remains authoritative.

        The optional issued catalogue supplies indexable native document IDs,
        not permission. Every operation derives its subset from the current
        bound scope, under guards before and after the mask computation.
        """
        self._current(context, guard)
        if self._catalog is None:
            return None
        ValidatedCatalog.require(self._catalog, self.snapshot, issuer=self.database)
        try:
            mask = tuple(context.scope.mask_for(self.snapshot, self._jira_doc_ids))
        except (ValueError, TypeError):
            raise EvidenceValidationError('Native scope requires one boolean per canonical Jira document') from None
        if len(mask) != len(self._jira_doc_ids) or any(type(value) is not bool for value in mask):
            raise EvidenceValidationError('Native scope requires one boolean per canonical Jira document')
        allowed = tuple(doc_id for doc_id, permitted in zip(self._jira_doc_ids, mask) if permitted)
        self._current(context, guard)
        if not allowed:
            return sa.false()
        # One bound array avoids a SQL-text/parameter explosion; built-in text
        # equality is indexable before the unchanged document RLS function.
        return schema.tickets.c.doc_id == sa.any_(sa.bindparam(
            'authorized_jira_doc_ids', value=allowed, type_=ARRAY(sa.Text())))

    def _feed(self, connection):
        table = schema.ticket_feeds
        row = connection.execute(sa.select(*(table.c[name] for name in _FEED_FIELDS)).where(
            table.c.snapshot_id == self.snapshot.snapshot_id, table.c.source_type == "jira")).mappings().one_or_none()
        if row is None or row["state"] != "ready" or row["imported_records"] != row["expected_records"]:
            return None
        if row["corpus_sha256"] != self.snapshot.corpus_sha256:
            raise EvidenceValidationError("Jira feed differs from the canonical snapshot")
        return row

    @staticmethod
    def _base(feed, scope_predicate=None):
        table = schema.tickets
        return (table.c.snapshot_id == feed["snapshot_id"], table.c.feed_id == feed["feed_id"],
                table.c.coverage == feed["coverage"]) + (() if scope_predicate is None else (scope_predicate,))

    def _predicates(self, feed, request, scope_predicate=None):
        return self._base(feed, scope_predicate) + tuple(
            schema.tickets.c[key] == value for key, value in sorted(request.filters.items()))

    def _covered(self, connection, feed, request, scope_predicate=None):
        # Coverage is checked BEFORE filters and only over visible records. A
        # missing hidden field neither leaks metadata nor invalidates a complete
        # authorized relation; NULL must never silently become a zero match.
        fields = sorted(set(request.filters) & FIELDS.keys())
        if not fields:
            return True
        missing = sa.select(sa.literal(1)).select_from(schema.tickets).where(
            *self._base(feed, scope_predicate),
            sa.or_(*(schema.tickets.c[name].is_(None) for name in fields)))
        return not connection.execute(sa.select(sa.exists(missing))).scalar_one()

    @staticmethod
    def _unavailable():
        return ToolResult(coverage="unknown", limitation="Verified Jira records or required field coverage are unavailable for this scope.")

    def _record(self, context, row, feed):
        if not context.scope.permits(row["doc_id"]):
            raise AccessDenied("Jira record is unavailable in this scope")
        provenance = row["field_provenance"]
        return StructuredRecord(scope=context.binding,
            citation_id="jira-record:" + feed["feed_id"] + ":" + row["doc_id"],
            record_id=row["ticket_id"], doc_id=row["doc_id"], revision_id=row["revision_id"],
            native_key=row["native_key"], status=row["status"], priority=row["priority"], project_id=row["project_id"],
            source=RecordSource(feed_id=feed["feed_id"], **provenance["source"]),
            field_provenance=tuple(RecordField.model_validate(value) for value in provenance["fields"]))

    def execute(self, request: ToolRequest, *, context: RequestContext, guard, run_id=None) -> ToolResult:
        self._current(context, guard)
        request = self._request(request)
        scope_predicate = self._scope_predicate(context, guard)
        try:
            with self.database.transaction(context) as connection:
                feed = self._feed(connection)
                if feed is None or not self._covered(connection, feed, request, scope_predicate):
                    result = self._unavailable()
                elif request.operation == "count":
                    result = self._count(connection, context, request, feed, run_id, scope_predicate)
                else:
                    rows = connection.execute(sa.select(schema.tickets).where(*self._predicates(feed, request, scope_predicate))
                        .order_by(schema.tickets.c.ticket_id).limit(request.limit + 1)).mappings().all()
                    result = ToolResult(records=tuple(self._record(context, row, feed) for row in rows[:request.limit]),
                        coverage=feed["coverage"], truncated=len(rows) > request.limit,
                        ambiguous=request.operation == "lookup" and "native_key" in request.filters and len(rows) > 1,
                        limitation=CARDINALITY)
        except (sa.exc.SQLAlchemyError, DatabaseRoleError):
            raise PolicyUnavailable("Verified Jira data is unavailable") from None
        self._current(context, guard)
        if result.evidence:
            validate_grounded_evidence(context, result.evidence, guard, self.registry(context, guard))
        if result.records:
            self.validate_records(context, result.records, guard=guard)
        self._current(context, guard)
        return result

    def validate_records(self, context, records, *, guard):
        """Native fields require exact canonical lookup, not provenance alone."""
        self._current(context, guard)
        if not isinstance(records, (tuple, list)) or len(records) > 100 or any(
                not isinstance(value, StructuredRecord) for value in records):
            raise EvidenceValidationError("Native record result is invalid or exceeds its bound")
        try:
            checked = tuple(StructuredRecord.model_validate(value.model_dump(warnings=False)) for value in records)
        except (ValidationError, ValueError, TypeError):
            raise EvidenceValidationError("Native record result is invalid") from None
        ids = [value.record_id for value in checked]
        if len(ids) != len(set(ids)) or any(value.scope != context.binding for value in checked):
            raise AccessDenied("Native record is unavailable in this scope")
        scope_predicate = self._scope_predicate(context, guard)
        try:
            with self.database.transaction(context) as connection:
                feed = self._feed(connection)
                if feed is None:
                    raise AccessDenied("Verified Jira feed is unavailable")
                rows = connection.execute(sa.select(schema.tickets).where(*self._base(feed, scope_predicate),
                    schema.tickets.c.ticket_id.in_(ids))).mappings().all()
                actual = {row["ticket_id"]: self._record(context, row, feed) for row in rows}
        except (sa.exc.SQLAlchemyError, DatabaseRoleError):
            raise PolicyUnavailable("Verified Jira data is unavailable") from None
        self._current(context, guard)
        if any(actual.get(value.record_id) != value for value in checked):
            raise AccessDenied("Native record differs from its authoritative source")

    def _count(self, connection, context, request, feed, run_id, scope_predicate=None):
        predicates = self._predicates(feed, request, scope_predicate)
        table = schema.tickets
        count = connection.execute(sa.select(sa.func.count()).select_from(table).where(*predicates)).scalar_one()
        filters = tuple(BoundFilter(field=key, value=value) for key, value in sorted(request.filters.items()))
        lineage = hashlib.sha256(b"rag-jira-count-lineage-v1\n" + canonical({
            "scope": context.binding.model_dump(mode="json"), "feed_id": feed["feed_id"],
            "source_set_sha256": feed["source_set_sha256"], "template_id": TEMPLATE_ID,
            "template_version": TEMPLATE_VERSION, "filters": [value.model_dump() for value in filters],
            "cardinality": "canonical_source_records", "count": count}) + b"\n")
        # Complete matching relation, never the lookup limit or a retrieved top-k.
        seen = 0
        with connection.execution_options(stream_results=True).execute(sa.select(table.c.ticket_id).where(*predicates)
                .order_by(table.c.ticket_id)) as rows:
            while batch := rows.fetchmany(1000):
                for row in batch:
                    lineage.update(canonical(row[0]) + b"\n")
                    seen += 1
        connection.execution_options(stream_results=False)
        if seen != count:
            raise StaleContext("Jira authorization changed during aggregation")
        owned = run_id is None
        run_id = uuid4().hex if owned else run_id
        now = datetime.now(timezone.utc)
        if owned:
            connection.execute(schema.query_runs.insert().values(run_id=run_id, owner_user_id=context.identity.user_id,
                snapshot_id=self.snapshot.snapshot_id, scope_id=context.binding.scope_id,
                policy_revision=context.binding.policy_revision, session_version=context.binding.session_version,
                config_sha256=hashlib.sha256(canonical({"template": TEMPLATE_ID, "version": TEMPLATE_VERSION})).hexdigest(),
                mode="baseline", question="Structured Jira canonical-record count", created_at=now))
        else:
            run = connection.execute(sa.select(schema.query_runs).where(
                schema.query_runs.c.run_id == run_id).with_for_update()).mappings().one_or_none()
            if run is None or (run["owner_user_id"], run["scope_id"], run["snapshot_id"], run["policy_revision"], run["session_version"]) != (
                context.identity.user_id, context.binding.scope_id, self.snapshot.snapshot_id,
                context.binding.policy_revision, context.binding.session_version):
                raise AccessDenied("Aggregate run is unavailable in this scope")
            if run['status'] != 'pending':
                raise AccessDenied("Aggregate run no longer accepts tool activity")
        artifact_id = uuid4().hex
        aggregate = AggregateEvidence(artifact_id=artifact_id, citation_id="aggregate:" + artifact_id,
            scope=context.binding, template_id=TEMPLATE_ID, template_version=TEMPLATE_VERSION,
            filters=filters, count=count, lineage_sha256=lineage.hexdigest(), coverage=feed["coverage"], created_at=now)
        connection.execute(schema.aggregate_results.insert().values(artifact_id=aggregate.artifact_id,
            citation_id=aggregate.citation_id, run_id=run_id, owner_user_id=context.identity.user_id,
            snapshot_id=self.snapshot.snapshot_id, scope_id=context.binding.scope_id,
            policy_revision=context.binding.policy_revision, session_version=context.binding.session_version,
            template_id=TEMPLATE_ID, template_version=TEMPLATE_VERSION,
            filters=[item.model_dump() for item in filters], operation="count", count=count,
            lineage_sha256=aggregate.lineage_sha256, coverage=feed["coverage"], created_at=now))
        if owned:
            connection.execute(schema.query_runs.update().where(schema.query_runs.c.run_id == run_id).values(
                status="answered", finished_at=now, estimated_cost_usd=0))
        return ToolResult(evidence=(aggregate,), count=count, coverage=feed["coverage"], limitation=CARDINALITY)

    def registry(self, context, guard):
        tool = self
        class Registry:
            def resolve(self, artifact_id):
                tool._current(context, guard)
                try:
                    with tool.database.transaction(context) as connection:
                        row = connection.execute(sa.select(schema.aggregate_results).where(
                            schema.aggregate_results.c.artifact_id == artifact_id)).mappings().one_or_none()
                except (sa.exc.SQLAlchemyError, DatabaseRoleError):
                    raise PolicyUnavailable("Verified Jira artifact is unavailable") from None
                tool._current(context, guard)
                if row is None:
                    return None
                if (row["owner_user_id"], row["scope_id"], row["snapshot_id"], row["policy_revision"], row["session_version"]) != (
                    context.identity.user_id, context.binding.scope_id, tool.snapshot.snapshot_id,
                    context.binding.policy_revision, context.binding.session_version):
                    raise AccessDenied("Aggregate artifact is unavailable in this scope")
                return AggregateEvidence(artifact_id=row["artifact_id"], citation_id=row["citation_id"],
                    scope=context.binding, template_id=row["template_id"], template_version=row["template_version"],
                    filters=tuple(BoundFilter.model_validate(value) for value in row["filters"]), count=row["count"],
                    lineage_sha256=row["lineage_sha256"], coverage=row["coverage"], created_at=row["created_at"])
        return Registry()

"""Current-owner run metadata under the existing exact-scope RLS policies.

No answers, raw questions, source bodies, model payloads or arbitrary trace JSON
are stored here. The legacy required question column holds a SHA256 marker.
Invalidation can leave a pending run; there is no privileged lifecycle fallback.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.access.contracts import (AccessDenied, EvidenceValidationError, PolicyUnavailable,
                                  RequestContext)
from src.contracts import Contract, Identifier, Sha256, Usage
from src.db import schema
from src.db.runtime import Database, DatabaseRoleError

RunStatus = Literal['pending', 'answered', 'insufficient_evidence', 'error', 'clarification']
ShortId = Annotated[str, Field(min_length=1, max_length=1024, pattern=r'^\S+$')]


class RunStepMetadata(Contract):
    tool: Annotated[str, Field(pattern=r'^[A-Za-z0-9_:-]{1,64}$')]
    round_number: Annotated[int, Field(strict=True, ge=1, le=10)]
    evidence_ids: tuple[ShortId, ...] = Field(default=(), max_length=1000)
    query_sha256: Sha256
    source_types: tuple[Annotated[str, Field(min_length=1, max_length=64, pattern=r'^\S+$')], ...] = Field(default=(), max_length=9)
    elapsed_ms: Annotated[float, Field(ge=0)]
    usage: Usage = Field(default_factory=Usage)
    input_payload_sha256: Sha256 | None = None

    @model_validator(mode='after')
    def bounded_metadata(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)) or len(self.source_types) != len(set(self.source_types)):
            raise ValueError('Duplicate run metadata references')
        if len(json.dumps(self.model_dump(mode='json'), allow_nan=False).encode()) > 512 * 1024:
            raise ValueError('Run metadata exceeds its bound')
        return self


class RunMetadata(Contract):
    run_id: Identifier
    snapshot_id: Identifier
    scope_id: Identifier
    policy_revision: Annotated[int, Field(strict=True, ge=0)]
    session_version: Annotated[int, Field(strict=True, ge=0)]
    config_sha256: Sha256
    mode: Literal['baseline', 'agentic']
    status: RunStatus
    created_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    usage: Usage
    latency_ms: Annotated[float, Field(ge=0)]
    step_count: Annotated[int, Field(strict=True, ge=0)]


class RunRepository:
    def __init__(self, database: Database, guard, *, clock=None):
        if not isinstance(database, Database) or not callable(getattr(guard, 'assert_current', None)):
            raise TypeError('Run storage requires the shared runtime database and freshness guard')
        self.database, self.guard = database, guard
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _current(self, context):
        if not isinstance(context, RequestContext):
            raise EvidenceValidationError('Run storage requires a resolved request context')
        self.guard.assert_current(context)
        context.assert_bound()

    @contextmanager
    def _transaction(self, context):
        self._current(context)
        try:
            with self.database.transaction(context) as connection:
                yield connection
        except (sa.exc.SQLAlchemyError, DatabaseRoleError):
            raise PolicyUnavailable('Current run storage is unavailable') from None
        finally:
            # The database transaction/pool slot is released before the guard
            # performs its own authoritative queries, including pool_size=1.
            self._current(context)

    @staticmethod
    def _id(run_id):
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 128 or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
            raise EvidenceValidationError('Invalid server run identity')
        return run_id

    def _row(self, connection, context, run_id, *, lock=False):
        statement = sa.select(schema.query_runs).where(schema.query_runs.c.run_id == self._id(run_id))
        if lock:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().one_or_none()
        binding = context.binding
        if row is None or (row['owner_user_id'], row['snapshot_id'], row['scope_id'],
                row['policy_revision'], row['session_version']) != (context.identity.user_id,
                binding.snapshot.snapshot_id, binding.scope_id, binding.policy_revision, binding.session_version):
            raise AccessDenied('Run is unavailable to this current scope')
        return row

    @staticmethod
    def _metadata(connection, row):
        count = connection.execute(sa.select(sa.func.count()).select_from(schema.run_steps)
            .where(schema.run_steps.c.run_id == row['run_id'])).scalar_one()
        return RunMetadata(**{key: row[key] for key in ('run_id', 'snapshot_id', 'scope_id',
            'policy_revision', 'session_version', 'config_sha256', 'mode', 'status', 'created_at',
            'finished_at', 'latency_ms')}, usage=Usage(**{key: row[key] for key in
                ('input_tokens', 'output_tokens', 'api_calls', 'estimated_cost_usd')}), step_count=count)

    def begin(self, context, run_id, *, mode, question, config_sha256):
        self._current(context)
        run_id = self._id(run_id)
        if mode not in ('baseline', 'agentic') or not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise EvidenceValidationError('Invalid admitted run request')
        if not isinstance(config_sha256, str) or not re.fullmatch('[0-9a-f]{64}', config_sha256):
            raise EvidenceValidationError('Invalid run configuration fingerprint')
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise EvidenceValidationError('Run clock must be aware')
        values = {'run_id': run_id, 'owner_user_id': context.identity.user_id,
            'snapshot_id': context.binding.snapshot.snapshot_id, 'scope_id': context.binding.scope_id,
            'policy_revision': context.binding.policy_revision, 'session_version': context.binding.session_version,
            'config_sha256': config_sha256, 'mode': mode,
            'question': 'sha256:' + hashlib.sha256(question.encode('utf-8')).hexdigest(), 'created_at': now}
        with self._transaction(context) as connection:
            created = connection.execute(insert(schema.query_runs).values(**values)
                .on_conflict_do_nothing(index_elements=['run_id']).returning(schema.query_runs.c.run_id)).scalar_one_or_none()
            if created is None:
                raise AccessDenied('Server run identity is already unavailable')
            return self._metadata(connection, self._row(connection, context, run_id))

    def require_pending(self, context, run_id):
        with self._transaction(context) as connection:
            if self._row(connection, context, run_id)['status'] != 'pending':
                raise AccessDenied('Run no longer accepts tool activity')

    @staticmethod
    def _references(connection, context, run_id, references):
        for citation in references:
            if citation.startswith('aggregate:') and citation.count(':') == 1:
                row = connection.execute(sa.select(schema.aggregate_results.c.artifact_id).where(
                    schema.aggregate_results.c.citation_id == citation,
                    schema.aggregate_results.c.run_id == run_id)).scalar_one_or_none()
                if row is None:
                    raise AccessDenied('Aggregate reference is unavailable in this run')
            elif citation.startswith('jira-record:'):
                parts = citation.split(':', 2)
                if len(parts) != 3 or not context.scope.permits(parts[2]):
                    raise AccessDenied('Native record reference is unavailable')
                row = connection.execute(sa.select(schema.tickets.c.ticket_id).where(
                    schema.tickets.c.snapshot_id == context.binding.snapshot.snapshot_id,
                    schema.tickets.c.feed_id == parts[1], schema.tickets.c.doc_id == parts[2])).scalar_one_or_none()
                if row is None:
                    raise AccessDenied('Native record reference is unavailable')
            else:
                parts = citation.rsplit(':', 2)
                if (len(parts) != 3 or not re.fullmatch(r'[0-9]+', parts[1]) or not re.fullmatch(r'[0-9]+', parts[2])
                        or not context.scope.permits(parts[0])):
                    raise AccessDenied('Passage reference is unavailable')
                start, end = int(parts[1]), int(parts[2])
                length = connection.execute(sa.select(schema.documents.c.text_length).where(
                    schema.documents.c.snapshot_id == context.binding.snapshot.snapshot_id,
                    schema.documents.c.doc_id == parts[0])).scalar_one_or_none()
                if length is None or not 0 <= start < end <= length:
                    raise EvidenceValidationError('Passage reference is outside canonical bounds')

    def append(self, context, run_id, step_number, payload):
        if type(step_number) is not int or not 1 <= step_number <= 1000 or not isinstance(payload, RunStepMetadata):
            raise EvidenceValidationError('Run steps require typed bounded metadata')
        try:
            payload = RunStepMetadata.model_validate(payload.model_dump(warnings=False))
        except ValueError:
            raise EvidenceValidationError('Invalid run step metadata') from None
        body = payload.model_dump(mode='json')
        with self._transaction(context) as connection:
            if self._row(connection, context, run_id, lock=True)['status'] != 'pending':
                raise AccessDenied('Only pending runs accept steps')
            count, maximum = connection.execute(sa.select(sa.func.count(), sa.func.max(schema.run_steps.c.step_number))
                .where(schema.run_steps.c.run_id == run_id)).one()
            if count != (maximum or 0):
                raise EvidenceValidationError('Existing run steps are not contiguous')
            existing = connection.execute(sa.select(schema.run_steps.c.payload).where(
                schema.run_steps.c.run_id == run_id, schema.run_steps.c.step_number == step_number)).scalar_one_or_none()
            if existing is not None:
                if existing != body:
                    raise EvidenceValidationError('Run step conflicts with its existing ordinal')
                return
            if step_number != (maximum or 0) + 1:
                raise EvidenceValidationError('Run step ordinals must be contiguous')
            self._references(connection, context, run_id, payload.evidence_ids)
            connection.execute(schema.run_steps.insert().values(run_id=run_id, step_number=step_number, payload=body))

    def finish(self, context, run_id, *, status, usage, latency_ms, finished_at):
        if status not in ('answered', 'insufficient_evidence', 'error', 'clarification') or not isinstance(usage, Usage):
            raise EvidenceValidationError('Run completion requires a typed terminal outcome')
        try:
            usage = Usage.model_validate(usage.model_dump(warnings=False))
            if (not isinstance(finished_at, datetime) or finished_at.tzinfo is None or finished_at.utcoffset() is None
                    or isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float))
                    or not 0 <= latency_ms < float('inf')):
                raise ValueError()
        except ValueError:
            raise EvidenceValidationError('Invalid run completion metadata') from None
        values = {**usage.model_dump(), 'status': status, 'latency_ms': latency_ms, 'finished_at': finished_at}
        with self._transaction(context) as connection:
            row = self._row(connection, context, run_id, lock=True)
            count, maximum = connection.execute(sa.select(sa.func.count(), sa.func.max(schema.run_steps.c.step_number))
                .where(schema.run_steps.c.run_id == run_id)).one()
            if count != (maximum or 0):
                raise EvidenceValidationError('Run cannot finish with noncontiguous steps')
            if finished_at < row['created_at']:
                raise EvidenceValidationError('Run completion precedes creation')
            if row['status'] != 'pending':
                if any(row[key] != value for key, value in values.items()):
                    raise AccessDenied('Run terminal outcome cannot be replaced')
                return self._metadata(connection, row)
            changed = connection.execute(schema.query_runs.update().where(
                schema.query_runs.c.run_id == run_id, schema.query_runs.c.status == 'pending')
                .values(**values).returning(schema.query_runs.c.run_id)).scalar_one_or_none()
            if changed is None:
                raise AccessDenied('Run completion is no longer available')
            return self._metadata(connection, self._row(connection, context, run_id))

    def get_metadata(self, context, run_id):
        with self._transaction(context) as connection:
            return self._metadata(connection, self._row(connection, context, run_id))

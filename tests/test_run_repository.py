"""Owned, bounded current-run metadata under the actual runtime-role RLS.

Root executes these opt-in PG tests privately. They create only disposable tiny
fixture databases; there are no live models or credentials in saved payloads.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from threading import Barrier
from uuid import uuid4

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from src.access.contracts import AccessDenied, EvidenceValidationError, PolicyUnavailable, SessionExpired, StaleContext
from src.access.policy import PolicyResolver
from src.auth.accounts import disable_account
from src.contracts import ToolRequest, Usage
from src.db import schema
from src.storage.runs import RunRepository, RunStepMetadata
from tests.db_support import pg_urls, url_string
from tests.jira_support import env, ready, source


QUESTION = 'PRIVATE USER QUESTION MUST NOT BECOME STORED HISTORY'
CONFIG = 'a' * 64


def repository(data):
    return RunRepository(data.database, data.resolver)


def begin(data, repo=None, *, principal='intern', run_id=None):
    repo = repo or repository(data)
    run_id = run_id or 'run-' + uuid4().hex
    value = repo.begin(data.contexts[principal], run_id, mode='baseline', question=QUESTION, config_sha256=CONFIG)
    return repo, run_id, value


def step(*, refs=(), tool='text_search', digest=None):
    return RunStepMetadata(tool=tool, round_number=1, evidence_ids=refs,
        query_sha256=digest or hashlib.sha256(b'private retrieval query').hexdigest(),
        source_types=('jira',), elapsed_ms=2., usage=Usage(estimated_cost_usd=0))


def finish(repo, context, run_id, *, status='answered', finished_at=None, usage=None):
    return repo.finish(context, run_id, status=status, usage=usage or Usage(input_tokens=3,
        output_tokens=2, api_calls=0, estimated_cost_usd=0), latency_ms=25.,
        finished_at=finished_at or datetime.now(timezone.utc))


def test_real_owner_can_begin_append_finish_and_read_only_whitelisted_metadata(ready):
    repo, run_id, initial = begin(ready)
    context = ready.contexts['intern']
    assert initial.run_id == run_id and initial.status == 'pending' and initial.step_count == 0
    repo.append(context, run_id, 1, step(refs=('one:0:8',)))
    ended = finish(repo, context, run_id)
    assert ended.status == 'answered' and ended.step_count == 1
    assert ended.usage.input_tokens == 3 and ended.usage.estimated_cost_usd == 0
    current = repo.get_metadata(context, run_id)
    assert current == ended
    assert current.scope_id == context.binding.scope_id
    assert current.snapshot_id == ready.source.snapshot.snapshot_id
    assert not any(hasattr(current, name) for name in ('question', 'steps', 'answer', 'sources', 'payload'))
    with ready.owner.connect() as connection:
        row = connection.execute(sa.select(schema.query_runs).where(schema.query_runs.c.run_id == run_id)).mappings().one()
        payload = connection.execute(sa.select(schema.run_steps.c.payload).where(schema.run_steps.c.run_id == run_id)).scalar_one()
    assert row['question'] == 'sha256:' + hashlib.sha256(QUESTION.encode()).hexdigest()
    assert QUESTION not in json.dumps(payload) and 'private retrieval query' not in json.dumps(payload)
    assert set(payload) <= set(RunStepMetadata.model_fields)


def test_run_metadata_and_appends_are_invisible_to_other_account_or_unbound_context(ready):
    repo, run_id, _ = begin(ready)
    other = ready.contexts['executive']
    for operation in (lambda: repo.get_metadata(other, run_id),
                      lambda: repo.append(other, run_id, 1, step()),
                      lambda: finish(repo, other, run_id)):
        with pytest.raises(AccessDenied):
            operation()
    with ready.database.transaction() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.query_runs)).scalar_one() == 0
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.run_steps)).scalar_one() == 0
    with pytest.raises((AccessDenied, EvidenceValidationError, TypeError)):
        repo.get_metadata(None, run_id)


def test_append_requires_contiguous_numbers_and_exact_retries_are_idempotent(ready):
    repo, run_id, _ = begin(ready)
    context = ready.contexts['intern']
    with pytest.raises((AccessDenied, EvidenceValidationError)):
        repo.append(context, run_id, 2, step())
    first = step(refs=('one:0:8',))
    repo.append(context, run_id, 1, first)
    repo.append(context, run_id, 1, first)
    assert repo.get_metadata(context, run_id).step_count == 1
    with pytest.raises((AccessDenied, EvidenceValidationError)):
        repo.append(context, run_id, 1, step(digest='f' * 64))
    repo.append(context, run_id, 2, step(tool='generate'))
    assert repo.get_metadata(context, run_id).step_count == 2


def test_terminal_finish_is_idempotent_but_never_reopens_or_accepts_new_steps(ready):
    repo, run_id, _ = begin(ready)
    context = ready.contexts['intern']
    when = datetime.now(timezone.utc)
    original = finish(repo, context, run_id, finished_at=when)
    assert finish(repo, context, run_id, finished_at=when) == original
    for operation in (lambda: finish(repo, context, run_id, status='error', finished_at=when),
                      lambda: finish(repo, context, run_id, finished_at=when + timedelta(seconds=1)),
                      lambda: repo.append(context, run_id, 1, step())):
        with pytest.raises((AccessDenied, EvidenceValidationError)):
            operation()
    assert repo.get_metadata(context, run_id) == original
    with pytest.raises((AccessDenied, EvidenceValidationError, ValidationError, ValueError)):
        finish(repo, context, run_id, status='pending')
    with pytest.raises(AccessDenied):
        ready.tool.execute(ToolRequest(source_type='jira', operation='count'),
            context=context, guard=ready.resolver, run_id=run_id)


def test_duplicate_concurrent_step_append_is_one_committed_step(ready):
    repo, run_id, _ = begin(ready)
    barrier = Barrier(2)

    def append():
        barrier.wait(timeout=5)
        repo.append(ready.contexts['intern'], run_id, 1, step())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(append) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert repo.get_metadata(ready.contexts['intern'], run_id).step_count == 1


def test_competing_terminal_outcomes_commit_once_and_never_overwrite_the_winner(ready):
    repo, run_id, _ = begin(ready)
    barrier, when = Barrier(2), datetime.now(timezone.utc)

    def complete(status):
        barrier.wait(timeout=5)
        try:
            finish(repo, ready.contexts['intern'], run_id, status=status, finished_at=when)
            return 'finished'
        except AccessDenied:
            return 'denied'

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(complete, status) for status in ('answered', 'error')]
        assert sorted(future.result(timeout=10) for future in results) == ['denied', 'finished']
    assert repo.get_metadata(ready.contexts['intern'], run_id).status in {'answered', 'error'}


@pytest.mark.parametrize('mutation', ['revoke', 'logout', 'disable', 'expire'])
def test_invalidated_run_remains_incomplete_and_cannot_be_finished_by_old_or_new_scope(ready, mutation):
    repo, run_id, _ = begin(ready)
    old = ready.contexts['intern']
    if mutation == 'revoke':
        ready.repository.revoke(ready.manifest)
        fresh = ready.resolver.resolve(ready.identities['intern'])
        expected = StaleContext
    else:
        with ready.owner.begin() as connection:
            if mutation == 'logout':
                connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == old.identity.session_id)
                    .values(revoked_at=datetime.now(timezone.utc)))
            elif mutation == 'expire':
                now = datetime.now(timezone.utc)
                connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == old.identity.session_id)
                    .values(created_at=now - timedelta(hours=1), expires_at=now - timedelta(seconds=1)))
            else:
                connection.execute(schema.users.update().where(schema.users.c.user_id == old.identity.user_id).values(is_active=False))
        fresh, expected = None, SessionExpired
    for operation in (lambda: repo.get_metadata(old, run_id),
                      lambda: repo.append(old, run_id, 1, step()),
                      lambda: finish(repo, old, run_id)):
        with pytest.raises(expected):
            operation()
    if fresh is not None:
        for operation in (lambda: repo.get_metadata(fresh, run_id), lambda: finish(repo, fresh, run_id)):
            with pytest.raises(AccessDenied):
                operation()
    with ready.owner.connect() as connection:
        row = connection.execute(sa.select(schema.query_runs).where(schema.query_runs.c.run_id == run_id)).mappings().one()
    assert row['status'] == 'pending' and row['finished_at'] is None
    assert row['scope_id'] == old.binding.scope_id and row['policy_revision'] == old.binding.policy_revision


def test_new_session_same_account_cannot_rebind_an_existing_run(ready):
    from tests.jira_support import PASSWORD
    repo, run_id, _ = begin(ready)
    grant = ready.auth.login('intern', PASSWORD, client_ip='127.0.0.1')
    fresh_identity = ready.auth.authenticate(grant.token, touch=False).identity
    fresh = ready.resolver.resolve(fresh_identity)
    assert fresh.identity.user_id == ready.contexts['intern'].identity.user_id
    assert fresh.binding.scope_id != ready.contexts['intern'].binding.scope_id
    with pytest.raises(AccessDenied):
        repo.get_metadata(fresh, run_id)
    with pytest.raises(AccessDenied):
        finish(repo, fresh, run_id)


def test_idle_touch_same_session_and_binding_can_finish_the_original_run(ready):
    from tests.jira_support import PASSWORD
    repo = repository(ready)
    grant = ready.auth.login('intern', PASSWORD, client_ip='127.0.0.1')
    first_identity = ready.auth.authenticate(grant.token, touch=False).identity
    first = ready.resolver.resolve(first_identity)
    run_id = 'run-' + uuid4().hex
    repo.begin(first, run_id, mode='baseline', question=QUESTION, config_sha256=CONFIG)
    current = ready.resolver.resolve(ready.auth.authenticate(grant.token, touch=True).identity)
    assert current.binding == first.binding
    assert finish(repo, current, run_id).status == 'answered'


def test_count_artifact_reference_must_belong_to_this_exact_run_even_for_same_owner_scope(ready):
    repo, first, _ = begin(ready)
    _, second, _ = begin(ready, repo)
    context = ready.contexts['intern']
    result = ready.tool.execute(ToolRequest(source_type='jira', operation='count', filters={'status': 'Absent'}),
        context=context, guard=ready.resolver, run_id=first)
    assert result.count == 0
    reference = step(refs=(result.evidence[0].citation_id,), tool='jira')
    repo.append(context, first, 1, reference)
    with pytest.raises((AccessDenied, EvidenceValidationError)):
        repo.append(context, second, 1, reference)
    assert repo.get_metadata(context, second).step_count == 0


def test_record_and_passage_references_are_authoritative_not_arbitrary_strings(ready):
    repo, run_id, _ = begin(ready)
    context = ready.contexts['intern']
    record = ready.tool.execute(ToolRequest(source_type='jira', operation='lookup', filters={'doc_id': 'one'}),
        context=context, guard=ready.resolver).records[0]
    repo.append(context, run_id, 1, step(refs=(record.citation_id, 'one:0:8')))
    for reference in ('hidden:0:8', 'one:0:99999', 'one:9:8', 'jira-record:' + 'f' * 64 + ':one', 'made-up-reference'):
        with pytest.raises((AccessDenied, EvidenceValidationError)):
            repo.append(context, run_id, 2, step(refs=(reference,)))
    assert repo.get_metadata(context, run_id).step_count == 1


def test_copied_payload_cannot_smuggle_raw_history_or_invalid_metadata(ready):
    repo, run_id, _ = begin(ready)
    for payload in (step().model_copy(update={'query_sha256': 'not-a-sha'}),
                    step().model_copy(update={'elapsed_ms': float('nan')})):
        with pytest.raises((EvidenceValidationError, ValidationError, ValueError)):
            repo.append(ready.contexts['intern'], run_id, 1, payload)
    with pytest.raises(ValidationError):
        RunStepMetadata.model_validate({**step().model_dump(), 'question': QUESTION})
    assert repo.get_metadata(ready.contexts['intern'], run_id).step_count == 0


def test_database_outage_is_sanitized_and_cannot_be_a_successful_metadata_read(ready, monkeypatch):
    repo, run_id, _ = begin(ready)

    @contextmanager
    def outage(context=None):
        raise sa.exc.SQLAlchemyError('Synthetic private database failure marker')
        yield

    monkeypatch.setattr(ready.database, 'transaction', outage)
    for operation in (lambda: repo.get_metadata(ready.contexts['intern'], run_id),
                      lambda: finish(repo, ready.contexts['intern'], run_id)):
        with pytest.raises(PolicyUnavailable) as failure:
            operation()
        assert 'private database failure marker' not in str(failure.value)

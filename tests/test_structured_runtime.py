"""Real tiny Jira/runtime/store integration with scripted local providers.

Only root executes the opt-in PostgreSQL cases. No live model, downloaded index
or production account is used; routing/provider outputs are deterministic.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from threading import Event

import pytest
import sqlalchemy as sa

from src.access.contracts import AccessDenied, AccessError, SessionExpired, StaleContext
from src.access.policy import PolicyResolver
from src.agents.routing import Route, ToolCall
from src.api.runtime import AuthenticatedRuntime, WebSettings
from src.contracts import AggregateEvidence, PassageEvidence, QueryRequest, StructuredRecord, Usage
from src.db import schema
from src.evaluation.experiment_contracts import ModelProvenance
from src.retrieval.store import StoreBM25Retriever
from src.retrieval.tools import JiraTool
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import pg_urls
from tests.jira_support import env, ready, source
from tests.test_authenticated_runtime import MetadataIndex, RecordingProvider


class ScriptedRouter:
    def __init__(self, *, operation='count', filters=None, mixed=False, lookup_too=False, unsupported=False):
        self.operation, self.filters, self.mixed = operation, filters or {}, mixed
        self.lookup_too, self.unsupported = lookup_too, unsupported
        self.calls = []

    def export_attestation(self):
        return {'kind': 'scripted_fixture', 'router_id': 'wire-jira-conformance-v1',
            'behavior': {'operation': self.operation, 'filters': dict(self.filters),
                'mixed': self.mixed, 'lookup_too': self.lookup_too, 'unsupported': self.unsupported}}

    def route(self, request, feedback, tools):
        self.calls.append((request, dict(tools)))
        assert 'jira' in tools and tools['jira'].source_type == 'jira'
        assert set(tools['jira'].operations) == {'lookup', 'filter', 'count'}
        calls = [ToolCall(tool='not_registered' if self.unsupported else 'jira', operation=self.operation, filters=self.filters)]
        if self.lookup_too:
            calls.append(ToolCall(tool='jira', operation='lookup', filters={'doc_id': 'one'}))
        return Route(reason='Scripted local conformance route.',
            text_queries=(request.query,) if self.mixed else (), tool_calls=tuple(calls)), Usage(estimated_cost_usd=0)


def runtime(data, router, *, provider=True, top_k=5, attest_router=True):
    store = PostgresDocumentStore(data.database, data.source.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    resolver = PolicyResolver(data.database, data.source.snapshot, catalog=proof)
    retriever = StoreBM25Retriever(MetadataIndex(data.source.snapshot, proof), store, catalog=proof)
    configured = RecordingProvider() if provider is True else provider
    # Functional PG checks cross the Mac/Docker TCP boundary; serving latency
    # is measured separately inside Docker. Use the normal request deadline.
    settings = WebSettings(mode='fixture' if configured is not None else 'retrieval',
        auth_mode='local', demo_access=False, max_seconds=120, max_rounds=2, top_k=top_k)
    actual = AuthenticatedRuntime(settings, retriever, resolver, database=data.database,
        provider=configured, router=router,
        provider_identity=ModelProvenance(provider='fixture', model_id='fixture-generator')
            if configured is not None else None, provider_max_output_tokens=1024,
        router_attestation=router.export_attestation() if attest_router else None)
    return actual, configured, resolver


def request(question='What do the permitted Jira records show?', **kwargs):
    return QueryRequest(question=question, mode='baseline', **kwargs)


def test_lookup_uses_native_fields_and_provenance_without_fabricating_passages(ready):
    actual, provider, resolver = runtime(ready, ScriptedRouter(operation='lookup', filters={'doc_id': 'one'}))
    response = actual.query(request(), ready.identities['intern'])
    assert response.contract_version == '3' and response.answer.status == 'answered'
    assert len(response.sources) == 1 and isinstance(response.sources[0], StructuredRecord)
    record = response.sources[0]
    assert record.doc_id == 'one' and record.status == 'In Progress'
    assert not hasattr(record, 'chunk')
    assert record.source.feed_id == ready.source.inputs.feed_id
    assert record.scope.snapshot == ready.source.snapshot
    assert provider.prompts and 'In Progress' in json.dumps(provider.prompts)
    ready.tool.validate_records(resolver.resolve(ready.identities['intern']), (record,), guard=resolver)
    step = next(value for value in response.trace.steps if value.tool == 'jira')
    metadata = json.loads(step.query)
    assert metadata['request']['filters'] == {'doc_id': 'one'}
    assert 'user' not in metadata['request']
    assert metadata['result']['ambiguous'] is False and metadata['result']['coverage'] == 'synthetic_fixture'


@pytest.mark.parametrize('filters,count', [({'status': 'Exact absent status'}, 0), ({'native_key': 'SUP-100'}, 2)])
def test_grounded_counts_include_zero_and_count_source_records_not_unique_keys(ready, filters, count):
    actual, provider, _resolver = runtime(ready, ScriptedRouter(filters=filters))
    response = actual.query(request(), ready.identities['intern'])
    assert response.answer.status == 'answered' and response.trace.stop_reason == 'single_pass', (
        response.answer.status, response.trace.stop_reason, response.trace.total_latency_ms,
        tuple((step.tool, step.elapsed_ms) for step in response.trace.steps))
    assert response.contract_version == '2'
    assert len(response.sources) == 1 and isinstance(response.sources[0], AggregateEvidence)
    aggregate = response.sources[0]
    assert aggregate.count == count and aggregate.scope.snapshot == ready.source.snapshot
    assert not hasattr(aggregate, 'chunk')
    assert provider.prompts, 'A successful cardinality disclaimer must not stop generation'
    with ready.owner.connect() as connection:
        stored = connection.execute(sa.select(schema.aggregate_results).where(
            schema.aggregate_results.c.artifact_id == aggregate.artifact_id)).mappings().one()
        run = connection.execute(sa.select(schema.query_runs).where(
            schema.query_runs.c.run_id == response.trace.run_id)).mappings().one()
    assert stored['run_id'] == response.trace.run_id and stored['count'] == count
    assert run['status'] == 'answered' and run['scope_id'] == aggregate.scope.scope_id


@pytest.mark.parametrize('kind', ['ambiguous', 'truncated', 'unknown', 'unsupported'])
def test_incomplete_or_ambiguous_native_results_stop_before_generation(ready, kind):
    router = ScriptedRouter(operation='lookup' if kind == 'ambiguous' else 'filter' if kind == 'truncated' else 'count',
        filters={'native_key': 'SUP-100'} if kind == 'ambiguous' else {'status': 'In Progress'} if kind == 'unknown' else {},
        unsupported=kind == 'unsupported')
    actual, provider, _ = runtime(ready, router, top_k=1 if kind == 'truncated' else 5)
    response = actual.query(request(), ready.identities['executive' if kind == 'unknown' else 'intern'])
    assert response.answer.status != 'answered'
    assert provider.prompts == []
    assert all(not isinstance(value, AggregateEvidence) for value in response.sources)
    if kind == 'ambiguous':
        assert response.trace.stop_reason == 'clarification'
        assert {value.doc_id for value in response.sources} == {'one', 'two'}
    if kind == 'unknown':
        assert response.sources == ()
    if kind in {'ambiguous', 'truncated', 'unknown'}:
        result = json.loads(next(value.query for value in response.trace.steps if value.tool == 'jira'))['result']
        if kind == 'unknown':
            assert result['coverage'] == 'unknown'
        else:
            assert result['ambiguous' if kind == 'ambiguous' else 'truncated'] is True


def test_mixed_request_captures_exact_passage_record_and_count_used_by_generator(ready):
    actual, provider, _ = runtime(ready, ScriptedRouter(filters={'status': 'Exact absent status'}, mixed=True, lookup_too=True))
    captured = actual.query_with_capture(request('Explain the prose and give exact native status and count.'), ready.identities['intern'])
    response = captured.response
    assert response.contract_version == '3' and response.answer.status == 'answered'
    assert {value.kind for value in response.sources} == {'passage', 'record', 'aggregate'}
    assert {value.kind for value in captured.generation_evidence} == {'passage', 'record', 'aggregate'}
    assert captured.run_id == response.trace.run_id and captured.binding.snapshot == ready.source.snapshot
    assert captured.payload_sha256 and len(captured.payload_sha256) == 64
    assert all(value.chunk.doc_id in {'one', 'two'} for value in response.sources if isinstance(value, PassageEvidence))
    assert provider.prompts
    with ready.owner.connect() as connection:
        payloads = connection.execute(sa.select(schema.run_steps.c.payload).where(
            schema.run_steps.c.run_id == captured.run_id)).scalars().all()
    assert payloads
    serialized = json.dumps(payloads)
    assert 'Exact source prose' not in serialized and 'Explain the prose' not in serialized
    assert all(set(payload) <= {'tool', 'round_number', 'evidence_ids', 'query_sha256', 'source_types',
        'elapsed_ms', 'usage', 'input_payload_sha256'} for payload in payloads)


def test_retrieval_only_native_tools_do_not_invent_generation_input(ready):
    actual, provider, _ = runtime(ready, ScriptedRouter(operation='lookup', filters={'doc_id': 'one'}), provider=None)
    captured = actual.query_with_capture(request(), ready.identities['intern'])
    assert provider is None and captured.response.answer.status == 'insufficient_evidence'
    assert captured.response.answer.citation_ids == ()
    assert captured.response.trace.usage.api_calls == 0
    assert captured.generation_evidence == () and captured.payload_sha256 is None
    assert len(captured.response.sources) == 1 and isinstance(captured.response.sources[0], StructuredRecord)


def test_other_human_has_only_their_current_authorized_native_records(ready):
    actual, _provider, _ = runtime(ready, ScriptedRouter(operation='filter'))
    for principal, expected in (('intern', {'one', 'two'}), ('executive', {'one', 'two', 'hidden', 'unknown'})):
        response = actual.query(request(), ready.identities[principal])
        assert {value.doc_id for value in response.sources} == expected
        assert all(value.scope.user_id == ready.identities[principal].user_id for value in response.sources)


@pytest.mark.parametrize('filter', [{'source_types': ('slack',)}, {'after': datetime(2020, 1, 1, tzinfo=timezone.utc)}])
def test_native_tools_cannot_bypass_requested_platform_or_unsupported_date_filter(ready, filter):
    actual, provider, _ = runtime(ready, ScriptedRouter())
    try:
        response = actual.query(request(**filter), ready.identities['intern'])
    except AccessError:
        pass  # Admission may reject the source before the router is invoked.
    else:
        assert response.answer.status != 'answered' and not response.sources
    assert provider.prompts == []
    with ready.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.aggregate_results)).scalar_one() == 0


@pytest.mark.parametrize('mutation', ['grant', 'logout'])
def test_native_result_invalidation_during_provider_does_not_release_answer_or_force_finish(ready, mutation):
    entered, release = Event(), Event()
    provider = RecordingProvider(entered=entered, release=release)
    actual, _, _ = runtime(ready, ScriptedRouter(), provider=provider)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(actual.query, request(), ready.identities['intern'])
        try:
            # Mac-to-Docker PostgreSQL setup/authority round trips are not a
            # serving-latency benchmark. Invalidate only after dispatch begins.
            assert entered.wait(30)
            if mutation == 'grant':
                ready.repository.revoke(ready.manifest)
            else:
                with ready.owner.begin() as connection:
                    connection.execute(schema.sessions.update().where(
                        schema.sessions.c.session_id == ready.identities['intern'].session_id)
                        .values(revoked_at=datetime.now(timezone.utc)))
        finally:
            release.set()
        with pytest.raises(StaleContext if mutation == 'grant' else SessionExpired):
            future.result(timeout=10)
    with ready.owner.connect() as connection:
        rows = connection.execute(sa.select(schema.query_runs)).mappings().all()
    assert len(rows) == 1 and rows[0]['status'] == 'pending' and rows[0]['finished_at'] is None


def test_forged_native_fields_never_enter_provider_even_with_valid_citation_id(ready, monkeypatch):
    original = JiraTool.execute

    def altered(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result.records:
            return result.model_copy(update={'records': (result.records[0].model_copy(update={'status': 'FORGED SECRET'}),)})
        return result

    monkeypatch.setattr(JiraTool, 'execute', altered)
    actual, provider, _ = runtime(ready, ScriptedRouter(operation='lookup', filters={'doc_id': 'one'}))
    with pytest.raises(AccessError):
        actual.query(request(), ready.identities['intern'])
    assert provider.prompts == []

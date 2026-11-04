"""Authenticated query isolation using tiny real PostgreSQL and no live models."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from threading import Event
from types import SimpleNamespace

import pytest

sa = pytest.importorskip('sqlalchemy', reason='Authenticated runtime checks require DB dependencies')
pytest.importorskip('psycopg', reason='Authenticated runtime checks require psycopg')
pytest.importorskip('alembic', reason='Authenticated runtime checks require Alembic')

from src.access.contracts import (
    AccessDenied, AccessError, EvidenceValidationError, PolicyUnavailable,
    SessionExpired, SnapshotMismatch, StaleContext,
)
from src.access.policy import PolicyResolver
from src.api.runtime import AuthenticatedRuntime, RuntimeConfigurationError, WebSettings, build_runtime
from src.auth.accounts import disable_account
from src.contracts import Chunk, PassageEvidence, QueryRequest, Usage
from src.db import schema
from src.db.runtime import Database, DatabaseSettings
from src.generation.generator import Completion
from src.retrieval.store import StoreBM25Retriever
from src.storage.contracts import RankedCandidate
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import pg_urls, url_string
from tests.test_access_policy import (
    ENTRIES, TEXTS, acl_db, identity, raw_token, seed_explicit_grants,
)

QUESTION = 'What do the matching notes say?'
_DEFAULT = object()


class RecordingProvider:
    """Scripted output and captured payloads, never a model/provider connection."""
    reserved_usd = 0
    supports_structured_output = False

    def __init__(self, *, entered=None, release=None, request_more=False):
        self.prompts = []
        self.entered, self.release, self.request_more = entered, release, request_more
        self.critic_calls = 0

    def complete(self, system, prompt):
        payload = json.loads(prompt)
        self.prompts.append((system, payload))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(10), 'Test did not release the scripted provider'
        evidence = payload.get('evidence', [])
        if 'Check whether the supplied passages' in system:
            self.critic_calls += 1
            more = self.request_more and self.critic_calls == 1
            output = {'sufficient': not more, 'confidence': .95, 'reason': 'Scripted fixture assessment.',
                'missing_facts': ['More of the matching note'] if more else [],
                'read_more_doc_ids': [evidence[0]['doc_id']] if more and evidence else [],
                'suggested_queries': [], 'suggested_sources': []}
        else:
            citations = [row['citation_id'] for row in evidence]
            output = {'status': 'answered' if citations else 'insufficient_evidence',
                'text': 'Authorized notes. ' + ' '.join('[' + value + ']' for value in citations),
                'citation_ids': citations}
        return Completion(json.dumps(output), Usage(api_calls=0, estimated_cost_usd=0))


class MetadataIndex:
    """Deterministic metadata ranks; actual text is read through runtime-role RLS."""
    def __init__(self, snapshot, proof, initial_chars=None):
        self.doc_ids = proof.doc_ids
        self.manifest = {'version': 1, 'corpus_sha256': snapshot.corpus_sha256, 'method': 'synthetic-ranks'}
        self.initial_chars = initial_chars
        self.calls = []

    def rank_candidates(self, query, top_k, *, snapshot, catalog, mask):
        self.calls.append((query, tuple(bool(value) for value in mask)))
        return tuple(RankedCandidate(snapshot=snapshot, doc_id=entry.doc_id, revision_id=entry.revision_id,
            source_type=entry.source_type, start_char=0,
            end_char=min(entry.text_length, self.initial_chars) if self.initial_chars else entry.text_length,
            score=float(100-entry.position), retrieval_method='bm25', scoring_provenance='app-test-metadata-index')
            for entry in catalog if mask[entry.position] and entry.text_length)[:top_k]


class ObservedRetriever:
    def __init__(self, actual):
        self.actual, self.snapshot = actual, actual.snapshot
        self.catalog_handle = actual.catalog_handle
        self.searches, self.expansions = [], []
        self.alter = None

    def source_types(self):
        return self.actual.source_types()

    def search(self, context, query, *, guard, top_k, filters):
        self.searches.append((context, query, filters))
        result = self.actual.search(context, query, guard=guard, top_k=top_k, filters=filters)
        return self.alter(result) if self.alter else result

    def read_more(self, context, anchors, *, guard, max_chars, filters):
        self.expansions.append((context, anchors, filters))
        return self.actual.read_more(context, anchors, guard=guard, max_chars=max_chars, filters=filters)


@pytest.fixture
def app_db(acl_db):
    seed_explicit_grants(acl_db)
    return acl_db


def make_runtime(fixture, *, provider=_DEFAULT, initial_chars=None, settings=None):
    settings = settings or WebSettings(mode='fixture', auth_mode='local', demo_access=False,
        max_seconds=20, max_rounds=2, top_k=5)
    store = PostgresDocumentStore(fixture.database, fixture.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    resolver = PolicyResolver(fixture.database, fixture.snapshot, catalog=proof)
    index = MetadataIndex(fixture.snapshot, proof, initial_chars)
    retriever = ObservedRetriever(StoreBM25Retriever(index, store, catalog=proof))
    provider = RecordingProvider() if provider is _DEFAULT else provider
    runtime = AuthenticatedRuntime(settings, retriever, resolver, database=fixture.database, provider=provider)
    return SimpleNamespace(runtime=runtime, provider=provider, retriever=retriever, resolver=resolver,
        store=store, index=index, settings=settings)


def assert_prompt_scope(provider, expected):
    assert provider.prompts
    documents = {row['doc_id'] for _system, payload in provider.prompts for row in payload.get('evidence', [])}
    assert documents == set(expected)
    for _system, payload in provider.prompts:
        for row in payload.get('evidence', []):
            entry = ENTRIES[next(index for index, value in enumerate(ENTRIES) if value.doc_id == row['doc_id'])]
            canonical = TEXTS[entry.position]
            assert row['text'] in canonical
    return documents


def test_two_actual_humans_ask_same_question_and_only_permitted_text_enters_generation(app_db):
    bundle = make_runtime(app_db)
    assert bundle.runtime.database is app_db.database is app_db.auth.database
    for principal, expected in (('intern', {'general', 'atlas'}), ('executive', {'general', 'finance-exec'})):
        bundle.provider.prompts.clear()
        response = bundle.runtime.query(QueryRequest(question=QUESTION, mode='baseline'), identity(app_db, principal))
        assert response.snapshot == app_db.snapshot and response.answer.status == 'answered'
        assert {source.chunk.doc_id for source in response.sources} == expected
        assert_prompt_scope(bundle.provider, expected)
        for source in response.sources:
            entry = next(value for value in ENTRIES if value.doc_id == source.chunk.doc_id)
            assert source.revision_id == entry.revision_id and source.snapshot_id == app_db.snapshot.snapshot_id
    assert not hasattr(bundle.runtime, 'user')


def test_descriptive_ceo_metadata_never_grants_a_runtime_document(app_db):
    bundle = make_runtime(app_db)
    try:
        response = bundle.runtime.query(QueryRequest(question=QUESTION), identity(app_db, 'labels_only'))
    except AccessDenied:
        pass
    else:
        assert response.sources == () and response.answer.status == 'insufficient_evidence'
    assert bundle.provider.prompts == []


@pytest.mark.parametrize('violation', ['denied', 'snapshot', 'revision'])
def test_malicious_adapter_cannot_send_denied_or_forged_evidence_to_provider(app_db, violation):
    bundle = make_runtime(app_db)
    def invalid(rows):
        value = rows[0]
        if violation == 'denied':
            entry, text = ENTRIES[2], TEXTS[2]
            key = entry.doc_id + ':0:' + str(len(text))
            return (PassageEvidence(snapshot_id=app_db.snapshot.snapshot_id, revision_id=entry.revision_id,
                citation_id=key, score=1., retrieval_method='bm25', chunk=Chunk(chunk_id=key,
                doc_id=entry.doc_id, source_type=entry.source_type, text=text, title='Finance-Exec',
                start_char=0, end_char=len(text), metadata_origin='synthetic')),)
        return (value.model_copy(update={'snapshot_id': 'foreign-snapshot'} if violation == 'snapshot'
                                 else {'revision_id': 'f'*64}),)
    bundle.retriever.alter = invalid
    with pytest.raises(AccessError):
        bundle.runtime.query(QueryRequest(question=QUESTION), identity(app_db, 'intern'))
    assert bundle.provider.prompts == []


def test_source_filter_is_preserved_for_actual_search_prompt_and_response(app_db):
    bundle = make_runtime(app_db)
    response = bundle.runtime.query(QueryRequest(question=QUESTION, source_types=('jira',)), identity(app_db, 'intern'))
    assert {source.chunk.doc_id for source in response.sources} == {'general'}
    assert_prompt_scope(bundle.provider, {'general'})
    assert all(filters.source_types == ('jira',) for _context, _question, filters in bundle.retriever.searches)


def test_date_filter_denies_unknown_dates_without_provider_dispatch(app_db):
    bundle = make_runtime(app_db)
    after = datetime(2020, 1, 1, tzinfo=timezone.utc)
    response = bundle.runtime.query(QueryRequest(question=QUESTION, after=after), identity(app_db, 'intern'))
    assert response.sources == () and response.answer.status == 'insufficient_evidence'
    assert bundle.provider.prompts == []
    assert bundle.retriever.searches and all(filters.after == after for _context, _question, filters in bundle.retriever.searches)


def test_agent_read_more_keeps_original_user_source_filter_and_exact_canonical_offsets(app_db):
    provider = RecordingProvider(request_more=True)
    bundle = make_runtime(app_db, provider=provider, initial_chars=8)
    response = bundle.runtime.query(QueryRequest(question=QUESTION, mode='agentic', source_types=('jira',)), identity(app_db, 'intern'))
    assert bundle.retriever.expansions
    assert all(filters.source_types == ('jira',) for _ctx, _anchors, filters in bundle.retriever.expansions)
    assert_prompt_scope(provider, {'general'})
    for source in response.sources:
        assert source.chunk.doc_id == 'general'
        assert source.chunk.text == TEXTS[0][source.chunk.start_char:source.chunk.end_char]
    assert any(step.tool == 'read_more' for step in response.trace.steps)


@pytest.mark.parametrize('mutation', ['grant', 'logout', 'disable', 'expiry'])
def test_live_change_during_scripted_provider_aborts_instead_of_releasing_answer(app_db, mutation):
    entered, release = Event(), Event()
    bundle = make_runtime(app_db, provider=RecordingProvider(entered=entered, release=release))
    principal = identity(app_db, 'intern')
    expected = StaleContext if mutation == 'grant' else SessionExpired
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bundle.runtime.query, QueryRequest(question=QUESTION), principal)
        try:
            assert entered.wait(5)
            if mutation == 'grant':
                with app_db.owner.begin() as connection:
                    connection.execute(schema.document_grants.delete().where(
                        schema.document_grants.c.snapshot_id == app_db.snapshot.snapshot_id,
                        schema.document_grants.c.doc_id == 'general'))
            elif mutation == 'logout':
                app_db.auth.logout(raw_token(app_db, 'intern'), app_db.sessions['intern'].response.csrf_token)
            elif mutation == 'disable':
                disable_account(app_db.logins['intern'], url=url_string(app_db.urls.migration))
            else:
                now = datetime.now(timezone.utc)
                with app_db.owner.begin() as connection:
                    connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == principal.session_id)
                        .values(created_at=now-timedelta(hours=1), expires_at=now-timedelta(seconds=1)))
        finally:
            release.set()
        with pytest.raises(expected):
            future.result(timeout=5)
    assert_prompt_scope(bundle.provider, {'general', 'atlas'})
    assert app_db.database.engine.pool.checkedout() == 0


def test_authoritative_policy_sql_failure_never_uses_a_prior_runtime_permission(app_db):
    bundle = make_runtime(app_db)
    principal = identity(app_db, 'intern')
    bundle.runtime.query(QueryRequest(question=QUESTION), principal)
    bundle.provider.prompts.clear()
    with app_db.owner.begin() as connection:
        connection.exec_driver_sql('REVOKE SELECT ON rag.policy_state FROM rag_runtime')
    try:
        with pytest.raises(PolicyUnavailable):
            bundle.runtime.query(QueryRequest(question=QUESTION), principal)
    finally:
        with app_db.owner.begin() as connection:
            connection.exec_driver_sql('GRANT SELECT ON rag.policy_state TO rag_runtime')
    assert bundle.provider.prompts == []


def test_retrieval_only_response_is_snapshot_bound_and_safe_info_has_no_global_inventory(app_db):
    bundle = make_runtime(app_db, provider=None)
    response = bundle.runtime.query(QueryRequest(question=QUESTION), identity(app_db, 'intern'))
    assert response.snapshot == app_db.snapshot and {source.chunk.doc_id for source in response.sources} == {'general', 'atlas'}
    assert response.answer.status == 'insufficient_evidence'
    info = bundle.runtime.info()
    assert set(info) <= {'mode', 'retrieval', 'authentication', 'generation_enabled'}
    assert not {'documents', 'allowed_documents', 'source_types', 'model', 'budget_remaining_usd', 'embedding_model'} & info.keys()


def test_runtime_rejects_a_resolver_from_another_pool_even_with_same_database_dsn(app_db):
    bundle = make_runtime(app_db, provider=None)
    other = Database(DatabaseSettings(dsn=url_string(app_db.urls.runtime), pool_size=1))
    try:
        resolver = PolicyResolver(other, app_db.snapshot)
        with pytest.raises(RuntimeConfigurationError):
            AuthenticatedRuntime(bundle.settings, bundle.retriever, resolver, database=app_db.database)
    finally:
        other.close()


def test_authenticated_builder_rejects_fixture_mode_before_loading_demo_or_models(app_db, monkeypatch):
    import src.api.runtime as module
    def forbidden(*_args, **_kwargs):
        pytest.fail('Authenticated builder attempted a demo or model load')
    monkeypatch.setattr(module, 'load_retriever', forbidden)
    with pytest.raises(RuntimeConfigurationError):
        build_runtime(WebSettings(mode='fixture', auth_mode='local', demo_access=True),
            database=app_db.database, authenticated=True)


@pytest.mark.parametrize('empty_restriction', [False, True])
def test_explicit_authenticated_builder_uses_shared_pool_and_preserves_none_versus_empty_operator_scope(app_db, monkeypatch, tmp_path, empty_restriction):
    import src.api.runtime as module
    snapshot_file = tmp_path/'snapshot.json'
    snapshot_file.write_text(app_db.snapshot.model_dump_json())
    documents = tmp_path/'documents.json'
    if empty_restriction:
        documents.write_text('[]')
    def load(settings, store):
        assert store._database is app_db.database
        proof = store.validated_catalog()
        return ObservedRetriever(StoreBM25Retriever(MetadataIndex(app_db.snapshot, proof), store, catalog=proof))
    monkeypatch.setattr(module, 'load_store_retriever', load)
    settings = WebSettings(mode='retrieval', auth_mode='benchmark', demo_access=False,
        snapshot_file=snapshot_file, allowed_doc_ids_file=documents if empty_restriction else None)
    runtime = build_runtime(settings, database=app_db.database, authenticated=True)
    assert isinstance(runtime, AuthenticatedRuntime) and runtime.database is app_db.database
    if empty_restriction:
        with pytest.raises(AccessDenied):
            runtime.query(QueryRequest(question=QUESTION), identity(app_db, 'intern'))
    else:
        response = runtime.query(QueryRequest(question=QUESTION), identity(app_db, 'intern'))
        assert {source.chunk.doc_id for source in response.sources} == {'general', 'atlas'}


@pytest.mark.parametrize('missing', ['snapshot', 'database'])
def test_authenticated_builder_missing_prerequisites_never_loads_demo_or_models(monkeypatch, tmp_path, missing):
    import src.api.runtime as module
    def forbidden(*_args, **_kwargs):
        pytest.fail('Invalid authenticated builder reached a model/demo loader')
    monkeypatch.setattr(module, 'load_retriever', forbidden)
    monkeypatch.setattr(module, 'load_store_retriever', forbidden)
    settings = WebSettings(mode='retrieval', snapshot_file=tmp_path/'unused.json' if missing=='database' else None)
    with pytest.raises(RuntimeConfigurationError):
        build_runtime(settings, database=None, authenticated=True)


@pytest.mark.parametrize('retrieval', ['bm25', 'hybrid'])
def test_store_factory_uses_mmap_without_body_table_shared_proof_and_bounded_encoder_settings(app_db, monkeypatch, retrieval):
    from src.retrieval.factory import load_store_retriever
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import EmbeddingModel
    from src.retrieval.sharded_dense import ShardedDense
    bundle = make_runtime(app_db, provider=None)
    proof = bundle.store.validated_catalog()
    calls = []
    def lexical(cls, path, **kwargs):
        calls.append(('bm25', kwargs))
        return bundle.index
    model = {'name':'scripted-model','revision':'a'*40,'normalize':True}
    dense_index = SimpleNamespace(snapshot=app_db.snapshot, catalog_handle=proof,
        catalog=proof.entries, doc_ids=proof.doc_ids, manifest={'model':model})
    def dense(cls, path, **kwargs):
        calls.append(('dense', kwargs))
        return dense_index
    def encoder(cls, path, **kwargs):
        calls.append(('encoder', kwargs))
        return SimpleNamespace(identity=model)
    monkeypatch.setattr(CorpusBM25, 'load', classmethod(lexical))
    monkeypatch.setattr(ShardedDense, 'load', classmethod(dense))
    monkeypatch.setattr(EmbeddingModel, 'for_index', classmethod(encoder))
    result = load_store_retriever(WebSettings(mode='retrieval', retrieval=retrieval), bundle.store)
    assert calls[0] == ('bm25', {'corpus_path':None,'mmap':True})
    if retrieval=='hybrid':
        assert [name for name,_ in calls] == ['bm25','dense','encoder']
        assert calls[1][1]['catalog'] is proof and calls[1][1]['block_rows'] == 8192
        assert calls[1][1]['file_cache_policy'] == 'retain'
        assert calls[2][1] == {'device':'cpu','fp16':None,'batch_size':1}
        assert result.bm25.catalog_handle is result.dense.catalog_handle is proof
    else:
        assert len(calls)==1 and result.catalog_handle is proof


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '8192'])
def test_dense_block_setting_rejects_implicit_coercion(value):
    with pytest.raises(ValueError):
        WebSettings(dense_block_rows=value)


def test_dense_block_environment_has_one_explicit_decimal_conversion(monkeypatch):
    for field in WebSettings.model_fields:
        monkeypatch.delenv('RAG_'+field.upper(), raising=False)
    monkeypatch.setenv('RAG_DENSE_BLOCK_ROWS','8192')
    assert WebSettings.from_env().dense_block_rows==8192
    for invalid in ('true','8192.0',' 8192','١','0'):
        monkeypatch.setenv('RAG_DENSE_BLOCK_ROWS',invalid)
        with pytest.raises(ValueError):
            WebSettings.from_env()


def test_actual_tiny_hybrid_factory_and_authenticated_queries_share_canonical_store(app_db, monkeypatch, tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25
    from src.retrieval.corpus_dense import EmbeddingModel
    from src.retrieval.factory import load_store_retriever
    store = PostgresDocumentStore(app_db.database, app_db.snapshot, retain_catalog=True)
    proof = store.validated_catalog()
    bm25_directory, dense_directory = tmp_path/'bm25', tmp_path/'dense'
    config = BM25Config()
    model, stats = build_bm25(TEXTS, config)
    CorpusBM25(model, list(proof.doc_ids), config, manifest={'version':1,
        'config':config.model_dump(),'corpus_sha256':app_db.snapshot.corpus_sha256,'stats':stats}).save(bm25_directory)
    dense_directory.mkdir()
    model_identity = {'name':'fixture-encoder','revision':'c'*40,'normalize':True,
        'fp16':False,'max_seq_length':512,'query_prefix':''}
    (dense_directory/'manifest.json').write_text(json.dumps({'version':1,'complete':True,
        'documents':len(proof.entries),'shards':1,'corpus_sha256':app_db.snapshot.corpus_sha256,
        'model':model_identity}))
    (dense_directory/'doc_ids.json').write_text(json.dumps(proof.doc_ids))
    np.save(dense_directory/'shard_00000.npy',np.array([[1.,0.]]*len(proof.entries),dtype=np.float32))
    pq.write_table(pa.table({'doc_index':list(range(len(proof.entries))),
        'start_char':[0]*len(proof.entries),'end_char':[entry.text_length for entry in proof.entries]}),
        dense_directory/'shard_00000.parquet')
    calls=[]
    class Encoder:
        identity=model_identity
        def encode(self,texts,query=False):
            calls.append((texts,query))
            return np.array([[1.,0.]],dtype=np.float32)
    monkeypatch.setattr(EmbeddingModel,'for_index',classmethod(lambda cls,*args,**kwargs:Encoder()))
    settings=WebSettings(mode='retrieval',auth_mode='local',retrieval='hybrid',
        bm25_index=bm25_directory,dense_index=dense_directory,max_seconds=20)
    retriever=load_store_retriever(settings,store)
    assert retriever.bm25.index.corpus is None
    assert retriever.bm25.catalog_handle is retriever.dense.catalog_handle is proof
    resolver=PolicyResolver(app_db.database,app_db.snapshot,catalog=proof)
    provider=RecordingProvider()
    runtime=AuthenticatedRuntime(settings,retriever,resolver,database=app_db.database,provider=provider)
    for name,expected in (('intern',{'general','atlas'}),('executive',{'general','finance-exec'})):
        provider.prompts.clear()
        response=runtime.query(QueryRequest(question=QUESTION),identity(app_db,name))
        assert {source.chunk.doc_id for source in response.sources}==expected
        assert_prompt_scope(provider,expected)
    assert len(calls)==2 and all(query for _texts,query in calls)


def test_runtime_read_more_preserves_first_seen_duplicate_anchor_and_validates_all_anchors(app_db):
    from src.access.contracts import ScopedUserContext
    from src.api.runtime import _ScopedRetriever
    from src.contracts import SearchRequest
    bundle=make_runtime(app_db,provider=None)
    context=bundle.resolver.resolve(identity(app_db,'intern'))
    entry=ENTRIES[0]
    def anchor(start,end,score):
        value=RankedCandidate(snapshot=app_db.snapshot,doc_id=entry.doc_id,revision_id=entry.revision_id,
            source_type=entry.source_type,start_char=start,end_char=end,score=score,
            retrieval_method='bm25',scoring_provenance='duplicate-anchor-fixture')
        return bundle.store.read_passage(context,value,guard=bundle.resolver)
    first,second=anchor(0,8,.5),anchor(8,16,.8)
    request=SearchRequest(query=QUESTION,user=ScopedUserContext.from_context(context),top_k=5,source_types=('jira',))
    bridge=_ScopedRetriever(bundle.retriever,context,bundle.resolver,request,bundle.runtime._catalog)
    expanded=bridge.read_more(request,(first,second),max_chars=24)
    assert bundle.retriever.expansions[-1][1]==(first,second)
    assert len(expanded)==1 and expanded[0].score==first.score
    assert expanded[0].retrieval_method==first.retrieval_method
    value=expanded[0].chunk
    assert value.doc_id=='general' and value.text==TEXTS[0][value.start_char:value.end_char]
    invalid=second.model_copy(update={'revision_id':'f'*64})
    before=len(bundle.retriever.expansions)
    with pytest.raises(EvidenceValidationError):
        bridge.read_more(request,(first,invalid),max_chars=24)
    assert len(bundle.retriever.expansions)==before

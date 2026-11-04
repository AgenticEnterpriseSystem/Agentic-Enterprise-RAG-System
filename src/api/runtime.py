"""Shared retrieval infrastructure with isolated benchmark or request scopes."""
import json
import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.config import ROOT, RetrievalSettings
from src.retrieval.factory import (RetrievalConfigurationError as RuntimeConfigurationError,
                                   load_retriever as load_configured_retriever,
                                   load_store_retriever)
from src.access.contracts import (AccessDenied, EvidenceValidationError, RequestContext, ResolvedIdentity,
    ScopedUserContext, SnapshotMismatch, guarded_dispatch, validate_grounded_evidence)
from src.contracts import (Answer, Chunk, Evidence, QueryRequest, QueryResponse,
                           PassageEvidence, SearchRequest, SnapshotRef, Trace, TraceStep,
                           Usage, UserContext, validate_evidence)
from src.storage.contracts import ReadFilters, validate_read_more_bounds
from src.storage.fingerprints import ValidatedCatalog
from src.generation.generator import (AnthropicConfig, AnthropicProvider, Completion,
                                      GenerationConfig, GroundedGenerator, ProviderError)


class WebSettings(RetrievalSettings):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    mode: Literal['retrieval', 'fixture', 'live'] = 'retrieval'
    demo_access: bool = False
    auth_mode: Literal['benchmark', 'local'] = 'benchmark'
    allowed_doc_ids_file: Path | None = None
    allowed_sources: tuple[str, ...] = ()
    frontend_dist: Path = ROOT / 'frontend/dist'
    allowed_origins: tuple[str, ...] = ('http://localhost:8000', 'http://127.0.0.1:8000',
                                        'http://localhost:5173', 'http://127.0.0.1:5173')
    allow_paid: bool = False
    model: str = ''
    budget_usd: float = Field(default=0, ge=0)
    input_price: float = Field(default=0, ge=0)
    output_price: float = Field(default=0, ge=0)
    question_budget_usd: float = Field(default=.1, gt=0)
    max_seconds: float = Field(default=120, gt=0, le=300)
    max_tokens: int = Field(default=16000, ge=256, le=64000)
    max_context_chars: int = Field(default=24000, ge=128, le=48000)
    max_rounds: int = Field(default=2, ge=1, le=5)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator('allowed_sources', 'allowed_origins', mode='before')
    @classmethod
    def csv(cls, value):
        return tuple(s.strip() for s in value.split(',') if s.strip()) if isinstance(value, str) else value

    @classmethod
    def from_env(cls):
        values = {k: os.environ['RAG_' + k.upper()]
                  for k in cls.model_fields if os.environ.get('RAG_' + k.upper(), '') != ''}
        # The public Python setting remains a strict integer. Environment values
        # have a single explicit decimal conversion rather than bool/float casts.
        if 'dense_block_rows' in values:
            raw = values['dense_block_rows']
            if not raw.isascii() or not raw.isdecimal():
                raise RuntimeConfigurationError('RAG_DENSE_BLOCK_ROWS must be a positive integer.')
            values['dense_block_rows'] = int(raw)
        return cls.model_validate(values)


class RuntimeErrorResponse(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class FixtureRetriever:
    """Synthetic UI exercise data, enabled only by explicit fixture mode."""
    doc_ids = ('fixture-policy',)
    source_types = ('confluence',)
    manifest = {'method': 'synthetic-fixture'}

    def search(self, request):
        if not request.user.permits('fixture-policy') or 'unavailable' in request.query.lower():
            return ()
        if request.source_types and 'confluence' not in request.source_types:
            return ()
        if request.after or request.before:
            return ()
        text = 'Synthetic demo policy: artifacts are retained for 90 days. Standard access turnaround is 4 business hours. Storage fee responsibility is not specified.'
        return (Evidence(citation_id='fixture-policy:0:' + str(len(text)),
            chunk=Chunk(chunk_id='fixture-policy:0:' + str(len(text)), doc_id='fixture-policy',
                source_type='confluence', title='Artifact escrow policy · synthetic fixture',
                metadata_origin='synthetic', text=text, start_char=0, end_char=len(text)),
            score=1, retrieval_method='bm25'),)


class FixtureProvider:
    reserved_usd = 0

    def complete(self, system, prompt):
        payload = json.loads(prompt)
        question = payload['question'].lower()
        if 'demo failure' in question:
            raise ProviderError(usage=Usage(estimated_cost_usd=0))
        partial = any(word in question for word in ('fee', 'billing', 'pays', 'conflict'))
        if system == CRITIC_PROMPT:
            value = dict(sufficient=not partial, confidence=.9 if not partial else .3,
                reason='Synthetic fixture covers retention and turnaround; fee responsibility is missing.' if partial else 'Synthetic fixture states the requested policy facts.',
                missing_facts=['Storage fee payer'] if partial else [],
                suggested_queries=['escrow storage fee payer'] if partial else [], suggested_sources=[])
        else:
            value = dict(status='insufficient_evidence' if partial else 'answered',
                text='Artifacts are retained for 90 days and standard access takes 4 business hours. ' +
                     ('The storage fee payer is not specified in this synthetic policy.' if partial else 'These are synthetic UI demonstration facts.') +
                     ' [' + payload['evidence'][0]['citation_id'] + ']',
                citation_ids=[payload['evidence'][0]['citation_id']])
        return Completion(json.dumps(value), Usage(api_calls=0, estimated_cost_usd=0))


def load_retriever(settings):
    """Use the existing retrievers; artifact checks happen before loading a model."""
    if settings.mode == 'fixture':
        return FixtureRetriever()
    return load_configured_retriever(settings)


class Runtime:
    def __init__(self, settings, retriever, provider=None):
        self.settings, self.retriever, self.provider = settings, retriever, provider
        ids = frozenset(retriever.doc_ids)
        self.source_types = tuple(getattr(retriever, 'source_types', ()))
        if not self.source_types:
            table = getattr(retriever, 'corpus', None)
            if table is None:
                table = retriever.bm25.corpus
            self.source_types = tuple(sorted(table.column('source_type').unique().to_pylist()))
        if settings.allowed_sources:
            if not set(settings.allowed_sources) <= set(self.source_types):
                raise RuntimeConfigurationError('RAG_ALLOWED_SOURCES contains unknown platforms.')
            self.source_types = settings.allowed_sources
        if settings.allowed_doc_ids_file:
            restricted = json.loads(settings.allowed_doc_ids_file.read_text())
            if not isinstance(restricted, list) or any(not isinstance(i, str) for i in restricted):
                raise RuntimeConfigurationError('RAG_ALLOWED_DOC_IDS_FILE must contain a JSON list of document IDs.')
            if not set(restricted) <= ids:
                raise RuntimeConfigurationError('RAG_ALLOWED_DOC_IDS_FILE contains document IDs outside this index.')
            ids = frozenset(restricted)
        self.user = UserContext(user_id='benchmark-demo', allowed_doc_ids=ids)
        generation = GenerationConfig(max_context_chars=settings.max_context_chars, retries=0)
        self.agent = (SearchAgent(retriever, EvidenceCritic(provider, generation),
            GroundedGenerator(provider, generation), config=AgentConfig(
                max_rounds=settings.max_rounds, max_context_chars=settings.max_context_chars,
                max_tokens=settings.max_tokens, max_seconds=settings.max_seconds,
                max_cost_usd=settings.question_budget_usd if settings.mode == 'live' else 0))
            if provider else None)

    def info(self):
        return {'mode': self.settings.mode, 'retrieval': self.settings.retrieval,
                'documents': len(self.retriever.doc_ids), 'allowed_documents': len(self.user.allowed_doc_ids),
                'source_types': self.source_types, 'generation_enabled': self.agent is not None,
                'scope': 'Explicit benchmark demo identity; not real authentication.',
                'model': self.settings.model if self.settings.mode == 'live' else None,
                'retrieval_config': self.retriever.manifest.get('config'),
                'embedding_model': self.retriever.manifest.get('dense', {}).get('model'),
                'budget_remaining_usd': max(0, self.settings.budget_usd-self.provider.reserved_usd)
                    if self.settings.mode == 'live' else None}

    def query(self, body: QueryRequest):
        if not self.user.allowed_doc_ids:
            raise RuntimeErrorResponse(403, 'scope_denied', 'This demo identity has no permitted documents.')
        sources = body.source_types or self.source_types
        if not set(sources) <= set(self.source_types):
            raise RuntimeErrorResponse(403, 'source_denied', 'Requested platforms are outside the server-resolved scope.')
        if self.settings.mode == 'live' and self.provider.reserved_usd >= self.settings.budget_usd:
            raise RuntimeErrorResponse(429, 'spending_limit', 'The server spending cap has been reached. An operator must review it before more calls.')
        request = SearchRequest(query=body.question.strip(), user=self.user, top_k=self.settings.top_k,
                                source_types=sources, after=body.after, before=body.before)
        if self.agent:
            response = self.agent.run(request, single_pass=body.mode == 'baseline')
        else:
            began = time.monotonic()
            hits = tuple(self.retriever.search(request))
            elapsed = (time.monotonic()-began)*1000
            response = QueryResponse(answer=Answer(status='insufficient_evidence',
                text='Retrieved passages only. Answer generation and critique are disabled in retrieval-only mode.'),
                sources=hits, trace=Trace(run_id=str(uuid4()), stop_reason='single_pass',
                    total_latency_ms=elapsed, usage=Usage(estimated_cost_usd=0),
                    steps=(TraceStep(round_number=1, query=request.query, tool='text_search',
                        evidence_ids=tuple(e.citation_id for e in hits), source_types=sources,
                        elapsed_ms=elapsed, usage=Usage(estimated_cost_usd=0)),)))
        # Validate the final application boundary even when an adapter is injected.
        response = QueryResponse.model_validate(response.model_dump())
        validate_evidence(self.user, response.sources)
        for item in response.sources:
            chunk = item.chunk
            if chunk.source_type not in sources or (request.after and (chunk.timestamp is None or chunk.timestamp < request.after)) or (request.before and (chunk.timestamp is None or chunk.timestamp > request.before)):
                raise ValueError('Response evidence violates request filters')
        return response


class _ScopedRetriever:
    """Request-local compatibility adapter; its user field grants no authority."""

    def __init__(self, backend, context, guard, request, catalog):
        self.backend, self.context, self.guard = backend, context, guard
        self.request, self.catalog = request, catalog
        self.filters = ReadFilters(source_types=request.source_types,
                                   after=request.after, before=request.before)

    def _current(self):
        self.guard.assert_current(self.context)
        self.context.assert_bound()

    def _restrictions(self, request):
        self._current()
        if not isinstance(request, SearchRequest):
            raise AccessDenied('A bound search request is required')
        ScopedUserContext.require_context(request.user, self.context)
        if (not isinstance(request.query, str) or not request.query.strip()
                or len(request.query) > 8000 or type(request.top_k) is not int
                or not 1 <= request.top_k <= self.request.top_k):
            raise AccessDenied('Search exceeds the admitted request bounds')
        try:
            filters = ReadFilters(source_types=request.source_types,
                                  after=request.after, before=request.before)
        except (ValueError, TypeError):
            raise AccessDenied('Invalid search restrictions') from None
        parent = self.filters
        # Empty nested filters inherit the admitted source restriction. Date
        # bounds must be carried forward explicitly and may only become tighter.
        if parent.source_types:
            if filters.source_types and not set(filters.source_types) <= set(parent.source_types):
                raise AccessDenied('Search cannot widen the admitted source scope')
            if not filters.source_types:
                filters = filters.model_copy(update={'source_types': parent.source_types})
        if ((parent.after and (filters.after is None or filters.after < parent.after))
                or (parent.before and (filters.before is None or filters.before > parent.before))):
            raise AccessDenied('Search cannot widen the admitted date scope')
        self._current()
        return filters

    def _passages(self, values, filters):
        self._current()
        checked = []
        for item in values:
            if not isinstance(item, PassageEvidence):
                raise EvidenceValidationError('Authenticated search requires passage evidence')
            try:
                item = PassageEvidence.model_validate(item.model_dump(warnings=False))
            except ValueError:
                raise EvidenceValidationError('Invalid retrieved passage evidence') from None
            chunk = item.chunk
            position = self.catalog.positions.get(chunk.doc_id)
            if position is None:
                raise EvidenceValidationError('Evidence is outside the canonical catalogue')
            canonical = self.catalog.entries[position]
            if (item.revision_id != canonical.revision_id or chunk.source_type != canonical.source_type
                    or chunk.timestamp != canonical.timestamp or chunk.metadata_origin != canonical.metadata_origin
                    or chunk.end_char > canonical.text_length
                    or item.citation_id != f'{chunk.doc_id}:{chunk.start_char}:{chunk.end_char}'
                    or chunk.chunk_id != item.citation_id):
                raise EvidenceValidationError('Evidence differs from canonical document metadata')
            if ((filters.source_types and chunk.source_type not in filters.source_types)
                    or (filters.after and (chunk.timestamp is None or chunk.timestamp < filters.after))
                    or (filters.before and (chunk.timestamp is None or chunk.timestamp > filters.before))):
                raise AccessDenied('Evidence is outside the admitted source/date scope')
            checked.append(item)
        result = tuple(checked)
        validate_grounded_evidence(self.context, result, self.guard)
        self._current()
        return result

    def search(self, request):
        filters = self._restrictions(request)
        hits = tuple(self.backend.search(self.context, request.query, guard=self.guard,
                                        top_k=request.top_k, filters=filters))
        self._current()
        if len(hits) > request.top_k:
            raise EvidenceValidationError('Search exceeded the admitted result limit')
        return self._passages(hits, filters)

    def read_more(self, request, anchors, *, max_chars=3000):
        filters = self._restrictions(request)
        anchors = tuple(anchors)
        try:
            validate_read_more_bounds(len(anchors), max_chars)
        except ValueError:
            raise EvidenceValidationError('Invalid document expansion bounds') from None
        # All anchors are checked before a backend may read even the first body.
        anchors = self._passages(anchors, filters)
        result = tuple(self.backend.read_more(self.context, anchors, guard=self.guard,
                                              max_chars=max_chars, filters=filters))
        result = self._passages(result, filters)
        parents = {item.chunk.doc_id: item for item in reversed(anchors)}
        seen = set()
        for item in result:
            chunk = item.chunk
            parent = parents.get(chunk.doc_id)
            if (parent is None or chunk.doc_id in seen or len(chunk.text) > max_chars
                    or item.revision_id != parent.revision_id or item.score != parent.score
                    or item.retrieval_method != parent.retrieval_method
                    or (chunk.start_char >= parent.chunk.start_char
                        and chunk.end_char <= parent.chunk.end_char)):
                raise EvidenceValidationError('Expansion left its matched document or made no progress')
            seen.add(chunk.doc_id)
        self._current()
        return result


class AuthenticatedRuntime:
    """Indexes are shared; identity, scope, bridge and agent belong to one query."""

    def __init__(self, settings, retriever, resolver, *, database, provider=None):
        self.settings = WebSettings.model_validate(settings.model_dump(warnings=False))
        if database is None or getattr(resolver, 'database', None) is not database:
            raise RuntimeConfigurationError('Authenticated runtime requires the shared authentication database.')
        snapshot = getattr(resolver, 'snapshot', None)
        if not isinstance(snapshot, SnapshotRef):
            raise RuntimeConfigurationError('Authenticated runtime requires a resolved snapshot.')
        self.snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        lexical, dense = getattr(retriever, 'bm25', None), getattr(retriever, 'dense', None)
        backend_snapshot = getattr(retriever, 'snapshot', None)
        if lexical is not None or dense is not None:
            lexical_snapshot = getattr(lexical, 'snapshot', None)
            if (lexical_snapshot is None or getattr(dense, 'snapshot', None) != lexical_snapshot
                    or (backend_snapshot is not None and backend_snapshot != lexical_snapshot)):
                raise RuntimeConfigurationError('Authenticated hybrid components must share one snapshot.')
            backend_snapshot = lexical_snapshot
        if backend_snapshot != self.snapshot:
            raise RuntimeConfigurationError('Authenticated retriever and policy snapshots must match.')
        handle = getattr(retriever, 'catalog_handle', None)
        if handle is None:
            handle = getattr(getattr(retriever, 'bm25', None), 'catalog_handle', None)
        try:
            self._catalog = ValidatedCatalog.require(handle, self.snapshot, issuer=database)
        except (ValueError, TypeError):
            raise RuntimeConfigurationError('Authenticated retrieval requires the shared canonical catalogue.') from None
        self.database, self.retriever, self.resolver, self.provider = database, retriever, resolver, provider
        sources = getattr(retriever, 'source_types', ())
        sources = sources() if callable(sources) else sources
        self._source_types = frozenset(sources)
        if not set(self.settings.allowed_sources) <= self._source_types:
            raise RuntimeConfigurationError('RAG_ALLOWED_SOURCES contains unknown platforms.')

    def info(self):
        return {'mode': self.settings.mode, 'retrieval': self.settings.retrieval,
                'authentication': 'local', 'generation_enabled': self.provider is not None}

    def query(self, body: QueryRequest, identity: ResolvedIdentity):
        if not isinstance(body, QueryRequest):
            raise AccessDenied('A validated question request is required')
        if not isinstance(identity, ResolvedIdentity):
            raise AccessDenied('A server-resolved identity is required')
        body = QueryRequest.model_validate(body.model_dump(warnings=False))
        identity = ResolvedIdentity.model_validate(identity.model_dump(warnings=False))
        context = self.resolver.resolve(identity)
        if not isinstance(context, RequestContext):
            raise EvidenceValidationError('Policy resolution requires a request context')
        if context.identity != identity:
            raise AccessDenied('Resolved context differs from the authenticated identity')
        self.resolver.assert_current(context)
        context.assert_bound()
        if context.binding.snapshot != self.snapshot:
            raise SnapshotMismatch('Policy and runtime snapshots differ')
        if getattr(context.scope, 'allowed_count', None) == 0:
            raise AccessDenied('This account has no permitted documents')
        sources = body.source_types or self.settings.allowed_sources
        if (not set(sources) <= self._source_types
                or (self.settings.allowed_sources
                    and not set(sources) <= set(self.settings.allowed_sources))):
            raise AccessDenied('Requested platforms are outside the server-resolved scope')
        if (self.settings.mode == 'live' and self.provider is not None
                and getattr(self.provider, 'reserved_usd', 0) >= self.settings.budget_usd):
            raise RuntimeErrorResponse(429, 'spending_limit',
                'The server spending cap has been reached. An operator must review it before more calls.')
        user = ScopedUserContext.from_context(context)
        request = SearchRequest(query=body.question.strip(), user=user, top_k=self.settings.top_k,
                                source_types=sources, after=body.after, before=body.before)
        bridge = _ScopedRetriever(self.retriever, context, self.resolver, request, self._catalog)
        with guarded_dispatch(context, self.resolver):
            if self.provider is not None:
                generation = GenerationConfig(max_context_chars=self.settings.max_context_chars, retries=0)
                agent = SearchAgent(bridge, EvidenceCritic(self.provider, generation),
                    GroundedGenerator(self.provider, generation), config=AgentConfig(
                        max_rounds=self.settings.max_rounds, max_context_chars=self.settings.max_context_chars,
                        max_tokens=self.settings.max_tokens, max_seconds=self.settings.max_seconds,
                        max_cost_usd=self.settings.question_budget_usd if self.settings.mode == 'live' else 0))
                response = agent.run(request, single_pass=body.mode == 'baseline',
                                     context=context, guard=self.resolver)
            else:
                began = time.monotonic()
                hits = bridge.search(request)
                elapsed = (time.monotonic() - began) * 1000
                response = QueryResponse(snapshot=context.binding.snapshot,
                    answer=Answer(status='insufficient_evidence',
                        text='Retrieved passages only. Answer generation and critique are disabled in retrieval-only mode.'),
                    sources=hits, trace=Trace(run_id=str(uuid4()), stop_reason='single_pass',
                        total_latency_ms=elapsed, usage=Usage(estimated_cost_usd=0),
                        steps=(TraceStep(round_number=1, query=request.query, tool='text_search',
                            evidence_ids=tuple(item.citation_id for item in hits), source_types=sources,
                            elapsed_ms=elapsed, usage=Usage(estimated_cost_usd=0)),)))
            if not isinstance(response, QueryResponse):
                raise EvidenceValidationError('Runtime returned an invalid response')
            try:
                response = QueryResponse.model_validate(response.model_dump(warnings=False))
            except ValueError:
                raise EvidenceValidationError('Runtime returned an invalid response') from None
            if response.snapshot != context.binding.snapshot:
                raise SnapshotMismatch('Response differs from the request snapshot')
            bridge._passages(response.sources, bridge.filters)
            self.resolver.assert_current(context)
            context.assert_bound()
            return response


def _json_file(path, *, max_bytes):
    """Bound operator configuration reads and reject ambiguous JSON objects."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('Invalid JSON constant')

    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError('Configuration is too large')
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (OSError, ValueError, TypeError, UnicodeError):
        raise RuntimeConfigurationError('The configured snapshot or document restriction file is invalid.') from None


def _provider(settings, *, allow_fixture):
    provider = None
    if settings.mode == 'live':
        if not settings.allow_paid or not settings.model or not os.environ.get('ANTHROPIC_API_KEY'):
            raise RuntimeConfigurationError('Live mode requires RAG_ALLOW_PAID=true, RAG_MODEL and ANTHROPIC_API_KEY.')
        if min(settings.budget_usd, settings.input_price, settings.output_price) <= 0:
            raise RuntimeConfigurationError('Live mode requires positive RAG_BUDGET_USD, RAG_INPUT_PRICE and RAG_OUTPUT_PRICE.')
        config = AnthropicConfig(model=settings.model, allow_paid=True, budget_usd=settings.budget_usd,
            input_usd_per_million=settings.input_price, output_usd_per_million=settings.output_price,
            max_output_tokens=1024, timeout_seconds=min(30, settings.max_seconds))
        provider = AnthropicProvider(config)
    elif settings.mode == 'fixture' and allow_fixture:
        provider = FixtureProvider()
    return provider


def build_runtime(settings, *, database=None, authenticated=False):
    if type(authenticated) is not bool:
        raise RuntimeConfigurationError('Authenticated runtime selection must be a server boolean.')
    settings = WebSettings.model_validate(settings.model_dump(warnings=False))
    if not authenticated:
        if settings.auth_mode == 'local':
            raise RuntimeConfigurationError('Local authentication requires the authenticated runtime.')
        if not settings.demo_access:
            raise RuntimeConfigurationError('Set RAG_DEMO_ACCESS=true to explicitly enable the benchmark demo identity.')
        provider = _provider(settings, allow_fixture=True)
        return Runtime(settings, load_retriever(settings), provider)

    if settings.mode == 'fixture':
        raise RuntimeConfigurationError('Synthetic fixture mode is unavailable with local authentication.')
    if settings.snapshot_file is None:
        raise RuntimeConfigurationError('Local authentication requires RAG_SNAPSHOT_FILE.')

    # Imports stay local so the isolated benchmark profile does not require a
    # PostgreSQL driver and cannot accidentally select a privileged fallback.
    from src.access.policy import PolicyResolver
    from src.db.runtime import Database
    from src.storage.postgres_store import PostgresDocumentStore
    if not isinstance(database, Database):
        raise RuntimeConfigurationError('Authenticated runtime requires the shared authentication database.')
    try:
        snapshot = SnapshotRef.model_validate(_json_file(settings.snapshot_file, max_bytes=65536))
    except ValueError:
        raise RuntimeConfigurationError('RAG_SNAPSHOT_FILE must contain one exact canonical snapshot.') from None
    documents = None
    if settings.allowed_doc_ids_file is not None:
        documents = _json_file(settings.allowed_doc_ids_file, max_bytes=64 * 1024 * 1024)
        if (not isinstance(documents, list)
                or any(not isinstance(item, str) or not item.strip() for item in documents)
                or len(documents) != len(set(documents))):
            raise RuntimeConfigurationError('RAG_ALLOWED_DOC_IDS_FILE must contain distinct nonblank document IDs.')
        documents = tuple(documents)
    provider = _provider(settings, allow_fixture=False)
    store = PostgresDocumentStore(database, snapshot, retain_catalog=True)
    resolver = PolicyResolver(database, snapshot, source_types=settings.allowed_sources,
                              doc_ids=documents, catalog=store.validated_catalog())
    return AuthenticatedRuntime(settings, load_store_retriever(settings, store), resolver,
                                database=database, provider=provider)

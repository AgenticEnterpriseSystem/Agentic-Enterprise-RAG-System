"""Shared retrieval infrastructure with isolated benchmark or request scopes."""
import json
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4
from weakref import WeakKeyDictionary

from pydantic import ConfigDict, Field, field_validator

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.config import ROOT, RetrievalSettings
from src.retrieval.factory import (RetrievalConfigurationError as RuntimeConfigurationError,
                                   load_retriever as load_configured_retriever,
                                   load_store_retriever)
from src.access.contracts import (AccessDenied, EvidenceValidationError, RequestContext, ResolvedIdentity,
    ScopedUserContext, SnapshotMismatch, guarded_dispatch, validate_grounded_evidence)
from src.contracts import (AggregateEvidence, Answer, Chunk, Evidence, QueryRequest, QueryResponse,
                           PassageEvidence, SearchRequest, SnapshotRef, Trace, TraceStep,
                           StructuredRecord, Usage, UserContext, validate_evidence)
from src.storage.contracts import ReadFilters, validate_read_more_bounds
from src.storage.fingerprints import ValidatedCatalog
from src.generation.generator import (AnthropicConfig, AnthropicProvider, Completion,
                                      GenerationConfig, GroundedGenerator, ProviderError)


log = logging.getLogger(__name__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


class RuntimeCapture:
    """Issued, immutable, memory-only input receipt; never an HTTP history DTO.

    Sensitive state is held in a weak issuance registry rather than assignable
    attributes. Copying or fabricating the visible receipt cannot issue a capture.
    """

    __slots__ = ('__weakref__',)

    def __new__(cls):
        raise TypeError('Runtime captures are issued by the authenticated runtime')

    def __init_subclass__(cls, **kwargs):
        raise TypeError('Runtime captures cannot be subclassed')

    def __copy__(self):
        raise TypeError('Runtime captures cannot be copied')

    def __deepcopy__(self, memo):
        raise TypeError('Runtime captures cannot be copied')

    def __reduce_ex__(self, protocol):
        raise TypeError('Runtime captures cannot be serialized')

    @property
    def response(self):
        value = _capture_state(self)['response']
        return QueryResponse.model_validate(value.model_dump(warnings=False))

    @property
    def generation_evidence(self):
        return tuple(type(item).model_validate(item.model_dump(warnings=False))
                     for item in _capture_state(self)['selected'])

    @property
    def payload_sha256(self):
        return _capture_state(self)['payload_sha256']

    @property
    def binding(self):
        value = _capture_state(self)['context'].binding
        return type(value).model_validate(value.model_dump(warnings=False))

    @property
    def run_id(self):
        return _capture_state(self)['response'].trace.run_id


_CAPTURES = WeakKeyDictionary()
_CAPTURE_LOCK = Lock()


def _capture_state(capture, runtime=None):
    with _CAPTURE_LOCK:
        state = _CAPTURES.get(capture) if type(capture) is RuntimeCapture else None
    if state is None or (runtime is not None and state['runtime'] is not runtime):
        raise AccessDenied('An issued current runtime capture is required')
    return state


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


class _RunBoundRegistry:
    """An aggregate issued to another run is never this request's evidence."""

    def __init__(self, runtime, context, run_id):
        self.runtime, self.context, self.run_id = runtime, context, run_id

    def resolve(self, artifact_id):
        import sqlalchemy as sa
        from src.db import schema
        from src.db.runtime import DatabaseRoleError
        from src.access.contracts import PolicyUnavailable

        guard, context = self.runtime.resolver, self.context
        guard.assert_current(context)
        try:
            with self.runtime.database.transaction(context) as connection:
                bound = connection.execute(sa.select(schema.aggregate_results.c.run_id).where(
                    schema.aggregate_results.c.artifact_id == artifact_id)).scalar_one_or_none()
        except (sa.exc.SQLAlchemyError, DatabaseRoleError):
            raise PolicyUnavailable('Current aggregate evidence is unavailable') from None
        guard.assert_current(context)
        if bound != self.run_id:
            raise AccessDenied('Aggregate evidence is unavailable in this run')
        result = self.runtime.jira_tool.registry(context, guard=guard).resolve(artifact_id)
        guard.assert_current(context)
        return result


class _RunJira:
    def __init__(self, runtime, context, run_id, search, registry):
        self.runtime, self.context, self.run_id = runtime, context, run_id
        self.search, self.registry = search, registry

    def execute(self, request):
        from src.contracts import ToolRequest
        if not isinstance(request, ToolRequest):
            raise EvidenceValidationError('A bound structured request is required')
        ScopedUserContext.require_context(request.user, self.context)
        if (request.source_type != 'jira' or type(request.limit) is not int
                or not 1 <= request.limit <= self.search.top_k
                or (self.search.source_types and 'jira' not in self.search.source_types)
                or self.search.after is not None or self.search.before is not None):
            raise AccessDenied('Structured request exceeds its admitted filters')
        self.runtime.runs.require_pending(self.context, self.run_id)
        result = self.runtime.jira_tool.execute(request, context=self.context,
            guard=self.runtime.resolver, run_id=self.run_id)
        self.runtime._native(self.context, (*result.evidence, *result.records), self.registry)
        return result


class AuthenticatedRuntime:
    """Indexes are shared; identity, scope, bridge and agent belong to one query."""

    def __init__(self, settings, retriever, resolver, *, database, provider=None,
                 router=None, provider_identity=None, provider_max_output_tokens=1024,
                 router_attestation=None):
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
        from src.retrieval.tools import JiraTool
        from src.storage.runs import RunRepository
        from src.evaluation.experiment_contracts import ModelProvenance
        if type(provider_max_output_tokens) is not int or provider_max_output_tokens < 1:
            raise RuntimeConfigurationError('Provider output limit must be a positive server integer.')
        if isinstance(provider, AnthropicProvider):
            actual_identity = ModelProvenance(provider='anthropic', model_id=provider.config.model)
            if provider_identity is not None and provider_identity != actual_identity:
                raise RuntimeConfigurationError('Provider identity differs from its configured model.')
            provider_identity = actual_identity
            provider_max_output_tokens = provider.config.max_output_tokens
        elif provider is None:
            provider_identity = ModelProvenance(provider='disabled', model_id='retrieval-only')
        elif provider_identity is not None:
            try:
                provider_identity = ModelProvenance.model_validate(provider_identity.model_dump(warnings=False))
            except (ValueError, AttributeError):
                raise RuntimeConfigurationError('Scripted provider identity must be explicitly typed.') from None
        self.router, self._provider_identity = router, provider_identity
        self._provider_max_output_tokens = provider_max_output_tokens
        self._router_attestation = None
        if router_attestation is not None:
            self._router_attestation = self._fixture_route(router_attestation)
            if not callable(getattr(router, 'export_attestation', None)):
                raise RuntimeConfigurationError('A fixture router must explicitly attest its current behavior.')
        self._router_profile()
        self.jira_tool = JiraTool(database, self.snapshot)
        self.runs = RunRepository(database, resolver)
        sources = getattr(retriever, 'source_types', ())
        sources = sources() if callable(sources) else sources
        self._source_types = frozenset(sources)
        if not set(self.settings.allowed_sources) <= self._source_types:
            raise RuntimeConfigurationError('RAG_ALLOWED_SOURCES contains unknown platforms.')

    def info(self):
        return {'mode': self.settings.mode, 'retrieval': self.settings.retrieval,
                'authentication': 'local', 'generation_enabled': self.provider is not None}

    def _native(self, context, evidence, registry):
        return validate_grounded_evidence(context, tuple(evidence), self.resolver, registry,
            record_validator=lambda records: self.jira_tool.validate_records(
                context, records, guard=self.resolver))

    @staticmethod
    def _fixture_route(value):
        if (type(value) is not dict or set(value) != {'kind', 'router_id', 'behavior'}
                or value.get('kind') != 'scripted_fixture'
                or not isinstance(value.get('router_id'), str)
                or not value['router_id'].strip() or len(value['router_id']) > 128
                or type(value.get('behavior')) is not dict):
            raise RuntimeConfigurationError('A canonical explicit fixture-route attestation is required.')
        try:
            encoded = _canonical(value)
            if len(encoded) > 16384:
                raise ValueError()
            return json.loads(encoded)
        except (ValueError, TypeError, RecursionError):
            raise RuntimeConfigurationError('Fixture-route attestation must be bounded finite JSON.') from None

    def _router_profile(self):
        from src.agents.routing import ModelRouter
        if self.router is None:
            return {'kind': 'text_router', 'version': '1'}
        if type(self.router) is ModelRouter:
            if self.router.provider is not self.provider:
                raise RuntimeConfigurationError('The model router must share the attested runtime provider.')
            return {'kind': 'model_router', 'version': '1'}
        if self._router_attestation is None:
            return None
        try:
            supplied = self.router.export_attestation()
        except Exception:
            raise EvidenceValidationError('Explicit fixture-route attestation is unavailable') from None
        current = self._fixture_route(supplied)
        if current != self._router_attestation:
            raise EvidenceValidationError('Explicit fixture-route behavior changed after server configuration')
        return current

    def query(self, body: QueryRequest, identity: ResolvedIdentity):
        return self.query_with_capture(body, identity).response

    def query_with_capture(self, body: QueryRequest, identity: ResolvedIdentity):
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
        # An explicitly attested route must still describe the behavior admitted
        # now; an entirely unattested injected route remains nonexportable.
        self._router_profile()
        from src.agents.routing import ToolSpec
        from src.storage.runs import RunStepMetadata
        run_id = str(uuid4())
        registry = _RunBoundRegistry(self, context, run_id)
        selected, payload_hash = (), None
        # Capture code bytes before dispatch; a later operator mapping must bind
        # to these same runtime modules rather than merely name a Git commit.
        source_files = self._source_files()
        try:
            export_profile = self._scientific_profile(body)
        except EvidenceValidationError:
            # Ordinary injected adapters/providers can still execute guarded
            # queries. They cannot acquire an invented experiment attestation.
            export_profile = None

        def on_input(evidence, payload):
            nonlocal selected, payload_hash
            if not isinstance(payload, dict) or payload.get('question') != request.query:
                raise EvidenceValidationError('Generation input differs from the admitted question')
            checked = tuple(type(item).model_validate(item.model_dump(warnings=False)) for item in evidence)
            bridge._passages(tuple(item for item in checked if isinstance(item, PassageEvidence)), bridge.filters)
            self._native(context, checked, registry)
            selected, payload_hash = checked, _sha(_canonical(payload))

        config_sha = _sha(_canonical({'settings': self.settings.model_dump(mode='json', exclude={
            'frontend_dist', 'allowed_origins'}), 'request': body.model_dump(mode='json', exclude={'question'}),
            'source_files': source_files, 'effective_profile': export_profile}))
        self.runs.begin(context, run_id, mode=body.mode, question=request.query, config_sha256=config_sha)
        try:
            with guarded_dispatch(context, self.resolver, aggregates=registry,
                    record_validator=lambda records: self.jira_tool.validate_records(context, records, guard=self.resolver)):
                generation = GenerationConfig(max_context_chars=self.settings.max_context_chars, retries=0)
                tools = {'jira': ToolSpec(_RunJira(self, context, run_id, request, registry), 'jira',
                    'Verified native Jira records and exact counts of canonical source records; native keys may repeat.',
                    ('lookup', 'filter', 'count'), ('doc_id', 'native_key', 'status', 'priority', 'project_id'))}
                agent = SearchAgent(bridge, EvidenceCritic(self.provider, generation),
                    GroundedGenerator(self.provider, generation, on_input=on_input), router=self.router,
                    tools=tools, config=AgentConfig(max_rounds=self.settings.max_rounds,
                        max_context_chars=self.settings.max_context_chars, max_tokens=self.settings.max_tokens,
                        max_seconds=self.settings.max_seconds,
                        max_cost_usd=self.settings.question_budget_usd if self.settings.mode == 'live' else 0))
                response = agent.run(request, single_pass=body.mode == 'baseline', context=context,
                    guard=self.resolver, run_id=run_id, retrieval_only=self.provider is None,
                    aggregates=registry, record_validator=lambda records: self.jira_tool.validate_records(
                        context, records, guard=self.resolver))
                if not isinstance(response, QueryResponse):
                    raise EvidenceValidationError('Runtime returned an invalid response')
                try:
                    response = QueryResponse.model_validate(response.model_dump(warnings=False))
                except ValueError:
                    raise EvidenceValidationError('Runtime returned an invalid response') from None
                if response.snapshot != context.binding.snapshot or response.trace.run_id != run_id:
                    raise SnapshotMismatch('Response differs from the request snapshot or server run')
                bridge._passages(tuple(item for item in response.sources if isinstance(item, PassageEvidence)), bridge.filters)
                self._native(context, response.sources, registry)
                if response.trace.stop_reason == 'clarification':
                    selected, payload_hash = (), None
                for ordinal, step in enumerate(response.trace.steps, 1):
                    self.runs.append(context, run_id, ordinal, RunStepMetadata(
                        tool=step.tool, round_number=step.round_number,
                        evidence_ids=step.evidence_ids, query_sha256=_sha(step.query.encode('utf-8')),
                        source_types=step.source_types, elapsed_ms=step.elapsed_ms, usage=step.usage,
                        input_payload_sha256=payload_hash if step.tool.split(':', 1)[0] == 'generate' else None))
                self.runs.finish(context, run_id,
                    status='clarification' if response.trace.stop_reason == 'clarification' else response.answer.status,
                    usage=response.trace.usage, latency_ms=response.trace.total_latency_ms,
                    finished_at=datetime.now(timezone.utc))
                self.resolver.assert_current(context)
                context.assert_bound()
        except Exception:
            # A failed authority check may make even the pending row invisible.
            # Do not elevate or manufacture a terminal result to repair history.
            log.warning('Run lifecycle incomplete; current authority did not permit completion')
            raise
        capture = object.__new__(RuntimeCapture)
        state = {'runtime': self, 'context': context, 'response': response, 'selected': selected,
            'payload_sha256': payload_hash, 'question': request.query, 'body': body,
            'registry': registry, 'source_files': source_files, 'profile': export_profile}
        with _CAPTURE_LOCK:
            _CAPTURES[capture] = state
        return capture

    def get_run_metadata(self, run_id, identity):
        if not isinstance(identity, ResolvedIdentity):
            raise AccessDenied('A current authenticated identity is required')
        identity = ResolvedIdentity.model_validate(identity.model_dump(warnings=False))
        return self.runs.get_metadata(self.resolver.resolve(identity), run_id)

    @staticmethod
    def _source_files():
        # Bound all application source, including policy/storage/tool validation.
        # Read only code bytes, never model/index contents or operator secrets.
        return {path.relative_to(ROOT).as_posix(): _sha(path.read_bytes())
                for path in sorted((ROOT / 'src').rglob('*.py'))}

    def export_profile(self, capture):
        state = self._export_state(capture)
        expected = state['profile']
        if expected is None or self._scientific_profile(state['body']) != expected:
            raise EvidenceValidationError('Actual experiment settings are unavailable or changed after capture')
        return {**json.loads(_canonical(expected)), 'source_files': dict(state['source_files'])}

    def _scientific_profile(self, body):
        from src.agents.routing import ModelRouter, ROUTER_PROMPT
        from src.evaluation.experiment_contracts import GenerationBudgets, ModelProvenance
        from src.retrieval.store import StoreBM25Retriever, StoreDenseRetriever, StoreHybridRetriever
        if self._provider_identity is None:
            raise EvidenceValidationError('Export requires the actual provider identity')
        router_profile = self._router_profile()
        if router_profile is None:
            raise EvidenceValidationError('Export requires an explicit attestation of custom router behavior')
        model_identity, output_limit = self._provider_identity, self._provider_max_output_tokens
        if isinstance(self.provider, AnthropicProvider):
            model_identity = ModelProvenance(provider='anthropic', model_id=self.provider.config.model)
            output_limit = self.provider.config.max_output_tokens
        backend = self.retriever
        embedding = None
        if isinstance(backend, StoreHybridRetriever):
            method = 'store-hybrid'
            index = {'bm25': json.loads(backend.bm25._manifest), 'dense': backend.dense.index.manifest}
            configuration = {'hybrid': backend.config.model_dump(mode='json')}
            dense = backend.dense
            if backend.reranker is not None:
                raise EvidenceValidationError('Export requires an explicitly attested reranker profile')
        elif isinstance(backend, StoreDenseRetriever):
            method, index, configuration, dense = 'store-dense', backend.index.manifest, {}, backend
        elif isinstance(backend, StoreBM25Retriever):
            method, index, configuration, dense = 'store-bm25', json.loads(backend._manifest), {}, None
        else:
            raise EvidenceValidationError('Export requires an attested store-backed retriever')
        if dense is not None:
            actual_model = dense.encoder.identity
            embedding = ModelProvenance(provider='sentence-transformers',
                model_id=actual_model['name'], revision=actual_model.get('revision')).model_dump(mode='json')
            configuration['dense_execution'] = {'block_rows': dense.index.block_rows,
                                                'file_cache_policy': dense.index.file_cache_policy}
        configuration.update(method=method, top_k=self.settings.top_k,
            router=router_profile,
            filters={'source_types': list(body.source_types or self.settings.allowed_sources),
                     'after': body.after.isoformat() if body.after else None,
                     'before': body.before.isoformat() if body.before else None})
        index = json.loads(_canonical(index))
        budgets = GenerationBudgets(top_k=self.settings.top_k, max_question_chars=8000,
            max_context_chars=self.settings.max_context_chars, max_output_tokens=output_limit,
            max_run_tokens=self.settings.max_tokens, max_run_seconds=self.settings.max_seconds,
            max_run_cost_usd=self.settings.question_budget_usd if self.settings.mode == 'live' else 0, retries=0)
        return {'generation_model': model_identity.model_dump(mode='json'),
            'budgets': budgets.model_dump(mode='json'), 'temperature': 0,
            'workflow': 'fixed' if body.mode == 'baseline' else 'adaptive',
            'max_rounds': 1 if body.mode == 'baseline' else self.settings.max_rounds,
            'generator_prompt_sha256': _sha(__import__('src.generation.generator', fromlist=['SYSTEM_PROMPT']).SYSTEM_PROMPT.encode()),
            'critic_prompt_sha256': _sha(CRITIC_PROMPT.encode()) if body.mode != 'baseline' else None,
            'router_prompt_sha256': _sha(ROUTER_PROMPT.encode()) if type(self.router) is ModelRouter else None,
            'retrieval': {'retriever_id': method, 'snapshot': self.snapshot.model_dump(mode='json'),
                'index_manifest_sha256': _sha(_canonical(index)), 'config_sha256': _sha(_canonical(configuration)),
                'embedding_model': embedding},
            'index_manifest': index, 'retrieval_config': configuration}

    def _export_state(self, capture):
        state = _capture_state(capture, self)
        context = state['context']
        self.resolver.assert_current(context)
        context.assert_bound()
        if self._source_files() != state['source_files']:
            raise EvidenceValidationError('Runtime source changed after input capture')
        self.runs.get_metadata(context, state['response'].trace.run_id)
        self._native(context, state['selected'], state['registry'])
        self._native(context, state['response'].sources, state['registry'])
        return state

    def export_capture(self, capture, *, manifest, verified_inputs, question_id, directory=None):
        from src.evaluation.experiment_contracts import ExperimentManifest, RuntimeCaseExport, RuntimeFailure
        from src.evaluation.experiment_inputs import VerifiedInputs
        from src.evaluation.experiment_io import RunDirectory
        state = self._export_state(capture)
        if not isinstance(manifest, ExperimentManifest) or not isinstance(verified_inputs, VerifiedInputs):
            raise EvidenceValidationError('Export requires verified experiment inputs')
        manifest = ExperimentManifest.model_validate(manifest.model_dump(warnings=False))
        checked = verified_inputs.recheck(manifest)
        if manifest.scope != state['context'].binding or manifest.retrieval.snapshot != self.snapshot:
            raise AccessDenied('Experiment mapping differs from the current capture scope')
        question = next((value for value in checked.questions if value.question_id == question_id), None)
        if question_id not in manifest.question_ids or question is None or question.question != state['question']:
            raise EvidenceValidationError('Question mapping differs from the captured runtime question')
        profile = self.export_profile(capture)
        declared = manifest.model_dump(mode='json')
        for key in ('generation_model', 'budgets', 'temperature', 'workflow', 'max_rounds',
                'generator_prompt_sha256', 'critic_prompt_sha256', 'router_prompt_sha256', 'retrieval'):
            if declared[key] != profile[key]:
                raise EvidenceValidationError('Experiment settings differ from the actual captured runtime')
        for relative, digest in state['source_files'].items():
            path = checked.files.source_repo / relative
            try:
                same = path.is_file() and _sha(path.read_bytes()) == digest
            except OSError:
                same = False
            if not same:
                raise EvidenceValidationError('Experiment source differs from the captured runtime')
        response = state['response']
        case = RuntimeCaseExport(manifest_id=manifest.manifest_id, run_id=response.trace.run_id,
            question_id=question_id, question=state['question'], question_sha256=_sha(state['question'].encode()),
            response=response, generation_evidence=state['selected'],
            error=RuntimeFailure(code='runtime_error', stage='runtime', retryable=False)
                if response.answer.status == 'error' else None)
        self._export_state(capture)
        if directory is not None:
            if not isinstance(directory, RunDirectory) or directory.manifest != manifest:
                raise EvidenceValidationError('Export directory differs from the verified manifest')
            directory.append_case(case)
            self._export_state(capture)
        return case


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
    from src.agents.routing import ModelRouter
    return AuthenticatedRuntime(settings, load_store_retriever(settings, store), resolver,
        database=database, provider=provider, router=ModelRouter(provider) if provider is not None else None)

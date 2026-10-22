"""One process-local retriever, demo scope and budgeted agent runtime."""
import json
import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.config import CORPUS_DIR, ROOT
from src.contracts import (Answer, Chunk, Evidence, QueryRequest, QueryResponse,
                           SearchRequest, Trace, TraceStep, Usage, UserContext, validate_evidence)
from src.generation.generator import (AnthropicConfig, AnthropicProvider, Completion,
                                      GenerationConfig, GroundedGenerator, ProviderError)


class WebSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    mode: Literal['retrieval', 'fixture', 'live'] = 'retrieval'
    demo_access: bool = False
    retrieval: Literal['bm25', 'hybrid'] = 'bm25'
    corpus: Path = CORPUS_DIR / 'documents.parquet'
    bm25_index: Path = CORPUS_DIR / 'indexes/bm25'
    dense_index: Path = CORPUS_DIR / 'indexes/dense'
    device: str = 'cpu'
    fp16: bool | None = None
    bm25_weight: float = Field(default=.7, ge=0, le=1)
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
        return cls.model_validate({k: os.environ['RAG_' + k.upper()]
                                   for k in cls.model_fields if os.environ.get('RAG_' + k.upper(), '') != ''})


class RuntimeErrorResponse(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class RuntimeConfigurationError(ValueError):
    """Operator-safe setup explanation, with no credentials or provider text."""


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
    from src.retrieval.corpus_bm25 import CorpusBM25
    bm25 = CorpusBM25.load(settings.bm25_index, corpus_path=settings.corpus)
    if settings.retrieval == 'bm25':
        return bm25
    from src.retrieval.corpus_dense import BGEModel, CorpusDense, INDEX_VERSION
    from src.retrieval.hybrid import HybridConfig, HybridRetriever
    manifest = json.loads((settings.dense_index / 'manifest.json').read_text())
    if manifest.get('version') != INDEX_VERSION or not manifest.get('complete'):
        raise RuntimeConfigurationError('Dense manifest is incomplete or unsupported. Supply a complete version-1 dense index.')
    if manifest.get('corpus_sha256') != bm25.manifest.get('corpus_sha256'):
        raise RuntimeConfigurationError('BM25 and dense indexes must match the same corpus file (corpus_sha256).')
    if not (settings.dense_index / 'doc_ids.json').is_file():
        raise FileNotFoundError('Dense document IDs are missing')
    if not isinstance(manifest.get('shards'), int) or manifest['shards'] < 1:
        raise RuntimeConfigurationError('Dense manifest must record a positive shard count.')
    if settings.fp16 is not None and settings.fp16 != manifest['model']['fp16']:
        raise RuntimeConfigurationError('Query precision must match the dense index manifest. Unset RAG_FP16 to use the recorded precision.')
    for i in range(manifest['shards']):
        for suffix in ('.npy', '.parquet'):
            if not (settings.dense_index / f'shard_{i:05d}{suffix}').is_file():
                raise RuntimeConfigurationError('Dense index is missing a shard. Supply every shard_XXXXX.npy and .parquet file.')
    try:
        model = BGEModel.for_index(settings.dense_index, settings.device, fp16=settings.fp16)
        dense = CorpusDense.load(settings.dense_index, model=model,
                                 corpus_path=settings.corpus, corpus=bm25.corpus)
    except ValueError:
        raise RuntimeConfigurationError('Dense model or index is incompatible. Check the pinned BGE name/revision, query encoding, precision, corpus order and shard dimensions.') from None
    return HybridRetriever(bm25, dense, config=HybridConfig(
        bm25_weight=settings.bm25_weight, dense_weight=1-settings.bm25_weight))


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


def build_runtime(settings):
    if not settings.demo_access:
        raise RuntimeConfigurationError('Set RAG_DEMO_ACCESS=true to explicitly enable the benchmark demo identity.')
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
    elif settings.mode == 'fixture':
        provider = FixtureProvider()
    return Runtime(settings, load_retriever(settings), provider)

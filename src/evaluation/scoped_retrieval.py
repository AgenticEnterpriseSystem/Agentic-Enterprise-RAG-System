"""Bounded evaluation over the accepted store/policy backend.

This module authenticates through existing services and never derives grants.
Only one question's candidate metadata is replayed; evidence bodies are neither
cached across questions nor recovered from legacy corpus loaders.
"""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from src.access.contracts import (AccessError, EvidenceValidationError, RequestContext,
    ResolvedIdentity, SnapshotMismatch, StaleContext, validate_grounded_evidence)
from src.config import ROOT, RetrievalSettings
from src.contracts import PassageEvidence, SnapshotRef
from src.storage.contracts import ReadFilters, validate_read_more_bounds
from src.storage.fingerprints import ValidatedCatalog


class ScopedEvaluationError(ValueError):
    """Sanitized evaluation/setup failure without provider or credential text."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    """Attest executable source bytes even in a lean image without Git."""
    names = ('src/config.py', 'src/contracts.py', 'src/access/contracts.py',
        'src/access/policy.py', 'src/access/repository.py', 'src/auth/sessions.py',
        'src/auth/accounts.py', 'src/auth/settings.py', 'src/db/runtime.py',
        'src/storage/contracts.py', 'src/storage/fingerprints.py', 'src/storage/postgres_store.py',
        'src/retrieval/factory.py', 'src/retrieval/corpus_bm25.py', 'src/retrieval/corpus_dense.py',
        'src/retrieval/sharded_dense.py', 'src/retrieval/store.py', 'src/retrieval/hybrid.py',
        'src/retrieval/fusion.py', 'src/retrieval/passages.py', 'src/retrieval/reranker.py',
        'src/evaluation/retrieval_metrics.py', 'src/evaluation/scoped_retrieval.py',
        'src/evaluation/run_retrieval.py', 'src/evaluation/run_hybrid.py')
    return {name: file_sha256(ROOT / name) for name in names}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScopedEvaluationError('Duplicate configuration fields')
        result[key] = value
    return result


def _json(path, *, limit, secret=False):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_size > limit
                    or (secret and stat.S_IMODE(info.st_mode) != 0o600)):
                raise ScopedEvaluationError('Invalid bounded input file')
            raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ScopedEvaluationError('Input file exceeds its bound')
        return json.loads(raw, object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (OSError, ValueError, TypeError):
        raise ScopedEvaluationError('Evaluation input or credential file is invalid') from None


def add_scope_arguments(parser):
    parser.add_argument('--snapshot-file', type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path, required=True,
                        help='Private ordinary0600 benchmark credentials or session-token JSON; never a CLI secret')
    parser.add_argument('--principal-kind', choices=('benchmark', 'session'), default='benchmark')
    parser.add_argument('--allowed-doc-ids-file', type=Path,
                        help='Optional operator narrowing; an explicit [] denies all documents')
    parser.add_argument('--source-types', default='', help='Comma-separated narrowing filter; question source annotations are not filters')
    parser.add_argument('--after', type=datetime.fromisoformat)
    parser.add_argument('--before', type=datetime.fromisoformat)
    parser.add_argument('--dense-block-rows', type=int, default=8192)
    parser.add_argument('--dense-file-cache-policy', choices=('retain', 'discard'), default='retain')
    parser.add_argument('--read-more-chars', type=int, default=0,
                        help='Also verify adjacent reads from at most two returned documents (1..6000; 0 disables)')
    parser.add_argument('--limit', type=int, help='Explicit partial/smoke subset of the selected split')


def scope_filters(args):
    return ReadFilters(source_types=tuple(value.strip() for value in args.source_types.split(',') if value.strip()),
                       after=args.after, before=args.before)


def sanitized_error(error):
    code = getattr(error, 'code', None)
    if code in {'scope_denied', 'stale_context', 'session_expired', 'authentication_required', 'policy_unavailable'}:
        return code
    if isinstance(error, AccessError):
        return 'access_denied'
    # Exception messages can include SQL parameters, source text or credentials.
    return 'retrieval_failed'


@dataclass(frozen=True)
class Observation:
    context: RequestContext
    hits: tuple
    read_more: tuple
    seconds: float


@dataclass(frozen=True)
class PreparedQuery:
    context: RequestContext
    query: str
    filters: ReadFilters
    depth: int
    rankings: tuple
    seconds: float


class _CandidateReplay:
    def __init__(self, actual, prepared, values):
        self.actual, self.prepared, self.values = actual, prepared, values

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def candidates(self, context, query, *, guard, top_k, filters):
        saved = self.prepared
        self.actual._current(context, guard)
        if (context is not saved.context or query != saved.query or filters != saved.filters
                or top_k > saved.depth):
            raise EvidenceValidationError('Candidate replay differs from its exact request')
        result = tuple(self.actual._checked_candidate(context, value, filters)
                       for value in self.values[:top_k])
        self.actual._current(context, guard)
        return result


class ScopedRetrievalRunner:
    def __init__(self, retriever, resolver, *, database, identity_provider,
                 principal_kind, settings, filters=ReadFilters(), read_more_chars=0, owns_database=False):
        self.settings = RetrievalSettings.model_validate(settings.model_dump(warnings=False))
        self.filters = ReadFilters.model_validate(filters.model_dump(warnings=False))
        if type(read_more_chars) is not int or not 0 <= read_more_chars <= 6000:
            raise ScopedEvaluationError('Adjacent read bound must be0..6000')
        if principal_kind not in ('benchmark', 'session') or resolver.database is not database:
            raise ScopedEvaluationError('Evaluation requires one authenticated runtime pool')
        self.database, self.retriever, self.resolver = database, retriever, resolver
        self.identity_provider, self.principal_kind = identity_provider, principal_kind
        self.read_more_chars, self._owns_database = read_more_chars, owns_database
        self.snapshot = SnapshotRef.model_validate(resolver.snapshot.model_dump(warnings=False))
        lexical = getattr(retriever, 'bm25', retriever)
        self.catalog = ValidatedCatalog.require(lexical.catalog_handle, self.snapshot, issuer=database)
        if retriever.snapshot != self.snapshot:
            raise SnapshotMismatch('Retriever differs from evaluation snapshot')
        self._binding = None
        self._source_hashes = source_hashes()
        self._initial = self.context(touch=False)

    @classmethod
    def open(cls, settings, *, snapshot_file, credentials_file, principal_kind='benchmark',
             doc_ids_file=None, filters=ReadFilters(), read_more_chars=0, database=None):
        # The command is explicitly offline. Set these before importing any Hub
        # or inference stack; missing local immutable model files fail closed.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from src.access.policy import PolicyResolver
        from src.access.repository import verify_benchmark
        from src.db.runtime import Database, DatabaseSettings
        from src.retrieval.factory import load_store_retriever
        from src.storage.postgres_store import PostgresDocumentStore

        owns = database is None
        database = database or Database(DatabaseSettings.from_env())
        try:
            snapshot = SnapshotRef.model_validate(_json(snapshot_file, limit=65536))
            documents = None if doc_ids_file is None else _json(doc_ids_file, limit=64 * 1024**2)
            if documents is not None and (not isinstance(documents, list)
                    or any(not isinstance(value, str) or not value.strip() for value in documents)
                    or len(documents) != len(set(documents))):
                raise ScopedEvaluationError('Invalid operator document restriction')
            if principal_kind == 'benchmark':
                identity = verify_benchmark(credentials_file, database)
                identity_provider = lambda touch: identity
            elif principal_kind == 'session':
                from src.auth.sessions import AuthService
                from src.auth.settings import AuthSettings
                secret = _json(credentials_file, limit=4096, secret=True)
                if (not isinstance(secret, dict) or set(secret) != {'format', 'token'}
                        or secret['format'] != 'rag-session-v1'
                        or not isinstance(secret['token'], str)
                        or not re.fullmatch(r'[A-Za-z0-9_-]{43}', secret['token'])):
                    raise ScopedEvaluationError('Invalid private session credential')
                service = AuthService(database, AuthSettings.from_env(enabled=True))
                identity_provider = lambda touch: service.authenticate(secret['token'], touch=touch).identity
            else:
                raise ScopedEvaluationError('Unknown principal kind')
            store = PostgresDocumentStore(database, snapshot, retain_catalog=True)
            resolver = PolicyResolver(database, snapshot, source_types=filters.source_types,
                doc_ids=documents, catalog=store.validated_catalog())
            return cls(load_store_retriever(settings, store), resolver, database=database,
                identity_provider=identity_provider, principal_kind=principal_kind, settings=settings,
                filters=filters, read_more_chars=read_more_chars, owns_database=owns)
        except BaseException:
            if owns:
                database.close()
            raise

    def close(self):
        if self._owns_database:
            self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def current(self, context):
        self.resolver.assert_current(context)
        context.assert_bound()
        if context.binding != self._binding:
            raise StaleContext('Evaluation binding changed; create a new run')

    def context(self, *, touch=True):
        identity = self.identity_provider(touch)
        if not isinstance(identity, ResolvedIdentity):
            raise EvidenceValidationError('A trusted evaluation identity is required')
        context = self.resolver.resolve(identity)
        if context.identity != identity or context.binding.snapshot != self.snapshot:
            raise EvidenceValidationError('Evaluation context differs from authenticated identity')
        if self._binding is None:
            self._binding = context.binding
        self.current(context)
        self._latest = context
        return context

    def assert_ready(self):
        self.current(self._latest)

    def adapter(self, method):
        if method == 'bm25':
            return getattr(self.retriever, 'bm25', self.retriever)
        if method == 'dense' and hasattr(self.retriever, 'dense'):
            return self.retriever.dense
        if method == 'hybrid' and hasattr(self.retriever, 'dense'):
            return self.retriever
        raise ScopedEvaluationError('The requested bounded backend is unavailable')

    def labels(self, question, context):
        self.current(context)
        supplied = question.get('expected_doc_ids', [])
        if not isinstance(supplied, list) or any(not isinstance(value, str) or not value.strip() for value in supplied):
            raise ScopedEvaluationError('Invalid reference labels')
        original = tuple(dict.fromkeys(supplied))
        eligible, unknown = [], 0
        for doc in original:
            position = self.catalog.positions.get(doc)
            if position is None:
                unknown += 1
            elif context.scope.permits(doc) and self.filters.matches(self.catalog.entries[position]):
                eligible.append(doc)
        status = ('unknown_reference' if unknown else 'unlabeled' if not original
                  else 'unanswerable' if not eligible else 'scored')
        self.current(context)
        return {'relevant_doc_ids': eligible if status == 'scored' else [],
                'original_reference_count': len(original), 'eligible_reference_count': len(eligible),
                'unknown_reference_count': unknown, 'scoring_status': status,
                'relevance_basis': 'authorized_reference_intersection'}

    def _observe(self, adapter, context, hits, began):
        def checked(values):
            result = []
            for value in values:
                if not isinstance(value, PassageEvidence):
                    raise EvidenceValidationError('Evaluation requires canonical passage evidence')
                value = PassageEvidence.model_validate(value.model_dump(warnings=False))
                chunk = value.chunk
                position = self.catalog.positions.get(chunk.doc_id)
                if position is None:
                    raise EvidenceValidationError('Evaluation passage is outside the catalogue')
                entry = self.catalog.entries[position]
                if (value.revision_id != entry.revision_id or chunk.source_type != entry.source_type
                        or chunk.timestamp != entry.timestamp or chunk.metadata_origin != entry.metadata_origin
                        or chunk.end_char > entry.text_length or not self.filters.matches(entry)
                        or value.citation_id != f'{chunk.doc_id}:{chunk.start_char}:{chunk.end_char}'
                        or chunk.chunk_id != value.citation_id):
                    raise EvidenceValidationError('Evaluation passage differs from canonical metadata or filters')
                result.append(value)
            result = tuple(result)
            validate_grounded_evidence(context, result, self.resolver)
            return result

        hits = checked(hits)
        expansion = ()
        if self.read_more_chars:
            anchors, seen = [], set()
            for hit in hits:
                if hit.chunk.doc_id not in seen:
                    anchors.append(hit)
                    seen.add(hit.chunk.doc_id)
                if len(anchors) == 2:
                    break
            validate_read_more_bounds(len(anchors), self.read_more_chars)
            expansion = checked(adapter.read_more(context, tuple(anchors), guard=self.resolver,
                max_chars=self.read_more_chars, filters=self.filters))
            parents = {value.chunk.doc_id: value for value in anchors}
            seen = set()
            for value in expansion:
                parent = parents.get(value.chunk.doc_id)
                if (parent is None or value.chunk.doc_id in seen or len(value.chunk.text) > self.read_more_chars
                        or value.score != parent.score or value.retrieval_method != parent.retrieval_method
                        or (value.chunk.start_char >= parent.chunk.start_char
                            and value.chunk.end_char <= parent.chunk.end_char)):
                    raise EvidenceValidationError('Adjacent read left its matched document or made no progress')
                seen.add(value.chunk.doc_id)
        self.current(context)
        return Observation(context, hits, expansion, time.perf_counter() - began)

    def search(self, context, query, *, method, top_k):
        began = time.perf_counter()
        self.current(context)
        adapter = self.adapter(method)
        hits = adapter.search(context, query, guard=self.resolver, top_k=top_k, filters=self.filters)
        return self._observe(adapter, context, hits, began)

    def prepare(self, context, query, *, depth):
        began = time.perf_counter()
        self.current(context)
        rankings = tuple(adapter.candidates(context, query, guard=self.resolver,
            top_k=depth, filters=self.filters) for adapter in (self.adapter('bm25'), self.adapter('dense')))
        self.current(context)
        return PreparedQuery(context, query, self.filters, depth, rankings, time.perf_counter() - began)

    def hybrid(self, prepared, *, config, top_k, reranker=None):
        from src.retrieval.store import StoreHybridRetriever
        began = time.perf_counter()
        self.current(prepared.context)
        adapters = tuple(_CandidateReplay(self.adapter(method), prepared, ranking)
                         for method, ranking in zip(('bm25', 'dense'), prepared.rankings))
        hybrid = StoreHybridRetriever(*adapters, config=config, reranker=reranker)
        hits = hybrid.search(prepared.context, prepared.query, guard=self.resolver,
                             top_k=top_k, filters=prepared.filters)
        result = self._observe(hybrid, prepared.context, hits, began)
        return Observation(result.context, result.hits, result.read_more, result.seconds + prepared.seconds)

    def baseline(self, prepared, *, method, top_k):
        began = time.perf_counter()
        self.current(prepared.context)
        adapter = self.adapter(method)
        ranking = prepared.rankings[0 if method == 'bm25' else 1]
        hits = tuple(adapter._materialize(prepared.context, candidate, prepared.query,
            self.resolver, prepared.filters) for candidate in ranking[:top_k])
        result = self._observe(adapter, prepared.context, hits, began)
        return Observation(result.context, result.hits, result.read_more, result.seconds + prepared.seconds)

    def manifest(self):
        if source_hashes() != self._source_hashes:
            raise ScopedEvaluationError('Evaluation source changed; create a new run')
        indexes = {'bm25': self.adapter('bm25').index.manifest}
        if hasattr(self.retriever, 'dense'):
            indexes['dense'] = self.adapter('dense').index.manifest
        context = self._latest
        self.current(context)
        return {'backend': 'postgres-store-bounded-v1', 'snapshot': self.snapshot.model_dump(mode='json'),
            'principal': {'kind': self.principal_kind, 'user_id': context.identity.user_id,
                          'session_id': context.identity.session_id, 'session_version': context.identity.session_version},
            'scope': self._binding.model_dump(mode='json'),
            'allowed_documents': getattr(context.scope, 'allowed_count', None),
            'bitmap_sha256': getattr(context.scope, 'bitmap_sha256', None),
            'filters': self.filters.model_dump(mode='json'), 'read_more_chars': self.read_more_chars,
            'indexes': indexes, 'config': self.settings.model_dump(mode='json'),
            'source_sha256': dict(self._source_hashes),
            'offline_models': True, 'file_cache_policy': self.settings.dense_file_cache_policy,
            'block_rows': self.settings.dense_block_rows,
            'cache_condition': 'One process, OS cache not cleared; no cold-disk claim. Candidate metadata shared within one question only.',
            'identity_policy': 'Current session/policy checked for every query and sensitive operation; one frozen run binding'}


def result_row(question, observation=None, *, labels=None, error=None):
    labels = labels or {'relevant_doc_ids': [], 'original_reference_count': len(set(question.get('expected_doc_ids', []))),
        'eligible_reference_count': None, 'unknown_reference_count': None,
        'scoring_status': 'access_unavailable', 'relevance_basis': 'authorized_reference_intersection'}
    hits = observation.hits if observation else ()
    return {'question_id': question['question_id'], 'question_type': question.get('question_type', 'unknown'),
        'source_types': question.get('source_types', []), **labels,
        'retrieved_doc_ids': [item.chunk.doc_id for item in hits],
        'scores': [round(item.score, 4) for item in hits], 'error': error,
        'read_more': [{'doc_id': item.chunk.doc_id, 'citation_id': item.citation_id,
            'revision_id': item.revision_id, 'start_char': item.chunk.start_char, 'end_char': item.chunk.end_char,
            'text_sha256': hashlib.sha256(item.chunk.text.encode('utf-8')).hexdigest()}
            for item in (observation.read_more if observation else ())]}


def label_summary(rows):
    statuses = {name: sum(row['scoring_status'] == name for row in rows)
                for name in ('scored', 'unanswerable', 'unlabeled', 'unknown_reference', 'access_unavailable')}
    return {'basis': 'authorized_reference_intersection', 'question_status_counts': statuses,
        'original_references': sum(row['original_reference_count'] for row in rows),
        'eligible_references': sum(row['eligible_reference_count'] or 0 for row in rows),
        'unknown_references': sum(row['unknown_reference_count'] or 0 for row in rows),
        'matches_original_reference_denominators': all(row['unknown_reference_count'] == 0
            and row['eligible_reference_count'] == row['original_reference_count'] for row in rows),
        'policy': 'No eligible references means unanswerable/not scored; any unknown reference excludes the entire question from quality metrics. Queries and errors remain counted.'}


class RunOutput:
    """Exclusive streaming output; incomplete attempts remain visibly incomplete."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.predictions = (self.path / 'predictions.jsonl').open('x', encoding='utf-8')
        self.upstream = (self.path / 'upstream_answers.jsonl').open('x', encoding='utf-8')
        self.rows, self.latencies = [], []

    def append(self, row, seconds):
        self.predictions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.upstream.write(json.dumps({'question_id': row['question_id'],
            'document_ids': row['retrieved_doc_ids'][:10]}, ensure_ascii=False) + '\n')
        self.predictions.flush()
        self.upstream.flush()
        self.rows.append(row)
        self.latencies.append(seconds)

    def finish(self, summary, manifest):
        for name, value in (('summary.json', summary), ('manifest.json', {'complete': True, **manifest})):
            with (self.path / name).open('x', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')

    def close(self):
        self.predictions.close()
        self.upstream.close()

"""Bounded, content-free workflow observations, separate from product traces.

Only explicitly named scalar/count fields enter the recorder or JSON log queue.
Emitters never write to stdout or wait for a logging consumer. Diagnostics access
control belongs to the authenticated API; this module exposes no endpoint.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
from queue import Empty, Full, Queue
import re
import sys
import threading
import time
from uuid import UUID, uuid4


STAGES = frozenset({
    'request', 'http', 'worker', 'auth', 'auth.authenticate', 'scope', 'policy',
    'policy.resolve', 'policy.grants', 'policy.bitmap', 'policy.register',
    'agent', 'route', 'text_search', 'hybrid', 'bm25', 'dense', 'fusion',
    'materialize', 'rerank', 'selection', 'context_select', 'critic', 'generate',
    'read_more', 'tool', 'database', 'validation', 'other',
})
EVENTS = frozenset({'start', 'end', 'error', 'skipped', 'decision', 'retry', 'bound', 'reduced'})
STATUSES = frozenset({'in_progress', 'ok', 'error', 'denied', 'timeout', 'cancelled', 'skipped'})
REASONS = frozenset({
    'sufficient', 'insufficient', 'missing_evidence', 'missing_facts', 'read_more',
    'reformulation', 'suggested_queries', 'round_limit', 'no_new_evidence', 'budget',
    'doc_limit', 'char_limit', 'token_limit', 'context_reduced',
    'below_relevance_threshold', 'none', 'document_limit', 'char_budget',
    'retained_document_limit', 'retained_char_budget', 'frozen_context_mismatch',
    'disabled', 'unavailable', 'not_applicable', 'clarification', 'cropped',
    'single_pass', 'error', 'other',
})
ERROR_CODES = frozenset({
    'agent_error', 'truncated_output', 'invalid_json', 'invalid_schema',
    'provider_error', 'invalid_citation', 'missing_claims', 'invalid_input',
    'scope_denied', 'policy_unavailable', 'policy_changed', 'session_expired',
    'session_revoked', 'csrf_failed', 'unauthorized', 'authentication_required',
    'not_authenticated', 'invalid_credentials', 'runtime_error', 'runtime_failed',
    'runtime_unavailable', 'not_ready', 'busy', 'concurrency_limit',
    'budget_exceeded', 'invalid_request', 'validation_error', 'internal_error',
    'forbidden', 'not_found', 'timeout', 'origin_denied', 'stale_context',
    'spending_limit', 'query_failed', 'other',
})
COUNT_FIELDS = frozenset({
    'candidate_count', 'bm25_candidate_count', 'dense_candidate_count', 'fused_count',
    'returned_count', 'unique_document_count', 'selected_document_count',
    'selected_passage_count', 'excluded_document_count', 'excluded_passage_count',
    'retained_document_count', 'retained_passage_count', 'document_limit',
    'pool_documents', 'pool_passages', 'pool_native_items', 'selected_documents',
    'selected_passages', 'native_items', 'selected_native_items', 'selected_chars',
    'char_budget', 'excluded_documents', 'excluded_passages', 'cropped_passages',
    'retained_documents', 'retained_passages', 'input_tokens', 'output_tokens',
    'api_calls', 'grant_page_count', 'grant_row_count',
})
_COMMON_ERROR_NAMES = frozenset({
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'TimeoutError',
    'CancelledError', 'AccessDenied', 'SessionExpired', 'SessionRevoked',
    'PolicyUnavailable', 'EvidenceValidationError', 'RuntimeConfigurationError',
    'RunBudgetExceeded', 'AgentError', 'ProviderError', 'ValidationError',
    'JSONDecodeError', 'OperationalError', 'DatabaseRoleError', 'StaleContext',
    'SnapshotMismatch', 'HTTPException', 'PermissionError', 'ContextBudgetError',
    'FrozenContextError', 'OtherError',
})
_PROGRESS_REQUEST_FIELDS = ('status', 'http_status_code', 'worker_status', 'started_at', 'completed_at',
                            'duration_ms', 'rounds', 'error_code', 'error_type', 'usage')
_PROGRESS_STAGE_FIELDS = ('sequence', 'stage', 'event', 'status', 'round_number', 'duration_ms',
                          'reason_code', 'stop_reason', 'sufficient', 'error_code', 'error_type',
                          'returned_count', 'unique_document_count', 'selected_document_count')
_CURRENT: ContextVar[RequestObservation | None] = ContextVar('rag_request_observation', default=None)


def _uuid(value):
    try:
        return str(UUID(value)) if type(value) is str else None
    except (ValueError, TypeError, AttributeError):
        return None


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


def _error_name(value):
    if isinstance(value, BaseException):
        value = type(value)
    if isinstance(value, type) and issubclass(value, BaseException):
        name = value.__name__
        return name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', name) else 'OtherError'
    # Strings cannot carry an exception message, a token or a query into logs.
    return value if type(value) is str and value in _COMMON_ERROR_NAMES else None


def _safe_fields(fields):
    result = {}
    for key, value in fields.items():
        if key in COUNT_FIELDS and type(value) is int and 0 <= value <= 10**12:
            result[key] = value
        elif key in {'duration_ms', 'estimated_cost_usd'} and _number(value):
            result[key] = float(value)
        elif key == 'round_number' and type(value) is int and 1 <= value <= 1000:
            result[key] = value
        elif key == 'http_status_code' and type(value) is int and 100 <= value <= 599:
            result[key] = value
        elif key in {'sufficient', 'reranker_enabled', 'duration_is_cumulative'} and type(value) is bool:
            result[key] = value
        elif key in {'retry_reason', 'exclusion_reason', 'reason', 'reason_code', 'stop_reason'} and type(value) is str and value in REASONS:
            result[key] = value
        elif key == 'error_code' and type(value) is str and value in ERROR_CODES:
            result[key] = value
        elif key == 'provider_stop_reason' and type(value) is str and value in {
                'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'pause_turn',
                'refusal', 'model_context_window_exceeded', 'other'}:
            result[key] = value
        elif key == 'database_sqlstate' and type(value) is str and value in {
                '25P03', '57014', '55P03', '08000', '08003', '08006', '57P01', '57P02', '57P03'}:
            result[key] = value
        elif key == 'database_phase' and type(value) is str and value in {'execute', 'materialize', 'revalidate'}:
            result[key] = value
        elif key in {'error', 'error_type'}:
            name = _error_name(value)
            if name:
                result['error_type'] = name
        elif key == 'exclusion_reasons' and type(value) is dict:
            counts = {name: count for name, count in value.items()
                      if type(name) is str and name in REASONS
                      and type(count) is int and 0 <= count <= 10**12}
            if counts:
                result[key] = counts
    return result


def _usage(value):
    output = {key: None for key in ('input_tokens', 'output_tokens', 'api_calls', 'estimated_cost_usd')}
    for key in output:
        try:
            item = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
            if key == 'estimated_cost_usd':
                output[key] = float(item) if _number(item) else None
            elif type(item) is int and item >= 0:
                output[key] = item
        except Exception:
            pass
    return output


class _QuietHandler(logging.StreamHandler):
    def handleError(self, record):
        pass


class Recorder:
    """One process-local bounded history and nonblocking JSON log queue."""
    def __init__(self, *, max_requests=500, max_events_per_request=128,
                 retention_seconds=3600, queue_size=1024, log=True, sink=None,
                 clock=time.monotonic):
        self.max_requests = max(1, min(5000, int(max_requests)))
        self.max_events_per_request = max(1, min(1000, int(max_events_per_request)))
        self.retention_seconds = max(1, min(86400, float(retention_seconds)))
        self._clock = clock
        self._lock = threading.RLock()
        self._requests = OrderedDict()
        # token -> (request_id, owner); bounded because every entry needs a retained request.
        self._progress = {}
        self._queue = Queue(maxsize=max(1, min(10000, int(queue_size))))
        self._closed = threading.Event()
        self._dropped_logs = self._logging_failures = self._evicted_requests = 0
        self._dropped_request_events = 0
        self._logging = bool(log)
        self._sink = sink
        if self._logging:
            if sink is None:
                logger = logging.Logger('rag.observability', level=logging.INFO)
                handler = _QuietHandler(sys.stdout)
                handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(handler)
                logger.propagate = False
                self._sink = logger.info
            threading.Thread(target=self._drain, name='rag-observability', daemon=True).start()

    def _now(self):
        try:
            value = self._clock()
            if _number(value):
                return float(value)
        except Exception:
            pass
        return time.monotonic()

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat()

    def _prune(self, now):
        for key, value in tuple(self._requests.items()):
            if now - value._started > self.retention_seconds:
                del self._requests[key]
                self._evicted_requests += 1
        while len(self._requests) > self.max_requests:
            self._requests.popitem(last=False)
            self._evicted_requests += 1
        for token, (request_id, _owner) in tuple(self._progress.items()):
            if request_id not in self._requests:
                del self._progress[token]

    def start_request(self, request_id=None):
        observation = RequestObservation(self, _uuid(request_id) or str(uuid4()))
        with self._lock:
            if observation.request_id in self._requests:
                observation.request_id = str(uuid4())
            self._requests[observation.request_id] = observation
            self._prune(self._now())
        observation.emit('http', 'start', status='in_progress')
        return observation

    def _enqueue(self, event):
        if not self._logging or self._closed.is_set():
            return
        try:
            self._queue.put_nowait(json.dumps(event, separators=(',', ':'), allow_nan=False))
        except (Full, ValueError, TypeError):
            with self._lock:
                self._dropped_logs += 1

    def _drain(self):
        while not self._closed.is_set():
            try:
                item = self._queue.get(timeout=.1)
            except Empty:
                continue
            try:
                self._sink(item)
            except Exception:
                with self._lock:
                    self._logging_failures += 1
            finally:
                self._queue.task_done()

    def list(self, *, limit=50, offset=0):
        try:
            limit = min(100, max(1, limit)) if type(limit) is int else 50
            offset = max(0, offset) if type(offset) is int else 0
            with self._lock:
                self._prune(self._now())
                rows = list(reversed(self._requests.values()))
                selected = [item._summary() for item in rows[offset:offset + limit]]
                after = offset + len(selected)
                return {'requests': selected, 'next_offset': after if after < len(rows) else None,
                        'total': len(rows)}
        except Exception:
            return {'requests': [], 'next_offset': None, 'total': 0}

    def get(self, request_id):
        try:
            if _uuid(request_id) != request_id:
                return None
            with self._lock:
                self._prune(self._now())
                observation = self._requests.get(request_id)
                if observation is None:
                    return None
                return {'request': observation._summary(),
                        'stages': [dict(item, **({'exclusion_reasons': dict(item['exclusion_reasons'])}
                                   if 'exclusion_reasons' in item else {})) for item in observation._events]}
        except Exception:
            return None

    def bind_progress(self, observation, token, owner):
        try:
            key = _uuid(token)
            if key is None or not isinstance(observation, RequestObservation):
                return
            with self._lock:
                self._prune(self._now())
                existing = self._progress.get(key)
                if existing is not None and existing[0] != observation.request_id:
                    return
                if observation.request_id not in self._requests:
                    return
                # One token per request keeps the registry no larger than the request history.
                if observation._progress_token not in (None, key):
                    return
                observation._progress_token = key
                self._progress[key] = (observation.request_id, owner)
        except Exception:
            pass

    def progress(self, token, owner):
        try:
            key = _uuid(token)
            if key is None:
                return None
            with self._lock:
                self._prune(self._now())
                bound = self._progress.get(key)
                if bound is None or bound[1] != owner:
                    return None
                observation = self._requests.get(bound[0])
                if observation is None:
                    return None
                summary = observation._summary()
                return {'request': {name: summary[name] for name in _PROGRESS_REQUEST_FIELDS},
                        'stages': [{name: item[name] for name in _PROGRESS_STAGE_FIELDS if name in item}
                                   for item in observation._events]}
        except Exception:
            return None

    def stats(self):
        with self._lock:
            self._prune(self._now())
            return {'retained_requests': len(self._requests), 'max_requests': self.max_requests,
                    'max_events_per_request': self.max_events_per_request,
                    'retention_seconds': self.retention_seconds, 'log_queue_size': self._queue.qsize(),
                    'dropped_log_events': self._dropped_logs, 'logging_failures': self._logging_failures,
                    'dropped_request_events': self._dropped_request_events,
                    'evicted_requests': self._evicted_requests}

    def close(self):
        # Never join a thread whose stdout consumer might be blocked.
        self._closed.set()


class RequestObservation:
    def __init__(self, recorder, request_id):
        self.recorder, self.request_id = recorder, request_id
        self.run_id = None
        self._progress_token = None
        self._started = recorder._now()
        self._started_at = recorder._timestamp()
        self._completed_at = None
        self._http_end = self._worker_start = self._worker_end = None
        self._http_status = None
        self._status = 'in_progress'
        self._http_outcome = None
        self._worker_status = 'not_started'
        self._events = []
        self._sequence = self._dropped = self._rounds = 0
        self._error_type = self._error_code = None
        self._counts = {}
        self._usage = _usage(None)

    def _summary(self):
        ended = max((value for value in (self._http_end, self._worker_end) if value is not None), default=None)
        return {'request_id': self.request_id, 'run_id': self.run_id,
                'started_at': self._started_at, 'completed_at': self._completed_at,
                'status': self._status, 'http_status_code': self._http_status,
                'http_finished': self._http_end is not None, 'worker_status': self._worker_status,
                'duration_ms': max(0, ((ended if self._completed_at else self.recorder._now()) - self._started) * 1000),
                'http_duration_ms': None if self._http_end is None else max(0, (self._http_end - self._started) * 1000),
                'worker_duration_ms': None if self._worker_end is None else max(0, (self._worker_end - self._worker_start) * 1000),
                'rounds': self._rounds, 'event_count': len(self._events), 'dropped_event_count': self._dropped,
                'error_type': self._error_type, 'error_code': self._error_code,
                'counts': dict(self._counts), 'usage': dict(self._usage)}

    def emit(self, stage, event='end', status=None, **fields):
        try:
            stage = stage if type(stage) is str and stage in STAGES else 'other'
            event = event if type(event) is str and event in EVENTS else 'end'
            status = status if type(status) is str and status in STATUSES else None
            safe = _safe_fields(fields)
            with self.recorder._lock:
                self._sequence += 1
                self._rounds = max(self._rounds, safe.get('round_number', 0))
                self._counts.update({key: value for key, value in safe.items() if key in COUNT_FIELDS})
                self._error_type = safe.get('error_type', self._error_type)
                self._error_code = safe.get('error_code', self._error_code)
                item = {'request_id': self.request_id, 'run_id': self.run_id, 'sequence': self._sequence,
                        'stage': stage, 'event': event, 'timestamp': self.recorder._timestamp(),
                        'status': status, **safe}
                if len(self._events) < self.recorder.max_events_per_request:
                    self._events.append(item)
                else:
                    self._dropped += 1
                    self.recorder._dropped_request_events += 1
                    return
            self.recorder._enqueue(item)
        except Exception:
            pass

    def bind_run(self, run_id):
        try:
            value = _uuid(run_id)
            if value is not None:
                with self.recorder._lock:
                    if self.run_id is None or self.run_id == value:
                        self.run_id = value
                self.emit('request', 'bound', status='in_progress')
        except Exception:
            pass

    def worker_start(self):
        try:
            with self.recorder._lock:
                if self._worker_end is not None:
                    return
                if self._worker_start is None:
                    self._worker_start = self.recorder._now()
                    self._worker_status = 'running'
                    self._completed_at = None
            self.emit('worker', 'start', status='in_progress')
        except Exception:
            pass

    def _complete(self):
        outcomes = [self._http_outcome, self._worker_status]
        self._status = next((value for value in ('error', 'timeout', 'denied', 'cancelled')
                            if value in outcomes), self._http_outcome or 'in_progress')
        if self._http_end is not None and self._worker_status != 'running':
            self._completed_at = self.recorder._timestamp()

    def http_finish(self, status_code, status=None, error=None, **fields):
        try:
            code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
            status = status if type(status) is str and status in STATUSES else ('error' if code is None or code >= 500
                     else 'denied' if code in (401, 403) else 'error' if code >= 400 else 'ok')
            with self.recorder._lock:
                if self._http_end is not None:
                    return
                self._http_end = self.recorder._now()
                self._http_status = code
                self._http_outcome = status
                self._complete()
            self.emit('http', 'end', status=status, error=error, **{**fields,
                'http_status_code': code, 'duration_ms': max(0, (self._http_end - self._started) * 1000)})
        except Exception:
            pass

    def worker_finish(self, status='ok', usage=None, error=None, **fields):
        try:
            status = status if type(status) is str and status in STATUSES else 'error'
            with self.recorder._lock:
                if self._worker_end is not None:
                    return
                now = self.recorder._now()
                self._worker_start = self._worker_start if self._worker_start is not None else now
                self._worker_end = now
                self._worker_status = status
                self._usage = _usage(usage)
                self._complete()
            self.emit('worker', 'end', status=status, error=error, **{**fields, **self._usage,
                'duration_ms': max(0, (self._worker_end - self._worker_start) * 1000)})
        except Exception:
            pass


@contextmanager
def observation_context(observation):
    token = _CURRENT.set(observation if isinstance(observation, RequestObservation) else None)
    try:
        yield observation
    finally:
        _CURRENT.reset(token)


def emit(stage, event='end', status=None, **fields):
    try:
        observation = _CURRENT.get()
        if observation is not None:
            observation.emit(stage, event, status, **fields)
    except Exception:
        pass


def bind_run(run_id):
    try:
        observation = _CURRENT.get()
        if observation is not None:
            observation.bind_run(run_id)
    except Exception:
        pass


def bind_progress(token, owner):
    try:
        observation = _CURRENT.get()
        if observation is not None:
            observation.recorder.bind_progress(observation, token, owner)
    except Exception:
        pass


def current_observation():
    try:
        return _CURRENT.get()
    except Exception:
        return None


class _Span:
    def __init__(self, fields):
        self.fields = fields

    def set(self, **fields):
        try:
            self.fields.update(_safe_fields(fields))
        except Exception:
            pass


def _exception_fields(error):
    fields = {}
    for attribute, key in (('code', 'error_code'), ('provider_stop_reason', 'provider_stop_reason'),
                           ('database_sqlstate', 'database_sqlstate'), ('database_phase', 'database_phase')):
        try:
            fields[key] = getattr(error, attribute, None)
        except Exception:
            pass
    return _safe_fields(fields)


@contextmanager
def span(stage, round_number=None, **fields):
    started = time.monotonic()
    values = _safe_fields({**fields, 'round_number': round_number})
    detail = _Span(values)
    emit(stage, 'start', status='in_progress', **values)
    try:
        yield detail
    except BaseException as error:
        emit(stage, 'error', status='cancelled' if type(error).__name__ == 'CancelledError' else 'error',
             **{**detail.fields, **_exception_fields(error),
                'duration_ms': max(0, (time.monotonic() - started) * 1000), 'error': error})
        raise
    else:
        emit(stage, 'end', status='ok', **{**detail.fields,
             'duration_ms': max(0, (time.monotonic() - started) * 1000)})

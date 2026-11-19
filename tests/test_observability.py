"""Content privacy, bounded resources and request/thread lifecycle observations."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import threading
import time
from uuid import uuid4

import pytest

from src.observability import (Recorder, bind_run, current_observation, emit,
                               observation_context, span)


def test_allowlisted_metadata_never_retains_query_documents_credentials_or_exception_prose():
    recorder = Recorder(log=False)
    secret = 'PRIVATE query / document / password / token / key'
    observation = recorder.start_request(secret)
    with observation_context(observation):
        emit(secret, secret, status=secret, query=secret, doc_id=secret, api_key=secret,
             error=ValueError(secret), candidate_count=50, selected_documents=True,
             reason=[secret], error_type=secret, evidence_sha256=secret,
             exclusion_reasons={'document_limit': 3, secret: 8})
    detail = recorder.get(observation.request_id)
    encoded = json.dumps(detail)
    assert secret not in encoded and 'query' not in detail['stages'][-1]
    assert detail['stages'][-1]['error_type'] == 'ValueError'
    assert detail['stages'][-1]['candidate_count'] == 50
    assert 'selected_documents' not in detail['stages'][-1]
    assert detail['stages'][-1]['exclusion_reasons'] == {'document_limit': 3}
    assert detail['request']['request_id'] != secret


def test_aggregate_policy_counts_and_timing_are_typed_and_content_free():
    recorder = Recorder(log=False)
    observation = recorder.start_request()
    with observation_context(observation):
        emit('policy.grants', grant_page_count=63, grant_row_count=511958,
             duration_ms=1234.5, duration_is_cumulative=True, user_id='PRIVATE-USER',
             grant_documents=['PRIVATE-DOCUMENT'])
        emit('policy.bitmap', grant_page_count=True, grant_row_count='511958',
             duration_is_cumulative='true')
    detail = recorder.get(observation.request_id)
    query, malformed = detail['stages'][-2:]
    assert query['grant_page_count'] == 63 and query['grant_row_count'] == 511958
    assert query['duration_is_cumulative'] is True and query['duration_ms'] == 1234.5
    assert detail['request']['counts']['grant_row_count'] == 511958
    assert all(field not in malformed for field in ('grant_page_count', 'grant_row_count', 'duration_is_cumulative'))
    assert 'PRIVATE-' not in json.dumps(detail)


def test_span_failure_preserves_original_exception_and_observer_failure_cannot_break_work(monkeypatch):
    recorder = Recorder(log=False)
    observation = recorder.start_request()
    error = RuntimeError('PRIVATE exception details')
    with observation_context(observation):
        with pytest.raises(RuntimeError) as raised:
            with span('dense', round_number=2):
                raise error
        assert raised.value is error
        monkeypatch.setattr(observation, 'emit', lambda *_a, **_k: (_ for _ in ()).throw(ValueError('broken observer')))
        with span('bm25') as measured:
            measured.set(candidate_count=3)
            answer = 42
    assert answer == 42
    rows = recorder.get(observation.request_id)['stages']
    assert rows[-1]['event'] == 'error' and rows[-1]['error_type'] == 'RuntimeError'
    assert rows[-1]['round_number'] == 2 and rows[-1]['duration_ms'] >= 0


def test_http_failure_before_worker_and_late_worker_completion_are_distinct():
    clock = [10.0]
    recorder = Recorder(log=False, clock=lambda: clock[0])
    denied = recorder.start_request()
    clock[0] += 1
    denied.http_finish(503, error=RuntimeError('PRIVATE DSN'), error_code='policy_unavailable')
    row = recorder.get(denied.request_id)['request']
    assert row['completed_at'] and row['worker_status'] == 'not_started'
    assert row['http_duration_ms'] == 1000 and row['error_code'] == 'policy_unavailable'
    observation = recorder.start_request()
    observation.worker_start()
    clock[0] += 2
    observation.http_finish(499, status='cancelled')
    row = recorder.get(observation.request_id)['request']
    assert row['completed_at'] is None and row['worker_status'] == 'running'
    clock[0] += 3
    observation.worker_finish('error', usage={'input_tokens': 20, 'output_tokens': 0,
        'api_calls': 1, 'estimated_cost_usd': None}, error=TimeoutError('PRIVATE'))
    row = recorder.get(observation.request_id)['request']
    assert row['completed_at'] and row['status'] == 'error'
    assert row['http_status_code'] == 499 and row['http_duration_ms'] == 2000
    assert row['worker_duration_ms'] == 5000 and row['duration_ms'] == 5000
    assert row['usage'] == {'input_tokens': 20, 'output_tokens': 0, 'api_calls': 1, 'estimated_cost_usd': None}
    stages = recorder.get(observation.request_id)['stages']
    http_end = next(item for item in stages if item['stage'] == 'http' and item['event'] == 'end')
    worker_end = stages[-1]
    assert http_end['duration_ms'] == 2000
    assert worker_end['duration_ms'] == 5000 and worker_end['input_tokens'] == 20
    assert worker_end['api_calls'] == 1 and 'estimated_cost_usd' not in worker_end


def test_worker_error_cannot_be_hidden_by_successful_http_transport():
    recorder = Recorder(log=False)
    observation = recorder.start_request()
    observation.worker_start()
    observation.worker_finish('error', error=ValueError('private'))
    observation.http_finish(200)
    assert recorder.get(observation.request_id)['request']['status'] == 'error'


def test_late_worker_after_http_disconnect_reopens_pending_lifecycle_and_terminal_calls_are_idempotent():
    clock = [1.0]
    recorder = Recorder(log=False, clock=lambda: clock[0])
    observation = recorder.start_request()
    observation.http_finish(499, status='cancelled')
    clock[0] += 2
    observation.worker_start()
    assert recorder.get(observation.request_id)['request']['completed_at'] is None
    clock[0] += 3
    observation.worker_finish(usage={'api_calls': 2, 'estimated_cost_usd': .01})
    first = recorder.get(observation.request_id)['request']
    observation.worker_finish('error')
    observation.worker_start()
    observation.http_finish(200)
    assert recorder.get(observation.request_id)['request'] == first
    assert first['worker_duration_ms'] == 3000 and first['duration_ms'] == 5000
    assert first['http_status_code'] == 499 and first['usage']['estimated_cost_usd'] == .01


def test_bounded_history_event_caps_expiry_pagination_and_copied_dtos():
    clock = [1.0]
    recorder = Recorder(log=False, max_requests=2, max_events_per_request=3,
                        retention_seconds=10, clock=lambda: clock[0])
    first = recorder.start_request()
    second = recorder.start_request()
    third = recorder.start_request()
    assert recorder.get(first.request_id) is None
    for _ in range(20):
        third.emit('selection', selected_documents=8)
    third.worker_finish(usage={'input_tokens': 10, 'output_tokens': 2, 'api_calls': 2, 'estimated_cost_usd': .01})
    third.http_finish(200)
    page = recorder.list(limit=1)
    assert page['requests'][0]['request_id'] == third.request_id and page['next_offset'] == 1
    assert recorder.list(limit=1, offset=1)['requests'][0]['request_id'] == second.request_id
    detail = recorder.get(third.request_id)
    assert len(detail['stages']) == 3 and detail['request']['dropped_event_count'] > 0
    assert detail['request']['completed_at'] and detail['request']['counts']['selected_documents'] == 8
    detail['request']['counts']['selected_documents'] = 999
    detail['request']['usage']['estimated_cost_usd'] = 999
    assert recorder.get(third.request_id)['request']['counts']['selected_documents'] == 8
    assert recorder.get(third.request_id)['request']['usage']['estimated_cost_usd'] == .01
    clock[0] += 11
    assert recorder.list()['total'] == 0 and recorder.stats()['evicted_requests'] == 3


def test_parallel_requests_and_explicit_copy_context_keep_one_request_identity():
    recorder = Recorder(log=False)
    observations = [recorder.start_request() for _ in range(2)]
    barrier = threading.Barrier(2)
    def work(observation, count):
        with observation_context(observation):
            barrier.wait()
            with span('bm25', round_number=count) as measured:
                measured.set(candidate_count=count)
            bind_run(str(uuid4()))
            assert current_observation() is observation
        assert current_observation() is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda pair: work(*pair), zip(observations, (1, 2))))
        with observation_context(observations[0]):
            copied = copy_context()
            pool.submit(copied.run, emit, 'dense', candidate_count=50).result()
    assert current_observation() is None
    for observation in observations:
        rows = recorder.get(observation.request_id)
        assert all(row['request_id'] == observation.request_id for row in rows['stages'])
        assert rows['request']['run_id'] is not None
    assert recorder.get(observations[1].request_id)['request']['counts']['candidate_count'] == 2


def test_blocked_stdout_and_full_log_queue_never_block_emit_or_shutdown():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def blocked(_message):
        entered.set()
        release.wait(3)
    recorder = Recorder(queue_size=1, sink=blocked)
    try:
        observation = recorder.start_request()
        assert entered.wait(1)
        def produce():
            for _ in range(100): observation.emit('bm25', candidate_count=50)
            recorder.close()
            finished.set()
        threading.Thread(target=produce, daemon=True).start()
        assert finished.wait(1)
        assert recorder.stats()['dropped_log_events'] > 0
    finally:
        release.set()
        recorder.close()


def test_json_queue_sink_failure_is_nonthrowing_and_contains_only_sanitized_values():
    received, seen = threading.Event(), []
    def failing(message):
        seen.append(json.loads(message))
        received.set()
        raise OSError('PRIVATE stdout failure')
    recorder = Recorder(sink=failing)
    try:
        observation = recorder.start_request()
        observation.emit('critic', 'decision', sufficient=False, reason_code='missing_evidence',
                         query='PRIVATE', error=ValueError('PRIVATE'), estimated_cost_usd=float('nan'))
        assert received.wait(1)
        end = time.monotonic() + 1
        while recorder.stats()['logging_failures'] < 1 and time.monotonic() < end:
            time.sleep(.001)
        assert recorder.stats()['logging_failures'] >= 1
        assert 'PRIVATE' not in json.dumps(seen)
        assert recorder.get(observation.request_id)['request']['error_type'] == 'ValueError'
    finally:
        recorder.close()


def test_store_hybrid_stages_preserve_actual_results_candidates_scope_and_skipped_reranker():
    from tests.test_store_retrieval import fixture_store, hybrid, context, Guard
    from src.retrieval.hybrid import HybridConfig
    store = fixture_store()
    retriever, index, _ = hybrid(store, config=HybridConfig(candidate_k=50, rerank_k=50))
    request = context(store, {'alpha', 'beta'})
    expected = retriever.search(request, 'needle', guard=Guard(), top_k=2)
    recorder = Recorder(log=False)
    observation = recorder.start_request()
    with observation_context(observation):
        actual = retriever.search(request, 'needle', guard=Guard(), top_k=2)
    assert actual == expected and all(call[1] == 50 for call in index.calls)
    assert 'secret' not in store.body_reads
    stages = recorder.get(observation.request_id)['stages']
    assert {'hybrid', 'bm25', 'dense', 'fusion', 'materialize', 'rerank'} <= {row['stage'] for row in stages}
    assert any(row['stage'] == 'rerank' and row['event'] == 'skipped' for row in stages)
    complete = next(row for row in stages if row['stage'] == 'hybrid' and row['event'] == 'end')
    assert complete['candidate_count'] == 4 and complete['returned_count'] == complete['fused_count'] == 2
    assert all(value not in json.dumps(stages) for value in ('needle', 'alpha', 'beta', 'secret'))

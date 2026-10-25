"""Measure three development workloads against an already running web app.

Uses live model calls and the server's existing spending limits. No automatic
retries or answer-quality claims. Raw responses remain in the chosen run folder.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


WORKLOADS = (
    ('retention', 'What is the default retention period for the ProofPoint pentest artifact audit-proxy escrow?'),
    ('escrow_details', 'For the ProofPoint pentest artifact audit-proxy escrow, what is the default retention period, standard access turnaround, and payer for AuditWorks storage fees?'),
    ('multipart_limits', 'What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?'),
)


def request_json(url: str, payload: JsonObject | None = None,
                 timeout: float = 10) -> tuple[int, JsonObject]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


def summarize(records: list[JsonObject]) -> list[JsonObject]:
    result: list[JsonObject] = []
    for name, question in WORKLOADS:
        runs = [r for r in records if r['workload'] == name]
        measured = [r for r in runs if r['http_status'] == 200]
        costs = [r['estimated_cost_usd'] for r in measured if r['estimated_cost_usd'] is not None]
        result.append(dict(workload=name, question=question, attempts=len(runs), completed=len(measured),
            median_latency_seconds=statistics.median(r['latency_seconds'] for r in measured) if measured else None,
            min_latency_seconds=min((r['latency_seconds'] for r in measured), default=None),
            max_latency_seconds=max((r['latency_seconds'] for r in measured), default=None),
            mean_estimated_api_cost_usd=statistics.mean(costs) if costs else None,
            cost_scored_runs=len(costs),
            answers=[r['answer_status'] for r in runs],
            stop_reasons=[r['stop_reason'] for r in runs],
            searches=[r['searches'] for r in runs],
            rounds=[r['rounds'] for r in runs],
            failed_attempts=[r for r in runs if r['http_status'] != 200]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--retrieval', choices=('bm25', 'hybrid'), default='hybrid')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=150)
    parser.add_argument('--minimum-remaining-budget', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Continue unattempted requests; never retry saved failures')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    status, info = request_json(args.url + '/api/ready')
    if status != 200 or info.get('mode') != 'live' or info.get('retrieval') != args.retrieval or info.get('busy'):
        parser.error('App must be ready, idle, live, and using the requested retriever')
    manifest: JsonObject
    records: list[JsonObject]
    if args.resume:
        manifest = json.loads((args.output / 'manifest.json').read_text())
        records = json.loads((args.output / 'runs.json').read_text())
        if args.repeats != manifest['repeats'] or any(info.get(k) != manifest['readiness'].get(k)
                for k in ('mode', 'retrieval', 'model', 'documents', 'retrieval_config', 'embedding_model')):
            parser.error('Resume requires the original repeat count and server configuration')
        manifest.setdefault('interruptions', []).append(dict(
            previous_stop_reason=manifest.get('stop_reason'), resumed_at=datetime.now(timezone.utc).isoformat()))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), checkout_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(), readiness=info,
            protocol='Three workloads, sequential requests; order rotated per repeat; no retries or paid warm-up.',
            repeats=args.repeats, mode='agentic', timeout_seconds=args.timeout,
            timing='Client HTTP wall time, excludes server/index startup; first request retained.',
            cost='Server-reported token-based API estimate; excludes local compute and judge costs.',
            quality='Answer status is model-reported, not a correctness score.')
        records = []
    manifest.setdefault('script_versions', []).append(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output / 'manifest.json', manifest)
    stop = None
    for repeat in range(args.repeats):
        ordered = WORKLOADS[repeat % len(WORKLOADS):] + WORKLOADS[:repeat % len(WORKLOADS)]
        for name, question in ordered:
            if any(r['repeat'] == repeat + 1 and r['workload'] == name for r in records):
                continue
            status, ready = request_json(args.url + '/api/ready')
            if (status != 200 or ready.get('busy') or ready.get('mode') != 'live'
                    or ready.get('retrieval') != args.retrieval or ready.get('model') != info.get('model')):
                stop = 'Server unavailable, busy, or configuration changed'
                break
            if ready.get('budget_remaining_usd', 0) < args.minimum_remaining_budget:
                stop = 'Remaining session reservation budget below benchmark threshold; no reset or increase'
                break
            began = time.perf_counter()
            body: JsonObject
            try:
                status, body = request_json(args.url + '/api/query', {'question': question, 'mode': 'agentic'}, args.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                status, body = 0, {'transport_error': type(error).__name__}
            elapsed = time.perf_counter() - began
            save(args.output / f'{repeat + 1}_{name}.json', body)
            trace, answer = body.get('trace', {}), body.get('answer', {})
            steps, usage = trace.get('steps', []), trace.get('usage', {})
            record = dict(repeat=repeat + 1, workload=name, http_status=status,
                latency_seconds=elapsed, server_latency_seconds=trace.get('total_latency_ms', 0) / 1000,
                answer_status=answer.get('status'), stop_reason=trace.get('stop_reason'),
                searches=sum(s['tool'] == 'text_search' for s in steps),
                rounds=max((s['round_number'] for s in steps), default=0),
                model_completions=sum(s['tool'] in ('critic', 'generate') for s in steps),
                sources=len(body.get('sources', [])), citations=len(answer.get('citation_ids', [])),
                input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                api_calls=usage.get('api_calls'), estimated_cost_usd=usage.get('estimated_cost_usd'))
            records.append(record)
            save(args.output / 'runs.json', records)
            save(args.output / 'summary.json', summarize(records))
            print(json.dumps(record), flush=True)
            if status != 200:
                stop = 'Request failed; stopped without retrying or dropping the failure'
                break
        if stop:
            break
    manifest['stop_reason'] = stop or ('completed_with_errors' if any(r['http_status'] != 200 for r in records) else 'completed')
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'manifest.json', manifest)
    print(json.dumps({'stop_reason': manifest['stop_reason'], 'summary': summarize(records)}), flush=True)


if __name__ == '__main__':
    main()

"""Run a small development-only agent pilot; paid calls require --live."""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.config import CORPUS_DIR, ROOT, runtime_versions
from src.contracts import Answer, QueryResponse, Trace, TraceStep, Usage, UserContext, SearchRequest, validate_evidence
from src.agents.budget import RunBudgetExceeded
from src.evaluation.run_retrieval import git_state, load_questions
from src.evaluation.splits import select_questions
from src.generation.generator import (
    AnthropicConfig, AnthropicProvider, GenerationConfig, GroundedGenerator,
    ProviderError, SYSTEM_PROMPT as GENERATOR_PROMPT, combined_usage,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def development_sample(questions, count, question_ids=None):
    """Explicit IDs must already belong to the development split."""
    if question_ids is None:
        return select_questions(questions, count)
    available = {q['question_id']: q for q in questions}
    if not 1 <= len(question_ids) <= 10 or len(set(question_ids)) != len(question_ids):
        raise ValueError('Choose 1–10 distinct development question IDs')
    if any(qid not in available for qid in question_ids):
        raise ValueError('Every requested question ID must belong to the development split')
    return [available[qid] for qid in question_ids]


def timing_breakdown(trace):
    """Keep actual recorded durations; do not infer quality from stop status."""
    groups = {'search': 0.0, 'critic': 0.0, 'generate': 0.0, 'other': 0.0}
    for step in trace.steps:
        tool = step.tool.split(':', 1)[0]
        key = 'search' if tool == 'text_search' else tool if tool in groups else 'other'
        groups[key] += step.elapsed_ms
    return groups


def prepare_review(questions, retriever, output):
    """Freeze first-search evidence and references without calling a model."""
    if not questions:
        raise ValueError('Review packet requires at least one question')
    output.mkdir(parents=True, exist_ok=False)
    user = UserContext(user_id='benchmark-pilot', allowed_doc_ids=frozenset(retriever.doc_ids))
    search = RecordingRetriever(retriever)
    review = []
    for number, question in enumerate(questions, 1):
        began = time.monotonic()
        search.search(SearchRequest(query=question['question'], user=user, top_k=5))
        filename = f'case_{number:02}.json'
        write_json(output / filename, {'question_id': question['question_id'],
            'question': question['question'], 'search': search.searches[-1],
            'retrieval_latency_ms': (time.monotonic() - began) * 1000,
            'response': None, 'model_calls': []})
        review.append({'case': filename, 'reference': question,
            'evidence_sufficient': None, 'answer_correctness_0_to_2': None,
            'answer_completeness_0_to_2': None, 'citation_support_0_to_2': None,
            'notes': ''})
    write_json(output / 'review.json', review)
    summary = {'completed': len(review), 'paid_calls': False,
               'mode': 'retrieval-review-packet', 'answers_generated': 0}
    write_json(output / 'summary.json', summary)
    return summary


class RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def complete(self, system, prompt):
        record = {'system': system, 'prompt': json.loads(prompt)}
        try:
            result = self.provider.complete(system, prompt)
        except (ProviderError, RunBudgetExceeded) as exc:
            self.calls.append({**record, 'error': type(exc).__name__, 'usage': exc.usage.model_dump()})
            raise
        self.calls.append({**record, 'response': result.text, 'truncated': result.truncated,
                           'usage': result.usage.model_dump()})
        return result


class RecordingRetriever:
    def __init__(self, retriever):
        self.retriever, self.searches = retriever, []

    def search(self, request):
        hits = self.retriever.search(request)
        validate_evidence(request.user, hits)
        for item in hits:
            chunk = item.chunk
            if ((request.source_types and chunk.source_type not in request.source_types)
                    or (request.after and (chunk.timestamp is None or chunk.timestamp < request.after))
                    or (request.before and (chunk.timestamp is None or chunk.timestamp > request.before))):
                raise ValueError('Retrieved evidence violates requested filters')
        self.searches.append({'query': request.query, 'source_types': request.source_types,
                              'evidence': [e.model_dump(mode='json') for e in hits]})
        return hits


def debug_steps(response):
    """Explain new evidence per search without treating critic approval as truth."""
    seen, result = set(), []
    for step in response.trace.steps:
        row = step.model_dump(mode='json')
        if step.tool not in ('route', 'critic', 'generate') and ':' not in step.tool:
            row['new_evidence_ids'] = sorted(set(step.evidence_ids) - seen)
            seen.update(step.evidence_ids)
        result.append(row)
    return result


def show_response(response, emit=print):
    for step in debug_steps(response):
        if 'new_evidence_ids' in step:
            emit(f"Round {step['round_number']} — {step['tool']}: {step['query']}")
            emit(f"  {len(step['evidence_ids'])} hits; {len(step['new_evidence_ids'])} new evidence IDs")
        if step['critic']:
            critic = step['critic']
            emit(f"  Critic: {'sufficient' if critic['sufficient'] else 'insufficient'} "
                 f"(uncalibrated confidence {critic['confidence']:.2f}) — {critic['reason']}")
            if critic['missing_facts']:
                emit('  Missing: ' + '; '.join(critic['missing_facts']))
            if critic['suggested_queries']:
                emit('  Suggested searches: ' + '; '.join(critic['suggested_queries']))
    emit(f"Stop: {response.trace.stop_reason}; answer status: {response.answer.status}")
    emit(response.answer.text)
    cited = set(response.answer.citation_ids)
    for item in response.sources:
        mark = 'cited' if item.citation_id in cited else 'retrieved'
        emit(f"[{item.citation_id}] {mark} — {item.chunk.source_type}: {item.chunk.title or item.chunk.doc_id}")
        emit('  ' + item.chunk.text[:400].replace('\n', ' '))
    usage = response.trace.usage
    cost = 'unknown' if usage.estimated_cost_usd is None else f'${usage.estimated_cost_usd:.6f}'
    emit(f"Usage: {usage.input_tokens + usage.output_tokens} tokens; {usage.api_calls} API calls; "
         f"estimated cost {cost}; {response.trace.total_latency_ms / 1000:.2f}s")


def run_interactive(retriever, provider, config, output, *, live=False, read=input, emit=print):
    """Custom questions: real retrieval by default, actual agent only with --live."""
    if live and provider is None:
        raise ValueError('Live interactive mode requires a provider')
    output.mkdir(parents=True, exist_ok=False)
    user = UserContext(user_id='benchmark-pilot', allowed_doc_ids=frozenset(retriever.doc_ids))
    recorded_search = RecordingRetriever(retriever)
    recorded_provider = RecordingProvider(provider) if live else None
    generation = GenerationConfig(max_context_chars=config.max_context_chars, retries=0)
    agent = None
    if recorded_provider is not None:
        agent = SearchAgent(recorded_search, EvidenceCritic(recorded_provider, generation),
                            GroundedGenerator(recorded_provider, generation), config=config)
    emit('Live agent mode.' if live else 'Retrieval-only mode: no critic, generated answers, or paid calls.')
    emit('Benchmark scope allows all indexed documents. Enter /quit to exit.')
    summaries = []
    summary = {'completed': 0, 'mode': 'live-agent' if live else 'retrieval-only',
               'stops': {}, 'usage': combined_usage([]).model_dump(), 'total_latency_ms': 0}
    write_json(output / 'summary.json', summary)
    while True:
        try:
            question = read('Question> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ('/quit', '/exit'):
            break
        if not question:
            continue
        if len(question) > config.max_question_chars:
            emit('Question exceeds the configured length limit.')
            continue
        recorded_search.searches.clear()
        if recorded_provider:
            recorded_provider.calls.clear()
        request = SearchRequest(query=question, user=user, top_k=5)
        if agent:
            response = agent.run(request)
        else:
            began = time.monotonic()
            try:
                hits = recorded_search.search(request)
                elapsed = (time.monotonic() - began) * 1000
                response = QueryResponse(answer=Answer(status='insufficient_evidence',
                    text='Retrieved evidence only. Use --live to run the critic and answer generator.'),
                    sources=hits, trace=Trace(run_id=f'interactive-{len(summaries)+1}', stop_reason='single_pass',
                        total_latency_ms=elapsed, usage=Usage(estimated_cost_usd=0),
                        steps=(TraceStep(round_number=1, query=question, tool='text_search',
                            evidence_ids=tuple(e.citation_id for e in hits), elapsed_ms=elapsed,
                            usage=Usage(estimated_cost_usd=0)),)))
            except Exception:
                response = QueryResponse(answer=Answer(status='error', text='Retrieval failed.'),
                    trace=Trace(run_id=f'interactive-{len(summaries)+1}', stop_reason='error',
                                usage=Usage(estimated_cost_usd=0)))
        filename = f'case_{len(summaries)+1:02}.json'
        write_json(output / filename, {'question': question, 'mode': summary['mode'],
            'response': response.model_dump(mode='json'), 'debug_steps': debug_steps(response),
            'searches': recorded_search.searches,
            'model_calls': recorded_provider.calls if recorded_provider else []})
        summaries.append(response.trace)
        summary.update(completed=len(summaries), stops=dict(Counter(t.stop_reason for t in summaries)),
                       usage=combined_usage([t.usage for t in summaries]).model_dump(),
                       total_latency_ms=sum(t.total_latency_ms for t in summaries))
        write_json(output / 'summary.json', summary)
        show_response(response, emit)
        emit(f'Saved: {output / filename}')
        if response.trace.stop_reason in ('error', 'budget'):
            emit('Session stopped after a component error or budget limit.')
            break
    return summary


def run_pilot(questions, retriever, provider, config, output, *, workflow='agent',
              critic_prompt=CRITIC_PROMPT):
    """Write each case immediately; reference labels never enter the agent request."""
    if not questions:
        raise ValueError("Pilot requires at least one question")
    if workflow not in ('baseline', 'agent'):
        raise ValueError('Unknown workflow')
    output.mkdir(parents=True, exist_ok=False)
    user = UserContext(user_id='benchmark-pilot', allowed_doc_ids=frozenset(retriever.doc_ids))
    recorded_provider, recorded_search = RecordingProvider(provider), RecordingRetriever(retriever)
    generation = GenerationConfig(max_context_chars=config.max_context_chars, retries=0)
    agent = SearchAgent(recorded_search, EvidenceCritic(recorded_provider, generation, system_prompt=critic_prompt),
                        GroundedGenerator(recorded_provider, generation), config=config)
    responses, review = [], []
    summary = {"completed": 0, "requested": len(questions), "stops": {},
               "usage": combined_usage([]).model_dump(), "total_latency_ms": 0}
    for q in questions:
        recorded_provider.calls.clear()
        recorded_search.searches.clear()
        response = agent.run(SearchRequest(query=q['question'], user=user, top_k=5),
                             single_pass=workflow == 'baseline')
        responses.append(response)
        case = {'question_id': q['question_id'], 'question': q['question'], 'workflow': workflow,
                'question_type': q['question_type'], 'response': response.model_dump(mode='json'),
                'timing_ms': timing_breakdown(response.trace),
                'debug_steps': debug_steps(response),
                'searches': recorded_search.searches, 'model_calls': recorded_provider.calls}
        # Numeric filenames avoid interpreting externally supplied question IDs as paths.
        write_json(output / f'case_{len(responses):02}.json', case)
        review.append({'case': f'case_{len(responses):02}.json',
                       'question_id': q['question_id'], 'reference': q,
                       'critic_correct': None, 'followup_helpful': None,
                       'answer_correctness_0_to_2': None, 'answer_completeness_0_to_2': None,
                       'citation_support_0_to_2': None, 'notes': ''})
        write_json(output / 'review.json', review)
        summary = {'completed': len(responses), 'requested': len(questions),
                   'stops': dict(Counter(r.trace.stop_reason for r in responses)),
                   'usage': combined_usage([r.trace.usage for r in responses]).model_dump(),
                   'total_latency_ms': sum(r.trace.total_latency_ms for r in responses)}
        write_json(output / 'summary.json', summary)
        print(f"{q['question_id']}: {response.trace.stop_reason}", flush=True)
        # Stop a pilot on service/component errors or exhausted/unknown budgets.
        if response.trace.stop_reason in ('error', 'budget'):
            break
    return summary


def run_comparison(questions, retriever, provider, config, output, *, critic_prompt=CRITIC_PROMPT):
    """Same development sample and provider; one total spending cap across both arms."""
    if not questions:
        raise ValueError('Comparison requires at least one question')
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for workflow in ('baseline', 'agent'):
        summaries[workflow] = run_pilot(questions, retriever, provider, config,
                                        output / workflow, workflow=workflow, critic_prompt=critic_prompt)
        # A partial/error arm is not a valid paired comparison. Preserve its logs.
        if any(summaries[workflow]['stops'].get(reason, 0) for reason in ('error', 'budget')):
            break
    pairs = []
    for number, question in enumerate(questions, 1):
        paths = [output / arm / f'case_{number:02}.json' for arm in ('baseline', 'agent')]
        if not all(path.exists() for path in paths):
            continue
        cases = [json.loads(path.read_text()) for path in paths]
        pair = {'question_id': question['question_id']}
        for arm, case in zip(('baseline', 'agent'), cases):
            response = case['response']
            searches = case['searches']
            first_ids = {e['citation_id'] for e in searches[0]['evidence']} if searches else set()
            later_ids = {e['citation_id'] for s in searches[1:] for e in s['evidence']} - first_ids
            pair[arm] = {'answer': response['answer'], 'stop_reason': response['trace']['stop_reason'],
                         'searches': len(searches), 'new_followup_evidence_ids': sorted(later_ids),
                         'latency_ms': response['trace']['total_latency_ms'],
                         'timing_ms': case['timing_ms'],
                         'usage': response['trace']['usage']}
        pairs.append(pair)
    report = {'summaries': summaries, 'paired_cases': len(pairs), 'pairs': pairs,
              'limitation': 'New evidence IDs and critic approval do not establish answer quality. '
                            'Review full case evidence and citations; baseline runs first, so timings may include cache effects.'}
    write_json(output / 'comparison.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=5, choices=range(1, 11), metavar='1..10')
    p.add_argument('--question-ids', nargs='+', help='Exact development IDs, in order; overrides --count')
    p.add_argument('--prepare', action='store_true', help='Save first-search evidence and review references without model calls')
    p.add_argument('--critic-prompt', type=Path, help='Use a saved critic prompt for a controlled development batch run')
    p.add_argument('--workflow', choices=('baseline', 'agent', 'compare'), default='agent',
                   help='Baseline skips critic/follow-ups; compare saves both arms on identical questions')
    p.add_argument('--questions', type=Path, default=CORPUS_DIR / 'questions.jsonl')
    p.add_argument('--splits', type=Path, default=ROOT / 'evaluation/splits.json')
    from src.retrieval.factory import add_retrieval_arguments
    add_retrieval_arguments(p, index_flag='--index')
    p.add_argument('--retrieval', choices=('bm25', 'hybrid'), default='bm25')
    p.add_argument('--output', type=Path)
    p.add_argument('--live', action='store_true', help='Allow paid model calls')
    p.add_argument('--interactive', action='store_true', help='Enter custom questions; retrieval only unless --live')
    p.add_argument('--model')
    p.add_argument('--budget-usd', type=float, default=0)
    p.add_argument('--question-budget-usd', type=float, default=0.10,
                   help='Per-question cap, shared across its critic and generation calls')
    p.add_argument('--input-price', type=float, default=0, help='USD per million input tokens')
    p.add_argument('--output-price', type=float, default=0, help='USD per million output tokens')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.prepare and (args.live or args.interactive):
        p.error('--prepare cannot be combined with --live or --interactive')
    if args.interactive and (args.question_ids or args.critic_prompt):
        p.error('--question-ids and --critic-prompt apply only to development batch runs')
    critic_prompt = args.critic_prompt.read_text() if args.critic_prompt else CRITIC_PROMPT
    if not critic_prompt.strip():
        p.error('Critic prompt must not be empty')
    if not 0 <= args.bm25_weight <= 1:
        p.error('--bm25-weight must be between 0 and 1')
    if args.interactive and args.workflow != 'agent':
        p.error('--workflow baseline/compare requires the development batch pilot')
    required = [args.corpus, args.bm25_index / 'manifest.json']
    if not args.interactive:
        required.extend([args.questions, args.splits])
    if args.retrieval == 'hybrid':
        required.append(args.dense_index / 'manifest.json')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        p.error('Missing local artifacts:\n' + '\n'.join(missing) +
                '\nObtain the full-corpus BM25 artifacts from Hira or build them using the README.')
    if args.live and (not args.model or not os.environ.get('ANTHROPIC_API_KEY')):
        p.error('--live requires --model and ANTHROPIC_API_KEY in the environment')
    if args.live and min(args.budget_usd, args.input_price, args.output_price) <= 0:
        p.error('--live requires a positive total budget and both token prices')
    if not 0 < args.question_budget_usd < float('inf'):
        p.error('--question-budget-usd must be positive and finite')
    from src.retrieval.corpus_bm25 import file_sha256

    try:
        questions = ([] if args.interactive else development_sample(
            load_questions(args.questions, args.splits, 'development'), args.count, args.question_ids))
    except ValueError as exc:
        p.error(str(exc))
    from src.retrieval.factory import load_retriever, settings_from_args
    retriever = load_retriever(settings_from_args(args))
    if not args.live and not args.interactive and not args.prepare:
        print(json.dumps({'ready': True, 'paid_calls': False, 'retrieval': args.retrieval,
                          'documents': len(retriever.doc_ids),
                          'questions': [{'id': q['question_id'], 'type': q['question_type']} for q in questions]}, indent=2))
        return
    output = args.output or ROOT / '.cache/agent-pilot' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists():
        p.error('Output already exists; use a new directory to preserve previous results')
    provider_config = AnthropicConfig(model=args.model, allow_paid=True, budget_usd=args.budget_usd,
        input_usd_per_million=args.input_price, output_usd_per_million=args.output_price,
        max_output_tokens=1024, timeout_seconds=30) if args.live else None
    config = AgentConfig(max_rounds=2, max_cost_usd=args.question_budget_usd, max_seconds=120, max_tokens=16000)
    manifest = {'code': git_state(), 'runtime': runtime_versions(),
                'provider': provider_config.model_dump() if provider_config else None, 'temperature': 0,
                'mode': 'retrieval-review-packet' if args.prepare else 'interactive' if args.interactive else 'development-pilot', 'paid_calls': args.live,
                'workflow': args.workflow,
                'agent': config.model_dump(), 'top_k': 5, 'router': 'TextRouter',
                'scope': 'benchmark-only access to all documents in this index; no production ACL claims',
                'question_ids': [q['question_id'] for q in questions],
                'questions_sha256': file_sha256(args.questions) if not args.interactive else None,
                'splits_sha256': file_sha256(args.splits) if not args.interactive else None,
                'index': retriever.manifest,
                'prompts': {'critic': critic_prompt, 'generator': GENERATOR_PROMPT},
                'critic_prompt_sha256': sha256(critic_prompt.encode()).hexdigest(),
                'source_hashes': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / 'src').rglob('*.py'))}}
    provider = AnthropicProvider(provider_config) if provider_config else None
    try:
        summary = (prepare_review(questions, retriever, output) if args.prepare else
                   run_interactive(retriever, provider, config, output, live=args.live)
                   if args.interactive else
                   run_comparison(questions, retriever, provider, config, output, critic_prompt=critic_prompt)
                   if args.workflow == 'compare' else
                   run_pilot(questions, retriever, provider, config, output, workflow=args.workflow, critic_prompt=critic_prompt))
        print(json.dumps(summary, indent=2))
    finally:
        if output.exists():
            manifest['reserved_cost_usd'] = provider.reserved_usd if provider else 0
            write_json(output / 'manifest.json', manifest)
    print(f'Inspect results in {output}')


if __name__ == '__main__':
    main()

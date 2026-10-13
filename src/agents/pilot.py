"""Run a small development-only agent pilot; paid calls require --live."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic, SYSTEM_PROMPT as CRITIC_PROMPT
from src.config import CORPUS_DIR, ROOT, runtime_versions
from src.contracts import UserContext, SearchRequest
from src.evaluation.run_retrieval import git_state, load_questions
from src.evaluation.splits import select_questions
from src.generation.generator import (
    AnthropicConfig, AnthropicProvider, GenerationConfig, GroundedGenerator,
    ProviderError, SYSTEM_PROMPT as GENERATOR_PROMPT, combined_usage,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def complete(self, system, prompt):
        record = {'system': system, 'prompt': json.loads(prompt)}
        try:
            result = self.provider.complete(system, prompt)
        except ProviderError as exc:
            self.calls.append({**record, 'error': 'ProviderError', 'usage': exc.usage.model_dump()})
            raise
        self.calls.append({**record, 'response': result.text, 'truncated': result.truncated,
                           'usage': result.usage.model_dump()})
        return result


class RecordingRetriever:
    def __init__(self, retriever):
        self.retriever, self.searches = retriever, []

    def search(self, request):
        hits = self.retriever.search(request)
        self.searches.append({'query': request.query, 'source_types': request.source_types,
                              'evidence': [e.model_dump(mode='json') for e in hits]})
        return hits


def run_pilot(questions, retriever, provider, config, output):
    """Write each case immediately; reference labels never enter the agent request."""
    if not questions:
        raise ValueError("Pilot requires at least one question")
    output.mkdir(parents=True, exist_ok=False)
    user = UserContext(user_id='benchmark-pilot', allowed_doc_ids=frozenset(retriever.doc_ids))
    recorded_provider, recorded_search = RecordingProvider(provider), RecordingRetriever(retriever)
    generation = GenerationConfig(max_context_chars=config.max_context_chars, retries=0)
    agent = SearchAgent(recorded_search, EvidenceCritic(recorded_provider, generation),
                        GroundedGenerator(recorded_provider, generation), config=config)
    responses, review = [], []
    summary = {"completed": 0, "requested": len(questions), "stops": {},
               "usage": combined_usage([]).model_dump(), "total_latency_ms": 0}
    for q in questions:
        recorded_provider.calls.clear()
        recorded_search.searches.clear()
        response = agent.run(SearchRequest(query=q['question'], user=user, top_k=5))
        responses.append(response)
        case = {'question_id': q['question_id'], 'question': q['question'],
                'question_type': q['question_type'], 'response': response.model_dump(mode='json'),
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=5, choices=range(1, 11), metavar='1..10')
    p.add_argument('--questions', type=Path, default=CORPUS_DIR / 'questions.jsonl')
    p.add_argument('--splits', type=Path, default=ROOT / 'evaluation/splits.json')
    p.add_argument('--corpus', type=Path, default=CORPUS_DIR / 'documents.parquet')
    p.add_argument('--index', type=Path, default=CORPUS_DIR / 'indexes/bm25')
    p.add_argument('--retrieval', choices=('bm25', 'hybrid'), default='bm25')
    p.add_argument('--dense-index', type=Path, default=CORPUS_DIR / 'indexes/dense')
    p.add_argument('--device', default='cpu', help='Device for hybrid query embeddings')
    p.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=None,
                   help='Match dense index precision; default is fp16 on CUDA, fp32 otherwise')
    p.add_argument('--output', type=Path)
    p.add_argument('--live', action='store_true', help='Allow paid model calls')
    p.add_argument('--model')
    p.add_argument('--budget-usd', type=float, default=0)
    p.add_argument('--input-price', type=float, default=0, help='USD per million input tokens')
    p.add_argument('--output-price', type=float, default=0, help='USD per million output tokens')
    return p


def main():
    p = parser()
    args = p.parse_args()
    required = [args.questions, args.splits, args.corpus, args.index / 'manifest.json']
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
    from src.retrieval.corpus_bm25 import CorpusBM25, file_sha256

    questions = select_questions(load_questions(args.questions, args.splits, 'development'), args.count)
    # Loading also verifies the actual corpus hash against the index manifest.
    retriever = CorpusBM25.load(args.index, corpus_path=args.corpus)
    if args.retrieval == 'hybrid':
        from src.retrieval.corpus_dense import BGEModel, CorpusDense
        from src.retrieval.hybrid import HybridRetriever

        dense = CorpusDense.load(args.dense_index, model=BGEModel(args.device, fp16=args.fp16),
                                 corpus_path=args.corpus, corpus=retriever.corpus)
        retriever = HybridRetriever(retriever, dense)
    if not args.live:
        print(json.dumps({'ready': True, 'paid_calls': False, 'retrieval': args.retrieval,
                          'documents': len(retriever.doc_ids),
                          'questions': [{'id': q['question_id'], 'type': q['question_type']} for q in questions]}, indent=2))
        return
    output = args.output or ROOT / '.cache/agent-pilot' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists():
        p.error('Output already exists; use a new directory to preserve previous results')
    provider_config = AnthropicConfig(model=args.model, allow_paid=True, budget_usd=args.budget_usd,
        input_usd_per_million=args.input_price, output_usd_per_million=args.output_price,
        max_output_tokens=1024, timeout_seconds=30)
    config = AgentConfig(max_rounds=2, max_cost_usd=args.budget_usd, max_seconds=120, max_tokens=16000)
    manifest = {'code': git_state(), 'runtime': runtime_versions(),
                'provider': provider_config.model_dump(), 'temperature': 0,
                'agent': config.model_dump(), 'top_k': 5, 'router': 'TextRouter',
                'scope': 'benchmark-only access to all documents in this index; no production ACL claims',
                'question_ids': [q['question_id'] for q in questions],
                'questions_sha256': file_sha256(args.questions), 'splits_sha256': file_sha256(args.splits),
                'index': retriever.manifest,
                'prompts': {'critic': CRITIC_PROMPT, 'generator': GENERATOR_PROMPT},
                'source_hashes': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / 'src').rglob('*.py'))}}
    provider = AnthropicProvider(provider_config)  # One budget shared by all questions/calls.
    try:
        summary = run_pilot(questions, retriever, provider, config, output)
        print(json.dumps(summary, indent=2))
    finally:
        if output.exists():
            manifest['reserved_cost_usd'] = provider.reserved_usd
            write_json(output / 'manifest.json', manifest)
    print(f'Inspect results in {output}')


if __name__ == '__main__':
    main()

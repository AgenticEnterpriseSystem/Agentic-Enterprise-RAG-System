"""CAL-01 fixed-evidence critic coverage experiment.

The critic is run twice on identical synthetic evidence: once choosing its own
fact checklist ("own", the deployed behavior) and once given the annotated
requested facts ("given"). Evidence is fixed and no answer is generated, so
retrieval gaps and generator omissions are excluded by construction. Errors that
disappear under "given" are planning errors; errors that remain are critic
judgment errors. Fixture labels are AI-proposed, not human judgments.

Without --live this only validates fixtures and reports the planned call count.
Results contain aggregate counts only, never questions, evidence or model text.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Literal

from pydantic import Field, model_validator

from src.agents.search_critic import (
    AgentError, EvidenceCritic, MAX_REQUESTED_FACTS, REQUESTED_FACTS_ADDENDUM, SYSTEM_PROMPT,
)
from src.contracts import (
    Chunk, Contract, CriticDecision, Evidence, GenerationRequest, Identifier, UserContext,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / 'tests/fixtures/critic_coverage_cases.json'
Condition = Literal['own', 'given']
CONDITIONS: tuple[Condition, ...] = ('own', 'given')
NOT_MEASURED = (
    'retrieval_gap: evidence is fixed per case',
    'generator_omission: no answer is generated',
    'human_agreement: fixture labels are AI-proposed',
)


class RequestedFact(Contract):
    label: Identifier
    answered: bool
    # Lowercase substrings used to match the critic's own free-text labels.
    match_terms: tuple[Identifier, ...] = Field(min_length=1)


class CoverageCase(Contract):
    id: Identifier
    question: Identifier
    evidence: Identifier
    requested_facts: tuple[RequestedFact, ...] = Field(min_length=1, max_length=MAX_REQUESTED_FACTS)

    @model_validator(mode='after')
    def unique_labels(self):
        if len({f.label for f in self.requested_facts}) != len(self.requested_facts):
            raise ValueError(f'Duplicate requested fact label in case {self.id}')
        return self

    @property
    def expected_sufficient(self) -> bool:
        return all(f.answered for f in self.requested_facts)

    def request(self) -> GenerationRequest:
        chunk = Chunk(chunk_id=f'{self.id}:0:{len(self.evidence)}', doc_id=self.id, source_type='docs',
                      metadata_origin='synthetic', text=self.evidence, start_char=0, end_char=len(self.evidence))
        return GenerationRequest(
            question=self.question,
            user=UserContext(user_id='critic-coverage', allowed_doc_ids=frozenset({self.id})),
            evidence=(Evidence(citation_id=chunk.chunk_id, chunk=chunk, score=1, retrieval_method='bm25'),))


class CaseOutcome(Contract):
    condition: Condition
    expected_sufficient: bool
    action: Literal['approve', 'continue', 'clarify', 'error']
    error_code: str | None = None
    causes: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0


def load_cases(path: Path = DEFAULT_CASES) -> tuple[CoverageCase, ...]:
    data = json.loads(path.read_text(encoding='utf-8'))
    cases = tuple(CoverageCase.model_validate(c) for c in data['cases'])
    if len({c.id for c in cases}) != len(cases):
        raise ValueError('Duplicate case id')
    return cases


def _matches(fact: RequestedFact, label: str, condition: Condition) -> bool:
    if condition == 'given':
        return label == fact.label
    lowered = label.lower()
    return any(term.lower() in lowered for term in fact.match_terms)


def classify(case: CoverageCase, condition: Condition, decision: CriticDecision) -> CaseOutcome:
    """Place one decision in a stopping-error direction and attribute its cause."""
    action = 'approve' if decision.sufficient else 'clarify' if decision.clarification_question else 'continue'
    causes: set[str] = set()
    if action == 'approve' and not case.expected_sufficient:
        # Premature approval: a requested fact the evidence lacks was accepted.
        for fact in case.requested_facts:
            if fact.answered:
                continue
            listed = [f for f in decision.facts if _matches(fact, f.fact, condition)]
            causes.add('critic_judgment' if listed else 'planning_omission')
    elif action != 'approve' and case.expected_sufficient:
        # Unnecessary continuation: every requested fact was present.
        listed = {f.fact for f in decision.facts}
        for gap in decision.missing_facts:
            requested = [f for f in case.requested_facts if _matches(f, gap, condition)]
            if condition == 'given' and requested and gap not in listed:
                causes.add('checklist_noncompliance')
            else:
                causes.add('critic_judgment' if requested else 'over_asking')
        if not causes:
            causes.add('critic_judgment')
    return CaseOutcome(condition=condition, expected_sufficient=case.expected_sufficient,
                       action=action, causes=tuple(sorted(causes)))


def run_case(critic: EvidenceCritic, case: CoverageCase, condition: Condition) -> CaseOutcome:
    labels = tuple(f.label for f in case.requested_facts) if condition == 'given' else None
    try:
        decision, usage = critic.assess(case.request(), requested_facts=labels)
    except AgentError as error:
        usage = error.usage
        return CaseOutcome(condition=condition, expected_sufficient=case.expected_sufficient,
                           action='error', error_code=error.code,
                           input_tokens=getattr(usage, 'input_tokens', 0) or 0,
                           output_tokens=getattr(usage, 'output_tokens', 0) or 0,
                           cost_usd=getattr(usage, 'estimated_cost_usd', None))
    return classify(case, condition, decision).model_copy(update={
        'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens,
        'cost_usd': usage.estimated_cost_usd})


def run(critic: EvidenceCritic, cases, conditions=CONDITIONS):
    """Interleave conditions per case; stop at the first provider failure."""
    outcomes, stopped = [], False
    for case in cases:
        for condition in conditions:
            outcome = run_case(critic, case, condition)
            outcomes.append(outcome)
            if outcome.error_code == 'provider_error':
                stopped = True
                break
        if stopped:
            break
    return outcomes, stopped


def aggregate(outcomes, condition: Condition) -> dict:
    rows = [o for o in outcomes if o.condition == condition]
    done = [o for o in rows if o.action != 'error']
    should_stop = [o for o in done if not o.expected_sufficient]
    should_answer = [o for o in done if o.expected_sufficient]
    premature = [o for o in should_stop if o.action == 'approve']
    unnecessary = [o for o in should_answer if o.action != 'approve']
    costs = [o.cost_usd for o in rows]
    return {
        'attempted': len(rows),
        'completed': len(done),
        'errors': dict(Counter(o.error_code for o in rows if o.action == 'error')),
        'missing_fact_cases': len(should_stop),
        'premature_approvals': len(premature),
        'premature_approval_causes': dict(Counter(c for o in premature for c in o.causes)),
        'complete_cases': len(should_answer),
        'unnecessary_continuations': len(unnecessary),
        'unnecessary_continuation_causes': dict(Counter(c for o in unnecessary for c in o.causes)),
        'clarifications': sum(o.action == 'clarify' for o in done),
        'input_tokens': sum(o.input_tokens for o in rows),
        'output_tokens': sum(o.output_tokens for o in rows),
        # None means at least one call had unknown cost, not zero.
        'cost_usd': None if any(c is None for c in costs) else round(sum(costs), 6),
    }


def report(outcomes, stopped: bool, cases, conditions, manifest: dict) -> dict:
    return {'manifest': manifest, 'cases': len(cases), 'stopped_early': stopped,
            'not_measured': list(NOT_MEASURED),
            'conditions': {c: aggregate(outcomes, c) for c in conditions}}


def _git_state() -> dict:
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--output', type=Path, help='Aggregate JSON path; must not already exist')
    p.add_argument('--live', action='store_true', help='Allow paid model calls')
    p.add_argument('--model')
    p.add_argument('--budget-usd', type=float, default=0)
    p.add_argument('--input-price', type=float, default=0, help='USD per million input tokens')
    p.add_argument('--output-price', type=float, default=0, help='USD per million output tokens')
    args = p.parse_args(argv)
    cases = load_cases(args.cases)
    conditions = tuple(dict.fromkeys(args.conditions))
    if not args.live:
        print(json.dumps({'ready': True, 'paid_calls': False, 'cases': len(cases),
                          'conditions': list(conditions), 'planned_model_calls': len(cases) * len(conditions),
                          'missing_fact_cases': sum(not c.expected_sufficient for c in cases),
                          'complete_cases': sum(c.expected_sufficient for c in cases)}, indent=2))
        return 0
    if not args.model or not args.output or min(args.budget_usd, args.input_price, args.output_price) <= 0:
        p.error('--live requires --model, --output, a positive budget and both token prices')
    if args.output.exists():
        p.error('Output already exists; choose a new path to preserve earlier results')
    from src.generation.generator import AnthropicConfig, AnthropicProvider
    config = AnthropicConfig(model=args.model, allow_paid=True, budget_usd=args.budget_usd,
                             input_usd_per_million=args.input_price, output_usd_per_million=args.output_price,
                             max_output_tokens=2048, timeout_seconds=60)
    outcomes, stopped = run(EvidenceCritic(AnthropicProvider(config)), cases, conditions)
    hashed = lambda text: sha256(text.encode()).hexdigest()
    manifest = {'code': _git_state(), 'provider': config.model_dump(),
                'cases_sha256': sha256(args.cases.read_bytes()).hexdigest(),
                'critic_prompt_sha256': hashed(SYSTEM_PROMPT),
                'requested_facts_addendum_sha256': hashed(REQUESTED_FACTS_ADDENDUM)}
    result = report(outcomes, stopped, cases, conditions, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result['conditions'], indent=2))
    return 1 if stopped else 0


if __name__ == '__main__':
    raise SystemExit(main())

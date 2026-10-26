"""Bounded search → evidence assessment → answer orchestration."""

import time
from uuid import uuid4

from pydantic import Field

from src.agents.context import select_agent_context
from src.agents.budget import CallBudget, RunBudgetExceeded, call_budget
from src.agents.routing import TextRouter
from src.agents.search_critic import AgentError
from src.contracts import (
    Answer, Contract, GenerationRequest, QueryResponse, SearchRequest, Trace,
    TraceStep, Usage, validate_evidence,
)
from src.generation.generator import combined_usage, select_context


class AgentConfig(Contract):
    max_rounds: int = Field(default=3, ge=1, le=10)
    max_context_chars: int = Field(default=24000, ge=1)
    max_question_chars: int = Field(default=8000, ge=1)
    max_tokens: int = Field(default=8000, ge=1)
    max_seconds: float = Field(default=60, gt=0)
    max_cost_usd: float = Field(default=0, ge=0)
    confidence_threshold: float = Field(default=0.8, ge=0, le=1)


class SearchAgent:
    def __init__(self, retriever, critic, generator, *, router=None, tools=None,
                 config=None, clock=time.monotonic):
        self.retriever, self.critic, self.generator = retriever, critic, generator
        self.router, self.tools = router or TextRouter(), dict(tools or {})
        self.config, self.clock = config or AgentConfig(), clock

    def run(self, request: SearchRequest, *, single_pass=False) -> QueryResponse:
        started = self.clock()
        steps, usages, evidence = [], [], {}
        feedback = None
        context_order = []
        initial_ids = set()
        config = self.config
        round_number = 1

        def exhausted():
            usage = combined_usage(usages)
            return (self.clock() - started >= config.max_seconds
                    or usage.input_tokens + usage.output_tokens >= config.max_tokens
                    or (usage.estimated_cost_usd is None and usage.api_calls > 0)
                    or (usage.estimated_cost_usd is not None
                        and (usage.estimated_cost_usd > config.max_cost_usd
                             or (config.max_cost_usd > 0
                                 and usage.estimated_cost_usd >= config.max_cost_usd))))

        def model_call(operation, *args):
            usage = combined_usage(usages)
            allowance = CallBudget(
                remaining_tokens=max(0, config.max_tokens - usage.input_tokens - usage.output_tokens),
                remaining_seconds=max(0, config.max_seconds - (self.clock() - started)),
                remaining_cost_usd=max(0, config.max_cost_usd - (usage.estimated_cost_usd or 0)),
            )
            with call_budget(allowance):
                return operation(*args)

        def record(tool, began, *, query=None, hits=(), decision=None, usage=None, sources=()):
            usage = usage or Usage(estimated_cost_usd=0)
            usages.append(usage)
            steps.append(TraceStep(
                round_number=round_number, query=query or request.query, tool=tool,
                source_types=sources, evidence_ids=tuple(e.citation_id for e in hits),
                critic=decision, elapsed_ms=max(0, (self.clock() - began) * 1000), usage=usage,
            ))

        def finish(reason, text, answer=None, sources=None):
            return QueryResponse(
                answer=answer or Answer(status="error" if reason == "error" else "insufficient_evidence", text=text),
                sources=tuple(evidence.values()) if sources is None else tuple(sources),
                trace=Trace(run_id=run_id, steps=tuple(steps),
                            total_latency_ms=max(0, (self.clock() - started) * 1000),
                            usage=combined_usage(usages), stop_reason=reason),
            )

        def accept(hits, source_scope):
            hits = tuple(hits)
            validate_evidence(request.user, hits)
            checked_chunks = {e.chunk.chunk_id: e.chunk for e in evidence.values()}
            for item in hits:
                chunk = item.chunk
                if source_scope and chunk.source_type not in source_scope:
                    raise AgentError("Search returned evidence outside the source filter.")
                if request.after and (chunk.timestamp is None or chunk.timestamp < request.after):
                    raise AgentError("Search returned evidence outside the date filter.")
                if request.before and (chunk.timestamp is None or chunk.timestamp > request.before):
                    raise AgentError("Search returned evidence outside the date filter.")
                previous = evidence.get(item.citation_id)
                if previous and previous.chunk != chunk:
                    raise AgentError("A citation ID was reused for different evidence.")
                if chunk.chunk_id in checked_chunks and checked_chunks[chunk.chunk_id] != chunk:
                    raise AgentError("A chunk ID was reused for different evidence.")
                checked_chunks[chunk.chunk_id] = chunk
            known_chunks = {e.chunk.chunk_id for e in evidence.values()}
            for item in hits:
                if item.chunk.chunk_id not in known_chunks:
                    evidence.setdefault(item.citation_id, item)
                    known_chunks.add(item.chunk.chunk_id)
            return hits

        def partial_answer(reason, limitation):
            # Called only for normal search stops, never errors or exhausted budgets.
            nonlocal active_tool, began
            if exhausted():
                return finish("budget", "Search budget reached.")
            context = GenerationRequest(question=request.query, user=request.user,
                                        evidence=tuple(evidence.values()), search_limitation=limitation)
            usage = combined_usage(usages)
            remaining = config.max_tokens - usage.input_tokens - usage.output_tokens
            # Reserve output/prompt room and reduce context under a tight allowance.
            # This is a sizing heuristic; provider token counting remains authoritative.
            context_limit = min(config.max_context_chars, max(1, (remaining - 1500) * 2))
            context = context.model_copy(update={"evidence": tuple(select_agent_context(context, context_limit, initial_ids))})
            if not context.evidence:
                return finish(reason, limitation)
            active_tool, began = "generate", self.clock()
            answer, usage = model_call(self.generator.generate, context)
            record(active_tool, began, hits=context.evidence, usage=usage)
            if not set(answer.citation_ids) <= {e.citation_id for e in context.evidence}:
                raise AgentError("Generator returned an unknown citation.")
            if exhausted():
                return finish("budget", "Answer generation exceeded the run budget.")
            if answer.status == "error":
                return finish("error", "", answer, context.evidence)
            answer = answer.model_copy(update={"status": "insufficient_evidence",
                "text": answer.text + "\n\nSearch limitation: " + limitation})
            return finish(reason, "", answer, context.evidence)

        run_id = str(uuid4())
        if len(request.query) > config.max_question_chars:
            return finish("error", "Question exceeds the configured length limit.")
        for round_number in range(1, config.max_rounds + 1):
            if exhausted():
                return finish("budget", "Search budget reached.")
            before = len(evidence)
            active_tool, began = "route", self.clock()
            try:
                route, usage = model_call(self.router.route, request, feedback, self.tools)
                record("route", began, query=route.reason, usage=usage)
                if exhausted():
                    return finish("budget", "Search budget reached.")
                if route.limitation:
                    return finish("no_new_evidence", route.limitation)
                sources = route.source_types or request.source_types
                if request.source_types and not set(sources) <= set(request.source_types):
                    raise AgentError("Router tried to widen the source scope.")
                # Validate every operation before executing any part of this plan.
                searches = [SearchRequest(query=query, user=request.user, top_k=request.top_k,
                                          source_types=sources, after=request.after, before=request.before)
                            for query in route.text_queries]
                if any(len(search.query) > config.max_question_chars for search in searches):
                    raise AgentError("Search query exceeds the configured length limit.")
                calls = []
                for call in route.tool_calls:
                    if call.tool not in self.tools:
                        raise AgentError("Requested structured tool is unavailable.")
                    spec = self.tools[call.tool]
                    scoped = request.model_copy(update={"source_types": sources})
                    calls.append((call, spec, spec.request(call, scoped)))
                for search in searches:
                    if exhausted():
                        return finish("budget", "Search budget reached.")
                    active_tool, began = "text_search", self.clock()
                    hits = accept(self.retriever.search(search), sources)
                    record(active_tool, began, query=search.query, hits=hits, sources=sources)
                for call, spec, tool_request in calls:
                    if exhausted():
                        return finish("budget", "Search budget reached.")
                    active_tool, began = call.tool, self.clock()
                    result = spec.adapter.execute(tool_request)
                    # Counts need scoped, citable aggregate evidence from the adapter.
                    if result.count is not None and not result.evidence:
                        raise AgentError("Structured count has no supporting evidence.")
                    hits = accept(result.evidence, (spec.source_type,))
                    record(active_tool, began, query=tool_request.model_dump_json(exclude={"user"}),
                           hits=hits, sources=(spec.source_type,))
                if exhausted():
                    return finish("budget", "Search budget reached.")
                if len(evidence) == before:
                    return partial_answer("no_new_evidence", "Search found no new evidence; some requested facts may remain unresolved.")
                # Prefer the newest round without reversing each search's ranking.
                new_evidence = list(evidence.values())[before:]
                if round_number == 1:
                    initial_ids.update(e.citation_id for e in new_evidence)
                context_order = new_evidence + context_order
                context = GenerationRequest(question=request.query, user=request.user,
                                            evidence=tuple(context_order))
                selected = (select_agent_context(context, config.max_context_chars, initial_ids)
                            if round_number > 1 else select_context(context, config.max_context_chars))
                context = context.model_copy(update={"evidence": tuple(selected)})
                if not single_pass:
                    active_tool, began = "critic", self.clock()
                    if round_number == config.max_rounds:
                        usage_so_far = combined_usage(usages)
                        remaining = config.max_tokens - usage_so_far.input_tokens - usage_so_far.output_tokens
                        # Share the remaining allowance with the final answer rather
                        # than spending nearly all of it assessing more evidence.
                        critic_limit = min(config.max_context_chars, max(1, (remaining // 2 - 1500) * 2))
                        critic_pool = GenerationRequest(question=request.query, user=request.user, evidence=tuple(evidence.values()))
                        context = critic_pool.model_copy(update={"evidence": tuple(select_agent_context(critic_pool, critic_limit, initial_ids))})
                    # An approval applies only to this exact evidence set. Use it
                    # for generation too, including the reduced final-round pool.
                    feedback, usage = model_call(self.critic.assess, context)
                    record(active_tool, began, hits=context.evidence, decision=feedback, usage=usage)
                if exhausted():
                    return finish("budget", "Search budget reached.")
                if single_pass or (feedback is not None and feedback.sufficient
                                   and not feedback.missing_facts
                                   and feedback.confidence >= config.confidence_threshold):
                    active_tool, began = "generate", self.clock()
                    answer, usage = model_call(self.generator.generate, context)
                    record(active_tool, began, hits=context.evidence, usage=usage)
                    if not set(answer.citation_ids) <= {e.citation_id for e in context.evidence}:
                        raise AgentError("Generator returned an unknown citation.")
                    if exhausted():
                        return finish("budget", "Answer generation exceeded the run budget.")
                    reason = {"error": "error", "answered": "single_pass" if single_pass else "sufficient",
                              "insufficient_evidence": "no_new_evidence"}[answer.status]
                    return finish(reason, "", answer, context.evidence)
                if round_number == config.max_rounds:
                    return partial_answer("round_limit", "Search round limit reached before evidence was judged sufficient.")
            except RunBudgetExceeded as exc:
                record(active_tool + ":budget", began, usage=exc.usage)
                return finish("budget", "Model call exceeds remaining run budget.")
            except Exception as exc:
                # Preserve accounting without exposing provider messages or source text.
                record(active_tool + ":error", began,
                       usage=exc.usage if isinstance(exc, AgentError) else Usage(estimated_cost_usd=None))
                return finish("error", str(exc) if isinstance(exc, AgentError) else "A search component failed.")
        return finish("round_limit", "Search limit reached; the evidence is still insufficient.")

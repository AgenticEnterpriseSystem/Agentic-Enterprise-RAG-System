"""Live timing run: the web app's agent on development questions, with and without the reranker.

Runs each question through the same runtime the app uses (hybrid search, critic, follow-up
searches, answer), once without and once with the optional reranker, and records the total
time, the number of searches, how each run stopped and its cost. A run longer than the app's
time limit (RAG_MAX_SECONDS, default 200 s) would have timed out in the web app.

To measure how much time questions really need, give the agent more room than the app's limit
with --max-seconds (at most 300): every question then runs to its natural end, and the report
says how many would exceed the app's limit and by how much. Up to the app's limit, the agent
takes the same steps either way.

**Paid:** it calls the Anthropic API with the key, model and prices in your .env (live-mode
settings; see docs/WEB_APP.md). Expect about $0.02-0.04 per question per variant;
--budget-usd caps the total. Examples:

    python scripts/time_agent.py --budget-usd 1                     # 10 questions, both variants
    python scripts/time_agent.py --count 20 --arms reranker --max-seconds 300 --budget-usd 1.5

The first 10 questions are those of the 2026-10-21 agent record, so search counts can be
compared with that run; --count above 10 adds development questions sampled with seed 5980.
On macOS, set DYLD_LIBRARY_PATH as for the tests first.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_web import build_environment  # noqa: E402

RECORD = ROOT / "evaluation/agent_development/20261021.json"


def question_ids(args, development):
    """The record's questions first, then a fixed sample of other development questions."""
    if args.question_ids:
        return args.question_ids
    record = json.loads(RECORD.read_text())
    chosen = [case["question_id"] for case in record["experiments"][0]["cases"]][: args.count]
    others = sorted(set(development) - set(chosen))
    random.Random(5980).shuffle(others)
    return chosen + others[: max(0, args.count - len(chosen))]


def doc_id(citation_id):
    """Citation IDs are `<doc_id>:<start>:<end>`."""
    return citation_id.rsplit(":", 2)[0]


def diagnose(response, expected):
    """Where the reference documents were lost, from the run's own trace.

    found: in any search's results; shown: in a context the critic or the answer step read;
    cited: cited by the answer. Unanswered questions are labelled no_answer_expected (the
    question has no reference documents, so "insufficient evidence" is the right outcome),
    never_found (no reference document was retrieved), not_shown (retrieved but left out of the
    critic's context) or judged_insufficient (the critic saw at least one and still asked for more).
    """
    expected = set(expected)
    steps = response.trace.steps
    found = {doc_id(c) for s in steps if s.tool == "text_search" for c in s.evidence_ids} & expected
    shown = {doc_id(c) for s in steps if s.tool in ("critic", "generate") for c in s.evidence_ids} & expected
    cited = {doc_id(c) for c in response.answer.citation_ids} & expected
    first = next((s for s in steps if s.tool == "text_search"), None)
    critics = [{"round": s.round_number, "sufficient": s.critic.sufficient,
                "confidence": s.critic.confidence, "reason": s.critic.reason,
                "missing_facts": list(s.critic.missing_facts),
                "suggested_queries": list(s.critic.suggested_queries),
                "references_shown": len({doc_id(c) for c in s.evidence_ids} & expected),
                "excerpts_shown": len(s.evidence_ids)}
               for s in steps if s.tool == "critic" and s.critic is not None]
    generate = [s for s in steps if s.tool == "generate"]
    if response.answer.status == "answered":
        label = "answered"
    elif response.answer.status == "error" or response.trace.stop_reason == "error":
        label = "error"
    elif not expected:
        label = "no_answer_expected"
    elif not found:
        label = "never_found"
    elif not shown:
        label = "not_shown"
    else:
        label = "judged_insufficient"
    return {
        "diagnosis": label, "references": len(expected), "references_found": len(found),
        "references_found_first_search": len({doc_id(c) for c in first.evidence_ids} & expected) if first else 0,
        "references_shown": len(shown), "references_cited": len(cited),
        "queries": [s.query for s in steps if s.tool == "text_search"],
        "critic": critics,
        # A generate step that used the whole output limit (MAX_OUTPUT_TOKENS) was probably cut off.
        "answer_output_tokens": generate[-1].usage.output_tokens if generate else None,
        "answer_text": response.answer.text,
        "error_steps": [s.tool for s in steps if s.tool.endswith(":error") or s.tool.endswith(":budget")],
        # Output tokens of a failed model call; the full output limit means its reply was cut off.
        "error_output_tokens": [s.usage.output_tokens for s in steps if s.tool.endswith(":error")],
    }


FLAG_PATTERNS = {
    # Phrases in the critic's reason or missing facts; a rough first sort, not a judgement.
    "excerpt_cut_off": r"truncat|cut[s]? off|cut-off|incomplete in the (?:evidence|excerpt)|excerpt ends",
    "conflicting_sources": r"conflict|contradict|inconsistent|supersede|differing|disagree",
}


def flags(record):
    """Rough causes for an unanswered question, from the critic's words and the trace."""
    import re
    found = set()
    for c in record["critic"]:
        text = " ".join([c["reason"], *c["missing_facts"]]).lower()
        found |= {name for name, pattern in FLAG_PATTERNS.items() if re.search(pattern, text)}
    critics = record["critic"]
    if critics and critics[-1]["sufficient"] and record["status"] != "answered" and record["diagnosis"] != "error":
        found.add("answer_step_refused")  # The critic approved; the answer step declined.
    shown = [c["references_shown"] for c in critics]
    if len(shown) > 1 and max(shown[:-1]) > shown[-1]:
        found.add("reference_dropped_in_round_2")
    return sorted(found)


def run_one(runtime, question, mode, expected=()):
    from src.api.runtime import RuntimeErrorResponse
    from src.contracts import QueryRequest

    began = time.perf_counter()
    try:
        response = runtime.query(QueryRequest(question=question, mode=mode))
    except RuntimeErrorResponse as error:
        return {"seconds": time.perf_counter() - began, "error": error.code}
    seconds = time.perf_counter() - began
    steps = response.trace.steps
    by_tool = {}
    for step in steps:
        tool = step.tool.split(":")[0]
        by_tool[tool] = by_tool.get(tool, 0) + step.elapsed_ms / 1000
    result = {"seconds": seconds, "searches": sum(s.tool == "text_search" for s in steps),
            "search_seconds": by_tool.get("text_search", 0.0),
            "model_seconds": sum(v for k, v in by_tool.items() if k in ("critic", "generate", "route")),
            "stop_reason": response.trace.stop_reason, "status": response.answer.status,
            "citations": len(response.answer.citation_ids),
            "cost_usd": response.trace.usage.estimated_cost_usd, **diagnose(response, expected)}
    result["flags"] = flags(result)
    return result


def describe(name, runs, limit, agent_limit):
    done = [r for r in runs if "error" not in r]
    if not done:
        print(f"  {name}: no completed runs")
        return
    print(f"  {name}, seconds per question (longest first):")
    for r in sorted(done, key=lambda r: -r["seconds"]):
        cut = r["stop_reason"] == "budget" and r["seconds"] >= agent_limit
        flag = (f"  needs more than {agent_limit:.0f}s (stopped)" if cut
                else f"  {r['seconds'] - limit:+.0f}s over the {limit:.0f}s limit" if r["seconds"] > limit else "")
        print(f"    {r['question_id']} {r['seconds']:6.1f}s  {r['searches']} searches "
              f"({r['search_seconds']:.0f}s searching, {r['model_seconds']:.0f}s model)  {r['stop_reason']}{flag}")
    seconds = [r["seconds"] for r in done]
    searches = [r["searches"] for r in done]
    stops = {}
    for r in done:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    cost = sum(r["cost_usd"] or 0 for r in done)
    print(f"  {name}: {len(done)} questions | total time median {statistics.median(seconds):.1f}s, "
          f"mean {statistics.mean(seconds):.1f}s, max {max(seconds):.1f}s | over {limit:.0f}s: "
          f"{sum(s > limit for s in seconds)} | searches {sorted(searches)} (median "
          f"{statistics.median(searches):g}) | search time median "
          f"{statistics.median(r['search_seconds'] for r in done):.1f}s | answered "
          f"{sum(r['status'] == 'answered' for r in done)} | stops {stops} | cost ${cost:.3f}")
    over = sorted(s for s in seconds if s > limit)
    if over:
        stopped = max(seconds) >= agent_limit
        print(f"  -> {len(over)} of {len(done)} would time out at {limit:.0f}s; "
              + (f"some were still running at the {agent_limit:.0f}s cap, so they need more than that"
                 if stopped else f"a limit of {max(seconds):.0f}s would let all of them finish"))
    else:
        print(f"  -> all {len(done)} finished within {limit:.0f}s (longest {max(seconds):.0f}s)")
    explain(done)


def short(text, width=110):
    return text if len(text) <= width else text[:width - 1] + "…"


def explain(done):
    """Why questions went unanswered, from where their reference documents were lost."""
    labels = {}
    for r in done:
        labels[r["diagnosis"]] = labels.get(r["diagnosis"], 0) + 1
    print(f"  Diagnosis: {labels}")
    print("    no_answer_expected = the question has no reference documents (insufficient is right); "
          "never_found = no reference document in any search; not_shown = retrieved but not in "
          "the critic's context; judged_insufficient = the critic saw one and asked for more")
    names = ["answered", "no_answer_expected", "never_found", "not_shown", "judged_insufficient", "error"]
    print(f"\n    {'question type':26} {'n':>3} " + " ".join(f"{n[:12]:>12}" for n in names))
    for qtype in sorted({r["question_type"] for r in done}):
        group = [r for r in done if r["question_type"] == qtype]
        print(f"    {qtype:26} {len(group):>3} "
              + " ".join(f"{sum(r['diagnosis'] == n for r in group):>12}" for n in names))
    tagged = {}
    for r in done:
        for f in r.get("flags", []):
            tagged.setdefault(f, []).append(r["question_id"])
    print("\n    Likely causes among unanswered questions (from the critic's words; a question can have several):")
    for name in ("excerpt_cut_off", "reference_dropped_in_round_2", "conflicting_sources", "answer_step_refused"):
        ids = tagged.get(name, [])
        print(f"      {name:30} {len(ids):>3}  {' '.join(ids)}")
    print()
    for r in done:
        if r["diagnosis"] == "answered":
            continue
        print(f"    {r['question_id']} [{r['question_type']}] {r['diagnosis']} {r.get('flags', [])}: references found "
              f"{r['references_found']}/{r['references']} (first search {r['references_found_first_search']}), "
              f"shown {r['references_shown']}")
        if r["critic"]:
            last = r["critic"][-1]
            print(f"      critic (round {last['round']}): {short(last['reason'])}")
            if last["missing_facts"]:
                print(f"      missing: {short('; '.join(last['missing_facts']))}")
        if r["diagnosis"] == "error":
            print(f"      error: {short(r['answer_text'])} | steps {r['error_steps']} | "
                  f"failed call output tokens {r['error_output_tokens']} | answer output tokens "
                  f"{r['answer_output_tokens']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question-ids", nargs="+", help="Development question IDs (default: the 2026-10-21 record)")
    parser.add_argument("--count", type=int, default=10,
                        help="How many questions (the record's 10, then sampled development questions)")
    parser.add_argument("--arms", default="no reranker,reranker",
                        help='Comma-separated variants: "no reranker", "reranker"')
    parser.add_argument("--max-seconds", type=float,
                        help="Agent time limit for this run (default: RAG_MAX_SECONDS); set it above "
                             "the app's limit to measure how long questions really need")
    parser.add_argument("--modes", default="agentic", help="Comma-separated: agentic, baseline (Fixed RAG)")
    parser.add_argument("--reranker", default="bge-reranker-base", choices=["bge-reranker-base", "bge-reranker-v2-m3"])
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--rerank-max-windows", type=int, default=4)
    parser.add_argument("--budget-usd", type=float, default=1.0, help="Total spending cap for this run")
    parser.add_argument("--output", type=Path, help="Results folder (default .cache/agent-timing/<time>)")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not set(modes) <= {"agentic", "baseline"}:
        parser.error("--modes takes agentic and/or baseline")
    wanted_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not wanted_arms or not set(wanted_arms) <= {"no reranker", "reranker"}:
        parser.error('--arms takes "no reranker" and/or "reranker"')

    os.environ.update(build_environment(ROOT, os.environ, "hybrid"))
    os.environ["RAG_MODE"] = "live"
    os.environ.pop("RAG_RERANKER", None)  # The run itself turns the reranker on and off.
    from src.api.runtime import (MAX_OUTPUT_TOKENS, AnthropicConfig, AnthropicProvider, Runtime,
                                 WebSettings, attach_reranker, load_retriever)
    from src.evaluation.run_retrieval import load_questions

    app_settings = WebSettings.from_env()
    limit = app_settings.max_seconds  # The web app's limit, which results are judged against.
    settings = WebSettings(**{**app_settings.model_dump(), "budget_usd": args.budget_usd,
                              "max_seconds": args.max_seconds or limit,
                              "reranker": ""})  # The "no reranker" arm.
    if not (settings.allow_paid and settings.model and os.environ.get("ANTHROPIC_API_KEY")
            and settings.input_price > 0 and settings.output_price > 0 and settings.demo_access):
        parser.error("Needs live-mode settings in .env: RAG_ALLOW_PAID=true, RAG_MODEL, "
                     "ANTHROPIC_API_KEY, RAG_INPUT_PRICE, RAG_OUTPUT_PRICE, RAG_DEMO_ACCESS=true")
    reranked_settings = WebSettings(**{**settings.model_dump(), "reranker": args.reranker,
                                       "rerank_k": args.rerank_k,
                                       "rerank_max_windows": args.rerank_max_windows})
    questions = {q["question_id"]: q for q in load_questions(
        ROOT / ".cache/corpus/questions.jsonl", ROOT / "evaluation/splits.json", "development")}
    wanted = question_ids(args, questions)
    missing = [q for q in wanted if q not in questions]
    if missing:
        parser.error(f"Not development questions: {missing}")

    print("Loading the indexes and the reranker (a few minutes)...", flush=True)
    plain_retriever = load_retriever(settings)
    reranked_retriever = attach_reranker(plain_retriever, reranked_settings)
    provider = AnthropicProvider(AnthropicConfig(
        model=settings.model, allow_paid=True, budget_usd=args.budget_usd,
        input_usd_per_million=settings.input_price, output_usd_per_million=settings.output_price,
        max_output_tokens=MAX_OUTPUT_TOKENS, timeout_seconds=min(30, settings.max_seconds)))
    arms = {"no reranker": Runtime(settings, plain_retriever, provider),
            "reranker": Runtime(reranked_settings, reranked_retriever, provider)}
    arms = {name: arms[name] for name in arms if name in wanted_arms}
    print(f"Reranker: {args.reranker}, {args.rerank_k} candidates, up to {args.rerank_max_windows} "
          f"windows each, on {settings.device}; model {settings.model}; cap ${args.budget_usd:g}; "
          f"{len(wanted)} questions; agent limit {settings.max_seconds:.0f}s, app limit {limit:.0f}s; "
          f"top {settings.top_k}, {settings.max_rounds} rounds, {settings.max_context_chars} context "
          f"characters, {settings.max_tokens} tokens",
          flush=True)

    results = {mode: {arm: [] for arm in arms} for mode in modes}
    stopped = False
    for index, qid in enumerate(wanted, 1):
        question = questions[qid]["question"]
        for mode in modes:
            # Alternate which variant runs first, so caching does not favour one of them.
            order = list(arms) if index % 2 else list(reversed(arms))
            for arm in order:
                q = questions[qid]
                result = {"question_id": qid, "question_type": q["question_type"],
                          **run_one(arms[arm], question, mode, q["expected_doc_ids"])}
                results[mode][arm].append(result)
                note = result.get("error") or (f"{result['searches']} searches, {result['stop_reason']}, "
                                               f"{result['status']}")
                print(f"{index:>2}/{len(wanted)} {qid} {mode:8} {arm:11} {result['seconds']:6.1f}s  {note}",
                      flush=True)
                stopped = stopped or result.get("error") == "spending_limit"
            if stopped:
                break
        if stopped:
            print("Spending cap reached; stopping early.")
            break

    output = args.output or ROOT / ".cache/agent-timing" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({
        "model": settings.model, "max_seconds": settings.max_seconds, "app_max_seconds": limit,
        "device": settings.device, "top_k": settings.top_k, "max_rounds": settings.max_rounds,
        "max_context_chars": settings.max_context_chars, "max_tokens": settings.max_tokens,
        "reranker": reranked_retriever.manifest["reranker"],
        "rerank_config": reranked_retriever.manifest["config"], "results": results}, indent=2) + "\n")
    print(f"\nApp time limit per question: {limit:.0f}s (agent ran with {settings.max_seconds:.0f}s). "
          f"Results: {output / 'results.json'}")
    for mode in modes:
        print(f"\n{'Adaptive agent' if mode == 'agentic' else 'Fixed RAG'}:")
        for arm, runs in results[mode].items():
            describe(arm, runs, limit, settings.max_seconds)


if __name__ == "__main__":
    main()

"""Live timing run: the web app's agent on development questions, with and without the reranker.

Runs each question through the same runtime the app uses (hybrid search, critic, follow-up
searches, answer), once without and once with the optional reranker, and records the total
time, the number of searches, how each run stopped and its cost. A run longer than the app's
time limit (RAG_MAX_SECONDS, default 120 s) would have timed out in the web app.

**Paid:** it calls the Anthropic API with the key, model and prices in your .env (live-mode
settings; see docs/WEB_APP.md). Expect about $0.02-0.04 per question per variant;
--budget-usd caps the total. Example (about 30-40 minutes on a laptop):

    python scripts/time_agent.py --budget-usd 1

By default it uses the 10 development questions of the 2026-10-21 agent record, so search
counts can be compared with that run. On macOS, set DYLD_LIBRARY_PATH as for the tests first.
"""
import argparse
import json
import os
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


def question_ids(args):
    if args.question_ids:
        return args.question_ids
    record = json.loads(RECORD.read_text())
    return [case["question_id"] for case in record["experiments"][0]["cases"]][: args.count]


def run_one(runtime, question, mode):
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
    return {"seconds": seconds, "searches": sum(s.tool == "text_search" for s in steps),
            "search_seconds": by_tool.get("text_search", 0.0),
            "model_seconds": sum(v for k, v in by_tool.items() if k in ("critic", "generate", "route")),
            "stop_reason": response.trace.stop_reason, "status": response.answer.status,
            "citations": len(response.answer.citation_ids),
            "cost_usd": response.trace.usage.estimated_cost_usd}


def describe(name, runs, limit):
    done = [r for r in runs if "error" not in r]
    if not done:
        print(f"  {name}: no completed runs")
        return
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question-ids", nargs="+", help="Development question IDs (default: the 2026-10-21 record)")
    parser.add_argument("--count", type=int, default=10, help="How many of the default questions")
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

    os.environ.update(build_environment(ROOT, os.environ, "hybrid"))
    os.environ["RAG_MODE"] = "live"
    os.environ.pop("RAG_RERANKER", None)  # The run itself turns the reranker on and off.
    from src.api.runtime import (AnthropicConfig, AnthropicProvider, Runtime, WebSettings,
                                 attach_reranker, load_retriever)
    from src.evaluation.run_retrieval import load_questions

    settings = WebSettings(**{**WebSettings.from_env().model_dump(), "budget_usd": args.budget_usd})
    if not (settings.allow_paid and settings.model and os.environ.get("ANTHROPIC_API_KEY")
            and settings.input_price > 0 and settings.output_price > 0 and settings.demo_access):
        parser.error("Needs live-mode settings in .env: RAG_ALLOW_PAID=true, RAG_MODEL, "
                     "ANTHROPIC_API_KEY, RAG_INPUT_PRICE, RAG_OUTPUT_PRICE, RAG_DEMO_ACCESS=true")
    reranked_settings = WebSettings(**{**settings.model_dump(), "reranker": args.reranker,
                                       "rerank_k": args.rerank_k,
                                       "rerank_max_windows": args.rerank_max_windows})
    wanted = question_ids(args)
    questions = {q["question_id"]: q for q in load_questions(
        ROOT / ".cache/corpus/questions.jsonl", ROOT / "evaluation/splits.json", "development")}
    missing = [q for q in wanted if q not in questions]
    if missing:
        parser.error(f"Not development questions: {missing}")

    print("Loading the indexes and the reranker (a few minutes)...", flush=True)
    plain_retriever = load_retriever(settings)
    reranked_retriever = attach_reranker(plain_retriever, reranked_settings)
    provider = AnthropicProvider(AnthropicConfig(
        model=settings.model, allow_paid=True, budget_usd=args.budget_usd,
        input_usd_per_million=settings.input_price, output_usd_per_million=settings.output_price,
        max_output_tokens=1024, timeout_seconds=min(30, settings.max_seconds)))
    arms = {"no reranker": Runtime(settings, plain_retriever, provider),
            "reranker": Runtime(reranked_settings, reranked_retriever, provider)}
    print(f"Reranker: {args.reranker}, {args.rerank_k} candidates, up to {args.rerank_max_windows} "
          f"windows each, on {settings.device}; model {settings.model}; cap ${args.budget_usd:g}", flush=True)

    results = {mode: {arm: [] for arm in arms} for mode in modes}
    stopped = False
    for index, qid in enumerate(wanted, 1):
        question = questions[qid]["question"]
        for mode in modes:
            # Alternate which variant runs first, so caching does not favour one of them.
            order = list(arms) if index % 2 else list(reversed(arms))
            for arm in order:
                result = {"question_id": qid, **run_one(arms[arm], question, mode)}
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
        "model": settings.model, "max_seconds": settings.max_seconds, "device": settings.device,
        "reranker": reranked_retriever.manifest["reranker"],
        "rerank_config": reranked_retriever.manifest["config"], "results": results}, indent=2) + "\n")
    print(f"\nTime limit per question: {settings.max_seconds:.0f}s. Results: {output / 'results.json'}")
    for mode in modes:
        print(f"\n{'Adaptive agent' if mode == 'agentic' else 'Fixed RAG'}:")
        for arm, runs in results[mode].items():
            describe(arm, runs, settings.max_seconds)


if __name__ == "__main__":
    main()

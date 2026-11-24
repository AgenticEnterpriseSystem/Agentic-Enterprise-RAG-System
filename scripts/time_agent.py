"""Live experiment run: the web app's agent on development questions (hira/testing).

Runs each question through the same runtime the app uses (hybrid search, checklist critic,
read-more and follow-up searches, answer) and records the total time, the number of searches,
how each run stopped, the answer and its cost. Afterwards the answers are graded against the
gold answers (scripts/grade_answers.py) unless --no-grade is given.

--config picks one of the planned experiment configurations (see CONFIGS below and
docs/CRITIC_EVIDENCE_EXPERIMENTS.md). It sets every experiment setting explicitly, so values
in .env or the shell cannot leak into a run:

    python scripts/time_agent.py --config A --count 95 --start 70 --max-seconds 200 --budget-usd 2
    python scripts/time_agent.py --config 4 --count 95 --start 70 --max-seconds 200 --budget-usd 2

To measure how much time questions really need, give the agent more room than the app's limit
with --max-seconds (at most 300): every question then runs to its natural end, and the report
says how many would exceed the app's limit and by how much. Up to the app's limit, the agent
takes the same steps either way.

**Paid:** it calls the Anthropic API with the key, model and prices in your .env (live-mode
settings; see docs/WEB_APP.md). Expect about $0.02-0.04 per question per variant;
--budget-usd caps the total. Examples:

    python scripts/time_agent.py --budget-usd 1                     # 10 questions, current settings

The first 10 questions are those of the 2026-10-21 agent record, so search counts can be
compared with that run; --count above 10 adds development questions sampled with seed 5980.
On macOS, set DYLD_LIBRARY_PATH as for the tests first.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_web import build_environment  # noqa: E402

RECORD = ROOT / "evaluation/agent_development/20261021.json"

# Planned configurations (docs/CRITIC_EVIDENCE_EXPERIMENTS.md). "A" is main as released; every
# other entry changes only the settings it lists. Longer excerpts need a larger context and
# token allowance, or most of each excerpt would be cropped: 8 x 6,000 = 48,000 characters.
MAIN = {"RAG_PASSAGE_CHARS": "", "RAG_READ_MORE_CHARS": "3000", "RAG_CLARIFICATION": "true",
        "RAG_MATCHING_SOURCE": "false", "RAG_EVIDENCE_DOC_LIMIT": "8",
        "RAG_MODEL_MAX_OUTPUT_TOKENS": "1024", "RAG_CRITIC_MAX_OUTPUT_TOKENS": "2048",
        "RAG_MAX_TOKENS": "16000", "RAG_MAX_CONTEXT_CHARS": "24000", "RAG_MAX_ROUNDS": "2",
        "RAG_TOP_K": "5"}
LONG = {"RAG_PASSAGE_CHARS": "6000", "RAG_MAX_CONTEXT_CHARS": "48000", "RAG_MAX_TOKENS": "32000"}
BEST = {**LONG, "RAG_MATCHING_SOURCE": "true", "RAG_MODEL_MAX_OUTPUT_TOKENS": "2048"}
CONFIGS = {
    "A": ("main as is", {}),
    "1": ("A + answer output 2,048", {"RAG_MODEL_MAX_OUTPUT_TOKENS": "2048"}),
    "2": ("A + 5 documents", {"RAG_EVIDENCE_DOC_LIMIT": "5"}),
    "3": ("A + read-more 6,000", {"RAG_READ_MORE_CHARS": "6000"}),
    "4": ("A + 6,000-char excerpts", LONG),
    "5": ("A + 6,000-char excerpts, no read-more", {**LONG, "RAG_READ_MORE_CHARS": "0"}),
    "6": ("A + matching-source rule", {"RAG_MATCHING_SOURCE": "true"}),
    "7": ("6,000 + matching source + 2,048 out", BEST),
    "8": ("7 with 5 documents", {**BEST, "RAG_EVIDENCE_DOC_LIMIT": "5"}),
    "9": ("7 without clarification", {**BEST, "RAG_CLARIFICATION": "false"}),
}


def question_ids(args, development):
    """The record's questions first, then a fixed sample of other development questions."""
    if args.question_ids:
        return args.question_ids
    record = json.loads(RECORD.read_text())
    chosen = [case["question_id"] for case in record["experiments"][0]["cases"]][: args.count]
    others = sorted(set(development) - set(chosen))
    random.Random(5980).shuffle(others)
    return (chosen + others[: max(0, args.count - len(chosen))])[args.start:]


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
    elif response.trace.stop_reason == "clarification":
        label = "clarification"  # The agent asked the user to clarify (jane/dev's agent).
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
              "read_more": sum(s.tool == "read_more" for s in steps),
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
    print("    clarification = the agent asked the user to clarify (jane/dev); "
          "no_answer_expected = the question has no reference documents (insufficient is right); "
          "never_found = no reference document in any search; not_shown = retrieved but not in "
          "the critic's context; judged_insufficient = the critic saw one and asked for more")
    names = ["answered", "no_answer_expected", "never_found", "not_shown", "judged_insufficient", "error"]
    if any(r["diagnosis"] == "clarification" for r in done):
        names.insert(1, "clarification")
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
    parser.add_argument("--config", choices=list(CONFIGS),
                        help="A planned experiment configuration: " + "; ".join(
                            f"{k} = {v[0]}" for k, v in CONFIGS.items()))
    parser.add_argument("--label", help="Name of this run in reports (default: the configuration)")
    parser.add_argument("--no-grade", action="store_true", help="Skip grading the answers afterwards")
    parser.add_argument("--grade-budget-usd", type=float, default=0.3, help="Spending cap for grading")
    parser.add_argument("--question-ids", nargs="+", help="Development question IDs (default: the 2026-10-21 record)")
    parser.add_argument("--count", type=int, default=10,
                        help="How many questions (the record's 10, then sampled development questions)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip this many of the selected questions (e.g. --count 120 --start 70 "
                             "runs only the 50 after the first 70)")
    parser.add_argument("--arms", default="no reranker",
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
    label = args.label
    if args.config:
        name, changes = CONFIGS[args.config]
        os.environ.update({**MAIN, **changes})
        label = label or f"{args.config}: {name}"
    os.environ["RAG_MODE"] = "live"
    os.environ.pop("RAG_RERANKER", None)  # The run itself turns the reranker on and off.
    import src.api.runtime as runtime_module
    from src.api.runtime import AnthropicConfig, AnthropicProvider, Runtime, WebSettings, load_retriever
    from src.evaluation.run_retrieval import load_questions

    # The same script runs on other branches' runtimes (e.g. jane/dev, which has no reranker
    # option, its own output limits and a "clarification" ending); features are detected.
    attach_reranker = getattr(runtime_module, "attach_reranker", None)
    fields = WebSettings.model_fields
    if "reranker" in wanted_arms and attach_reranker is None:
        parser.error('This branch\'s runtime has no reranker option; use --arms "no reranker"')
    app_settings = WebSettings.from_env()
    limit = app_settings.max_seconds  # The web app's limit, which results are judged against.
    settings = WebSettings(**{**app_settings.model_dump(), "budget_usd": args.budget_usd,
                              "max_seconds": args.max_seconds or limit,
                              **({"reranker": ""} if "reranker" in fields else {})})  # No reranker.
    if not (settings.allow_paid and settings.model and os.environ.get("ANTHROPIC_API_KEY")
            and settings.input_price > 0 and settings.output_price > 0 and settings.demo_access):
        parser.error("Needs live-mode settings in .env: RAG_ALLOW_PAID=true, RAG_MODEL, "
                     "ANTHROPIC_API_KEY, RAG_INPUT_PRICE, RAG_OUTPUT_PRICE, RAG_DEMO_ACCESS=true")
    reranked_settings = (WebSettings(**{**settings.model_dump(), "reranker": args.reranker,
                                        "rerank_k": args.rerank_k,
                                        "rerank_max_windows": args.rerank_max_windows})
                         if "reranker" in wanted_arms else None)
    questions = {q["question_id"]: q for q in load_questions(
        ROOT / ".cache/corpus/questions.jsonl", ROOT / "evaluation/splits.json", "development")}
    wanted = question_ids(args, questions)
    missing = [q for q in wanted if q not in questions]
    if missing:
        parser.error(f"Not development questions: {missing}")

    print("Loading the indexes" + (" and the reranker" if reranked_settings else "")
          + " (a few minutes)...", flush=True)
    plain_retriever = load_retriever(settings)
    reranked_retriever = attach_reranker(plain_retriever, reranked_settings) if reranked_settings else None
    # Output tokens per model call: this branch's constant, or the setting other branches use.
    max_output = getattr(runtime_module, "MAX_OUTPUT_TOKENS", None) or getattr(
        settings, "model_max_output_tokens", 1024)
    provider = AnthropicProvider(AnthropicConfig(
        model=settings.model, allow_paid=True, budget_usd=args.budget_usd,
        input_usd_per_million=settings.input_price, output_usd_per_million=settings.output_price,
        max_output_tokens=max_output, timeout_seconds=min(30, settings.max_seconds)))
    arms = {"no reranker": Runtime(settings, plain_retriever, provider)}
    if reranked_settings:
        arms["reranker"] = Runtime(reranked_settings, reranked_retriever, provider)
    arms = {name: arms[name] for name in arms if name in wanted_arms}
    passage_chars = getattr(settings, "passage_chars", None)
    extras = {name: getattr(settings, name) for name in (
        "evidence_doc_limit", "critic_max_output_tokens") if name in fields}
    experiment = {name: getattr(settings, name, None) for name in (
        "passage_chars", "read_more_chars", "clarification", "matching_source")}
    print(f"Run: {label or 'current settings'}; " + ", ".join(f"{k} {v}" for k, v in experiment.items()))
    print((f"Reranker: {args.reranker}, {args.rerank_k} candidates, up to {args.rerank_max_windows} "
           "windows each" if reranked_settings else "No reranker")
          + f", on {settings.device}; model {settings.model}; cap ${args.budget_usd:g}; "
          f"{len(wanted)} questions; agent limit {settings.max_seconds:.0f}s, app limit {limit:.0f}s; "
          f"top {settings.top_k}, {settings.max_rounds} rounds, {settings.max_context_chars} context "
          f"characters, {settings.max_tokens} tokens, {max_output} output tokens per call, excerpts "
          f"{passage_chars or 'default (3,000 BM25+ / BGE passage)'}"
          + "".join(f", {k} {v}" for k, v in extras.items()), flush=True)

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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / ".cache/agent-timing" / (
        stamp + (f"-config-{args.config}" if args.config else ""))
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({
        "label": label, "config": args.config, **experiment,
        "question_ids": wanted, "model": settings.model, "max_seconds": settings.max_seconds, "app_max_seconds": limit,
        "device": settings.device, "top_k": settings.top_k, "max_rounds": settings.max_rounds,
        "max_context_chars": settings.max_context_chars, "max_tokens": settings.max_tokens,
        "passage_chars": passage_chars, "max_output_tokens": max_output, **extras,
        "code_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip() or None,
        "reranker": reranked_retriever.manifest["reranker"] if reranked_retriever else None,
        "rerank_config": (reranked_retriever or plain_retriever).manifest.get("config"),
        "results": results}, indent=2) + "\n")
    print(f"\nApp time limit per question: {limit:.0f}s (agent ran with {settings.max_seconds:.0f}s). "
          f"Results: {output / 'results.json'}")
    for mode in modes:
        print(f"\n{'Adaptive agent' if mode == 'agentic' else 'Fixed RAG'}:")
        for arm, runs in results[mode].items():
            describe(arm, runs, limit, settings.max_seconds)
    if not args.no_grade:
        import grade_answers
        print("\nGrading the answers against the gold answers...")
        data, records, grades, cost = grade_answers.grade_file(
            output / "results.json", {qid: q for qid, q in questions.items()},
            lambda: grade_answers.make_judge(args.grade_budget_usd))
        c = grade_answers.summarize(records, grades)
        print(f"{label or 'This run'}: {c['questions']} questions | answered {c['answered']}: correct "
              f"{c['correct']}, partly {c['partly']}, wrong {c['wrong']} | clarification "
              f"{c['clarification']}, insufficient {c['insufficient']}, error {c['error']} | "
              f"grading ${cost:.3f}")
        print("Compare runs: python scripts/grade_answers.py .cache/agent-timing/*/results.json --by-question")


if __name__ == "__main__":
    main()

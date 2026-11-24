"""Grade agent answers against the gold answers and compare experiment runs side by side.

Reads one or more `results.json` files written by scripts/time_agent.py. Each answered
question is graded by a model judge against the question's gold answer:

    correct  every central fact of the gold answer is in the answer, and nothing contradicts it
    partly   the main answer is right, but a substantive detail is missing or wrong
    wrong    the central claim contradicts the gold answer, or the answer is about a different
             customer, incident, project or design than the one asked about

Unanswered questions are not sent to the judge; they count by how the run ended
(clarification, insufficient evidence, error). Grades are saved in `grades.json` next to each
results file, so re-running the comparison costs nothing; --regrade grades again.

**Paid** (small): one call per answered question with the model in .env, about $0.002-0.005
each. Examples:

    python scripts/grade_answers.py .cache/agent-timing/*/results.json
    python scripts/grade_answers.py run-A/results.json run-4/results.json --by-question
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

VERDICTS = ("correct", "partly", "wrong")
JUDGE_PROMPT = """You grade a question-answering system against a reference (gold) answer.
Compare the generated answer's substantive claims with the reference answer only; do not use
outside knowledge, and ignore citation markers, style and length.

- correct: the answer states every central fact of the reference answer (values, names,
  dates, decisions, causes) and contradicts none of it. Extra correct detail is fine.
- partly: the main point matches the reference, but a substantive requested detail is
  missing, or one secondary detail is wrong.
- wrong: the central claim contradicts or does not match the reference, or the answer is
  about a different customer, incident, project or design than the reference answer.

Return one JSON object: {"verdict": "correct" | "partly" | "wrong", "reason": "<one sentence>"}."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
          "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)},
                         "reason": {"type": "string"}}}


def load_gold():
    from src.evaluation.run_retrieval import load_questions

    questions = load_questions(ROOT / ".cache/corpus/questions.jsonl", ROOT / "evaluation/splits.json",
                               "development")
    return {q["question_id"]: q for q in questions}


def make_judge(budget_usd):
    """The Anthropic provider of the live settings in .env, capped at `budget_usd`."""
    from run_web import build_environment

    os.environ.update(build_environment(ROOT, os.environ, "hybrid"))
    from src.api.runtime import AnthropicConfig, AnthropicProvider, WebSettings

    settings = WebSettings.from_env()
    if not (settings.allow_paid and settings.model and os.environ.get("ANTHROPIC_API_KEY")
            and settings.input_price > 0 and settings.output_price > 0):
        sys.exit("Grading needs live-mode settings in .env: RAG_ALLOW_PAID=true, RAG_MODEL, "
                 "ANTHROPIC_API_KEY, RAG_INPUT_PRICE, RAG_OUTPUT_PRICE")
    return AnthropicProvider(AnthropicConfig(
        model=settings.model, allow_paid=True, budget_usd=budget_usd, max_output_tokens=300,
        input_usd_per_million=settings.input_price, output_usd_per_million=settings.output_price,
        timeout_seconds=60))


def grade_one(judge, question, answer):
    prompt = json.dumps({"question": question["question"], "reference_answer": question["gold_answer"],
                         "generated_answer": answer})
    completion = judge.complete_structured(JUDGE_PROMPT, prompt, SCHEMA)
    value = json.loads(completion.text)
    if value.get("verdict") not in VERDICTS:
        raise ValueError(f"Judge returned {value!r}")
    return value["verdict"], value.get("reason", ""), completion.usage.estimated_cost_usd or 0


def answer_key(record):
    return hashlib.sha256(f"{record['question_id']}\n{record.get('answer_text', '')}".encode()).hexdigest()[:16]


def outcome(record):
    """How a run ended, for questions the judge does not grade."""
    if "error" in record or record.get("diagnosis") == "error":
        return "error"
    if record.get("status") == "answered":
        return None
    return "clarification" if record.get("stop_reason") == "clarification" else "insufficient"


def grade_file(path, gold, judge_factory, regrade=False):
    """Grade every answered record in one results.json; returns (records, grades, judge cost)."""
    data = json.loads(path.read_text())
    records = [r for mode in data["results"].values() for arm in mode.values() for r in arm]
    grades_path = path.parent / "grades.json"
    grades = {} if regrade or not grades_path.exists() else json.loads(grades_path.read_text())
    cost = 0.0
    for record in records:
        key = answer_key(record)
        if outcome(record) is not None or key in grades:
            continue
        question = gold.get(record["question_id"])
        if question is None or not question.get("gold_answer"):
            grades[key] = {"question_id": record["question_id"], "verdict": "ungraded",
                           "reason": "No gold answer for this question."}
            continue
        verdict, reason, spent = grade_one(judge_factory(), question, record.get("answer_text", ""))
        cost += spent
        grades[key] = {"question_id": record["question_id"], "verdict": verdict, "reason": reason}
        print(f"  {path.parent.name} {record['question_id']}: {verdict} - {reason}", flush=True)
        grades_path.write_text(json.dumps(grades, indent=1) + "\n")
    grades_path.write_text(json.dumps(grades, indent=1) + "\n")
    return data, records, grades, cost


def label_of(data, path):
    return data.get("label") or path.parent.name


def summarize(records, grades):
    counts = {name: 0 for name in (*VERDICTS, "ungraded", "clarification", "insufficient", "error")}
    for record in records:
        kind = outcome(record) or grades.get(answer_key(record), {}).get("verdict", "ungraded")
        counts[kind] += 1
    done = [r for r in records if "seconds" in r]
    counts["questions"] = len(records)
    counts["answered"] = sum(r.get("status") == "answered" for r in records)
    counts["cited_reference"] = sum(r.get("status") == "answered" and r.get("references_cited", 0) > 0
                                    for r in records)
    counts["median_s"] = statistics.median(r["seconds"] for r in done) if done else 0
    counts["cost"] = sum(r.get("cost_usd") or 0 for r in records)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="results.json files from time_agent.py")
    parser.add_argument("--budget-usd", type=float, default=0.5, help="Spending cap for grading calls")
    parser.add_argument("--regrade", action="store_true", help="Ignore saved grades and grade again")
    parser.add_argument("--by-question", action="store_true", help="Also print each question's outcome per run")
    args = parser.parse_args()

    gold = load_gold()
    judge = []

    def judge_factory():  # Load the provider only when something needs grading.
        if not judge:
            judge.append(make_judge(args.budget_usd))
        return judge[0]

    runs, total = [], 0.0
    for path in args.results:
        data, records, grades, cost = grade_file(path, gold, judge_factory, args.regrade)
        total += cost
        runs.append((label_of(data, path), data, records, grades))

    print(f"\nGrading cost this time: ${total:.3f}\n")
    header = (f"{'run':34} {'n':>3} {'answered':>8} {'correct':>7} {'partly':>6} {'wrong':>5} "
              f"{'clarify':>7} {'insuff.':>7} {'error':>5} {'cited ref':>9} {'median s':>8} {'cost':>7}")
    print(header)
    print("-" * len(header))
    for label, data, records, grades in runs:
        c = summarize(records, grades)
        print(f"{label[:34]:34} {c['questions']:>3} {c['answered']:>8} {c['correct']:>7} {c['partly']:>6} "
              f"{c['wrong']:>5} {c['clarification']:>7} {c['insufficient']:>7} {c['error']:>5} "
              f"{c['cited_reference']:>9} {c['median_s']:>8.1f} ${c['cost']:>6.3f}")
    print("\ncorrect/partly/wrong: the judge's grade of answered questions against the gold answer; "
          "cited ref: answers citing at least one reference document.")

    if args.by_question:
        short = {"correct": "OK", "partly": "part", "wrong": "WRONG", "clarification": "clar",
                 "insufficient": "-", "error": "err", "ungraded": "?"}
        ids = list(dict.fromkeys(r["question_id"] for _, _, records, _ in runs for r in records))
        print(f"\n{'question':12} " + " ".join(f"{label[:10]:>10}" for label, *_ in runs))
        for qid in ids:
            cells = []
            for _, _, records, grades in runs:
                record = next((r for r in records if r["question_id"] == qid), None)
                kind = (None if record is None
                        else outcome(record) or grades.get(answer_key(record), {}).get("verdict", "ungraded"))
                cells.append(short.get(kind, "") if kind else "")
            print(f"{qid:12} " + " ".join(f"{cell:>10}" for cell in cells))


if __name__ == "__main__":
    main()

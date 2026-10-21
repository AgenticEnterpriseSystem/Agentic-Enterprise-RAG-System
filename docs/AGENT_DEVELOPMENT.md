# Agent development checks — October 21, 2026

The agent now supports fixed-search comparison, interactive inspection, partial
answers after normal search stops, and balanced passage selection across original
and follow-up searches. Citation validation and provider budget checks remain
active. This work does not establish better overall answer quality.

## What changed

- Separate critic document descriptions from platform source filters. Invalid
  suggestions retain the requested scope; descriptions can become search terms.
- Add `--workflow baseline|agent|compare`. Both arms use the same retriever, model,
  generator prompt, question sample and context ceiling. The fixed arm skips the
  critic; the agent may search again. Errors/budget stops preserve completed logs.
- At normal round/no-new-evidence stops, attempt a cited partial answer if evidence
  and budget remain. Keep `insufficient_evidence` and the search stop reason.
- Reserve answer room by bounding final critic context. Partial context sizing is
  heuristic; exact provider token preflight still enforces the per-question cap.
- Share multi-round context space between original and relevant follow-up evidence.
  Rank by question-term coverage, title/acronym matches and document diversity.
  Preserve a lead plus a query window in long passages with exact offsets.
- Accept one whole-response JSON code fence, then apply the existing strict schema
  and citation checks. Extra prose, malformed responses and unknown citations fail.

## Saved measurements

[Compact results](../evaluation/agent_development/20261021.json) record question IDs,
model, settings, runtime versions, source/prompt hashes, statuses, usage and timing.
Full passages and case traces stay in local ignored storage. The runs were made in
an isolated integration checkout before these agent changes were ported to
`jane/dev`; the compact results identify the measured snapshots.

All live comparisons used **BM25 over 511,958 documents and Haiku 4.5**, not
GPT-OSS or hybrid retrieval. Reference answers were used only for review, never
as runtime agent inputs. Alex's saved 20-question GPT-OSS experiment informed
initial debugging cases; his evaluator did not score these comparison runs.

- Eight synthetic critic fixtures: old prompt matched 5/8 AI-proposed sufficiency
  labels; revised prompt matched 8/8. These were development examples, not human
  labels, independent accuracy measurement, or formal calibration.
- Ten development pairs selected by deterministic category round-robin: follow-up
  searches found additional labeled reference documents in two cases, but both
  were omitted from the final context before the selection fix. Estimated serving
  cost: fixed $0.061950, agent $0.159843. Mean observed latency: 5.13s versus 14.54s.
  Baseline ran first; cache effects limit timing interpretation. Answered status
  is not a correctness score.
- Two-case selection regression: both added reference documents reached the
  generator. `qst_0384` recovered `budget_ms_per_step` and `budget_bytes_per_step`
  in priority order, but adjustment amounts remained missing. `qst_0342` produced
  an unsupplied citation ID and was rejected. That error is not a quality win.

Independent human review, calibration, hybrid answer comparison and final-split
experiments remain pending. Hira's newer BGE-base loader/index work and Alex's
Ollama evaluator remain on their separate branches; they are not imported by this
agent-only change. The hybrid weight option is configurable; selecting 0.7 does
not reproduce Hira's BGE-base configuration with Jane's BGE-small index.

## Run checks and comparisons

Offline checks make no model calls:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m src.agents.pilot
.venv/bin/python -m src.agents.pilot --interactive
```

To create a fresh paired development run, configure `ANTHROPIC_API_KEY` in the
process environment, use current model pricing, and choose a new output directory:

```sh
.venv/bin/python -m src.agents.pilot --live --workflow compare --count 3 \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS \
  --input-price INPUT_DOLLARS_PER_MILLION \
  --output-price OUTPUT_DOLLARS_PER_MILLION \
  --output .cache/agent-pilot/NEW_RUN_NAME
```

The total dollar cap is shared by both arms. Default limits are two search rounds,
16,000 tokens and 120 seconds per question, five results per search, and 24,000
context characters. Model output is capped at 1,024 tokens. `--count` allows 1–10;
`--retrieval` defaults to BM25. Outputs include per-arm cases/review files,
`comparison.json`, and a manifest. Review errors, supplied passages and citations;
new evidence IDs alone do not prove better answers. This generic command selects
a category round-robin sample; the earlier two-case run used a local derived
subset whose question IDs and hash are recorded in the compact results.

Remaining engineering issues include the DEK citation failure, missing passage
sections such as compaction adjustment amounts, and overcautious/conflicting
policy interpretation. Keep those limitations visible when sharing results.

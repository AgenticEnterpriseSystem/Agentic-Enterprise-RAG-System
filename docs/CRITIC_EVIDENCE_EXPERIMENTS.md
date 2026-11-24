# Critic and evidence experiments (`hira/testing`)

`main` (Jane's release candidate) and `hira/dev` differ mainly in the **critic** and the
**evidence** it reads. This branch is `main` plus switches for each difference, so that every
combination can be run on the same questions with the same code. **Every default keeps
`main`'s behaviour**; nothing here changes the released system until a result is adopted.

## What this branch adds to `main`

| Switch (environment variable) | Default (= `main`) | What it changes |
|---|---|---|
| `RAG_PASSAGE_CHARS` | unset (3,000-character BM25+ slice or BGE passage) | Each returned search result is re-cut from the whole document to this many characters around the match (`hira/dev`'s 6,000-character excerpts) |
| `RAG_READ_MORE_CHARS` | 3000 | Characters per read-more expansion (at most 6,000); `0` turns read-more off |
| `RAG_CLARIFICATION` | true | `false`: an ambiguous question keeps searching / ends as insufficient instead of asking the user |
| `RAG_MATCHING_SOURCE` | false | `true`: adds a rule to the checklist critic: a passage about a similar but *different* customer, incident, project or design never supports a fact |
| Spending reservation | — | Bug fix from `hira/dev`: each call's worst-case reservation is settled to its actual cost, so long runs no longer hit the spending cap early |

Settings that `main` already had are used as they are: `RAG_EVIDENCE_DOC_LIMIT` (5/8/10),
`RAG_MODEL_MAX_OUTPUT_TOKENS`, `RAG_MAX_TOKENS`, `RAG_MAX_CONTEXT_CHARS`.

Tools: `scripts/time_agent.py --config <run>` sets every experiment setting for one planned run
(so `.env` or shell values cannot leak in), runs the questions, and grades the answers;
`scripts/grade_answers.py` grades and compares any number of runs.

## Grading

A model judge compares each **answered** question with its gold answer:
**correct** (all central facts, nothing contradicted), **partly** (main point right, a
substantive detail missing or wrong) or **wrong** (contradicts the gold answer, or is about a
different customer/incident/design). Unanswered questions are counted by how they ended:
clarification, insufficient evidence or error. Grades are saved next to each `results.json`.
The judge costs about $0.002–0.005 per answered question.

**Decision rule:** prefer the configuration with the most correct answers whose wrong-answer
rate is no worse than `hira/dev`'s (2 of 25). A confident wrong answer counts as worse than
"insufficient evidence". Runs vary by 2–3 answers per 25 questions, so Phase 1 only screens.

## The runs

All runs use the **25 fresh development questions** (`--count 95 --start 70`) and the same
200-second agent limit (`--max-seconds 200`), so no configuration is cut off by time.

**Phase 1: one change at a time on top of `main`.**

| Run | Configuration | Question it answers |
|---|---|---|
| A | `main` as is | Baseline (re-run under the same limit and grading) |
| B | `hira/dev` as is | Reference: already run; grade its existing `results.json` |
| 1 | A + answer output 2,048 tokens | Were answers cut off? |
| 2 | A + 5 documents | Does 8 vs 5 documents matter for quality or cost? |
| 3 | A + read-more up to 6,000 characters | Does reading more on demand match longer excerpts? |
| 4 | A + 6,000-character excerpts (48,000-character context, 32,000 tokens) | Do longer excerpts help the checklist critic? |
| 5 | Run 4 with read-more off | Can longer excerpts replace read-more? |
| 6 | A + matching-source rule | Fewer wrong answers without losing correct ones? |

Runs 4, 5, 7, 8 and 9 raise the context to 48,000 characters and the token allowance to 32,000:
with 8 documents of 6,000 characters, `main`'s 24,000-character context would crop most of each
excerpt.

**Phase 2: combinations** (adjust after Phase 1).

| Run | Configuration |
|---|---|
| 7 | 6,000-character excerpts + matching-source rule + 2,048 output tokens |
| 8 | Run 7 with 5 documents |
| 9 | Run 7 without clarification |

**Phase 3:** the best 2–3 configurations on 75–100 development questions not used so far
(`--count 195 --start 95`). The final split stays untouched for one last check.

## Commands

From the repository root on `hira/testing`, with live-mode settings in `.env` (on macOS, set
`DYLD_LIBRARY_PATH` first, as for the tests):

```bash
git fetch origin && git checkout hira/testing && git pull
python scripts/time_agent.py --config A --count 95 --start 70 --max-seconds 200 --budget-usd 2
python scripts/time_agent.py --config 1 --count 95 --start 70 --max-seconds 200 --budget-usd 2
# ... configs 2 to 6 the same way

# Grade hira/dev's earlier 25-question run (B) and compare everything:
python scripts/grade_answers.py <hira/dev run>/results.json .cache/agent-timing/*-config-*/results.json --by-question
```

Each run writes `.cache/agent-timing/<time>-config-<run>/results.json` and `grades.json`.

## Results

To be filled in as runs finish.

| Run | Answered | Correct | Partly | Wrong | Clarification | Insufficient | Error | Median time | Cost |
|---|---|---|---|---|---|---|---|---|---|
| A | | | | | | | | | |
| B | | | | | | | | | |
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 3 | | | | | | | | | |
| 4 | | | | | | | | | |
| 5 | | | | | | | | | |
| 6 | | | | | | | | | |

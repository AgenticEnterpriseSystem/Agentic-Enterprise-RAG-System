# Evaluation

## What works now

- Retrieval scoring: Recall@5/10/20 and nDCG, overall and by question/source slice.
- Offline tests: schema, citation references, permissions, and agent behavior.
- A fixed 20-question development comparison: selected evidence, context size and
  search timing; a separate dense-load check records process peak memory.

**C3 LLM answer evaluation is paused.** Answer correctness, completeness and
claim-level citation support have not been established by these checks.

## Commands and outputs

Use the [README everyday commands](../README.md#everyday-commands) for tests,
BM25 readiness and `python -m src.evaluation.compare_fixes --label my-check`.
Use a fresh label; outputs stay in `.cache/fix-comparison/`. Add `--memory` for
an isolated full dense-index load without a query model. `--legacy-ties` only
reproduces the earlier cutoff-rule comparison.

Full-development retrieval uses `python -m src.evaluation.run_retrieval`; see
[README setup](../README.md#quick-start). Pass `--output` with a fresh directory,
as the default paths target historical results under `evaluation/results/`.
Evaluation code lives in `src/evaluation/`; frozen inputs/results in `evaluation/`;
correctness tests in `tests/`; large local artifacts and new outputs in `.cache/`.

Reference-document retention only checks IDs, not whether passages support answers.
Single-run timing/memory observations depend on host load and filesystem caches.
The offline comparison makes no LLM calls, downloads or index builds.

## Frozen splits

| Split | Questions | Purpose |
| --- | ---: | --- |
| Development | 300 | Debugging, baselines, model selection |
| Calibration | 100 | Critic thresholds and sufficiency labels |
| Final | 100 | Evaluate only after settings are frozen |

IDs are in `evaluation/splits.json`, seed 5980. Shared references or identical
normalized questions stay together, including transitive overlap. Grouping takes
priority over the 60/20/20 category balance; projects/near-duplicates may still
cross splits. The separate regression set has 60 questions, six per category;
the comparison command uses its own fixed 20-question sample. Do not tune on final
labels. Documents form one shared corpus; gold answers never enter runtime inputs.

## Comparison and scoring rules

- Keep corpus, questions, access policy, top-k and context limits equal. For
  answer comparisons, also match the generator and prompt. Record code, model,
  package versions, corpus hash, seed and settings; count retries once.
- Deduplicate chunks by document ID before scoring ranks. Recall is retrieved
  references divided by labeled references; nDCG uses binary relevance discounted
  by `log2(rank + 1)` and normalizes by the ideal ranking.
- No reference IDs means N/A for retrieval, with exclusion counts. Empty labels
  do not establish unanswerability. Keep missing corpus references in denominators.
  Eligible retrieval errors score zero; count errors and timeouts.
- Separate synthetic access tests from unrestricted benchmark scores. Schema or
  citation-reference validity does not establish correctness; access leaks fail.
- Report sample counts, overall/category/source scores and median/p95 latency,
  calls, rounds, tokens and cost. Source slices can overlap. Use paired comparisons
  and confidence intervals; avoid strong conclusions from small samples. Unknown
  cost is null, and judge costs are separate from serving costs.

<details>
<summary>Planned C3 answer rubric and review — paused</summary>

Planned measures: correctness (no substantive factual errors), completeness
(reference facts covered), citation support (supported claims divided by claims),
and appropriate abstention. Unjustified refusal fails correctness. No answer facts
means N/A for completeness; no factual claims means N/A for citation support.
Report correct and unnecessary abstentions separately.

Score each applicable dimension separately from 0 to 2. A usable answer needs 2
on each; an average should not conceal factual errors.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim | Main answer right, substantive detail wrong or unverifiable | All substantive claims correct |
| Completeness | Main information missing | Required details missing | Required facts, constraints, and conflicts covered |
| Citation support | Central claims unsupported | Some claims unsupported or ambiguously cited | Every factual claim supported by its citation |
| Abstention/limits | Unsupported answer or unjustified refusal | Uncertainty noted, key limitation missed | Answers when justified; explains missing/conflicting evidence otherwise |

Two reviewers independently score 20 development examples, including successes,
failures, and boundary cases. Record agreement by dimension, resolve differences,
and keep the adjudicated examples as an anchor set. Compare any LLM judge against
these labels; record its model, prompt, and reasons. Proposed acceptance: 80%
exact agreement per dimension before scaling up. This small-sample threshold is
for calibration, not proof of reliability. The Search Critic must not be its own
sole evaluator.

## Error analysis

Check schema validity, citation references, and permissions separately from answer
quality. Any invalid citation or access leak fails its check. Access tests use
synthetic permissions, separate from benchmark scores.

Inspect 20–30 failed development examples, or all if fewer. Record question/run
IDs, ranked evidence, selected context, answer, failed dimension, suspected cause,
proposed fix, and owner. Identify three to five recurring patterns when supported:
missing corpus evidence, retrieval miss, wrong chunk, generation error, or bad
abstention. Note uncertainty; a selected failure sample cannot estimate overall
failure rates. Later add critic confusion counts and useful/redundant search
rounds. Evaluate fixes on the same development set.


</details>

# Compare human ratings with the LLM evaluator

Check whether the LLM judge grades saved answers reasonably. You review answers that already exist. No model call is needed.

The current pilot is **one question with three saved answers** (BM25, dense, hybrid). It is a spot check. It does not show overall judge reliability or which retrieval method is best.

## 1. Get the files

Ask the operator for these, from the same review packet:

| File | Purpose |
| --- | --- |
| `packet.json` | Questions, answers, reference answers and evidence |
| `rubric.md` | The full grading rules |
| `blank-judgments.json` | Template for your ratings |

On Jane's machine they are in `local-v2/eval-ext/20261115/sequential-pilot-05/human-review-01/`. They are not in Git, so a teammate needs them separately. The packet hides the machine grades, but its IDs can reveal the retrieval method, so it is not fully blinded.

## 2. Review independently

Copy `blank-judgments.json` (for example `hira-judgments.json`). Each reviewer uses their own copy. Do not change any `identity` field.

Before you see machine grades, read each case in this order:

1. **Question.** What was asked, and with what constraints?
2. **Reference answer.** What does the benchmark expect? The system never saw it.
3. **Generated answer.** What did it answer, leave out or refuse?
4. **Evidence and citations.** What did the system have, and do the cited passages support its claims?

Judge the answer text, not its status label. An `insufficient_evidence` answer may still make claims. Use only the captured evidence to decide whether answering or refusing was justified.

## 3. Score four dimensions

Score each on its own. See `rubric.md` for unclear cases.

| Dimension | Judge | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| Correctness | Matches the reference? | Central answer wrong or absent | Partly correct, important error | Correct answer, or an appropriate non-answer to an unanswerable question |
| Completeness | Covers required facts and constraints? | Essential answer missing | Important details missing | All essential parts covered |
| Citation support | Do cited passages support the claims? | Central claims unsupported | Support partly missing or ambiguous | All claims supported |
| Abstention/limits | Was answering or refusing justified? | Unsupported guess or unjustified refusal | Important limit missed | Answers or declines appropriately and explains limits |

A responsible refusal on a question that has a benchmark answer still gets **0 for correctness and completeness**. Its abstention score is separate.

Use JSON `null` and a reason for N/A:

- Citation support is N/A for a bare refusal with no factual claims.
- Completeness is N/A only if it truly cannot be assessed. A refusal or unanswerable question does not qualify.
- Correctness and abstention always need a number.

## 4. Save your review

For each reviewed row:

- Put your real name in `reviewer` and the UTC time in `reviewed_at`.
- Set `review_origin` to `fresh_human_review`.
- Fill the four scores, each with a short reason naming the decisive fact, omission or citation.
- Set `review_status` to `REVIEWED`.

Leave unfinished rows `UNREVIEWED`. An empty score is not N/A. In a separate note, say whether you saw the machine scores first. Do not reuse old GPT-OSS ratings or another model's output as a human judgment.

## 5. Compare with the LLM grades

Match each dimension to the machine grade for the **same full case identity**, not just the same question ID. Record the human score, LLM score, whether they match, and why they differ. Keep N/A apart from numbers. With two human reviewers, keep both original ratings and record any agreed rating separately.

- Exact agreement = matching scores / comparable scores.
- Mean absolute error = average of `abs(human - LLM)`.
- Report N/A disagreements separately.
- With three cases from one question, report each finding with its denominator. Do not claim broad reliability from a percentage.

This guide covers final answers. [Critic calibration](EVALUATION.md#critic-calibration-capture) reviews search and stopping decisions. For automated grading commands see [the evaluator workflow](EVALUATION.md#offline-bridge-to-the-rc2-evaluator).

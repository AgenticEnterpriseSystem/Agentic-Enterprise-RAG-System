# Compare human ratings with the LLM evaluator

Use this guide to check whether the LLM judge grades saved answers reasonably.
You review answers that already exist; no new model call is needed.

The current pilot contains **one question with three saved answers**: BM25,
dense and hybrid. Treat it as a three-case spot check. It does not establish
overall judge reliability or which retrieval method is best.

## 1. Get the review files

Ask the operator for these files from the same review packet:

| File | Purpose |
| --- | --- |
| `packet.json` | Exact questions, generated answers, reference answers and evidence |
| `rubric.md` | The complete grading rules |
| `blank-judgments.json` | Template for your own ratings |

On Jane's machine, they are in
`local-v2/eval-ext/20261115/sequential-pilot-05/human-review-01/`.
These local files are not included in Git; a teammate needs them separately.
The packet omits the machine grades, but its IDs can reveal the retrieval method.
Do not describe it as a fully blinded experiment.

## 2. Review independently

Make your own copy of `blank-judgments.json`, such as `hira-judgments.json`.
Keep every `identity` field unchanged so the rating stays attached to the exact
answer. Each reviewer uses a separate copy.

Before looking at the machine grades, read each case in this order:

1. **Question:** what information and constraints were requested?
2. **Reference answer:** what does the benchmark expect?
3. **Generated answer:** what did the system actually answer, omit or refuse?
4. **Supplied evidence and citations:** what information did the system have,
   and do its cited passages support its claims?

The reference answer is used for grading. It was not evidence available to the
answering system. Use only the captured evidence when judging whether answering
or refusing was justified.

Judge the actual answer text, not its status label. An `insufficient_evidence`
answer may still contain factual claims or a partial answer that needs checking.

## 3. Give four separate scores

Use the complete rubric when a case is unclear. These are its main questions:

| Dimension | What to judge | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| Correctness | Does it match the reference? | Central answer wrong or absent | Partly correct with an important error | Correct substantive answer, or appropriate non-answer to an intentionally unanswerable question |
| Completeness | Does it cover the required facts and constraints? | Essential answer missing | Important details missing | All essential parts addressed |
| Citation support | Do actually cited passages support the factual claims? | Central claims unsupported | Support partly missing or ambiguous | All factual claims supported |
| Abstention/limits | Did the evidence justify answering or refusing? | Unsupported guess or unjustified refusal | Important limitation missed | Answers or declines appropriately and explains material limits |

Score these independently. A question that has a benchmark answer can receive a
responsible refusal because retrieval was weak. That refusal still gets **0 for
correctness and completeness**, while its abstention score is judged separately.

A bare refusal with no substantive factual claims has **N/A citation support**;
write JSON `null` and explain why. Completeness can be `null` only when it truly
cannot be assessed, not simply because the question is intentionally unanswerable
or the answer refuses. Correctness and abstention require numeric scores.

## 4. Save a complete human review

For each reviewed row in your copy:

- Enter your real name in `reviewer`.
- Enter the review time in UTC in `reviewed_at`.
- Set `review_origin` to `fresh_human_review`.
- Fill the four scores and a short reason for each, naming the decisive fact,
  omission or citation where useful.
- Set `review_status` to `REVIEWED` after you finish the case.

Leave unfinished rows `UNREVIEWED`. An empty score is not automatically N/A.
In a separate note, record whether you saw the machine scores beforehand. Do not
reuse old GPT-OSS ratings or another model's output as a new human judgment.

## 5. Compare with the saved LLM grades

After saving your human ratings, compare each dimension with the machine grade
for the **same complete case identity**, not merely the same question ID.

Record the human score, LLM score, whether they match, and why they disagree.
Keep N/A separate from numeric scores. If two humans review the answers, keep
their original ratings and record any later agreed rating separately.

For numeric pairs, exact agreement is `matching scores / comparable scores`.
Mean absolute error is the average of `abs(human score - LLM score)`.
Report N/A disagreements separately. With only three cases sharing one question,
report the individual findings and denominators; do not claim broad reliability
from a percentage.

This reviews final answers. [Critic calibration](EVALUATION.md#critic-calibration-capture)
separately reviews search/stopping decisions. For automated grading commands,
see [the evaluator workflow](EVALUATION.md#offline-bridge-to-the-rc2-evaluator).

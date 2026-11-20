<a id="milestone-2-status-and-review-guide"></a>
# Milestone 2 review guide

Where to find the evaluation deliverables. For the authoritative progress count and remaining work, see [Current status](STATUS.md).

The app now has hybrid search, an adaptive critic loop, real user permissions and a separate Observability page.

**Working software and measured answer quality are different things.** 19 of 23 implementation steps are accepted. Human review, critic calibration and the final quality comparison are not done.

<a id="deliverables"></a>
## Deliverables

| Deliverable | Where | Established |
| --- | --- | --- |
| Dataset source, license, splits, limits | [Data Card](DATA_CARD.md) | Documented |
| Search metrics and reproduction | [Evaluation](EVALUATION.md) | Permission-aware search tools; historical full-corpus measurements |
| Simple baseline | [Saved BM25 results](../evaluation/results/bm25_full_development/) | BM25+ on 300 development questions; 282 labeled questions scored |
| Open-source reference baseline | [Saved dense results](../evaluation/results/dense_full_development/) | BGE-small + FAISS, historical protocol |
| Fixed BM25/dense/hybrid answers | [Fixed runs](EVALUATION.md#matched-fixed-bm25-dense-and-hybrid-runs) | 30 saved outcomes; accepted as baseline-file readiness, not a fully matched quality experiment |
| Answer rubric and judge | [Rubric](EVALUATION.md#answer-rubric), [external evaluator](EVALUATION.md#offline-bridge-to-the-rc2-evaluator) | Both implemented; three machine-graded cases still need human validation |
| Critic calibration | [Calibration](EVALUATION.md#critic-calibration-capture) | Four saved critic episodes and blank review rows; human labels and accepted freeze pending |
| Browser, permissions, missing-evidence behavior | [Current status](STATUS.md), [Setup](WEB_APP.md) | Functional checks, reported apart from quality scores |

Early reduced-corpus experiments are historical. Do not present them as the current app or mix them with newer results.

<a id="complete-the-independent-answer-review"></a>
## Next review steps

1. Read [Current status](STATUS.md) and pick the exact saved answer or critic packet.
2. Rate answers with the [rubric](EVALUATION.md#answer-review-fields-and-scores) and the [review guide](ANSWER_REVIEW.md). For critic episodes, judge whether the evidence supports the requested facts and whether the next action was right.
3. Review before seeing machine grades. Keep question, run and source identities with each rating.
4. Mark N/A where a dimension does not apply. Never enter zero for a missing review.
5. Record disagreements and any agreed rating separately.
6. Report how many cases were reviewed and how they were chosen, not only an average.

The proposed gate is 80% agreement per dimension. Agreement shows consistency, not correctness. The critic cannot be its own sole evaluator. Old GPT-OSS workbook labels do not apply to current Claude answers or critic decisions.

Commands: [evaluation commands](EVALUATION.md#run-checks-and-experiments) and the [critic packet APIs](../src/evaluation/critic_calibration.py). A generated template still needs real human ratings.

## What to say in a review meeting

"Hybrid search and the adaptive workflow work with user-based document access. We have historical retrieval measurements and saved fixed-run answer files. Browser checks exercised answers, citations and a missing-evidence clarification. We still need human critic labels, validation of the external judge and a controlled final comparison. We do not claim an answer-quality improvement from the functional checks."

A source-supported answer also disagreed with a benchmark date, so valid citations alone do not make an answer benchmark-correct. See [Current status](STATUS.md).

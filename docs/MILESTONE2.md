<a id="milestone-2-status-and-review-guide"></a>
# Milestone 2 review guide

Use this page to find the project's evaluation deliverables. Use [Current status](STATUS.md) for the authoritative progress count and remaining work. The current application includes hybrid search, an adaptive critic loop, real user permissions and a separate Observability page; it is no longer just a two-retriever passage demo.

**Functional implementation and measured answer quality are different deliverables.** The implementation plan currently has 19 of 23 accepted steps. That does not mean the remaining human review, critic calibration or final quality comparison is complete.

<a id="deliverables"></a>
## Find the deliverables

| Deliverable | Where to look | What is established |
| --- | --- | --- |
| Dataset source, license, splits and limitations | [Data Card](DATA_CARD.md) | Dataset and split documentation |
| Search metrics and reproduction steps | [Evaluation](EVALUATION.md) | Runnable permission-aware search tools and historical full-corpus measurements |
| Historical simple baseline | [Saved BM25 results](../evaluation/results/bm25_full_development/) | BM25+ retrieval on 300 development questions; 282 labeled questions scored |
| Historical open-source reference baseline | [Saved dense results](../evaluation/results/dense_full_development/) | Pretrained BGE-small + FAISS under the historical retrieval protocol |
| Fixed BM25/dense/hybrid answer files | [Fixed-run explanation](EVALUATION.md#matched-fixed-bm25-dense-and-hybrid-runs) | 30 saved outcomes; accepted as baseline-file/report readiness, not a fully matched quality experiment |
| Answer-quality rubric and judge | [Rubric](EVALUATION.md#answer-rubric) and [external evaluator](EVALUATION.md#offline-bridge-to-the-rc2-evaluator) | Rubric and guarded evaluator implemented; three machine-graded cases still need human validation |
| Critic calibration | [Calibration workflow](EVALUATION.md#critic-calibration-capture) | Four saved critic episodes and blank review rows; human labels and accepted freeze pending |
| Browser, permissions and missing-evidence behavior | [Current status](STATUS.md) and [application setup](WEB_APP.md) | Functional checks, with failures and limits reported separately from quality scores |

Early reduced-corpus diagnostic experiments are historical. They must not be presented as the current full-corpus application or mixed with the newer results.

<a id="complete-the-independent-answer-review"></a>
## What a teammate should review next

1. Read [Current status](STATUS.md) and choose the exact saved answer or critic packet being reviewed.
2. Use the [answer rubric](EVALUATION.md#answer-review-fields-and-scores) for answers. For critic episodes, review whether the supplied evidence supports the requested facts and whether the next action was appropriate.
3. Review independently before looking at machine grades. Keep question/run/source identities with every rating.
4. Mark a dimension N/A when it does not apply. Do not enter zero for missing review.
5. Compare reviewer disagreements and record any adjudicated decision separately.
6. Report the number and selection of reviewed cases, not only an average score.

The proposed answer-review gate is 80% agreement per dimension. Agreement measures consistency between reviewers, not correctness. The model critic cannot be its own sole evaluator. Old GPT-OSS workbook labels are not new labels for current Claude answers or critic decisions.

Use the supported [evaluation commands](EVALUATION.md#run-checks-and-experiments) and the tested [critic packet APIs](../src/evaluation/critic_calibration.py); a generated template still needs real human ratings.

## What we can say in a review meeting

“We have the hybrid search and adaptive workflow working with user-based document access. We have historical retrieval measurements and saved fixed-run answer files. The latest browser checks exercised answers/citations and a missing-evidence clarification. We still need human critic labels, validation of the external judge and a controlled final comparison. We are not claiming an overall answer-quality improvement from the functional checks.”

A source-supported answer also disagreed with a benchmark date, so valid citations alone are not enough to declare the answer benchmark-correct. See [Current status](STATUS.md) for that example and the exact source/build limits.

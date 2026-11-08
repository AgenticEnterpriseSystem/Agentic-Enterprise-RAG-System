# RAG Answer Evaluator — RC2 Engineering Release Candidate

**Status:** Engineering release candidate. Operationally tested; not independently validated as a production-grade evaluator.

## 1. Overview

This evaluator grades generated RAG answers using four rubric dimensions:

- Correctness
- Completeness
- Citation support
- Abstention and evidence limits

Each dimension receives a score from 0 to 2, with N/A permitted where the rubric allows it.

The evaluator has two stages:

1. **Scoring:** Grades the answer using the RC2 scoring prompt.
2. **Citation audit:** Independently examines cited evidence and identifies cases requiring human review.

The citation audit does not automatically modify the scoring results.

## 2. Release configuration

| Component | Configuration |
|---|---|
| Model | `gpt-oss:20b-cloud` |
| Scoring prompt | `docs/evaluation_prompts/answer_grading_rc2.md` |
| Audit prompt | `docs/evaluation_prompts/citation_audit_v5.md` |
| Audit mode | `all` |
| Runner | `scripts/run_evaluator_candidate.py` |
| Evaluator identifier | `v3-scoring-v6-review-candidate-1` |

The evaluator identifier reflects the underlying V3 scoring/V6 review architecture. The explicit RC2 scoring prompt determines the release configuration.

## 3. Required files

**Runner and implementation**

- `scripts/run_evaluator_candidate.py`
- `src/evaluation/evaluator_candidate.py`
- `src/evaluation/verified_judge.py`
- `src/evaluation/verified_judge_v6.py`

**Prompts**

- `docs/evaluation_prompts/answer_grading_rc2.md`
- `docs/evaluation_prompts/citation_audit_v5.md`

**Tests**

- `tests/test_evaluator_candidate.py`
- `tests/test_verified_judge.py`
- `tests/test_verified_judge_v6.py`

The project must have its existing Python dependencies installed and access to the configured Ollama model/provider.

## 4. Validate installation

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Then inspect the runner options:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluator_candidate --help
```

## 5. Run an RC2 evaluation

Example using the existing 20-question development batch:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluator_candidate `
  --answers outputs/alex_evaluation/bm25_gptoss_answers_20.json `
  --questions outputs/alex_evaluation/questions_20.json `
  --score-prompt docs/evaluation_prompts/answer_grading_rc2.md `
  --audit-prompt docs/evaluation_prompts/citation_audit_v5.md `
  --audit-mode all `
  --model gpt-oss:20b-cloud `
  --limit 20 `
  --output outputs/alex_evaluation/rc2_evaluation.json
```

Replace the input paths with the generated answers and questions to evaluate.

Use `--dry-run` to validate the selected inputs and configuration without making model calls.

The full audit configuration ordinarily requires two model calls per case, excluding retries.

## 6. Resume interrupted evaluations

Repeat the original command with:

```powershell
--resume
```

Resume requires the same selected question IDs, input files, prompts, model, and implementation.

The runner records SHA-256 hashes in a companion manifest and checks configuration compatibility before resuming.

Completed cases can be reused rather than evaluated again.

## 7. Outputs and review flags

The evaluator saves structured JSON containing rubric scores, explanations, audit findings, review priorities, usage information, errors, and provenance.

Review priorities include:

- **High:** Potentially consequential evidence, attribution, or scoring concerns.
- **Medium:** Issues requiring human interpretation.
- **None:** No final review flag triggered.

Review flags are triage signals, not confirmed grading errors.

The independent audit does not automatically override scores.

## 8. Verified engineering tests

Testing completed November 7, 2026.

| Test | Result |
|---|---|
| Full repository test suite | 173/173 passed |
| Single-question live smoke test | Passed |
| Five-question live evaluation | 5/5 completed |
| Resume test | 5/5 recognized as complete |
| Twenty-question live evaluation | 20/20 completed |
| Incomplete cases in 20-question run | 0 |
| Cases flagged for human review | 13/20 (65%) |

The review distribution was 12 high, 1 medium, and 7 unflagged.

These results demonstrate operational functionality on the tested configuration. They do not establish scoring accuracy or generalization to unseen datasets.

## 9. Reproducibility

The successful 20-question run generated a manifest containing SHA-256 hashes of inputs, prompts, and implementation files.

Verified prompt hashes:

**RC2 scoring prompt**

`4d85bdff94c966d767c4746fbaac8cfc579e2b21d688e840773cbe7220aab8d3`

**Citation audit prompt**

`9ee725c6147988ed7abea0cafde4ed887ed738e1055fb39630ea825884b765e7`

The manifest from `rc2_release_batch20_test.json.manifest.json` records the exact tested implementation hashes.

## 10. Known limitations

1. **Review sensitivity:** The evaluator flagged 65% of the tested development cases. Some warnings may be overly conservative.
2. **Human validation:** Agreement with independently adjudicated human labels has not been established for this release configuration.
3. **Scorer–auditor disagreements:** The auditor can flag unsupported or ambiguous claims even when the scorer awards full citation credit.
4. **Model variability:** Repeated model calls may produce different judgments.
5. **Provider dependency:** Live evaluation requires access to the configured model.
6. **Runtime:** The runner uses configurable HTTP timeouts and retries, but an HTTP timeout is not a guaranteed end-to-end deadline.

## 11. Release guidance

RC2 is suitable for team engineering evaluation and supervised experimentation.

It should not yet be treated as an independently validated, unattended production judge.

Before production deployment, use a separately human-adjudicated holdout to measure scoring agreement, serious-error detection, review precision and recall, and reliability at the intended scale.

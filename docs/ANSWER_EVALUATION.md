# BM25+ → GPT-OSS answer generation and evaluation

This runbook describes the 20-question **BM25+ retrieval → GPT-OSS answer generation → LLM-as-judge evaluation** baseline. Commands below are for **Windows PowerShell**, run from the **repository root**. The underlying retrieval predictions already exist in the team's repository: `evaluation/results/bm25_full_development/predictions.jsonl` (introduced in commit `5cd05ea`). The 20-question experiment **reuses** those predictions; it does not independently rerun BM25 unless you perform the optional retrieval step below.

## 1. Prerequisites

- Python 3.11 and dependencies from `requirements.txt`.
- Ollama installed and accessible at `http://localhost:11434`, with access/authentication for the model specified in `src/generation/ollama_provider.py` (`gpt-oss:20b-cloud` at the time of this experiment). This is a **cloud model called via the local Ollama API**, not necessarily local inference. Verify model availability before running.
- The EnterpriseRAG-Bench corpus and `generated_data/uuid_index.json` available locally under the configured `CORPUS_DIR / EnterpriseRAG-Bench`. The corpus is not stored in Git.
- The selected question file `outputs/alex_evaluation/questions_20.json`. It must contain **exactly 20 unique questions**. Each record needs `question_id`, `question`, `question_type`, and `gold_answer`. The selection is curated input, **not generated** by the three scripts below. Teammates must obtain the approved question selection separately if it is not committed.
- The team's tracked `evaluation/results/bm25_full_development/predictions.jsonl`.

Create/setup the environment if necessary:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip show ollama
```

If the `ollama` Python package is not installed, add it to the project's declared requirements and install it; do not rely only on an ad hoc local environment.

For a fresh checkout without the corpus, use the team's corpus setup procedure, currently:

```powershell
.\.venv\Scripts\python.exe -m src.data.corpus --clone
```

Ensure Ollama is running and the configured model is accessible. The provider uses a fixed local API host. Confirm any required cloud login separately; no credentials belong in Git.

## 2. Optional: recreate retrieval predictions from scratch

Only do this if you intend to regenerate the team's BM25 retrieval baseline rather than reuse its committed predictions. It can use substantial local disk, memory, and time. Check whether its runner will overwrite tracked outputs before doing so.

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.corpus_bm25
.\.venv\Scripts\python.exe -m src.evaluation.run_retrieval --method bm25 --split development
```

The answer-generation baseline below uses only the **top five** document IDs for each selected question from those predictions.

## 3. Prepare the 20-question evidence batch

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.prepare_answer_batch
```

Reads:
- `outputs/alex_evaluation/questions_20.json`
- `evaluation/results/bm25_full_development/predictions.jsonl`
- The corpus `generated_data/uuid_index.json` and referenced source documents

Writes:
- `outputs/alex_evaluation/bm25_cases_20.json`

**Caution:** The current implementation opens the output with `"w"`, so it **overwrites an existing evidence batch**. Back up the existing file if you want to preserve it. Its source-document path handling is Windows-specific and should be made portable before using it on Linux/Colab.

## 4. Generate GPT-OSS answers

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ollama_benchmark --limit 20
```

Reads `bm25_cases_20.json` and writes:

- `outputs/alex_evaluation/bm25_gptoss_answers_20.json`

Each record saves answer status, answer text, citations, selected evidence, and token usage. The generator enforces a **24,000-character** total evidence budget (not a token count). The provider's default model is `gpt-oss:20b-cloud` and temperature is set to 0. This does not guarantee identical model outputs across runs.

For a single question, the script also supports:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ollama_benchmark --question-id qst_0350
```

That writes a separate `bm25_gptoss_retry_qst_0350.json` file, **not** an automatic replacement in the full 20-question answers file. To incorporate a retry, you must deliberately reconcile the records and document the change.

**Caution:** This script uses exclusive-create mode (`"x"`); it refuses to overwrite an existing output file. Rename/back up the existing file before a full rerun.

## 5. Run the automated judge

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ollama_evaluation --limit 20
```

Reads:
- `outputs/alex_evaluation/bm25_gptoss_answers_20.json`
- `outputs/alex_evaluation/questions_20.json` for reference answers (`gold_answer`)

Writes:
- `outputs/alex_evaluation/bm25_gptoss_evaluation_20.json`

The judge prompt is `JUDGE_PROMPT` in `src/evaluation/answer_metrics.py`. It scores correctness, completeness, citation support, and abstention/limits on a **0–2 scale** and records explanations. The current runner calls the same default `OllamaProvider` as answer generation, so the same model serves as both generator and judge; record that limitation.

**Caution:** The evaluation runner **resumes** from existing results. It skips successful prior scores, even if you changed the prompt/model. For a genuinely fresh judge experiment, back up/rename the old evaluation output first.

## 6. Rerun safely without losing earlier results

Before regenerating a file, keep a dated copy somewhere outside the filenames hard-coded in the scripts, for example:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Rename-Item 'outputs/alex_evaluation/bm25_gptoss_answers_20.json' "bm25_gptoss_answers_20_$stamp.json"
Rename-Item 'outputs/alex_evaluation/bm25_gptoss_evaluation_20.json' "bm25_gptoss_evaluation_20_$stamp.json"
```

Only run these commands if the named files exist. If you also regenerate evidence, back up `bm25_cases_20.json` **before** Step 3. Then run Steps 3–5 in order. A **judge-only rerun** needs only the evaluation file renamed, followed by Step 5. The prior saved files are experimental records; don't commit the dated duplicates by default.

## 7. Human ratings, comparison, and reporting

The automated judge does **not** perform manual evaluation or compute agreement with the human rater. Keep a separate human-scored table keyed by `question_id`, using the same rubric definitions and marking genuinely non-applicable dimensions as N/A. Compare human and judge scores using:

- Exact agreement = identical scores / comparable human–judge score pairs
- Mean absolute error (MAE) on the 0–2 scale
- Quadratic weighted Cohen's kappa, by dimension when sample size supports it

Exclude N/A pairs instead of treating them as zeros. Store the comparison script and instructions in the repository if these numbers are to be fully reproducible; the current three-stage generation/evaluation workflow alone is **not** enough to rebuild the human-versus-judge agreement report. `scripts/create_manual_evaluation.py` and any separate comparison code should be documented/tested before advertising a specific command for that part.

## 8. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_answer_metrics.py -v
.\.venv\Scripts\python.exe -m pytest -q
```

At the time of this documentation, the user reported **19/19** answer-metrics tests and **133/133** full-project tests passing. After merging into `hira/dev` with the web app (2026-10-26), the full suite is **205 passed, 1 skipped**; the skipped answer-metrics test needs the raw EnterpriseRAG-Bench clone. These tests do not by themselves establish a successful fresh run against the live Ollama service.

## 9. Git sharing and data handling

Commit the scripts, tests, this guide, and an approved **20-question selection file** so teammates can reproduce the exact sample. The retrieval-predictions file is already part of the repository. Consider committing compact results with question IDs and scores. Raw `bm25_cases_20.json` and generation outputs embed retrieved document text; review dataset licensing, access rights, sensitive content, and size before committing them. Do **not** commit credentials, `.env`, model weights, the full corpus, embeddings/vector stores, Python caches, Excel temporary lock files, or unreviewed debug/retry files.

## 10. Known reproducibility limitations / recommended improvements

1. Pin dependency versions, Ollama model tag/version, corpus revision, and the Git commit used for the predictions.
2. Document how the exact 20 questions were sampled and ensure the approved selection is available.
3. Make output paths configurable or use timestamped run directories; don't mix old and new experiment outputs.
4. Make the Windows-specific source-path handling portable.
5. Verify that the same evidence is passed into the judge as was actually passed into the generator; log hashes or IDs for auditing.
6. Provide a tested command/script for the human-vs-judge comparison, including N/A handling and agreement statistics.
7. If moving to hybrid retrieval, generate a **separate** comparable batch using the same questions, evidence budget, generator, rubric, and judge configuration.

Add a link from the root `README.md`:

```markdown
- [BM25 + GPT-OSS answer evaluation](docs/ANSWER_EVALUATION.md): how to reproduce the 20-question generation and judge baseline.
```

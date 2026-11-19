<a id="bm25--gpt-oss-answer-generation-and-evaluation"></a>
# Historical BM25 + GPT-OSS answer evaluation

A record of the saved **20-question answer baseline** and the older scripts that made it. It is not setup for the current hybrid/Claude app. For current tools see [Evaluation](EVALUATION.md); for open work see [Current status](STATUS.md).

How it ran:

- Saved BM25 search results were reused.
- Answers came from `gpt-oss:20b-cloud`, and the same model graded them.
- Ollama exposes this cloud model through a local API. Inference does **not** stay on your computer.

## Saved results

Reading them needs no model call. Files are in [outputs/alex_evaluation](../outputs/alex_evaluation/).

| File | Contents |
| --- | --- |
| `questions_20.json` | The 20 curated questions and reference answers |
| `bm25_cases_20.json` | Questions and evidence built from saved BM25 predictions |
| `bm25_gptoss_answers_20.json` | Answers, evidence, citations and usage |
| `bm25_gptoss_evaluation_20.json` | Model-judge scores and reasons |
| `rag_evaluation_results.xlsx` | Historical human/judge comparison workbook |

The 20 question IDs were hand-picked, not sampled by these scripts. Predictions came from [the saved development run](../evaluation/results/bm25_full_development/predictions.jsonl) (commit `5cd05ea`). Only the top five document IDs per question were used.

| Dimension | Mean model score, 0 to 2 | Cases scored 2 |
| --- | ---: | ---: |
| Correctness | 1.30 | 11/20 |
| Completeness | 1.20 | 9/20 |
| Citation support | 1.55 | 15/20 |
| Abstention/limits | 1.70 | 16/20 |

Fifteen answers were given and five marked insufficient evidence. These are model-judge scores, not validated scores for the current app. The same model generated and judged. The committed code does not reproduce the human-agreement report.

<a id="1-prerequisites"></a>
## Before a rerun

**Cost and data warning.** These scripts do not enforce dollar caps. `--limit` counts questions, not USD, and the Ollama adapter records cost as unknown. Agree on cloud access, excerpt transfer and a budget first. For controlled paid evaluation use the [guarded evaluator](EVALUATION.md#offline-bridge-to-the-rc2-evaluator).

You need:

- Python 3.11 and the [requirements](../requirements.txt).
- The raw EnterpriseRAG-Bench checkout with `generated_data/uuid_index.json` (see [dataset setup](WEB_APP.md)).
- The 20-question selection, the saved predictions, and Ollama at `http://localhost:11434`.

```sh
python -m pip install -r requirements.txt
python -m pytest tests/test_answer_metrics.py -q
```

Work in a separate checkout, or back up the saved outputs first. The generator opens its output file **after** generating answers, so an existing filename fails after model calls were already made.

<a id="2-optional-recreate-retrieval-predictions-from-scratch"></a>
## 1. New retrieval results (optional)

Reusing the saved predictions reproduces the input selection without a new search. For a new permission-aware BM25 run, follow [the retrieval instructions](EVALUATION.md#match-the-web-retriever). It is a separate experiment with different access rules.

<a id="3-prepare-the-20-question-evidence-batch"></a>
## 2. New evidence file (no model calls)

Skip this to use the saved batch.

```sh
python -m src.evaluation.prepare_answer_batch \
  --output .cache/evaluation/legacy-bm25-cases-new.json
```

It reads the selected questions, saved predictions, UUID index and source documents, and refuses an existing output file. Optional flags: `--selected`, `--predictions`, `--benchmark`, `--expected-count`, `--top-k`. It is a legacy batch, not an authenticated export, and does not enforce the frozen-split rules. Never copy references into generation inputs.

<a id="4-generate-gpt-oss-answers"></a>
## 3. Generate answers

[The generator](../scripts/run_ollama_benchmark.py) reads the fixed file `outputs/alex_evaluation/bm25_cases_20.json`, not the Step 2 output. In your separate checkout, place the new batch at that path and back up the old answers first.

```sh
python -m scripts.run_ollama_benchmark --limit 20
```

It writes `bm25_gptoss_answers_20.json` using a 24,000-character evidence limit, `gpt-oss:20b-cloud` and temperature 0. Temperature 0 does not guarantee identical output.

One question:

```sh
python -m scripts.run_ollama_benchmark --question-id qst_0350
```

This writes `bm25_gptoss_retry_qst_0350.json` and does not replace the record in the 20-question file. `--limit 1` writes a file ending `_1.json`, which the judge does not read.

<a id="5-run-the-automated-judge"></a>
## 4. Judge the saved answers

[The judge](../scripts/run_ollama_evaluation.py) reads the fixed 20-answer file and takes references from `questions_20.json`.

```sh
python -m scripts.run_ollama_evaluation --limit 20
```

It writes `bm25_gptoss_evaluation_20.json` with reasons and usage. It scores the four dimensions of the [0 to 2 rubric](EVALUATION.md#answer-rubric) and uses the same default Ollama model as generation.

<a id="6-rerun-safely-without-losing-earlier-results"></a>
### Resume behavior

On a later run the judge keeps successful scores and intentional generation-error records, and retries failed judge rows. It does **not** notice a changed model or prompt, so a rerun can mix settings and cost new calls. For a fresh experiment, back up the originals and use a new output name. For a retry, record which rows changed. The [guarded runner](EVALUATION.md#offline-bridge-to-the-rc2-evaluator) has plans and identity checks and never replays uncertain attempts.

<a id="7-human-ratings-comparison-and-reporting"></a>
## 5. Human comparison

These scripts do no human review and produce no agreement report. To compare, follow [Compare human ratings with the LLM evaluator](ANSWER_REVIEW.md) and the [same rubric](EVALUATION.md#answer-review-fields-and-scores). Current critic review uses [the calibration APIs](EVALUATION.md#critic-calibration-capture).

<a id="8-tests"></a>
## Tests

`python -m pytest tests/test_answer_metrics.py -q` checks the rubric parser and helpers. Passing it does not verify a cloud model run.

<a id="9-git-sharing-and-data-handling"></a>
<a id="10-known-reproducibility-limitations--recommended-improvements"></a>
### What to record and what to commit

Record the question selection and how it was chosen, corpus revision, source commit, prediction-file fingerprint, model and version, prompt, selected evidence, limits, usage and output paths. Keep these results apart from new hybrid/Claude experiments.

Commit source, tests, instructions and approved compact summaries. Keep credentials, `.env`, model weights, the full corpus, indexes, private review packets and debug files out of Git. The saved answer files contain fictional corpus excerpts. Do not assume future evidence files are safe to publish.

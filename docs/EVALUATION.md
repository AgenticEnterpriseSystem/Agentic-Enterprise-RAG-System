# Evaluation

Saved retrieval scores measure document ranking, not answer correctness.
Independent answer-quality evaluation of the current hybrid app remains incomplete.
See [retrieval results](FULL_CORPUS.md) for measured scores, model settings and
hardware, and [saved artifacts](../evaluation/README.md) for predictions and summaries.

## Run retrieval evaluation

Install the [app dependencies](WEB_APP.md#build-bm25-locally) and obtain the matching
corpus/index bundle. Also supply `.cache/corpus/questions.jsonl` from the benchmark
[dataset](DATA_CARD.md); it is separate from the web artifact bundle. Run from the
repository root with the virtual environment active:

```sh
python -m src.evaluation.run_hybrid \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --bm25-index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --questions .cache/corpus/questions.jsonl --split development \
  --output-dir .cache/evaluation/hybrid-development
```

Use a new output directory for each run. This performs retrieval without paid
answer-generation calls. Output includes ranked predictions, scores, settings,
code versions and index manifests.

The web runtime and evaluations share `RetrievalSettings` and manifest validation.
Defaults are BM25 weight 0.7, dense weight 0.3, 50 candidates per retriever,
fused pool 50, RRF k=60, CPU encoding, manifest-selected precision and no reranker.
CLI evaluations do not read `.env`; pass web overrides explicitly:

| Web setting | Evaluation option |
| --- | --- |
| `RAG_CANDIDATE_K` | `--candidate-k` |
| `RAG_RERANK_K` | `--rerank-k` |
| `RAG_RRF_K` | `--rrf-k` |
| `RAG_FP16` | `--fp16` / `--no-fp16` |
| BM25 weight | `--bm25-weights` (comma-separated values) |

Rerankers are opt-in with `--rerankers`. Retrieval scoring requests 20 results;
the web app starts with five passages for an answer. Avoid loading a second full
index alongside the web backend on a memory-constrained machine.

## Review generated answers

Save a small development review packet without model calls:

```sh
python -m src.agents.pilot --prepare --count 3 --retrieval hybrid \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --questions .cache/corpus/questions.jsonl \
  --output .cache/agent-pilot/review
```

The packet contains retrieved evidence and separate reference answers for review.
References are never sent to the running agent. Use `--question-ids` instead of
`--count` to repeat exact development questions. Synthetic boundary cases are in
[the critic fixtures](../tests/fixtures/critic_development_cases.json).

For a paid comparison, export `ANTHROPIC_API_KEY` securely into the shell. The
pilot does not load `.env`. Replace placeholders with your model, current prices
and allocated budget:

```sh
python -m src.agents.pilot --live --workflow compare --count 3 --retrieval hybrid \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --questions .cache/corpus/questions.jsonl \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS --question-budget-usd 0.10 \
  --input-price INPUT_USD_PER_MILLION --output-price OUTPUT_USD_PER_MILLION \
  --output .cache/agent-pilot/comparison
```

`compare` runs Fixed RAG and the adaptive agent on the same questions, sharing one
total cap. Errors or budget stops end the run and preserve completed cases. CLI
and web caps are independent; allocate their combined usage within your budget.
Use a fresh output directory and retain failed attempts. Outputs include answers,
evidence, model inputs/outputs, timings, token usage and configuration manifests.

Score each answer against its sources and reference facts:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim | Important errors or unsupported details | All substantive claims correct |
| Completeness | Main answer missing | Required details missing | Required facts and qualifications covered |
| Citation support | Central claims unsupported | Some claims unsupported | All factual claims supported |
| Abstention | Unsupported answer or unjustified refusal | Important limitation missed | Answers or explains gaps appropriately |

Use N/A for completeness without reference facts, or citation support without
factual claims. Have two reviewers independently score a representative sample,
resolve disagreements, and report agreement. Validate any LLM judge against those
human ratings before treating its scores as trustworthy. The search critic is not
an independent answer evaluator, and `answered` status is not a correctness score.

## Splits and scoring

- Use the frozen [splits](../evaluation/splits.json): 300 development, 100
  calibration and 100 final questions. Tune on development; freeze settings before
  calibration/final evaluation. Shared reference documents and identical questions
  are grouped together, though broader project overlap can remain.
- Match corpus, questions, permissions, retrieval settings, context limits and
  generator/model. Record changed settings and index/model revisions.
- Report scored counts, exclusions and category/source slices. Questions without
  reference-document labels are N/A for retrieval, not automatically unanswerable.
  Missing corpus references stay in denominators; eligible errors score zero.
- Record latency, calls, rounds, tokens and estimated API cost, including failures.
  State hardware, token prices, cache/warm-up conditions and exclusions. Unknown
  cost stays null; local compute and judge costs should be reported separately.
  Repeat paired timings in reversed order to reduce ordering effects.
- Review failures by retrieval miss, omitted passage, incorrect generation or bad
  abstention. Retest fixes with genuine-conflict and missing-evidence examples.
  A selected error sample does not establish an overall failure rate.

## Current limitations

The critic can reject useful evidence, select the wrong incident, or accept an
initial hypothesis as a confirmed cause. Selected passages can omit key details,
and valid citation IDs do not prove that a claim is supported. Search and budget
limits can produce partial answers. Clarification is supported, but conversation
state is not retained. See [System](SYSTEM.md) for runtime behavior and limits.

Run `python -m pytest -q` for component checks. Tests cover API boundaries,
permissions, evidence handling, provider failures and budgets; they do not measure
model accuracy. Full-corpus Docker/GPU validation and independent answer-quality
assessment remain incomplete.

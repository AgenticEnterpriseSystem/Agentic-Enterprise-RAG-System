<a id="evaluation-and-results"></a>
# Evaluation

How we measure Redwood, how to run each check, and what the results do and do not show. For open tasks see [Current status](STATUS.md). To run the app first, see [Setup](WEB_APP.md). Terms are in the [glossary](README.md).

**Implemented workflows pass specific functional checks. Better answer quality has not been demonstrated.** A found document, an HTTP 200, a valid citation ID or a high judge score does not prove an answer is correct.

## Rules that protect the results

- **Do not touch the final split.** Never use its labels to tune anything.
- **Keep credentials and private review files out of Git.**
- **Ask before any paid model call.** Help commands and config values do not authorize spending. An old approval does not cover a new batch.
- **Unknown cost is not zero.** Report it as unknown.

## Choose a check

| Question | Use | Calls a model? |
| --- | --- | --- |
| Does the code work? | Python tests | No |
| Can search find the reference documents? | [Retrieval checks](#match-the-web-retriever) | No (dense search runs a local embedding model) |
| What did each search method answer? | [Fixed runs](#matched-fixed-bm25-dense-and-hybrid-runs) | Only if live generation is enabled |
| Are the answers correct and supported? | [Rubric](#answer-rubric), then the [external evaluator](#offline-bridge-to-the-rc2-evaluator) | A model judge is a separate paid step |
| Does the critic stop at the right time? | [Critic calibration](#critic-calibration-capture) | Capturing decisions can; reviewing saved ones does not |

Run commands from the repository root. Paths such as `/private/run` are placeholders. Use a new output directory for each experiment.

<a id="splits-and-scoring"></a>
## Splits and scoring

[The frozen split file](../evaluation/splits.json) uses seed 5980.

| Split | Questions | Purpose |
| --- | ---: | --- |
| Development | 300 | Find problems and improve the system |
| Calibration | 100 | Choose settings before the final test |
| Final | 100 | Test the frozen system only |

- Identical normalized questions, and questions that share reference documents, stay in the same split.
- All splits search the same document collection.
- The 60-question golden development subset is separate from any small pilot sample.
- **Recall@10**: the fraction of reference documents in the top 10 results. Two reference documents with one found is 0.5.
- **nDCG@10**: like Recall@10, but finding a document higher in the list scores better.

For a fair comparison, keep the questions, user access, indexes, generator, prompts and budgets the same. Record every deliberate difference. Report failures, missing labels, question counts, latency, rounds, requests, tokens and cost. Operating-system caches are not cleared, so reverse the run order or repeat runs before making timing claims.

<a id="application-verification"></a>
<a id="current-agent-evidence-flow"></a>
## What is verified

The app checks the user's live permissions before retrieval and again before evidence is used or released. The critic can ask for more search, adjacent text, or clarification. The critic and generator see the same frozen evidence. A clarification keeps no conversation; the user submits a new full question.

The context holds up to **8 unique documents** by default (configurable to 5 or 10). Several passages from one document count once. Budgets can select fewer.

Verified:

- Real authenticated browser flows.
- Generated answers and citations on an earlier build.
- On the final build, a missing-evidence clarification after two critic rounds and `read_more`. It made no answer-generator call.
- One source-supported date disagreed with the dataset reference (April 7 vs April 6).

Not verified:

- A controlled 5-versus-8 document quality comparison.
- A passing remote CI run. CI is configured to run Python tests, build the frontend and run a tiny Docker fixture with real accounts. The fixture does not show full-corpus performance or a fresh teammate install.

See [Current status](STATUS.md) for limits and remaining work.

<a id="run-checks-and-experiments"></a>
<a id="local-correctness-checks-no-paid-calls"></a>
## Checks with no paid model

1. Install dependencies using [Setup](WEB_APP.md).
2. Run the tests. Database integration tests need the separate test PostgreSQL setup.
3. Read saved results before deciding you need a new run.

```sh
python -m pytest -q
python -m src.evaluation.run_retrieval --help
python -m src.evaluation.run_hybrid --help
python -m src.evaluation.run_fixed --help
python -m src.evaluation.run_evaluator --help
```

On a memory-limited machine, do not load a second copy of the indexes next to the running app. Coordinate expensive local runs with whoever runs the stack.

<a id="match-the-web-retriever"></a>
<a id="full-development-retrieval-scoring"></a>
## Retrieval checks with the app's permission rules

These commands use the same PostgreSQL-backed retrieval as the app. They do not read `.env`. The default Docker database is not exposed to the host, so ask the stack operator to prepare the environment.

You need:

1. The BM25 and dense index bundle, and the offline embedding-model cache.
2. A ready snapshot and canonical document import in PostgreSQL.
3. Runtime-only database settings, never the migration or bootstrap login.
4. The question file, the split file and an existing private evaluator credential.

Credentials:

- A benchmark credential uses the `rag-benchmark-v1` format. Browser login refuses it.
- An existing human session uses `--principal-kind session` and a mode-0600 file containing exactly `{"format":"rag-session-v1","token":"..."}`. Get it from the operator's approved workflow.
- These tools create no account or grant.
- Never put a password, token or database URL in command arguments.

Start with a small development selection:

```sh
python -m src.evaluation.run_retrieval --method bm25 \
  --bm25-index .cache/hybrid-artifacts/indexes/bm25 \
  --snapshot-file /path/ready-snapshot.json \
  --credentials-file /path/private/evaluator-session.json --principal-kind session \
  --questions .cache/corpus/questions.jsonl --splits evaluation/splits.json \
  --split development --limit 3 --output .cache/evaluation/bm25-check-new

python -m src.evaluation.run_hybrid \
  --bm25-index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --snapshot-file /path/ready-snapshot.json \
  --credentials-file /path/private/evaluator-session.json --principal-kind session \
  --questions .cache/corpus/questions.jsonl --splits evaluation/splits.json \
  --split development --limit 3 --output-dir .cache/evaluation/hybrid-check-new
```

Drop `--limit 3` only to run the whole split. Use `--method dense` (with both the BM25 and dense index paths) to score dense rankings only.

Default hybrid settings: BM25 weight 0.7, dense weight 0.3, 50 candidates per retriever, rerank pool 50, RRF k=60, **no reranker**. RRF (reciprocal rank fusion) merges the two rankings. See [the factory](../src/retrieval/factory.py) and [settings](../src/config.py).

| Option | Meaning |
| --- | --- |
| `--candidate-k`, `--rerank-k`, `--rrf-k` | Match the web retrieval settings |
| `--fp16` / `--no-fp16` | Query precision; a mismatch with the index fails |
| `--allowed-doc-ids-file`, `--source-types` | Narrow existing access; cannot grant access |
| `--after`, `--before` | Timezone-aware date filter; unknown dates fail |
| `--read-more-chars 3000` | Also check adjacent text in up to two returned documents |
| `--bm25-weights` | Hybrid only: comma-separated weight sweep |
| `--rerankers bge-reranker-base@EXACT_40_HEX_REVISION` | Enable one pinned offline reranker |

Calibration or final reranking needs a fixed numeric `--rerank-bm25-weight`. Do not pick a new best weight from those labels.

### Reading the output

Each finished run has predictions, metrics and `manifest.json`. The manifest records the backend, snapshot, permissions, filters, model and index settings, file hashes and versions. It never stores the session secret. An interrupted run may have no manifest.

- Scoring uses only reference documents the user can access. No accessible reference means N/A, not a success.
- An unknown reference excludes the question from quality scoring. Errors and counts stay visible.
- Denied reference IDs are not exported.
- A permission change stops the run.
- Latency includes permission resolution. p95 is the nearest-rank percentile.

<a id="retrieval-results"></a>
## Historical full-corpus retrieval results

These searched **511,958 documents** for 300 development questions. Eighteen had no reference labels, so **282 were scored**. They used the earlier unrestricted protocol, not a permission-scoped test.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| Hybrid BM25+ / BGE-base | 0.641 | 0.722 | 0.760 | 0.604 |
| Hybrid BM25+ / Arctic | 0.638 | 0.723 | 0.768 | 0.628 |

- BGE-base hybrid beat BM25+ by 0.029 Recall@10 (95% bootstrap interval +0.005 to +0.055; 19 questions better, 7 worse). Equal hybrid weights scored 0.700.
- One calibration check scored 94 labeled questions: BM25+ 0.712, BGE-base hybrid 0.767, Arctic hybrid 0.788 Recall@10. BGE-base's gain was 0.056 (interval +0.013 to +0.106). Do not tune further on this split. The final split stayed untouched.
- Arctic beat BGE-base alone by 0.067 Recall@10. Their hybrids differed by 0.001 Recall@10 (nDCG@10 0.023 higher for Arctic). The app keeps BGE-base; the Arctic index is still available.
- Rerankers lowered hybrid Recall@10 to 0.637 (`bge-reranker-base`) and 0.669 (`bge-reranker-v2-m3`). They saw one 3,000-character section cut to 512 tokens, often without the title. This does not show every reranking approach is worse.

[Saved result directories](../evaluation/README.md) hold predictions, scores, slices, settings and timings. [Full-corpus results](FULL_CORPUS.md) covers screening, intervals and title experiments.

<a id="reproduction-settings"></a>
### Settings behind these numbers

- Dataset revision `d36685e273713975ee20299bbf1ab64165575b3c`. See [Data Card](DATA_CARD.md).
- BM25+: k1=1.5, b=0.75, delta=1, lowercase word tokens, top 20 documents.
- Dense: 500-token passages, 50-token overlap, exact cosine search, document score is its best passage.
- BGE-small: `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; title on first passage, GPU fp16, code `e924855`.
- BGE-base: `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; title on every passage, GPU fp16, code `d11eb1b`.
- Arctic: `Snowflake/snowflake-arctic-embed-l-v2.0` at `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`; `query: ` prefix, title on every passage, L4 bf16, code `38489e0`.
- Hybrid used real `HybridRetriever` searches with the defaults above. BM25 ran on CPU. Query encoding ran on Colab GPUs (the web app uses CPU), so timings are not a controlled hardware comparison.
- An earlier estimate fused saved top-20 rankings and scored 0.705 at weights 0.6 to 0.8. The live-search numbers above replace it.

<a id="answer-rubric"></a>
<a id="answer-review-fields-and-scores"></a>
<a id="answer-review-rubric-not-yet-completed"></a>
<a id="answer-review-rubric--not-yet-completed"></a>
## Answer rubric

Score each dimension on its own. A usable answer scores 2 on every applicable dimension.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Central claim wrong | Main answer right, important details wrong or unverifiable | All substantive claims correct |
| Completeness | Main information missing | Some requested details missing | Required facts, constraints and conflicts covered |
| Citation support | Central claims unsupported | Some support missing or ambiguous | Every factual claim supported |
| Abstention/limits | Unsupported answer or unjustified refusal | Important limitation missed | Answers when justified and explains unavailable or conflicting evidence |

- N/A is not zero.
- Citation support is N/A for a bare refusal with no factual claims.
- Completeness is N/A only if it truly cannot be assessed. An unanswerable question is still assessable.
- Correctness and abstention always need a number.
- Track appropriate and unnecessary refusals separately.

The steps for comparing a human with the judge are in [Compare human ratings with the LLM evaluator](ANSWER_REVIEW.md). In short:

1. Two reviewers score the same saved answer and evidence, without seeing machine grades.
2. Compare exact agreement and mean absolute error. Use weighted kappa only if the sample supports it.
3. Record adjudicated decisions separately and keep the original ratings.
4. Compare the judge with the human ratings. The critic is never its own sole evaluator.

The 80% agreement per dimension is a development gate, not proof the judge is reliable. A hand-picked 20-case review is not an unbiased estimate. For error analysis, inspect 20 to 30 failed development cases (or all if fewer). Sort them into retrieval misses, missing passages, generation mistakes and bad refusals.

<a id="answer-evaluation-first-baseline"></a>
## Historical answer baseline: BM25+ and GPT-OSS

On October 17, 20 curated development answers were generated from the top five BM25 documents with GPT-OSS 20B through Ollama. The same model graded them.

| Dimension | Mean score, 0 to 2 | Cases scored 2 |
| --- | ---: | ---: |
| Correctness | 1.30 | 11/20 |
| Completeness | 1.20 | 9/20 |
| Citation support | 1.55 | 15/20 |
| Abstention/limits | 1.70 | 16/20 |

Fifteen answers were given and five marked insufficient evidence. The judge was the generating model, so it may favor its own answers. These scores do not describe the current hybrid/Claude app. See [the reproduction guide](ANSWER_EVALUATION.md) and [saved files](../outputs/alex_evaluation/).

<a id="agent-development-evidence"></a>
## Historical agent experiments

[The October 21 record](../evaluation/agent_development/20261021.json) lists question IDs, models, settings, hashes, usage and outcomes. It used **BM25 and Haiku 4.5**, not the current hybrid setup.

- Ten fixed/adaptive pairs: adaptive found extra reference documents in two cases, but useful passages were lost before generation. Mean latency was 5.13 s fixed vs 14.54 s adaptive. Estimated cost was $0.061950 vs $0.159843. Fixed ran first, so caching affects timing.
- Two-case context regression: both target documents kept. `qst_0384` recovered the two priority compaction settings but missed adjustment amounts. `qst_0342` used an unsupplied citation and was rejected.
- A revised critic matched 8/8 AI-proposed synthetic labels (5/8 before the revision). These are not human labels. The fixtures later grew to 14; the score applies only to the original eight.
- A scope/verbosity prompt experiment was reverted with no gain. [The frozen prompt](../evaluation/agent_development/critic_before_scope.txt) is historical, not the current prompt.

These older runs do not measure the deployed system. Do not merge different snapshots into one before/after result.

<a id="matched-fixed-bm25-dense-and-hybrid-runs"></a>
## Fixed runs: BM25, dense and hybrid answers

`src.evaluation.run_fixed` prepares BM25, dense and hybrid runs through the authenticated app. The 30 saved baseline outcomes (10 per method) are accepted as **baseline-file readiness**. They are not one fully matched experiment or a proven quality win. Of the 30, 25 are eligible for the passage judge, four contain unsupported native evidence and one is a runtime error. See [Current status](STATUS.md).

### Configuration

Save a new private config, for example `.cache/evaluation/fixed/config.json`. Paths resolve relative to it. Replace every path. Start with generation disabled.

```json
{
  "format": "rag-fixed-cli-v1",
  "sources": {
    "questions": "/data/questions.jsonl",
    "splits": "/path/to/checkout/evaluation/splits.json",
    "corpus": "/data/documents.parquet",
    "catalog": "/data/catalog.jsonl",
    "source_repo": "/path/to/checkout"
  },
  "authentication": {
    "snapshot_file": "/data/snapshot.json",
    "credentials_file": "/private/session.json",
    "principal_kind": "session"
  },
  "retrieval": {
    "bm25_index": "/indexes/bm25",
    "dense_index": "/indexes/dense-bge-base",
    "device": "cpu",
    "fp16": null,
    "dense_block_rows": 8192,
    "dense_file_cache_policy": "retain"
  },
  "web": {
    "top_k": 5,
    "max_context_chars": 24000,
    "max_tokens": 16000,
    "max_seconds": 120,
    "max_rounds": 2,
    "question_budget_usd": 0.1,
    "allowed_sources": []
  },
  "generation": {"provider": "disabled"},
  "selection": {"split": "development", "count": 20, "repetition": 1},
  "request_filters": {"source_types": []},
  "comparison": {"experiment_id": "fixed-development-check"},
  "output_root": "new-fixed-run"
}
```

- Only the `web` keys shown are allowed. Identity, model and payment overrides are rejected.
- `retrieval` also accepts `bm25_weight`, `candidate_k`, `rerank_k`, `rrf_k`. `fp16: null` follows the index (the current BGE-base bundle records `true`).
- `authentication` can add `allowed_doc_ids_file`. `request_filters` can add `after`/`before`. `selection` accepts exact `question_ids` instead of `count`.

### Run

```sh
python -m src.evaluation.run_fixed draft \
  --config .cache/evaluation/fixed/config.json --output .cache/evaluation/fixed/draft.json
python -m src.evaluation.run_fixed preflight \
  --config .cache/evaluation/fixed/config.json --output-root .cache/evaluation/fixed/preflight-new
python -m src.evaluation.run_fixed run --config .cache/evaluation/fixed/config.json
```

- `draft` selects question IDs and text. It does not verify permissions.
- `preflight` authenticates, loads one bounded index pair and builds manifests. It makes no provider calls.
- `run` with generation disabled only validates. It is not a quality experiment.
- Coordinate with the stack owner before loading full indexes. The CLI does not stop services.

For an approved live run, `generation` must declare `provider: "anthropic"`, the model, a positive `budget_usd`, `input_price_per_million` and `output_price_per_million`. Optional: `max_output_tokens`, `timeout_seconds`. The API key stays in the environment. Execution also needs `--allow-paid` and a `--max-total-cost-usd` that matches the plan.

Outputs are `plan.json`, manifests and input mappings, per-method run directories, private attempt journals and command receipts. References stay out of generation inputs. `resume` keeps the original manifests, session and settings. It rejects changed inputs and uncertain attempts, and does not replay recorded failures. Final-split selection also needs `allow_final: true` and an accepted freeze.

<a id="reproducible-offline-experiment-records"></a>
## Keep a run reproducible

The [input checker](../src/evaluation/experiment_inputs.py) and [record writer](../src/evaluation/experiment_io.py) verify files from a trusted producer. They do not generate answers or grant access.

A run needs an `ExperimentManifest` and an `InputFiles` mapping for the questions, splits, corpus, catalogue, index manifest, retrieval and comparison settings, permissions, runtime config, generator prompt and source checkout. Add critic/router prompts and the final freeze when required. Paths in the mapping are relative to its file. [The input tests](../tests/test_experiment_inputs.py) have a tiny full example.

```sh
python -m src.evaluation.experiment_io preflight --manifest /private/run/manifest.json --inputs /private/run/inputs.json
python -m src.evaluation.experiment_io validate --run /private/run/fixed --require-complete
python -m src.evaluation.experiment_io pair --fixed /private/run/fixed --adaptive /private/run/adaptive
python -m src.evaluation.experiment_io export-scoring --run /private/run/fixed --references /private/run/references.jsonl --output /private/run/scoring-new
```

Rules:

- Runtime questions hold only IDs and text. References join **after generation**.
- Hashes use exact bytes for the corpus, splits, index manifest and prompts. A changed file on resume is detected.
- Source identity covers Git HEAD, staged and unstaged changes and untracked files. Ignored data needs its own hashes. A checkout check is not a filesystem snapshot.
- New files are written exclusively and never replace earlier outcomes. Uncertain paid attempts are not replayed. Inspect a stale writer lock before removing it.
- Scoring keeps failures and joins references on the full case key `(manifest_id, run_id, question_id, question_sha256, runtime_case_sha256)`.
- Checksums detect corruption, not a malicious file owner, and do not prove current access.
- A final freeze must say `accepted: true`, bind settings per workflow, and name its calibration file and hash. It does not prove human calibration happened. No accepted CAL freeze exists yet.

<a id="offline-bridge-to-the-rc2-evaluator"></a>
## Grade saved answers with the external evaluator

The bridge sends saved answers to Alex's RC2 rubric and citation audit. RC2 is the name of the saved grading-prompt version, not a model. Grading compares the answer with its reference. The citation audit checks support in the cited evidence. The three evaluator modules and two prompt files are pinned at commit `38fb5088cb479eeb89b74175168b60872e047b25` and checked by Git blob hash. **Do not edit the prompt or library copies.** See [the bridge](../src/evaluation/evaluator_bridge.py) and [frozen prompts](evaluation_prompts/).

### 1. Build a judge bundle (no model calls)

```sh
python -m src.evaluation.evaluator_bridge project \
  --run /private/run/fixed \
  --scoring-inputs /private/run/scoring-new/scoring_inputs.jsonl \
  --output /private/run/rc2-new
```

Make one bundle per method, manifest and repetition. `--allow-partial` marks an incomplete run. The bundle holds answers, separate reference questions, a case ledger and fingerprints.

- Passage-only answers (full or partial) with an exact generation input and a reference can be projected.
- Native records or counts, runtime errors, clarifications and missing inputs, references or cases stay unscored. Native returned sources also exclude a case.

### 2. Plan, review and approve a live judge run

```sh
python -m src.evaluation.run_evaluator prepare --help
python -m src.evaluation.run_evaluator run --help
```

`prepare` needs the bundles, an optional exact selection, the judge model and prices, score/audit/case/total cost caps, a new plan path and a new run-output path. It only writes files. Check the selected answers and where the data goes before approving. `run` also needs the plan and approval hashes, a separate approval file, `--allow-paid` and the matching total cap. See [the CLI](../src/evaluation/run_evaluator.py) and [plan examples](../tests/test_run_evaluator_cli.py).

The runner:

- Reserves a whole case, journals each stage before sending it and disables SDK retries.
- Saves bounded raw responses before parsing.
- Stops on the first invalid or missing response, uncertain usage, or provider or budget failure. A valid grade is kept if its usage is incomplete, but no audit or later case is sent.
- Gives grading the references and the citation audit only the cited evidence.

Neither output is a human rating.

### 3. Attach finished results (no new judge call)

```sh
python -m src.evaluation.evaluator_bridge bind-results \
  --bundle /private/run/rc2-new --results /private/run/judge-results.json \
  --evaluator-manifest /private/run/judge-manifest.json --output /private/run/bound-results.json
```

Binding rechecks case identity, hashes, selected IDs, score dimensions and audit schema. Missing results and errors stay visible. A valid schema or matching quote does not prove factual support.

### What has been judged

A separately approved three-case pilot graded the saved BM25, dense and hybrid answers to **one question, `qst_0001`**.

- It completed three grades and three citation audits, kept six raw responses and made 12 SDK requests.
- Reported cost was **$0.048613** (cap $0.30). Invoices were not checked.
- All four machine dimensions scored 2/2 on all three cases. This is not validated perfect quality.
- Flags kept: BM25 has a refusal/uncited-evidence concern. Hybrid has uncertain audit coverage and an ambiguous-target concern. Dense has no human validation.
- Blank human review files exist. Their identities reveal the method, so blinding is not established.
- The other 22 eligible baseline cases are ungraded.

Two earlier attempts failed. One pilot had invalid grades, no audits, $0.032377 reported cost and no raw output kept. Another had provider errors, no grades, $0 reported generation cost and incomplete token counts. The runner then gained native schemas, raw capture and a fix for an empty SDK bearer-token setting.

<a id="critic-calibration-capture"></a>
## Calibrate the critic's stopping decisions

**CAL-01 asks whether the critic's decisions agree with human review.** It is not about matching documents to a reference answer.

[The collector](../src/evaluation/calibration_runs.py) records real authenticated adaptive runs and the exact evidence each critic call saw. [The calibration module](../src/evaluation/critic_calibration.py) picks calibration questions, validates captures, exports blank review packets and analyzes human judgments. These are Python APIs. There is no `python -m` collection command, so do not invent one.

1. Use calibration-only questions. Record question, source and settings identities.
2. Capture real critic inputs and decisions under current permissions.
3. Export the review packet. A ranking, fixed-run answer or synthetic fixture is not a critic judgment.
4. A human reviews the requested facts, their support and the next action.
5. Analyze only completed, attested human rows. A missing label is unavailable, not zero mistakes.
6. Review the freeze proposal. The module leaves it unaccepted; acceptance is a separate decision.

Saved collection:

- Nine outcomes from an original ten-question selection, with four critic episodes.
- `qst_0002` has a separate unresolved $0.05 intent. The other nine used a reviewed continuation of up to $0.45 within the original $0.50 total.
- No final answer was produced. Five cases stopped on budget, two on critic errors and two before critique after truncated Jira results. All stayed in round 1.
- Four decisions predicted eight supported facts. These are **not human labels**.
- Eight known whole-case costs total $0.087626. Known stages total $0.089757 (includes a router stage in an unknown-cost case). The totals overlap, so do not add them. One critic stage and the original interrupted question have unknown cost. Invoices are unverified.
- Four blank human judgments are available. Mistake metrics and an accepted freeze are not.

The private packets are shared separately, not in a fresh checkout. The code has changed since, so those captures keep their original source identity. Newer successful two-round browser flows do not supply these labels. See [Current status](STATUS.md).

<a id="paid-answer-comparison"></a>
<a id="artifact-readiness-no-generated-answers"></a>
<a id="evidencecontext-inspection-on-a-deterministic-20-question-sample"></a>
<a id="save-real-first-search-evidence-and-references-without-model-calls"></a>
## Older pilot tools (optional)

`src.agents.pilot` is a development profile, not the authenticated evaluator. It defaults to hybrid and does not load `.env`. `--prepare` saves first-search evidence and separate references without answer-model calls:

```sh
python -m src.agents.pilot --prepare \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --question-ids qst_0001 qst_0316 qst_0412 qst_0342 qst_0384 qst_0482 \
  --output .cache/agent-pilot/review-new
python -m src.evaluation.compare_fixes --label new-offline-check
```

The six questions cover defaults, escrow, conflicting reports, scope, completeness and missing production facts. The synthetic critic cases are in [the fixture file](../tests/fixtures/critic_development_cases.json). They are not human evaluation.

`pilot --live --workflow compare` runs fixed then adaptive on the `--question-ids` sample under a shared pilot cap, independent of the web cap. It stops on errors or budget limits. `--critic-prompt` tests a saved prompt. `--interactive` is retrieval-only unless `--live` is set. Use `--help` for the required model, price, path and budget arguments.

<a id="evaluation-and-results"></a>
# Evaluation: what we test and what the results mean

Use this page to understand the measurements, run a new check, or review saved answers. Start with [Current status](STATUS.md) for the latest completed work and open tasks. Use [Setup](WEB_APP.md) if you want to run the application first.

**Implemented workflows have passed specific functional checks; overall answer-quality improvement has not been demonstrated.** Finding a document, returning HTTP 200, producing a valid citation ID, or receiving a high model-judge score does not by itself prove that an answer is correct.

## Choose the check you need

| Your question | Use | Calls an answer model? |
| --- | --- | --- |
| Does the code behave correctly? | Python tests below | No live model calls in the ordinary test suite |
| Can search find the reference documents? | [Retrieval checks](#match-the-web-retriever) | No; dense search still runs the local embedding model |
| What answers did each search method produce? | [Fixed runs](#matched-fixed-bm25-dense-and-hybrid-runs) | Only when live generation is explicitly enabled |
| Are those answers correct and supported? | [Answer rubric](#answer-rubric), then human review and a validated judge | A model judge is a separate paid step |
| Does the critic know when to keep searching? | [Critic calibration](#critic-calibration-capture) | Capturing real decisions can require model calls; reviewing saved decisions does not |

Run commands from the repository root in the Python environment described in [Setup](WEB_APP.md). Replace example paths with your own. A path such as `/private/run` is a placeholder, not a directory supplied by this repository. Keep a new output directory for each experiment.

<a id="splits-and-scoring"></a>
## Understand the terms and the question splits

- **Retrieval** means finding documents. **Generation** means writing an answer from their evidence.
- **BM25** searches for matching words. **Dense search** compares numerical representations of text to find related meaning. **Hybrid search** combines their rankings.
- The **critic** is a model call that checks the supplied evidence. Its decisions are predictions, not independent human judgments.
- **Fixed RAG** uses one initial search plan, which can contain several text queries, then generates. The **adaptive agent** can inspect a critic's feedback and search or expand a document again.
- A **reference answer** is the dataset's expected answer. Some code calls it `gold`. It belongs in scoring, never in the input used to generate the answer.
- A **manifest** is a saved description of the exact inputs, settings, model, code and permissions used for a run. A **hash** is a fingerprint used to detect changed bytes.
- A **split** is a group of questions reserved for a particular purpose.

[The frozen split file](../evaluation/splits.json) uses seed 5980:

| Split | Questions | Purpose |
| --- | ---: | --- |
| Development | 300 | Find problems and improve the system |
| Calibration | 100 | Review stopping decisions and choose settings before the final test |
| Final | 100 | Evaluate the frozen system; do not use these labels to tune it |

Identical normalized questions and questions sharing reference documents stay together, including chains of overlap. Project overlap can still remain. All splits search the same document collection. The separate 60-question golden development subset is not the same as a small pilot sample.

**Recall@10** is the fraction of reference documents found in the first 10 results. If a question has two reference documents and search finds one, its Recall@10 is 0.5. **nDCG@10** also rewards placing relevant documents near the top: finding a useful document first is better than finding it tenth.

For fair comparisons, keep the questions, documents the user can access, indexes, generator, prompts and budgets the same. Record every deliberate difference. Report failures, missing labels, question counts, result slices, latency, rounds, model requests, tokens and known or unknown cost. Do not report an unknown cost as zero. Reverse run order or repeat runs before making timing claims; operating-system caches are not cleared automatically.

<a id="application-verification"></a>
<a id="current-agent-evidence-flow"></a>
## What is implemented, and what is still unproven?

The current app uses the user's live permissions before retrieval and again before evidence is used or released. The critic checks requested facts, can ask for more search or adjacent text, and can request clarification. Exact supported passages survive later rounds. The critic and generator use the same frozen evidence when generation follows an assessment. A clarification does not retain a conversation; the user must submit a new complete question.

The shared context ceiling defaults to **8 unique documents**, configurable to 5 or 10. Several passages from one document count once. This is a maximum, not a requirement to fill the context. Character and token budgets can select fewer documents. Native records keep their real fields; counts remain atomic results, not fabricated passages.

Current verification includes real authenticated browser flows, generated answers and citations on an earlier build, and a final-build missing-evidence clarification after two critic rounds and `read_more`. One source-supported date disagreed with the dataset reference: April 7 versus April 6. The final clarification made **no answer-generator call**. These are functional checks, not a controlled 5-versus-8 quality comparison. See [Current status](STATUS.md) for the exact limits and remaining work.

CI is configured to run Python tests, build the frontend and exercise a tiny real-account Docker fixture. A configured workflow is not evidence that a particular remote CI run passed. The small fixture does not establish full-corpus performance or a fresh teammate installation.

<a id="run-checks-and-experiments"></a>
<a id="local-correctness-checks-no-paid-calls"></a>
## Start with checks that do not call a paid model

1. Install the declared Python dependencies using [Setup](WEB_APP.md).
2. Run the relevant tests. Database integration tests need the separate test PostgreSQL configuration described in the test/deployment guides.
3. Read saved results before deciding that you need a new model run.

```sh
python -m pytest -q
python -m src.evaluation.run_retrieval --help
python -m src.evaluation.run_hybrid --help
python -m src.evaluation.run_fixed --help
python -m src.evaluation.run_evaluator --help
```

Help commands do not authorize spending. Do not load another full copy of the indexes beside the serving app on a memory-constrained machine. Coordinate expensive local runs with whoever owns the running stack.

<a id="match-the-web-retriever"></a>
<a id="full-development-retrieval-scoring"></a>
## Run retrieval checks with the application's permission rules

These commands use the same bounded PostgreSQL-backed retrieval factory as the authenticated app. They do **not** read `.env` automatically. They require an evaluation shell that can reach PostgreSQL and read the saved data/model files. The default Docker database is not exposed to the host; ask the stack operator to prepare that environment. For manual application testing, use the browser setup instead.

Before running, obtain:

1. The assembled BM25/dense index bundle and the exact offline embedding-model cache.
2. A ready snapshot and canonical document import in PostgreSQL.
3. Runtime-only database environment settings, never the migration or bootstrap login.
4. The question file, split file and an existing private evaluator credential.

A benchmark credential uses the reviewed `rag-benchmark-v1` format; browser login refuses it. An existing human session instead uses `--principal-kind session` and a file with mode 0600 containing exactly `{"format":"rag-session-v1","token":"..."}`. These tools create no account or permission grant. Browser login alone does not create this file. Obtain it through the operator's approved session workflow; do not put a password, token or database URL in command arguments.

Run a small, explicitly recorded development selection first:

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

Remove `--limit 3` only when you intend to run the whole split. Use `--method dense` with both matching BM25 metadata and dense index paths to score only dense rankings. The CLI uses the validated paired bundle; it does not fall back to a second full-resident FAISS loader.

Reciprocal rank fusion (**RRF**) combines the two search rankings. Default hybrid settings are BM25 weight 0.7, dense weight 0.3, 50 candidates from each retriever, a fused/rerank pool of 50, RRF k=60 and **no reranker**. Dense search uses 8192 rows per block and retain-cache advice. Precision follows the index manifest unless explicitly set. See [the factory](../src/retrieval/factory.py) and [settings](../src/config.py).

Useful options:

| Option | Meaning |
| --- | --- |
| `--candidate-k`, `--rerank-k`, `--rrf-k` | Match the corresponding web retrieval settings |
| `--fp16` / `--no-fp16` | Declare query precision; a mismatch with the index fails |
| `--allowed-doc-ids-file`, `--source-types` | Narrow existing access; they cannot grant access |
| `--after`, `--before` | Narrow by timezone-aware dates; unknown dates fail a date filter |
| `--read-more-chars 3000` | Check adjacent text in at most two returned documents without changing document-ranking metrics |
| `--bm25-weights` | Hybrid-only comma-separated weight sweep |
| `--rerankers bge-reranker-base@EXACT_40_HEX_REVISION` | Explicitly enable one pinned, offline reranker |

A weight sweep shares only the current question's candidate metadata. Bodies are not cached across the split. Calibration/final reranking needs a fixed numeric `--rerank-bm25-weight`; do not choose a new best weight from those labels. The evaluator normally requests 20 ranked documents for Recall@20. That does not mean the generator sees 20 documents.

### Read the new output correctly

Each completed run includes predictions, metrics and `manifest.json`. The manifest records the actual backend, snapshot, permission bitmap, policy/session versions, filters, model/index settings, source-file hashes, installed versions and timing conditions. It does not save the session secret. An interrupted run may lack its completion manifest.

Scoring uses reference documents that the authenticated user can access. No accessible reference means N/A, not successful retrieval. An unknown reference excludes that question from quality scoring; errors and query counts remain visible. Denied reference IDs are not exported. Source annotations on dataset questions are scoring slices, not automatic query filters. A permission change stops the run rather than mixing users or access versions.

Latency includes permission resolution and real work. A hybrid variant includes shared search time plus its own materialization/rerank/adjacent-read work. It does not add other variants' costs. p95 uses the empirical nearest-rank percentile; with two questions it is the slower query. Cold-disk performance is not established by starting a fresh process.

<a id="retrieval-results"></a>
## Historical full-corpus retrieval results

These measurements searched **511,958 documents** for 300 development questions. Eighteen had no reference-document labels, so **282 were scored**. They used the earlier unrestricted protocol, not a new permission-scoped answer-quality test.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| Hybrid BM25+ / BGE-base | 0.641 | 0.722 | 0.760 | 0.604 |
| Hybrid BM25+ / Arctic | 0.638 | 0.723 | 0.768 | 0.628 |

- BGE-base hybrid gained 0.029 Recall@10 over BM25+; the 95% bootstrap interval was +0.005 to +0.055, with 19 questions better and 7 worse. Equal hybrid weights scored 0.700.
- A single calibration check scored 94 labeled questions: BM25+ 0.712, BGE-base hybrid 0.767 and Arctic hybrid 0.788 Recall@10. BGE-base's gain was 0.056, interval +0.013 to +0.106. Do not use this split for further tuning; the final split remained untouched.
- Arctic beat BGE-base alone by 0.067 Recall@10, but their hybrids differed by only 0.001 Recall@10; Arctic's nDCG@10 was 0.023 higher. The app keeps BGE-base; the Arctic index remains available.
- Tested rerankers reduced hybrid Recall@10 to 0.637 (`bge-reranker-base`) and 0.669 (`bge-reranker-v2-m3`). They saw one 3,000-character document section, truncated to 512 tokens, often without the title. This does not show that every reranking approach is worse.

[Saved result directories](../evaluation/README.md) contain predictions, scores, category/source slices, settings and timing records. [Full-corpus results](FULL_CORPUS.md) explains screening, confidence intervals and the title experiments.

<a id="reproduction-settings"></a>
### Settings needed to interpret or reproduce those measurements

- Dataset revision: `d36685e273713975ee20299bbf1ab64165575b3c`; see [Data Card](DATA_CARD.md) for extraction and duplicates.
- BM25+: k1=1.5, b=0.75, delta=1, lowercase word tokens, top 20 documents. The vectorized builder is checked against `bm25s`; JAX stays on CPU.
- Dense passage size: 500 tokens with 50-token overlap; document score is its best passage score, using exact cosine search.
- BGE-small: `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; title on first passage, GPU fp16, code `e924855`.
- BGE-base: `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; title on every passage, GPU fp16, code `d11eb1b`.
- Arctic: `Snowflake/snowflake-arctic-embed-l-v2.0` at `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`; `query: ` prefix, title on every passage, L4 bf16, code `38489e0`.
- Hybrid measurements used actual `HybridRetriever` searches with the defaults above. BM25 ran on CPU and query encoding on Colab GPUs; the web app uses CPU. These timings are not a controlled hardware comparison.
- Colab/local Parquet file hashes differ despite matching document-content fingerprints. Runtime setup still requires the exact matching files. Earlier fusion of saved top 20 rankings scored 0.705 at weights 0.6–0.8; it was an estimate, superseded by the live-search measurements above.

<a id="answer-rubric"></a>
<a id="answer-review-fields-and-scores"></a>
<a id="answer-review-rubric-not-yet-completed"></a>
<a id="answer-review-rubric--not-yet-completed"></a>
## Review answer quality

Score each applicable dimension independently. A usable answer needs 2 on every applicable dimension.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Central claim wrong | Main answer right, important details wrong or unverifiable | All substantive claims correct |
| Completeness | Main information missing | Some requested details missing | Required facts, constraints and conflicts covered |
| Citation support | Central claims unsupported | Some support missing or ambiguous | Every factual claim supported |
| Abstention/limits | Unsupported answer or unjustified refusal | Important limitation missed | Answers when justified and explains unavailable or conflicting evidence |

Completeness is N/A without reference answer facts; citation support is N/A without factual claims. N/A is not zero. Track appropriate and unnecessary refusal separately.

1. Give two reviewers the **same exact saved answer and evidence**, without showing machine grades first.
2. Have them score independently and identify unsupported/missing claims.
3. Compare exact agreement and mean absolute error; use weighted kappa only when the sample supports it.
4. Discuss disagreements and record a separate adjudicated decision. Preserve the original ratings.
5. Compare the model judge with these human ratings. Do not use the critic as its own sole evaluator.

The proposed 80% agreement per dimension is a development gate, not proof of judge reliability. Sample successes, failures and boundary cases; a selected 20-case review is not an unbiased population estimate. For error analysis, inspect 20–30 failed development cases, or all if fewer, separating retrieval misses, missing passages, generation mistakes and bad refusal.

<a id="answer-evaluation-first-baseline"></a>
## Historical answer baseline: BM25+ and GPT-OSS

The October 17 baseline generated 20 curated development answers from the top five BM25 documents using GPT-OSS 20B through Ollama, then graded them with the same model.

| Dimension | Mean score, 0–2 | Cases scored 2 |
| --- | ---: | ---: |
| Correctness | 1.30 | 11/20 |
| Completeness | 1.20 | 9/20 |
| Citation support | 1.55 | 15/20 |
| Abstention/limits | 1.70 | 16/20 |

Fifteen answers were given and five marked insufficient evidence. The same-model judge can favor its own answers. Human comparison ratings exist in the historical workbook, but committed code does not fully reproduce the agreement report. These scores do not measure the current hybrid/Claude app. See [the historical reproduction guide](ANSWER_EVALUATION.md) and [saved files](../outputs/alex_evaluation/).

<a id="agent-development-evidence"></a>
## Historical agent experiments

[The October 21 record](../evaluation/agent_development/20261021.json) records exact question IDs, model/settings, code/prompt/source hashes, usage and outcomes. It used **BM25 and Haiku 4.5**, not the current hybrid configuration or GPT-OSS.

- Ten fixed/adaptive development pairs: additional reference documents appeared in two adaptive cases, but useful passages were initially lost before generation. Mean latency was 5.13 s fixed versus 14.54 s adaptive; estimated serving costs were $0.061950 and $0.159843. Fixed ran first, so cache effects limit timing conclusions.
- A two-case context regression retained both target documents. `qst_0384` recovered the two priority compaction settings but missed adjustment amounts; `qst_0342` used an unsupplied citation and was rejected.
- A revised critic matched 8/8 AI-proposed synthetic labels versus 5/8 before revision. These were not human labels or an independent accuracy estimate. The fixture set later grew to 14; the score applies only to the original eight and recorded hashes.
- A scope/verbosity prompt experiment was reverted without a demonstrated gain. [The frozen comparison prompt](../evaluation/agent_development/critic_before_scope.txt) is historical; it is not the current fact-checklist prompt.

Current prompts and code have changed. Do not relabel these older runs as measurements of the deployed system, or combine different snapshots into a controlled before/after result.

<a id="matched-fixed-bm25-dense-and-hybrid-runs"></a>
## Save comparable fixed-run answers

`src.evaluation.run_fixed` prepares BM25, dense and hybrid runs using the authenticated application. Its 30 saved baseline outcomes provide 10 records per method. They form an accepted **baseline-files/report readiness** step, not a single fully matched experiment or a proven quality win. Of those 30 outcomes, 25 are eligible for the passage judge, four contain unsupported native evidence and one is a runtime error. [Current status](STATUS.md) explains the remaining quality gates.

### Prepare the configuration

Save the following as a **new private configuration**, for example `.cache/evaluation/fixed/config.json`. Paths resolve relative to that file; replace every example path. Start with generation disabled.

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

The CLI's allowed `web` keys are exactly those shown; this is not the full web-server environment schema. It rejects identity/model/payment overrides there. Retrieval also accepts `bm25_weight`, `candidate_k`, `rerank_k` and `rrf_k`. `fp16:null` follows the index; the current BGE-base bundle records `true`. Authentication can add an `allowed_doc_ids_file` to narrow existing access. Filters can add aware `after`/`before` dates. Selection accepts exact `question_ids` instead of `count`.

### Run the preparation steps

```sh
python -m src.evaluation.run_fixed draft \
  --config .cache/evaluation/fixed/config.json --output .cache/evaluation/fixed/draft.json
python -m src.evaluation.run_fixed preflight \
  --config .cache/evaluation/fixed/config.json --output-root .cache/evaluation/fixed/preflight-new
python -m src.evaluation.run_fixed run --config .cache/evaluation/fixed/config.json
```

`draft` only selects question IDs/text and reports unverified permissions. `preflight` authenticates, loads one bounded index pair and creates actual manifests without provider calls. With generation disabled, `run` is **validation-only**; it is not an answer-quality experiment. Coordinate with the stack owner before loading full indexes; the CLI does not stop services for you.

For an approved live run, `generation` must explicitly declare `provider:"anthropic"`, model, positive `budget_usd`, `input_price_per_million` and `output_price_per_million`. Optional execution settings are `max_output_tokens` and `timeout_seconds`. The API key stays in the environment. Execution additionally requires `--allow-paid` and `--max-total-cost-usd` matching the plan. A configuration value or old approval does not authorize a new batch.

Outputs include `plan.json`, prepared manifests/input mappings, per-method run directories, private attempt journals and command receipts. References stay outside generation inputs. `resume` keeps the original manifests, session and settings; it rejects changed inputs or uncertain attempts. Recorded failures are completed attempts, not missing questions to replay. Benchmark sessions are renewed per process and cannot silently resume an older binding. Final selection additionally needs `allow_final:true` and an accepted freeze.

<a id="reproducible-offline-experiment-records"></a>
## Keep a run reproducible

The [input checker](../src/evaluation/experiment_inputs.py) and [record writer](../src/evaluation/experiment_io.py) do not generate answers or grant access. They verify files supplied by a trusted runtime producer.

A run needs an `ExperimentManifest` plus an `InputFiles` mapping for questions, splits, corpus, catalogue, index manifest, retrieval/comparison settings, permissions, runtime configuration, generator prompt and source checkout. Critic/router prompts and final freeze are included when required. Paths in a mapping JSON file are relative to that file. [Input tests](../tests/test_experiment_inputs.py) contain a complete tiny example; it is not a production dataset.

```sh
python -m src.evaluation.experiment_io preflight --manifest /private/run/manifest.json --inputs /private/run/inputs.json
python -m src.evaluation.experiment_io validate --run /private/run/fixed --require-complete
python -m src.evaluation.experiment_io pair --fixed /private/run/fixed --adaptive /private/run/adaptive
python -m src.evaluation.experiment_io export-scoring --run /private/run/fixed --references /private/run/references.jsonl --output /private/run/scoring-new
```

Technical rules worth keeping:

- Runtime questions contain only IDs/text. References join **after generation**, using the exact question text and runtime-case fingerprint.
- Raw corpus, split, index-manifest and UTF-8 prompt hashes use exact bytes. Catalogue JSONL uses ordered canonical catalogue records. Retrieval/comparison/permission JSON uses sorted keys, compact separators, UTF-8 and finite values; raw file hashes also detect changed formatting on resume.
- Runtime config contains `generation_model`, `retrieval`, `budgets`, `temperature`, `workflow` and `max_rounds`. Source identity includes Git HEAD, staged/unstaged changes and ordinary nonignored untracked files. Ignored data needs separate hashes; submodule or external symlink targets need explicit treatment. Checking a checkout is not making a filesystem snapshot.
- The producer calls `verify_inputs`, then writes through `RunDirectory`. New files are exclusive, synchronized to disk, and never silently replace prior outcomes. Resume preserves source/input/settings identity and does not replay uncertain paid attempts. Inspect a stale writer lock before removing it manually.
- Scoring preserves failures and joins references by the full saved identity. A whole case key is `(manifest_id, run_id, question_id, question_sha256, runtime_case_sha256)`, not merely a reused question ID.
- Checksums detect corruption and disagreement; they do not authenticate a malicious file owner or establish current document access. Capture export performs its own live permission check.
- A final freeze must say `accepted:true`, bind scientific settings per workflow, and identify a contained calibration file and its hash. This is an explicit acceptance record, not proof that human calibration happened. No accepted CAL freeze is available yet.

<a id="offline-bridge-to-the-rc2-evaluator"></a>
## Grade saved passage answers with the external evaluator

The bridge connects canonical saved answers to Alex's RC2 answer rubric and citation audit. RC2 names the saved grading-prompt version; it is not a separate model. Grading compares an answer with its reference, while the citation audit checks support in the cited evidence. It preserves the three evaluator modules and two prompt files at commit `38fb5088cb479eeb89b74175168b60872e047b25`. It checks their Git blob hashes and uses the real candidate parser. **Do not edit those prompt/library copies casually.** See [the bridge](../src/evaluation/evaluator_bridge.py) and [frozen prompts](evaluation_prompts/).

### 1. Prepare a judge bundle without model calls

```sh
python -m src.evaluation.evaluator_bridge project \
  --run /private/run/fixed \
  --scoring-inputs /private/run/scoring-new/scoring_inputs.jsonl \
  --output /private/run/rc2-new
```

Make a separate bundle per method/manifest/repetition. `--allow-partial` explicitly marks an incomplete run and still requires every committed outcome. The bundle contains answers, separate reference questions, a case-disposition ledger and source/input/prompt fingerprints.

Passage-only generated answers or partial answers can be projected when their exact generation input and a reference are available. Native records/counts, runtime errors, clarifications, missing inputs/references and missing cases remain explicitly unscored. Native returned sources also exclude a case; do not turn their fields into quotations.

### 2. Prepare, review and separately approve a live judge plan

```sh
python -m src.evaluation.run_evaluator prepare --help
python -m src.evaluation.run_evaluator run --help
```

`prepare` needs bundles, an optional exact selection, the judge model/prices, score/audit/case/total caps, a new plan path and a new run-output path. It only writes files. Review the exact selected answer identities and data destination before approving execution. `run` additionally needs exact plan/approval hashes, the separate approval file, `--allow-paid` and the matching total cap. See [CLI source](../src/evaluation/run_evaluator.py) and [tested plan examples](../tests/test_run_evaluator_cli.py).

The runner reserves a whole case once, journals each stage before dispatch, disables SDK retries and saves private bounded raw responses before parsing. The first invalid/missing response or uncertain usage stops further live admission. Answer grading receives references; citation audit receives only the answer's cited generation evidence. Neither output is a human rating.

### 3. Attach already-produced results without another judge call

```sh
python -m src.evaluation.evaluator_bridge bind-results \
  --bundle /private/run/rc2-new --results /private/run/judge-results.json \
  --evaluator-manifest /private/run/judge-manifest.json --output /private/run/bound-results.json
```

Binding rechecks the complete case identity, canonical projection, code/prompt/input hashes, selected IDs, score dimensions and audit schema. Missing results and judge/audit errors remain visible. A valid schema or matching quotation does not prove factual support.

### What has actually been judged?

A separately approved three-case pilot graded the saved BM25, dense and hybrid answers to **the same one question, `qst_0001`**. It completed three grades and three citation audits, retaining six raw responses and reporting 12 SDK requests including token counts. Reported cost was **$0.048613**, within its $0.30 cap; invoices were not independently checked.

All four machine dimensions were 2/2 on all three cases. Do not advertise that as validated perfect quality. BM25 has a refusal/uncited-evidence concern; hybrid has uncertain audit coverage and an ambiguous-target concern; dense still lacks human validation. Original flags/scores are retained. Blank human review materials exist, but full identities reveal the method, so reviewer blinding is not established. The other 22 eligible baseline cases remain ungraded.

Earlier failed attempts remain separate: one three-case pilot had invalid grades, no audits, and $0.032377 reported cost with no retained raw output; another three admissions produced provider errors, no grades, and $0 reported generation cost with incomplete token accounting. The guarded runner later added native schemas/raw capture and corrected an empty SDK bearer-token configuration. A successful token-count diagnostic was not itself a successful judge case.

<a id="critic-calibration-capture"></a>
## Calibrate the critic's stopping decisions

**CAL-01 asks whether the critic's decisions agree with human review.** It does not mean checking whether a retrieved document matches a reference answer.

[The collector](../src/evaluation/calibration_runs.py) uses real authenticated adaptive runs. Its private captures identify the exact evidence presented to each critic call. [The calibration module](../src/evaluation/critic_calibration.py) selects calibration questions, validates captures, exports blank review packets and analyzes human judgments. These are Python APIs, not a public `python -m ...` collection CLI; use the tested producer workflow rather than inventing a command.

1. Use calibration-only questions and record exact question/source/settings identities.
2. Capture real critic inputs and decisions under current document permissions.
3. Export the review packet. A retrieval ranking, fixed-run answer or synthetic fixture is not a critic judgment.
4. Have a human review the requested facts, their support and the appropriate next action.
5. Analyze only completed, attested human rows. Missing labels remain unavailable, not zero mistakes.
6. Review a source/settings-bound freeze proposal. The module always leaves it unaccepted; acceptance is a separate decision.

The saved collection has nine outcomes from an original ten-question selection and four critic episodes. The first question, `qst_0002`, has a separate unresolved $0.05 intent; the nine untouched questions used a reviewed continuation with up to $0.45, retaining the original $0.50 total. No final answer was produced: five cases stopped on budget, two on critic errors, and two before critique after truncated Jira results. All remained in round 1. Four decisions predicted eight supported facts; those predictions are **not human labels**.

Eight known whole-case costs total $0.087626. Known stages total $0.089757, including a router stage in an otherwise unknown-cost case. These totals overlap; do not add them. A critic stage and the original interrupted question remain unknown. Invoice amounts are unverified.

Four blank human judgments are available; mistake metrics and an accepted freeze are not. These private collected packets are provided separately rather than bundled with a fresh checkout. The current deployed code has since changed, so those older captures must retain their original source identity. Current successful two-round browser flows do not fill these human labels or retroactively calibrate that older packet. See [Current status](STATUS.md) for the next acceptance step.

<a id="paid-answer-comparison"></a>
<a id="artifact-readiness-no-generated-answers"></a>
<a id="evidencecontext-inspection-on-a-deterministic-20-question-sample"></a>
<a id="save-real-first-search-evidence-and-references-without-model-calls"></a>
## Optional older pilot tools

`src.agents.pilot` is a separate development/benchmark profile, not the authenticated production evaluator. It defaults to hybrid and does not load `.env`. Its `--prepare` operation saves real first-search evidence and separate review references without answer-model calls:

```sh
python -m src.agents.pilot --prepare \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --question-ids qst_0001 qst_0316 qst_0412 qst_0342 qst_0384 qst_0482 \
  --output .cache/agent-pilot/review-new
python -m src.evaluation.compare_fixes --label new-offline-check
```

The six development examples cover defaults, escrow, conflicting reports, scope, completeness and missing production facts. Synthetic critic cases are in [the fixture file](../tests/fixtures/critic_development_cases.json). They are not human evaluation.

A separately approved `pilot --live --workflow compare` runs fixed then adaptive with a shared pilot cap; `--question-ids` fixes the sample. It stops on errors/budget limits and preserves outputs. Its cap is independent of the web process's cap. `--critic-prompt` supports a separate saved prompt experiment; `--interactive` is retrieval-only unless `--live` is explicitly set. Use `--help` for all required model, price, path and budget arguments. Do not treat an old approval or a new example command as permission for additional paid requests.

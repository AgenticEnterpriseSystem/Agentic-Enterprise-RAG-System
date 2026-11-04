# Evaluation and results

**Retrieval is measured, including the web app's BM25+ and BGE-base hybrid. Overall
agent answer-quality improvement is not established.** Small live runs are development
checks, not final evaluation. Independent human review, judge validation, critic
calibration and comparable BGE-base hybrid answer evaluation remain pending.

## Match the web retriever

The authenticated web runtime and both retrieval evaluation commands use the
same store-backed bounded factory and current PostgreSQL policy. The older pilot
remains a separate development profile. Hybrid defaults are BM25 weight 0.7,
dense weight 0.3, 50 candidates per retriever, a fused/rerank pool of 50, RRF k=60,
CPU encoding, manifest-selected precision and no reranker. CLI evaluations do not
read `.env`: provide runtime-only database environment settings, a ready imported
snapshot, an explicit authenticated principal and matching artifact paths.

```sh
python -m src.evaluation.run_hybrid \
  --bm25-index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --snapshot-file /path/ready-snapshot.json \
  --credentials-file /path/private/benchmark-credentials.json \
  --principal-kind benchmark \
  --questions .cache/corpus/questions.jsonl --split development \
  --output-dir .cache/evaluation/hybrid-development
```

The benchmark credential is the existing private `rag-benchmark-v1` file from
reviewed operator provisioning; public login still refuses it. For an existing
human session, use `--principal-kind session` and an ordinary 0600 JSON file with
exactly `{"format":"rag-session-v1","token":"..."}`. The token is an existing
session secret, not a password or role claim. Never put a token, database URL or
password in CLI arguments. The runner calls the accepted AuthService and shares
one nonowner Database pool; it creates no accounts/grants and has no demo fallback.
Model loading is forced offline; pre-stage the exact manifest-pinned model cache.

Result directories are exclusive and never overwrite saved runs. The questions
file (`questions.jsonl` in the team Drive's `files_web_app` folder; see
[Setup](WEB_APP.md#get-the-artifacts)) is separate from the assembled web bundle.
Pass artifact paths explicitly; these commands read canonical bodies from the
store and do not load the raw corpus into Arrow or resident full FAISS.
Web settings `RAG_CANDIDATE_K`, `RAG_RERANK_K`, `RAG_RRF_K` and `RAG_FP16`
correspond to `--candidate-k`, `--rerank-k`, `--rrf-k` and `--fp16/--no-fp16`.
Dense defaults are 8192 rows per block and explicit retain-cache advice; neither
setting disables shard/model/snapshot validation.

`--bm25-weights` accepts a comma-separated sweep, sharing only one question's
candidate metadata across variants. Bodies are not cached across the split.
Rerankers are opt-in with
`--rerankers bge-reranker-base@EXACT_40_HEX_REVISION`; at most one pinned offline
reranker is allowed per invocation. Compare other rerankers in separate processes.
Best-development weight selection uses a second bounded pass instead of keeping
all-question candidates. Calibration/final reranking requires a fixed numeric
`--rerank-bm25-weight`. Existing notebook/library helpers retain their historical
development behavior; these command entry points have no legacy FAISS fallback.
Metrics still request 20 results for Recall@20, while the web answer context
starts with five.

`run_retrieval --method dense` scores only the dense rankings while loading the
accepted paired bundle; provide its matching BM25 metadata index as well as the
dense directory. This shares the validated catalogue without another loader.

`--allowed-doc-ids-file` and `--source-types` only narrow current explicit grants;
an empty document list denies everything. `--after`/`--before` require aware ISO
timestamps, and unknown dates fail a date restriction. Question `source_types`
annotations remain metric slices, not automatic query filters.
`--read-more-chars 3000` checks adjacent reads in at most two returned documents
and exports exact offsets/revision/text hashes without changing ranked-document
metrics. `--limit N` is recorded as an explicit partial smoke selection.

New predictions preserve the original IDs/scores/error fields and add original,
eligible and unknown reference counts plus `scoring_status`. Metrics use the
**authorized-reference intersection**, explicitly labeled in the summary. No
eligible reference means `unanswerable` and N/A, not a successful zero-recall
case. Any unknown reference excludes that whole question from quality scoring.
Queries and errors remain counted; denied reference IDs are not exported. Original
global reference denominators apply only when every reference is known and
eligible. Scoped scores and historical full-corpus scores have different bases.

Completed runs also have `manifest.json`, binding the actual backend, SnapshotRef,
principal/session row ID, policy/session versions, scope/bitmap, filters, input
hashes, index/model settings, installed versions, actual source-file hashes and
cache conditions. Session secrets are never serialized. A missing complete
manifest marks an interrupted attempt. Git-unavailable images report unknown
commit/dirty state; source-file hashes and the operator's frozen-source receipt
provide the source binding. A run freezes one permission binding; a policy or
session change fails closed rather than mixing successful permissions.

Latency includes identity/scope resolution and actual work. Hybrid variants each
include shared candidate-search time plus their own materialization/rerank/read
cost, not the sum of other variants. OS caches are not cleared, so first-process
timings are not cold-disk claims. Full-resident FAISS is reserved for the deliberate
separate `scripts/compare_retrieval_backends.py` reference under its memory guards.

## Retrieval results

Every method searched all **511,958 documents** for the 300 development questions.
Eighteen questions lack reference-document labels: **282 were scored**.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| **Hybrid: BM25+ and BGE-base (web default)** | **0.641** | **0.722** | 0.760 | 0.604 |
| Hybrid: BM25+ and Arctic | 0.638 | 0.723 | **0.768** | **0.628** |

Recall measures reference documents found; nDCG also rewards higher ranks.
Retrieval scores do not measure answer correctness or citation support.

- **The hybrid rows are live measurements of `HybridRetriever` with the web defaults:**
  BM25 weight 0.7, 50 candidates per retriever, a fused pool of 50, RRF k=60 and no
  reranker (`python -m src.evaluation.run_hybrid`, from
  [`notebooks/hybrid_colab.ipynb`](../notebooks/hybrid_colab.ipynb)). Queries were
  encoded on a Colab GPU; the web app encodes on CPU with the precision recorded in the
  index manifest.
- **BGE-base hybrid vs BM25+ alone:** Recall@10 +0.029 (95% bootstrap confidence interval
  +0.005 to +0.055; 19 questions better, 7 worse). Equal weights scored only 0.700.
- **Calibration** (94 scored questions, one check per setting): BM25+ 0.712, BGE-base
  hybrid 0.767, Arctic hybrid 0.788 Recall@10. The BGE-base hybrid's gain over BM25+ held
  (+0.056, interval +0.013 to +0.106). The calibration split should not be used for
  further tuning of these settings; the final split is untouched.
- **Arctic vs BGE-base:** better alone (+0.067 Recall@10 on development), but the hybrids
  tie on Recall@10 (+0.001) with slightly better Arctic ranking (nDCG@10 +0.023). The team
  keeps BGE-base; the Arctic index remains available.
- **Rerankers made results worse as tested:** `bge-reranker-base` lowered hybrid Recall@10
  to 0.637 and `bge-reranker-v2-m3` to 0.669. They saw one 3,000-character section of each
  document (cut to 512 tokens), often without its title, so long reference documents
  dropped. The web app does not enable a reranker.

Each result directory in [`evaluation/results/`](../evaluation/README.md) contains
category/source slices, timings, settings, code versions, index manifests, ranked
predictions and upstream-format outputs. Category tables, confidence intervals, the
title-on-every-passage experiment and the embedding model screening are in
[Retrieval results](FULL_CORPUS.md).

### Reproduction settings

- Dataset revision: `d36685e273713975ee20299bbf1ab64165575b3c`; extraction and duplicate
  handling are described in the [data card](DATA_CARD.md).
- BM25+: k1=1.5, b=0.75, delta=1, lowercase word tokens, top 20 documents.
  Its vectorized builder is tested against `bm25s` scores. JAX is kept on the CPU so that
  `bm25s` does not reserve GPU memory.
- Dense models use 500-token passages with 50-token overlap, max-over-passages document
  scores and exact cosine search. Models, pinned revisions and query prefixes are listed in
  `src/config.py` and recorded in each index manifest:
  - BGE-small: `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
    title on the first passage only, GPU fp16 (code `e924855`).
  - BGE-base: `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`,
    title on every passage, GPU fp16 (code `d11eb1b`). This is the web app's dense index.
  - Arctic: `Snowflake/snowflake-arctic-embed-l-v2.0` at
    `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, `query: ` prefix, title on every passage,
    L4 bf16 (code `38489e0`).
- Runs used CPU BM25 and Colab GPUs; timings and hardware differ, so they are not
  controlled latency comparisons. The Colab Parquet file hash differs from the local file,
  despite matching document content fingerprints. Runtime integration requires matching
  artifact hashes.
- Earlier fusion of saved top-20 rankings (0.705 with BM25 weights 0.6–0.8) was an
  offline estimate; the live hybrid measurement above replaces it.

## Agent development evidence

The [October 21 run record](../evaluation/agent_development/20261021.json) preserves
model/settings, question IDs, source/prompt hashes, runtime versions, usage and
per-case outcomes. These runs used **BM25 and Haiku 4.5**, not GPT-OSS or BGE-base
hybrid. They were recorded in an isolated integration checkout before changes
were ported; hashes identify the measured snapshots. The critic and generator
prompts have changed since these runs, including hybrid-context improvements
and the later fact-checklist and scoped-claim updates. These runs describe the
earlier prompts, not the current agent.

- Ten development pairs: follow-ups found additional reference documents in two
  cases, but their useful passages were initially dropped before generation.
  Mean latency was 5.13 s fixed versus 14.54 s agent; estimated total serving costs
  were $0.061950 and $0.159843. Fixed RAG ran first, so cache effects limit timing
  interpretation. Answered status is not a correctness score.
- A two-case passage-selection regression retained both target documents.
  `qst_0384` recovered the two priority compaction settings but missed adjustment
  amounts; `qst_0342` emitted an unsupplied citation and was rejected.
- An earlier critic prompt matched 8/8 AI-proposed synthetic labels versus 5/8
  before revision. These were prompt-development examples, not human judgments
  or an independent accuracy estimate. The fixture set now has 14 examples;
  the historical score applies only to the original eight and recorded hashes.
- A later scope/verbosity prompt experiment was reverted. At the time, the default
  matched the [saved pre-experiment prompt](../evaluation/agent_development/critic_before_scope.txt).
  No latency or accuracy improvement was established for that historical experiment.
  The current fact-checklist prompt has since changed; the saved prompt is a
  historical comparison artifact, not a copy of the current default.

Known issues include incomplete answers at round limits, ambiguous policy scope,
missed passage details and occasional invalid model-generated citations. The
current agent can stop with a focused clarification question. Conversation state
is not retained, so the user must submit a new complete question. More searches
or valid citation IDs alone are not quality gains.

Full traces and unfilled review packets remain in local ignored storage, including
`.cache/integration-audit-20261021/` and `.cache/critic-scope-development/`.
They are not included in a fresh clone. Historical packets assembled from multiple
snapshots must not be presented as a controlled before/after comparison.

## Run checks and experiments

Use the [native Python setup](WEB_APP.md#build-bm25-locally) and matching artifacts.
Run commands from the repository root. Use a **new output directory/label** each
run so historical results are preserved.

```sh
# Local correctness checks; no paid calls
python -m pytest -q

# Artifact readiness; no generated answers
python -m src.agents.pilot \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base

# Full-development retrieval scoring
python -m src.evaluation.run_retrieval --method bm25 --split development \
  --bm25-index .cache/hybrid-artifacts/indexes/bm25 \
  --snapshot-file /path/ready-snapshot.json \
  --credentials-file /path/private/benchmark-credentials.json \
  --output .cache/retrieval-runs/bm25-development-new

# Evidence/context inspection on a deterministic 20-question sample
python -m src.evaluation.compare_fixes --label new-check

# Save real first-search evidence and references without model calls
python -m src.agents.pilot --prepare \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --question-ids qst_0001 qst_0316 qst_0412 qst_0342 qst_0384 qst_0482 \
  --output .cache/agent-pilot/new-review
```

The pilot defaults to hybrid retrieval, like the web app, but does not read `.env`:
pass the assembled bundle's paths as above (its own defaults point at
`.cache/corpus/`). For a BM25 baseline, use `--retrieval bm25` with `--corpus` and
`--index`. Questions default to `.cache/corpus/questions.jsonl`.

The six explicit cases cover defaults, escrow, conflicting reports, policy
interpretation, multi-part completeness and missing production facts. Selection
rejects non-development IDs. References appear only in review files, not runtime
model/retrieval inputs. The synthetic scope fixtures are in
[critic_development_cases.json](../tests/fixtures/critic_development_cases.json).

For BGE retrieval scoring, use `--method dense --device cuda` with the matching
index. For offline dense memory measurement use `compare_fixes --memory` with a
fresh label; `--legacy-ties` replays an earlier cutoff rule. These offline checks
are not answer evaluation. Avoid loading a second full corpus alongside the web
backend on a memory-constrained machine.

### Paid answer comparison

Export `ANTHROPIC_API_KEY` into the shell; the pilot does not automatically read
`.env`. Replace the placeholders with an approved model, current token prices and
an allocated remaining budget:

```sh
python -m src.agents.pilot --live --workflow compare --count 3 \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --index .cache/hybrid-artifacts/indexes/bm25 \
  --dense-index .cache/hybrid-artifacts/indexes/dense-bge-base \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS --question-budget-usd 0.10 \
  --input-price INPUT_USD_PER_MILLION --output-price OUTPUT_USD_PER_MILLION \
  --output .cache/agent-pilot/new-comparison
```

`compare` runs fixed RAG then agent on the same sample with one shared total cap.
An error or budget stop ends the run early and preserves completed cases. Outputs
include per-arm cases/review files, `comparison.json`, and a manifest. Cases retain
exact evidence, model inputs/outputs and separate search/critic/generation timings.
Manifests identify code, prompts, runtime versions, corpus and model configuration.

Use `--question-ids` instead of `--count` for exact repeatable selection. For a
critic-only change, use `--workflow agent --critic-prompt PATH_TO_SAVED_PROMPT`
and compare with a separately saved candidate run, keeping other settings fixed.
CLI caps are independent of the web server cap; allocate their sum within the
overall budget. `--interactive` accepts custom questions and `/quit`; it is
retrieval-only unless `--live` is supplied. It cannot use batch workflow options.

## Splits and scoring

[Splits](../evaluation/splits.json) are frozen with seed 5980:
**300 development / 100 calibration / 100 final**. Shared reference documents and
identical normalized questions stay together, including transitive overlap.
Project overlap can remain. The 60-case golden development subset is distinct
from the pilot samples. Do not tune on calibration/final labels; freeze settings
before final evaluation. Documents form one shared corpus.

For comparisons, match corpus, questions, access policy, top-k, context limits,
generator and model. Record every changed setting, failures and exclusions.
Deduplicate by document ID for retrieval scoring. Questions without reference IDs
are N/A for retrieval, not automatically unanswerable. Missing corpus references
remain in denominators; eligible errors score zero. Report counts, category/source
slices, median/p95 latency, calls, rounds, tokens and cost. Unknown cost stays null;
judge costs are separate. Repeat paired runs in reversed order when assessing time.

### Answer review rubric — not yet completed

Score each applicable dimension independently; a usable answer needs 2 on each.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim | Main answer right; substantive details wrong/unverifiable | All substantive claims correct |
| Completeness | Main information missing | Required details missing | Required facts, constraints and conflicts covered |
| Citation support | Central claims unsupported | Some claims unsupported/ambiguous | Every factual claim supported |
| Abstention | Unsupported answer or unjustified refusal | Key limitation missed | Answers when justified; explains missing/conflicting evidence |

Completeness is N/A without answer facts; citation support is N/A without factual
claims. Track appropriate and unnecessary abstention separately. Two reviewers
should independently score 20 development successes, failures and boundary cases,
record agreement and adjudicate disagreements. Validate an LLM judge against those
labels; the proposed 80% per-dimension agreement threshold is a development gate,
not proof of reliability. The critic cannot be its own sole evaluator.

Inspect 20–30 failed development examples (or all if fewer), retaining run/question
IDs, ranked evidence, selected context, answer, failed dimension and suspected
cause. Distinguish retrieval misses, passage omission, generation errors and bad
abstention. Selected failures do not estimate an overall failure rate. Test fixes
on the same development set and retain real-conflict/unanswerable counterexamples.

## Answer evaluation: first baseline

[Answer evaluation](ANSWER_EVALUATION.md) runs the rubric above with an LLM judge.
Its first run (2026-10-17) answered 20 curated development questions with GPT-OSS
20B (through Ollama) from the top five BM25 documents, then graded each answer with
the same model. Results are in `outputs/alex_evaluation/`.

| Dimension | Mean (0–2) | Scored 2 |
| --- | ---: | ---: |
| Correctness | 1.30 | 11 of 20 |
| Completeness | 1.20 | 9 of 20 |
| Citation support | 1.55 | 15 of 20 |
| Abstention/limits | 1.70 | 16 of 20 |

Fifteen answers were given and five marked insufficient evidence. This is not a
measurement of the web configuration (hybrid retrieval with Claude), and the same
model wrote and graded the answers, which can inflate scores. Human scores for
comparison are in `outputs/alex_evaluation/rag_evaluation_results.xlsx`; agreement
statistics are not yet reproducible from committed code. A comparable run on the
hybrid retriever and the deployed generator is the next step.

## Current agent evidence flow

The current agent retains exact passages supporting requested facts across search
rounds and passes the assessed evidence unchanged to complete or partial answer
generation. The critic derives sufficiency from a fact checklist and validates
claim scope and citation IDs. Bounded `read_more` operations inspect adjacent text
from already-matched, authorized documents. Native structured providers constrain
JSON and supplied citations; plain providers, including Ollama, retain their
existing JSON-validation fallback. These safeguards do not establish answer
accuracy or an overall quality improvement.

## Application verification

Local checks exercised API validation/lifecycle, permissions, provider failures,
timeouts and budget accounting. Browser smoke checks covered citations, partial
answers and errors. Real full-corpus BM25 returned evidence without model calls;
small live BM25 requests produced cited answers and a round-limit partial answer.
These demonstrate integration, not independent correctness assessment.

Docker packaging was checked on Linux ARM64: fixture HTTP flows, served assets,
a two-document real BM25 mount, missing-artifact errors, and optional CPU hybrid
dependency imports. Full BGE-base inference, GPU and cross-platform verification
remain pending. Local records are `.cache/c5-verification/summary.json` and
`.cache/docker-verification/summary.json`; neither is shipped in a fresh clone.
CI is configured for Python tests, frontend build and a Docker fixture smoke test;
configuration alone does not establish that a remote CI run passed.


## Reproducible offline experiment records

The existing 20-case BM25 pilot above remains development evidence. New
snapshot-bound records use `src/evaluation/experiment_inputs.py` and
`src/evaluation/experiment_io.py`; these modules do not generate answers, load
indexes, resolve permissions, or invoke a judge. Application/runner integration
must supply the actual effective configuration and exact evidence passed to the
generator. Do not infer that evidence from the final displayed sources, or invent
snapshot/revision identities for historical runs.

Before recording a run, its producer supplies an `ExperimentManifest` and an
`InputFiles` mapping. The mapping names `questions`, `splits`, `corpus`, `catalog`,
`index_manifest`, `retrieval_config`, `comparison_config`, `scope`,
`runtime_config`, `generator_prompt`, and `source_repo`; `critic_prompt`,
`router_prompt`, and `final_freeze` are optional when the manifest allows them.
CLI mapping paths are relative to the mapping JSON file. See the real tiny-file
fixture in `tests/test_experiment_inputs.py` for a complete executable example;
it does not represent a production dataset or accepted calibration.

Input formats and fingerprints:

- Questions are a JSON array or `.jsonl` records. Runtime selection exposes only
  `question_id` and `question`; reference answers and source labels stay out of
  the runtime ledger. Splits are a JSON object with `splits` containing exactly
  `development`, `calibration`, and `final` ID lists, disjoint and covering all
  source questions. Explicit manifest IDs preserve execution order; count-based
  selection sorts IDs. The selected-set fingerprint hashes the canonical runtime
  questions sorted by ID, independently of execution order.
- Corpus, split, index-manifest and UTF-8 prompt fingerprints hash exact file
  bytes. Catalog JSONL contains ordered `CatalogEntry` records and uses the
  shared streaming catalog fingerprint with the declared document count.
  Retrieval/comparison configs and the whole scope JSON use canonical JSON:
  sorted keys, compact separators, UTF-8, finite values. The scope file contains
  `binding` plus the producer's scope representation. The ledger also retains
  raw file fingerprints so a reformatted file cannot silently change on resume.
- Runtime config contains exactly `generation_model`, `retrieval`, `budgets`,
  `temperature`, `workflow`, and `max_rounds`, matching the manifest. Git
  provenance records HEAD and hashes staged/unstaged patches and ordinary
  untracked, nonignored files; source contents and local input paths are not
  written into the runtime ledger. Ignored assets need their own input hashes.
  `source_repo` resolves to the whole Git root. Submodules are not recursively
  fingerprinted; tracked symlinks identify the link, not external target contents.
  This checkout has neither. Fingerprint any external dependencies explicitly.
  Verify an unchanged checkout; capture is not a filesystem snapshot.

Use offline checks after preparing actual manifest/input files:

```bash
python -m src.evaluation.experiment_io preflight --manifest /path/manifest.json --inputs /path/inputs.json
python -m src.evaluation.experiment_io validate --run /path/fixed-run --require-complete
python -m src.evaluation.experiment_io pair --fixed /path/fixed-run --adaptive /path/adaptive-run
python -m src.evaluation.experiment_io export-scoring --run /path/fixed-run --references /path/references.jsonl --output /path/new-scoring-export
```

A trusted producer uses `verify_inputs(manifest, files)` and then
`RunDirectory.create(path, manifest, verified_inputs)` as a context manager.
It appends validated `RuntimeCaseExport` outcomes, including sanitized failures,
budget stops, and clarifications. Run creation and resume recheck the actual
files and Git state. Output must be external to the source checkout or ignored
by Git. Creation is exclusive; immutable headers and ordinal case envelopes use
fsynced atomic no-replace publication on local macOS/Linux filesystems.
`RunDirectory.resume` rejects changed settings/inputs and only returns pending
question IDs; a recorded failure is a completed attempt, not a missing row.
After a crash, inspect the writer and records before explicitly removing a stale
`.writer.lock`. No recovery path silently discards files. Stored checksums detect
corruption/internal disagreement, not a malicious owner rewriting all records.
Read-only validation checks stored integrity; it does not reverify absent input
files or authenticate a scope.

Final selection requires a freeze JSON file with `accepted: true`, a
`settings_sha256` object mapping the workflow to
`scientific_settings_sha256(manifest)`, a relative `calibration_artifact` contained
under the freeze directory, and its exact `calibration_sha256`. Scientific
settings include code, scope, models, prompts and budgets while excluding run
identity, timestamp, split, question set and repetition. This is an explicit
operator attestation with a checked artifact, not proof that CAL-01's human
review gate passed. No final-run executor is added here. An optional freeze file
on development/calibration is only fingerprinted; it grants no final acceptance.

Scoring input is created in a separate, new directory only after a complete arm.
Reference JSONL rows contain `question_id`, exact `question`, and one or more of
`reference_answer`, `answer_facts`, `expected_doc_ids`. The exporter computes the
reference-file hash itself and binds each join to the runtime-case digest;
duplicate/missing IDs or different question text reject. Outputs are
`scoring_inputs.jsonl` and a handoff manifest. Failed outcomes remain present.
The optional legacy evaluator adapter accepts passage-only successful outcomes
with an available reference; aggregates/failures/clarifications retain the native
handoff instead. A valid export and a paired configuration are not quality scores.

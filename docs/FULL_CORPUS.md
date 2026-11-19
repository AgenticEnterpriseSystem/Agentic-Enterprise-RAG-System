<a id="baseline-retrieval-results"></a>

# Full dataset, indexes and retrieval results

Use this page to understand which data files the app needs, how search uses them,
and what the earlier retrieval experiments measured. To install the app, follow
[the setup guide](WEB_APP.md). For the latest application checks and unfinished
work, read [Current status](STATUS.md).

The **corpus** is the collection of documents being searched. An **index** is a
saved structure used to search that collection. Neither is included in a normal
Git clone. Accounts, passwords and document permissions are also separate from Git.

## 1. Get the files before installing

The app needs three inputs:

| Input | Suggested host location | Purpose |
| --- | --- | --- |
| Document file and prebuilt indexes | `.cache/hybrid-artifacts/` | Canonical text, keyword index and dense vectors |
| Cached embedding model | `.cache/models/` | Converts new questions into vectors locally |
| Pinned upstream source checkout | `.cache/local-source/` | Supplies original native Jira records |

An **embedding** is a numeric representation of text used for semantic similarity.
The document vectors in the index do not include the model weights needed to
encode a new question. Copy both the index and the complete model cache.

Get the team's `files_web_app` bundle and follow the exact directory layout in
[Get the three input directories](WEB_APP.md#get-the-artifacts). The current web
profile uses `BAAI/bge-base-en-v1.5`; an Arctic index from an experiment is not a
replacement for the configured BGE-base index. Do not combine shards from
separate model builds.

Use a machine with 16 GiB of memory and about 8 GiB available to Docker. Docker
copies the verified inputs into managed volumes, so budget disk space for the
originals, copies, images, database and backups. The setup guide explains the
build and copy headroom checks. A Git clone or successful download alone does
not establish that the app is installed.

## 2. Know which dataset is being used

The synthetic EnterpriseRAG-Bench dataset contains **511,958 canonical documents
from nine platforms**. “Canonical” means the one selected version of each document
that the database and citation offsets use.

| Item | Recorded value |
| --- | --- |
| Upstream revision | `d36685e273713975ee20299bbf1ab64165575b3c` |
| Count and content fingerprint | [`evaluation/corpus_manifest.json`](../evaluation/corpus_manifest.json) |
| Development / calibration / final split | 300 / 100 / 100 questions |
| Scored historical retrieval questions | 282 development; 94 calibration |

The upstream checkout contains 511,962 files. Four document IDs each appear in
two files; import keeps the file selected by upstream's `uuid_index.json`.
Eighteen development questions and six calibration questions lack reference-document
labels, so those historical retrieval tables use the smaller scored
counts above. Reference answers and reference documents are evaluation labels;
they are not supplied to the answering agent.

The dataset does not supply an authoritative company-wide access hierarchy for
this app. An operator must explicitly review and grant the documents a user can
access. For a teammate's full fictional-benchmark QA account, use the optional
[reviewed QA recipe](WEB_APP.md#full-benchmark-qa-access). That is a testing policy,
not an inferred real-company permission system.

## 3. Understand the current search path

The running app combines two searches:

1. **BM25+** scores keyword matches across a whole document.
2. **Dense search** compares a question vector with stored passage vectors.
3. **Reciprocal rank fusion (RRF)** combines their ranked lists. The current
   weights are 0.7 for BM25+ and 0.3 for dense search, with RRF constant 60.
4. The app reads exact permitted text for the winning documents. Live mode may
   then critique the evidence, read more or search again, and generate an answer.

Each branch requests up to 50 candidate documents. A **candidate** is a possible
match before the final evidence is selected. The managed factory has no reranker
enabled. A reranker would read candidate text and assign a new ordering.

Documents the user cannot access are excluded before BM25 selects candidates
and before dense similarity scoring. BM25 still computes shared-index scores
before applying its permission mask. Text reads also pass application checks and PostgreSQL
row-level security: database checks that reject unauthorized rows. Access and
snapshot freshness are checked again around model work and final output. A
cached catalogue contains metadata, not document bodies or permission authority.

The default shared evidence ceiling is eight unique documents, configurable as
5, 8 or 10. Several passages from one document count once. The character and
16,000-token question budgets may allow fewer documents; eight is not a quota.
The critic and generator use the same frozen selected evidence. The native tool's
`top_k=5`, each retriever's 50 candidates and this evidence ceiling are different
limits. See [application settings](WEB_APP.md#model-request-and-database-limits).

The current app has real login, hybrid retrieval and an adaptive workflow.
Historical retrieval tables below do not measure the current agent's answer
accuracy, its permission-specific recall or a controlled comparison of five
versus eight documents. Those limits and the actual live checks are in
[Current status](STATUS.md).

<a id="findings"></a>

## 4. Read the historical retrieval scores correctly

**Recall@10** measures how many reference documents appear in the first ten
results. **nDCG@10** also rewards placing relevant documents nearer the top.
These are document-retrieval metrics, not answer accuracy or citation correctness.
Scores below come from the recorded full-corpus experiments on 282 scored
development questions, unless a table says otherwise.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| Hybrid BM25+ / BGE-base, BM25 weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Hybrid BM25+ / Arctic, BM25 weight 0.7 | 0.638 | 0.723 | 0.768 | 0.628 |

BGE-base remains the configured dense model. The Arctic hybrid's Recall@10 gain
over it was too small to establish a useful improvement in those tests.

BM25+ beat the original BGE-small on Recall@10 for 88 questions, lost on 14 and
tied on 180. The mean difference was 0.22, with a 95% bootstrap confidence
interval of 0.17 to 0.28. BGE-small found reference documents for 10 of the 72
questions BM25+ missed completely in its top ten; both missed 62.

Reworded questions remained difficult: BM25+ Recall@10 was 0.413, original
BGE-small was 0.160 and the selected BGE-base hybrid was 0.400. Internal codes,
similar documents and source selection are possible explanations, not established
causes. These results do not show that reformulation has solved the problem.

<details>
<summary>Historical question-type scores: BM25+ versus original BGE-small</summary>

| Type | Questions | BM25+ R@10 | BGE R@10 | BM25+ nDCG@10 | BGE nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.505 | 0.674 | 0.375 |
| Semantic/reworded | 75 | 0.413 | 0.160 | 0.281 | 0.084 |
| Intra-document reasoning | 24 | 1.000 | 0.750 | 0.911 | 0.641 |
| Project related | 24 | 0.676 | 0.543 | 0.634 | 0.475 |
| Constrained | 18 | 0.917 | 0.861 | 0.820 | 0.705 |
| Completeness | 12 | 0.497 | 0.334 | 0.516 | 0.369 |
| Conflicting information | 12 | 0.792 | 0.625 | 0.810 | 0.512 |
| Miscellaneous | 12 | 0.917 | 0.750 | 0.886 | 0.641 |

The categories with 12–24 questions are small; treat their estimates cautiously.

</details>

<a id="saved-results"></a>

## Where the saved research outputs live

Historical result folders under [`evaluation/results/`](../evaluation/results/)
include `bm25_full_development`, `dense_full_development`,
`dense_titled_full_development` and `dense_base_titled_full_development`.
Each completed run has `summary.json` for metrics, `predictions.jsonl` for top-20
IDs/scores and `upstream_answers.jsonl` for top-ten benchmark-format results.
Those small files are not the document corpus, model cache or indexes.

See [Evaluation](EVALUATION.md) for current scoped commands and metric denominators,
[the Data Card](DATA_CARD.md) for provenance and [the documentation map](README.md)
for terminology. The final 100-question split remains reserved for settings that
have been frozen; earlier development tuning is not a final-test result.

<a id="reproducibility-details"></a>

## Technical reference: original index settings

| Component | Recorded configuration |
| --- | --- |
| BM25+ | k1=1.5, b=0.75, delta=1, lowercase word tokens; builder checked against `bm25s` |
| BGE-small | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| BGE-base | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, 768 dimensions |
| Passage construction | 500 tokens with 50-token overlap; later builds prefix the title on every passage, capped at 48 tokens |
| Dense scoring | Exact cosine similarity; each document receives its best passage score |
| Original GPU precision | fp16 for BGE; bf16 for the full Arctic build |

BM25+ originally ran on four CPUs. BGE-small ran on a Colab T4 from `e924855`;
the titled build used `2ab9ec8`, and BGE-base used `d11eb1b`. Colab rebuilt the
corpus file, so its file hash differed while its document-content fingerprint
matched. Historical GPU query times were about 130 ms for BGE-small and 250 ms
for BGE-base; BM25+ was about 20 ms on CPU. Hardware and run settings differed.
These are not controlled speed comparisons or current CPU web-app latencies.

<a id="experiment-title-on-every-passage"></a>
<a id="experiment-bge-base"></a>

## Earlier experiments: passage titles and BGE-base

Putting the title on every BGE-small passage raised Recall@10 by 0.013. Its 95%
paired bootstrap interval was −0.019 to +0.045: 17 questions improved, 13 worsened
and 252 stayed the same. This was not a statistically clear overall improvement.
A small 40-question pilot with roughly 40 candidates per question had suggested
0.367 → 0.507, overstating the benefit seen across the whole corpus.

BGE-base improved Recall@5 over titled BGE-small by 0.050 (interval +0.017 to
+0.085; 28 better, 11 worse). Recall@10 changed by only +0.005 (−0.031 to +0.041;
22 better, 16 worse). It found a reference for seven of BM25+'s 72 complete
misses, compared with nine for titled BGE-small.

Merging the saved BGE-base and BM25+ top-20 lists gave Recall@10 of 0.695 with
equal weights and 0.712 with BM25 weight 0.7. The latter gain over BM25+ was
+0.020 (+0.006 to +0.036; 12 better, two worse). Together the lists contained
77.7% of reference documents somewhere in their top twenties. This saved-list
merge is a different protocol from retrieving 50 candidates per branch below.
Earlier BGE-small merges scored 0.686 with equal weights and about 0.705 for
BM25 weights 0.6–0.8; the titled-small 0.7 merge was 0.701.

The existing build commands are:

```sh
python -m src.retrieval.corpus_dense --title-every-passage --help
python -m src.retrieval.corpus_dense --model bge-base --title-every-passage --help
```

These show required paths/options. Building embeds the full dataset and needs a
separate resource plan; it is not necessary for installing the prebuilt bundle.
The [dense notebook](../notebooks/dense_baseline_colab.ipynb) records the GPU path.

<a id="hybrid-search-and-reranking"></a>

## Earlier experiments: fusion and reranking

The historical hybrid run at `4856894` searched all documents once per branch,
kept 50 candidates from each, and reused them for each variant. It used a Colab
T4 and [the hybrid notebook](../notebooks/hybrid_colab.ipynb). The weights were
chosen on development questions.

| Development run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Equal-weight hybrid | 0.627 | 0.700 | 0.744 | 0.579 |
| BM25 weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Weight 0.7 + `bge-reranker-base` | 0.551 | 0.637 | 0.716 | 0.520 |
| Weight 0.7 + `bge-reranker-v2-m3` | 0.602 | 0.669 | 0.715 | 0.560 |

The selected hybrid improved Recall@10 over BM25+ by +0.029 (+0.005 to +0.055;
19 questions better, seven worse). Equal weights changed it by +0.008 (−0.024 to
+0.041; 21 better, 19 worse). The base reranker reduced it by 0.084 (interval
−0.129 to −0.041; 14 better, 46 worse); v2-m3 reduced it by 0.053 (−0.093 to
−0.012; 15 better, 37 worse).

The rerankers read 50 fused candidates in fp16 with a 512-token limit. Revisions
were `2cfc18c9415c912f9d8155881c133215df768a70` for base and
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` for v2-m3. Only one bounded window
per document was supplied. Of 45 references the base reranker moved from the
hybrid's top five to below its top ten, 43 exceeded 3,000 characters. Missing
sections or titles may explain some failures; the experiment does not prove a
single cause. Multi-window scoring, longer inputs and blending are ideas to test,
not current enabled behavior.

The first held-out calibration check used 94 scored questions:

| Calibration run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.653 | 0.712 | 0.792 | 0.592 |
| BGE-base | 0.425 | 0.487 | 0.584 | 0.389 |
| Selected hybrid | 0.658 | 0.767 | 0.821 | 0.625 |

The hybrid gained +0.056 Recall@10 over BM25+ (+0.013 to +0.106; seven better,
two worse). Gains mostly appeared at ranks 6–10; semantic questions did not
improve on development. This supports that historical retrieval setting, not
current adaptive answer-quality superiority. Do not keep tuning on calibration.

<a id="embedding-model-screening-on-a-sample"></a>

## Earlier experiments: screening stronger embedding models

A 20,364-document screening set kept all 431 development reference documents,
9,933 other BM25+/BGE top-20 documents and 10,000 seeded random documents. Its
list digest starts `6e6eeccc`. No calibration or final questions were used.
`src.evaluation.screen_models` ran from `5bf0525` on a Colab T4 with the same
500-token titled passages. The [screening notebook](../notebooks/model_screening_colab.ipynb)
and `evaluation/results/screening/` retain the procedure and sample manifests.

| Model | Alone R@5 | Alone R@10 | Alone R@20 | BM25+ misses found | Hybrid R@10 | Hybrid nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25+ | 0.635 | 0.680 | 0.774 | — | — | 0.595 |
| BGE-base, 110M parameters | 0.448 | 0.487 | 0.562 | 8 / 74 | 0.715 | 0.605 |
| GTE-modernbert-base, 149M | 0.535 | 0.633 | 0.705 | 18 / 74 | 0.724 | 0.635 |
| Arctic-embed-l-v2.0, 568M | 0.571 | 0.670 | 0.752 | 24 / 74 | 0.739 | 0.632 |
| Qwen3-Embedding-0.6B, 596M | 0.533 | 0.643 | 0.711 | 25 / 74 | 0.743 | 0.627 |

Compared with BGE-base alone, Recall@10 gains were Arctic +0.183 (+0.134 to
+0.233), Qwen3 +0.156 (+0.100 to +0.213) and GTE +0.146 (+0.098 to +0.194).
The gains over the BGE-base *hybrid* were smaller: Qwen3 +0.028 (−0.008 to
+0.064), Arctic +0.023 (−0.010 to +0.058) and GTE +0.009. The first two intervals
include zero.

The sample included the original models' strongest competing documents, but
not the new models' competitors. It therefore favored the new models. Use it
for screening, not as a full-corpus quality estimate.

<a id="full-corpus-arctic-embed-l-v20"></a>

## Earlier experiments: Arctic on the whole corpus

Arctic used revision `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, the `query: `
prefix and titled 500-token passages. It ran in bf16 on a Colab L4 from `38489e0`,
with 103 shards of 5,000 documents; the hybrid ran from `62002a6` without a reranker.
BGE-base used 26 shards. All 511,958 documents were included.

Arctic alone improved development Recall@10 over BGE-base by +0.067 (+0.026 to
+0.108; 35 better, 20 worse). It found 11 of BM25+'s 72 complete misses, compared
with seven for BGE-base. On calibration it scored 0.580 versus 0.487 (+0.093,
+0.008 to +0.181). Its sample Recall@10 of 0.670 fell to 0.554 on the full corpus,
showing how much the screening sample had overstated coverage.

| Split | Hybrid | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | BGE-base, weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Development | Arctic, equal weights | 0.642 | 0.703 | 0.761 | 0.612 |
| Development | Arctic, weight 0.7 | 0.638 | 0.723 | 0.768 | 0.628 |
| Calibration | BGE-base, weight 0.7 | 0.658 | 0.767 | 0.821 | 0.625 |
| Calibration | Arctic, weight 0.7 | 0.690 | 0.788 | 0.822 | 0.655 |

Arctic's hybrid gain over BGE-base was +0.001 Recall@10 on development (−0.021
to +0.023; nine better, ten worse), and +0.020 on calibration (−0.018 to +0.060).
Development nDCG@10 gained +0.023 (+0.002 to +0.046); calibration gained +0.030
(−0.014 to +0.076). This did not establish a calibration recall improvement.
Against BM25+ alone, Arctic's calibration gain was +0.076 (+0.030 to +0.130).

This was the second recorded calibration check, after BGE-base. Preserve that
history; do not treat it as untouched tuning data. Saved top-20 three-way fusion
scored 0.713 versus 0.712 for BGE-base hybrid and 0.710 for Arctic hybrid. Adding
Arctic increased top-20 union reference coverage from 0.775 to 0.805, but did not
establish a better merged top-ten setting. BGE-base remains configured.

<a id="bounded-dense-candidate-search"></a>
<a id="store-backed-retrieval-adapters"></a>

## Technical reference: bounded search and exact text reads

`ShardedDense` searches the existing version-1 `.npy` vector files in bounded
blocks instead of building one accumulating in-memory FAISS index. It validates
metadata, document order, finite vectors, spans, dimensions and counts against a
verified snapshot. A block of 8,192 × 768 float32 values contains 24 MiB of vector
payload; this is a calculation, not total process memory.

Each document keeps its best passage across every block and shard before top-k
selection. Exact-score ties use document ID, then earliest character span.
Numerical tolerances in cross-implementation checks do not create artificial
ties. An empty permitted selection skips vector scanning. Full access still
searches the complete shared vector collection; authorization is not a separate
index or duplicate embedding set per user.

The original dense manifest lacks per-shard cryptographic digests and a canonical
text revision proof. Structural validation and a verified external snapshot do
not prove which original text produced each embedding. Keep this provenance
limitation separate from the observed numerical parity below.

`StoreBM25Retriever`, `StoreDenseRetriever` and `StoreHybridRetriever` use a
`DocumentStore` and server-issued request context. The shared `ValidatedCatalog`
contains immutable metadata only and remains snapshot-checked. It saves duplicate
catalogue copies; it does not cache permission decisions or bypass freshness.

Hybrid text selection streams exact overlapping 3,000-character windows, using
the existing query-match/tie rule. A long document can require many bounded reads;
a bounded read size does not guarantee constant I/O. `read_more` expands only
matched authorized documents under the same snapshot/revision checks.
Standalone dense excerpts are capped at 3,000 characters; the legacy standalone
retriever returned its full winning span. That declared excerpt difference is
separate from document-ranking parity. Legacy retrievers remain available as
comparison components, not the current web factory.

<a id="full-corpus-postgresql-import-validation"></a>

## Recorded database import check

A Linux ARM64 Docker run on November 3, 2026 imported and verified all 511,958
documents and 1,746,899 indexed passage spans from 26 shards. All 37 bodies
containing NUL characters were preserved exactly. The ordered catalogue digest
was `5b27411a79bb1f04e6b67c5f86039864ac45aaf8ad312598f24f1c7bd8b6fa03`.

The operator had a 768 MiB memory/no-extra-swap limit and two CPUs; PostgreSQL
had 384 MiB, one CPU and 32 MiB shared buffers. Migration, verification and import
took 1,334.76 seconds. Linux high-water RSS was 352,040 KiB; a true cgroup peak
was unavailable. Exit status was zero and `OOMKilled=false`.

The 8 MiB payload-batch target observed a maximum of 8,388,490 bytes; the largest
document was 42,339 bytes. Database size was 3,527,521,971 bytes, excluding
separate WAL/volume overhead. A persistent-volume restart retained all counts
and the completed checkpoint. The restricted runtime pool could verify metadata
but could not read protected bodies without permission. No login or access
policy was inferred from importing data.

These are import checks with warmed file caches, not serving or answer-quality
results. Use [the setup guide](WEB_APP.md#canonical-document-import) for imports;
use [Status](STATUS.md) for later authentication and current application checks.

<a id="reproducible-full-corpus-parity-protocol"></a>

## Advanced: compare the old and bounded search implementations

**Parity** means that two implementations return the same required rankings and
source text. It does not mean the answers are correct. The existing
`scripts/compare_retrieval_backends.py` protocol has four stages and calls no
answer provider. It needs a ready imported snapshot, restricted runtime database
configuration, pinned local model and reviewed private benchmark credentials.
This is an operator experiment, not a beginner installation step.

Run in a prepared environment with the paths below mounted or replaced by your
actual inputs. Keep credentials mode 600. The reference implementation's text
reads also pass real PostgreSQL permission checks.

```sh
python -m scripts.compare_retrieval_backends encode \
  --questions /inputs/questions.jsonl --splits /app/evaluation/splits.json \
  --snapshot /inputs/snapshot.json --corpus /artifacts/documents.parquet \
  --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/shared
python -m scripts.compare_retrieval_backends reference \
  --shared /reports/shared --credentials-file /private/benchmark-credentials.json \
  --corpus /artifacts/documents.parquet \
  --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/reference
python -m scripts.compare_retrieval_backends candidate --scope full \
  --file-cache-policy retain \
  --shared /reports/shared --credentials-file /private/benchmark-credentials.json \
  --corpus /artifacts/documents.parquet \
  --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/candidate-full
```

Repeat `candidate` in fresh processes with `--scope narrow` and `--scope empty`,
new output paths and the same explicit file-cache policy. `retain` allows consumed
pages to remain eligible for operating-system cache reuse; memory sampling must
include that cache. The tool's default is `discard`. Do not combine policies and
call the result one comparable run. Neither policy establishes cold-disk timing.

The ten development IDs are `qst_0001`, `qst_0431`, `qst_0412`, `qst_0384`,
`qst_0471`, `qst_0482`, `qst_0301`, `qst_0451`, `qst_0342` and `qst_0176`.
The narrow access fixture has 32 documents: their 14 references plus the first
18 other IDs in lexical order. It is a reproducible permission test, not an
unbiased retrieval-quality sample. Full-access cases also test source hints and
after-January-1-2020 date filters; unknown dates fail a date bound.

The independent reference uses FAISS 1.15.1 document-maximum scores, including its
thread partitions and 1,024-row BLAS tiles. The candidate scans bounded blocks.
Comparison requires exact document/span ordering and source text, with score
`atol=rtol=2e-6`. Near scores do not excuse ranking mismatches. Both paths must
have matching query-encoder implementations; candidate encodings must be
byte-equal to the frozen vectors. All corpus, index, question, package and
executed-source fingerprints remain part of the reports.

Rank prefixes 5, 10 and 20 are checked for dense and hybrid results. The historical
timed hybrid call materializes five documents; that protocol is not the current
eight-document web setting. The independent RRF oracle uses weights 0.7/0.3,
constant 60, 50 candidates and no reranker. Source checks include exact Unicode
offsets, revisions, titles, provenance and adjacent expansions.

Each candidate process has one resolver and a resident CPU/fp16 BGE model.
Nonempty requests really re-encode questions. Timing includes input verification,
model/index initialization, fresh permission resolution, protected text reads and
expansion. Empty access/date-filter paths skip encoding and ranking when no
results can be admitted. The historical gates were startup ≤120 seconds, first
request ≤30 seconds and every later request ≤10 seconds, with three warm repeats.
These component gates differ from current agent/HTTP deadlines. Nested timings
are inclusive; do not sum overlapping guard and materialization times.

A supervising process must also produce a resource report for each scope, bound
to that candidate manifest. It records paired app/database samples every second,
raw sampler/supervisor digests, complete coverage, no triggered guard and OOM
counter changes. The acceptance envelope is app 6 GiB/no-extra-swap/two CPUs,
PostgreSQL 384 MiB/one CPU, paired sampled total ≤6.5 GiB, available VM memory
≥256 MiB and no OOM increase. Unknown kernel peaks remain unknown; never add
separately observed component maxima and call that a simultaneous peak.

After producing all three candidate and resource reports, compare them:

```sh
python -m scripts.compare_retrieval_backends compare \
  --shared /reports/shared --reference /reports/reference \
  --candidate /reports/candidate-full \
  --candidate /reports/candidate-narrow \
  --candidate /reports/candidate-empty \
  --corpus /artifacts/documents.parquet \
  --resources /reports/full-resource.json \
  --resources /reports/narrow-resource.json \
  --resources /reports/empty-resource.json --output /reports/comparison
```

Every output path must be new. Interrupted outputs stay incomplete; no automatic
resume or overwrite occurs. Missing resource evidence or a matrix, ranking,
text, latency or resource failure makes comparison fail. A reference setup with
three resolvers is harness overhead, not application readiness.

<a id="io-placement-is-a-measured-prerequisite"></a>
<a id="accepted-full-corpus-retrieval-measurement"></a>

## Recorded search parity and resource results

The complete November 3 run at source `a51e6a9` passed the protocol above across
all 511,958 documents, 1,746,899 passages and 26 shards. All 200 candidate cases
matched reference rankings at prefixes 5, 10 and 20. Checks verified 164 distinct
excerpts/expansions against canonical text. There were no paid model calls.

| Documents the fixture allowed | Process startup | First request | Maximum later request | Maximum paired app + DB sample |
| --- | ---: | ---: | ---: | ---: |
| Full | 64.746 s | 8.809 s | 7.477 s | 5.35 GiB |
| Narrow | 55.932 s | 3.340 s | 2.588 s | 5.30 GiB |
| Empty | 54.628 s | 2.096 s | 0.818 s | 5.05 GiB |

Verified read-only Linux volumes held indexes and canonical source files. File
cache was retained, the model stayed resident and queries were freshly encoded.
All resource proofs passed without OOM counter increases. These are fresh-process
measurements with filesystem-cache effects, not cold-disk or true kernel-peak
measurements. The final comparison manifest digest was
`0efa0a21215a66cab53e102e48dbbe8ed237c1650c95688f8b5f9ee77865a480`.

Earlier failures explain why storage placement is part of the protocol:

| Earlier configuration | Startup | First request | Later requests | Result |
| --- | ---: | ---: | --- | --- |
| Read-only Mac bind mounts | 388.334 s | 122.065 s | First repeat 110.580 s | Deliberately stopped; incomplete preflight |
| Managed indexes, raw Parquet on Mac bind; discard policy | 139.509 s | 15.098 s | Max 24.418 s; 35 above 10 s | Completed profile, failed serving gates |
| Optimized source `51bd422`, full fixture | 96.652 s | 12.716 s | Max 9.648 s | Full fixture passed timing; fresh empty startup failed at 138.474 s |

No OOM fired in the first two profiles. Their paired sampled app/database maxima
were 3,172,130,816 and 3,452,321,792 bytes. The managed-index profile read about
2,683,293,696 bytes per unfiltered warm request with unconditional dense cache
discard. A separate retained-catalogue component check built 511,958 entries in
13.227 seconds and avoided new metadata scans for three resolver wrappers. That
component result alone was not full-runtime acceptance.

The accepted November 3 result establishes retrieval parity within its measured
envelope. It does not establish current full-application latency, answer accuracy,
permission-cache cold timing or fair agent-versus-fixed comparisons. Later live
checks and remaining limitations are summarized in [Status](STATUS.md).

<a id="measure-the-existing-loader"></a>

## Advanced: profile the legacy loader separately

`scripts/profile_retrieval.py` measures legacy retrieval components. Its hybrid
mode calls the legacy factory; its dense mode uses the older model/FAISS loader.
It is useful for comparing implementations, but is not the current store-backed
web-app acceptance harness. Run one supervised attempt with explicit input paths
and a new output directory:

```sh
python scripts/profile_retrieval.py --mode hybrid \
  --corpus /data/documents.parquet \
  --bm25-index /data/indexes/bm25 \
  --dense-index /data/indexes/dense-bge-base \
  --output .cache/resource-profile/hybrid-attempt
```

Modes `inventory`, `corpus`, `model`, `bm25` and `dense` isolate particular
components. Inventory hashes and checks declared shards without building a
resident search index. Hybrid workloads exercise existing question-only
development queries, broad/narrow/denied fixture access, filters and expansion.
No login, critic, generator or judge is called.

Prepare the pinned model cache before imports. Record the image, actual source
hashes, mounts, CPU/memory/no-extra-swap limits and excluded dependencies. Disable
an inherited HTTP healthcheck for this non-HTTP process. Preserve phase events,
raw memory samples, logs and failed attempts. Use an independent container ceiling
as well as the profiler's RSS/cgroup/VM/wall-time watchdog.

Hashing warms file caches. Process RSS, cgroup totals including file cache,
sampled maxima and any kernel high-water counters describe different things.
Exit 137 alone does not prove an out-of-memory event. Keep machine-specific raw
reports private and ignored; publish a dated summary with its real limitations.
A resource profile is neither an answer-quality score nor a new application
acceptance result.

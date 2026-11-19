<a id="baseline-retrieval-results"></a>

# Full dataset, indexes and retrieval results

This page covers three things: what is in the data bundle, how to rebuild or
check it yourself, and what the earlier retrieval experiments measured.

- To run the app with the prepared bundle, use the [Quick start](QUICKSTART.md).
- For the latest app checks and open work, see [Current status](STATUS.md).
- Terms such as corpus, index and embedding are in the [glossary](README.md).

## 1. The data bundle

The app needs three inputs. None of them is in Git.

| Input | Default location | Purpose |
| --- | --- | --- |
| Document file and prebuilt indexes | `.cache/hybrid-artifacts/` | Document text, keyword index, dense vectors |
| Cached embedding model | `.cache/models/` | Turns new questions into vectors |
| Pinned upstream source checkout | `.cache/local-source/` | Original native Jira records |

- The exact folder layout is in
  [Get the three input directories](WEB_APP.md#get-the-artifacts).
- The web app uses `BAAI/bge-base-en-v1.5`. An Arctic index from an experiment
  does not replace the BGE-base index. Do not mix shards from different builds.
- Accounts, passwords and document permissions are not in the bundle.

## 2. The dataset

The synthetic EnterpriseRAG-Bench dataset has **511,958 canonical documents from
nine platforms**. "Canonical" means the one version of each document that the
database and citations use.

| Item | Value |
| --- | --- |
| Upstream revision | `d36685e273713975ee20299bbf1ab64165575b3c` |
| Count and content fingerprint | [`evaluation/corpus_manifest.json`](../evaluation/corpus_manifest.json) |
| Development / calibration / final split | 300 / 100 / 100 questions |
| Scored retrieval questions | 282 development; 94 calibration |

- The upstream checkout has 511,962 files. Four document IDs appear in two files
  each. Import keeps the file chosen by upstream's `uuid_index.json`.
- 18 development and 6 calibration questions have no reference-document labels.
  That is why the scored counts are smaller.
- Reference answers and documents are evaluation labels. The answering agent
  never sees them.
- The dataset has no company-wide access hierarchy. An operator must grant
  document access. For a full-benchmark QA account, see the
  [reviewed QA recipe](WEB_APP.md#full-benchmark-qa-access). That is a test
  policy, not a real permission system.

## 3. How search works

1. **BM25+** scores keyword matches across whole documents.
2. **Dense search** compares the question vector with stored passage vectors.
3. **Reciprocal rank fusion (RRF)** merges the two ranked lists. Weights are 0.7
   for BM25+ and 0.3 for dense. The RRF constant is 60.
4. The app reads the exact text of the winning documents. Live mode may then
   critique the evidence, read more or search again, and write an answer.

Each branch returns up to 50 candidates. No reranker is enabled.

**Permissions.** Documents the user cannot access are removed before candidates
are chosen. Text reads also pass application checks and PostgreSQL row-level
security. A cached catalogue holds metadata only, not document bodies or
permissions.

**Evidence limit.** The default is 8 unique documents (configurable: 5, 8 or 10).
Several passages from one document count once. The character and 16,000-token
budgets can allow fewer. The critic and generator see the same evidence. See
[application settings](WEB_APP.md#model-request-and-database-limits).

The tables below measure document retrieval only. They do not measure the
current agent's answer accuracy or permission-specific recall. See
[Current status](STATUS.md).

<a id="findings"></a>

## 4. Retrieval scores

**Recall@10** is the share of reference documents in the first ten results.
**nDCG@10** also rewards ranking them nearer the top. Neither measures answer
accuracy. Scores are on the 282 scored development questions unless stated.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| Hybrid BM25+ / BGE-base, BM25 weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Hybrid BM25+ / Arctic, BM25 weight 0.7 | 0.638 | 0.723 | 0.768 | 0.628 |

- BGE-base stays the configured dense model. The Arctic hybrid's Recall@10 gain
  over it was too small to count.
- BM25+ beat the original BGE-small on Recall@10 for 88 questions, lost on 14 and
  tied on 180. Mean difference 0.22 (95% bootstrap interval 0.17 to 0.28).
- BGE-small found a reference for 10 of the 72 questions BM25+ missed completely
  in its top ten. Both missed 62.
- Reworded questions stay hard. Recall@10: BM25+ 0.413, original BGE-small 0.160,
  BGE-base hybrid 0.400.

<details>
<summary>Scores by question type: BM25+ versus original BGE-small</summary>

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

Types with 12 to 24 questions are small. Treat them as rough.

</details>

<a id="saved-results"></a>

### Saved result files

Result folders are under [`evaluation/results/`](../evaluation/results/):
`bm25_full_development`, `dense_full_development`, `dense_titled_full_development`
and `dense_base_titled_full_development`. Each has `summary.json` (metrics),
`predictions.jsonl` (top-20 IDs and scores) and `upstream_answers.jsonl`
(top-ten, benchmark format).

See [Evaluation](EVALUATION.md) for current commands, the
[Data Card](DATA_CARD.md) for provenance. The final 100-question split is
reserved for frozen settings.

<a id="reproducibility-details"></a>

## 5. Index settings and build

| Component | Setting |
| --- | --- |
| BM25+ | k1=1.5, b=0.75, delta=1, lowercase word tokens; checked against `bm25s` |
| BGE-small | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| BGE-base | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, 768 dimensions |
| Passages | 500 tokens, 50-token overlap. Later builds prefix the title on every passage (capped at 48 tokens) |
| Dense scoring | Exact cosine similarity. Each document gets its best passage score |
| GPU precision | fp16 for BGE; bf16 for the full Arctic build |

- BM25+ ran on four CPUs. BGE-small ran on a Colab T4 from `e924855`. The titled
  build used `2ab9ec8`. BGE-base used `d11eb1b`.
- Colab rebuilt the corpus file, so its file hash differed. The document-content
  fingerprint matched.
- Query times: about 130 ms (BGE-small, GPU), 250 ms (BGE-base, GPU), 20 ms
  (BM25+, CPU). Hardware differed, so these are not a fair speed comparison.

To rebuild a dense index yourself, these commands show the required options:

```sh
python -m src.retrieval.corpus_dense --title-every-passage --help
python -m src.retrieval.corpus_dense --model bge-base --title-every-passage --help
```

Building embeds the whole dataset and needs a GPU plan of its own. It is not
needed for the prepared bundle. The
[dense notebook](../notebooks/dense_baseline_colab.ipynb) records the GPU path.

<a id="experiment-title-on-every-passage"></a>
<a id="experiment-bge-base"></a>

## 6. Experiment: title on every passage, and BGE-base

Intervals are 95% paired bootstrap.

| Change | Result |
| --- | --- |
| Title on every BGE-small passage | Recall@10 +0.013 (−0.019 to +0.045). 17 better, 13 worse, 252 same. Not clearly better. A 40-question pilot had suggested 0.367 to 0.507, which overstated it |
| BGE-base vs titled BGE-small, Recall@5 | +0.050 (+0.017 to +0.085); 28 better, 11 worse |
| BGE-base vs titled BGE-small, Recall@10 | +0.005 (−0.031 to +0.041); 22 better, 16 worse |
| BM25+ complete misses found | BGE-base 7 of 72; titled BGE-small 9 of 72 |
| Merge of saved BGE-base and BM25+ top-20 lists | Recall@10 0.695 (equal weights), 0.712 (BM25 weight 0.7). The 0.712 gain over BM25+ is +0.020 (+0.006 to +0.036); 12 better, 2 worse |
| Union of both top-20 lists | Holds 77.7% of reference documents |
| Earlier BGE-small merges | 0.686 (equal weights); about 0.705 (BM25 weights 0.6 to 0.8); 0.701 (titled small, 0.7) |

The saved-list merge is a different protocol from the 50-candidates-per-branch
runs below.

<a id="hybrid-search-and-reranking"></a>

## 7. Experiment: fusion and reranking

Run at `4856894` on a Colab T4 ([hybrid notebook](../notebooks/hybrid_colab.ipynb)).
Each branch searched all documents once and kept 50 candidates. Weights were
chosen on development questions.

| Development run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Equal-weight hybrid | 0.627 | 0.700 | 0.744 | 0.579 |
| BM25 weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Weight 0.7 + `bge-reranker-base` | 0.551 | 0.637 | 0.716 | 0.520 |
| Weight 0.7 + `bge-reranker-v2-m3` | 0.602 | 0.669 | 0.715 | 0.560 |

Recall@10 change:

| Compared | Change (interval) | Questions better / worse |
| --- | --- | --- |
| Selected hybrid vs BM25+ | +0.029 (+0.005 to +0.055) | 19 / 7 |
| Equal weights vs BM25+ | +0.008 (−0.024 to +0.041) | 21 / 19 |
| Base reranker vs hybrid | −0.084 (−0.129 to −0.041) | 14 / 46 |
| v2-m3 reranker vs hybrid | −0.053 (−0.093 to −0.012) | 15 / 37 |

- Rerankers read 50 fused candidates in fp16, 512-token limit. Revisions:
  base `2cfc18c9415c912f9d8155881c133215df768a70`, v2-m3
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Only one bounded window per document was given. Of 45 references the base
  reranker moved from the hybrid's top five to below its top ten, 43 exceeded
  3,000 characters. Missing sections or titles may explain some failures.
  Multi-window scoring is tested in [Hira's experiments](HIRA_DEV_EXPERIMENTS.md).

**Calibration check** (94 scored questions):

| Run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.653 | 0.712 | 0.792 | 0.592 |
| BGE-base | 0.425 | 0.487 | 0.584 | 0.389 |
| Selected hybrid | 0.658 | 0.767 | 0.821 | 0.625 |

The hybrid gained +0.056 Recall@10 over BM25+ (+0.013 to +0.106; 7 better, 2
worse). Gains were mostly at ranks 6 to 10. Semantic questions did not improve on
development. Do not keep tuning on calibration.

<a id="embedding-model-screening-on-a-sample"></a>

## 8. Experiment: screening stronger embedding models

The screening set had 20,364 documents: all 431 development reference documents,
9,933 other BM25+/BGE top-20 documents and 10,000 seeded random documents. List
digest starts `6e6eeccc`. No calibration or final questions were used.
`src.evaluation.screen_models` ran from `5bf0525` on a Colab T4 with titled
500-token passages. See the
[screening notebook](../notebooks/model_screening_colab.ipynb) and
`evaluation/results/screening/`.

| Model | Alone R@5 | Alone R@10 | Alone R@20 | BM25+ misses found | Hybrid R@10 | Hybrid nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25+ | 0.635 | 0.680 | 0.774 | — | — | 0.595 |
| BGE-base, 110M parameters | 0.448 | 0.487 | 0.562 | 8 / 74 | 0.715 | 0.605 |
| GTE-modernbert-base, 149M | 0.535 | 0.633 | 0.705 | 18 / 74 | 0.724 | 0.635 |
| Arctic-embed-l-v2.0, 568M | 0.571 | 0.670 | 0.752 | 24 / 74 | 0.739 | 0.632 |
| Qwen3-Embedding-0.6B, 596M | 0.533 | 0.643 | 0.711 | 25 / 74 | 0.743 | 0.627 |

Recall@10 gain over BGE-base alone: Arctic +0.183 (+0.134 to +0.233), Qwen3
+0.156 (+0.100 to +0.213), GTE +0.146 (+0.098 to +0.194).

Gain over the BGE-base hybrid: Qwen3 +0.028 (−0.008 to +0.064), Arctic +0.023
(−0.010 to +0.058), GTE +0.009. The first two intervals include zero.

The sample included the original models' strongest competing documents but not
the new models'. It favors the new models. Use it for screening only.

<a id="full-corpus-arctic-embed-l-v20"></a>

## 9. Experiment: Arctic on the whole corpus

Arctic: revision `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, `query: ` prefix,
titled 500-token passages, bf16 on a Colab L4 from `38489e0`, 103 shards of 5,000
documents. The hybrid ran from `62002a6`, no reranker. BGE-base used 26 shards.
All 511,958 documents were included.

| Split | Hybrid | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | BGE-base, weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Development | Arctic, equal weights | 0.642 | 0.703 | 0.761 | 0.612 |
| Development | Arctic, weight 0.7 | 0.638 | 0.723 | 0.768 | 0.628 |
| Calibration | BGE-base, weight 0.7 | 0.658 | 0.767 | 0.821 | 0.625 |
| Calibration | Arctic, weight 0.7 | 0.690 | 0.788 | 0.822 | 0.655 |

- Arctic alone vs BGE-base, development Recall@10: +0.067 (+0.026 to +0.108); 35
  better, 20 worse. It found 11 of BM25+'s 72 complete misses (BGE-base: 7).
- Arctic alone on calibration: 0.580 vs 0.487 (+0.093, +0.008 to +0.181).
- Arctic's sample Recall@10 of 0.670 fell to 0.554 on the full corpus.
- Arctic hybrid vs BGE-base hybrid, Recall@10: +0.001 on development (−0.021 to
  +0.023; 9 better, 10 worse); +0.020 on calibration (−0.018 to +0.060).
- nDCG@10: +0.023 on development (+0.002 to +0.046); +0.030 on calibration
  (−0.014 to +0.076). No clear calibration recall gain.
- Arctic hybrid vs BM25+ alone on calibration: +0.076 (+0.030 to +0.130).
- Saved top-20 three-way fusion scored 0.713 (BGE-base hybrid 0.712, Arctic
  hybrid 0.710). Adding Arctic raised top-20 union reference coverage from 0.775
  to 0.805.
- This was the second calibration check, after BGE-base. It is not untouched data.
- BGE-base remains configured.

<a id="bounded-dense-candidate-search"></a>
<a id="store-backed-retrieval-adapters"></a>

## 10. How the app searches the indexes

- `ShardedDense` reads the existing version-1 `.npy` vector files in bounded
  blocks. It does not build one in-memory FAISS index. It checks metadata,
  document order, vectors, spans, dimensions and counts against a verified
  snapshot.
- A block of 8,192 × 768 float32 values is 24 MiB of vectors. That is a
  calculation, not total process memory.
- Each document keeps its best passage across all blocks and shards. Ties break
  by document ID, then earliest character span.
- Authorization is not a separate index. All users search the same vectors.
- `StoreBM25Retriever`, `StoreDenseRetriever` and `StoreHybridRetriever` read
  through a `DocumentStore`. The shared `ValidatedCatalog` holds immutable
  metadata only. It does not cache permissions.
- Hybrid text selection reads overlapping 3,000-character windows. `read_more`
  expands only matched, authorized documents. Standalone dense excerpts are
  capped at 3,000 characters.
- Legacy retrievers stay as comparison components, not in the web factory.

**Known gap:** the original dense manifest has no per-shard digests and no proof
of which text produced each embedding.

<a id="full-corpus-postgresql-import-validation"></a>

## 11. Recorded database import check

A Linux ARM64 Docker run on November 3, 2026 imported and verified all 511,958
documents and 1,746,899 indexed passage spans from 26 shards.

| Item | Value |
| --- | --- |
| Bodies containing NUL characters | 37, preserved exactly |
| Ordered catalogue digest | `5b27411a79bb1f04e6b67c5f86039864ac45aaf8ad312598f24f1c7bd8b6fa03` |
| Operator limits | 768 MiB memory, no extra swap, 2 CPUs |
| PostgreSQL limits | 384 MiB, 1 CPU, 32 MiB shared buffers |
| Migration, verification and import time | 1,334.76 s |
| High-water RSS | 352,040 KiB (true cgroup peak unavailable) |
| Exit status | 0, `OOMKilled=false` |
| Largest payload batch (8 MiB target) | 8,388,490 bytes; largest document 42,339 bytes |
| Database size | 3,527,521,971 bytes (excluding WAL/volume overhead) |

- A restart with a persistent volume kept all counts and the completed checkpoint.
- The restricted runtime pool could read metadata but not protected bodies
  without permission.
- These are import checks with warm file caches, not serving or answer-quality
  results.

To import, use [the setup guide](WEB_APP.md#canonical-document-import).

<a id="reproducible-full-corpus-parity-protocol"></a>

## 12. Advanced: compare old and bounded search

**Parity** means two implementations return the same rankings and source text.
It does not mean the answers are correct. The `scripts/compare_retrieval_backends.py`
protocol has four stages and makes no paid model calls. It needs an imported
snapshot, restricted database configuration, a pinned local model and reviewed
private benchmark credentials. Keep credentials mode 600. This is an operator
experiment, not part of setup.

Replace the paths with your own inputs.

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

Repeat `candidate` in fresh processes with `--scope narrow` and `--scope empty`.
Use new output paths and the same `--file-cache-policy`. `retain` lets the OS
reuse its file cache. The tool's default is `discard`. Do not mix policies in one
comparison. Neither gives cold-disk timing.

**Test setup**

- Ten development questions: `qst_0001`, `qst_0431`, `qst_0412`, `qst_0384`,
  `qst_0471`, `qst_0482`, `qst_0301`, `qst_0451`, `qst_0342`, `qst_0176`.
- The narrow access fixture has 32 documents: the 14 references plus the first 18
  other IDs in lexical order. It tests permissions, not retrieval quality.
- Full-access cases also test source hints and after-January-1-2020 date
  filters. Unknown dates fail a date bound.
- The reference uses FAISS 1.15.1 document-maximum scores. Rankings and source
  text must match exactly. Scores must match within `atol=rtol=2e-6`.
- Rank prefixes 5, 10 and 20 are checked. The RRF oracle uses weights 0.7/0.3,
  constant 60, 50 candidates, no reranker.
- Timing gates: startup ≤120 s, first request ≤30 s, every later request ≤10 s,
  three warm repeats.
- A supervising process records app and database memory every second per scope.
  Acceptance: app 6 GiB, no extra swap, 2 CPUs; PostgreSQL 384 MiB, 1 CPU; paired
  sampled total ≤6.5 GiB; available VM memory ≥256 MiB; no OOM increase.

After all three candidate and resource reports exist, compare them:

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

Every output path must be new. There is no resume or overwrite. A missing report
or any ranking, text, latency or resource failure makes the comparison fail.

<a id="io-placement-is-a-measured-prerequisite"></a>
<a id="accepted-full-corpus-retrieval-measurement"></a>

## 13. Recorded parity and resource results

The November 3 run at source `a51e6a9` passed the protocol above on all 511,958
documents, 1,746,899 passages and 26 shards. All 200 candidate cases matched the
reference rankings at prefixes 5, 10 and 20. 164 distinct excerpts and expansions
matched canonical text. No paid model calls.

| Documents the fixture allowed | Process startup | First request | Max later request | Max paired app + DB sample |
| --- | ---: | ---: | ---: | ---: |
| Full | 64.746 s | 8.809 s | 7.477 s | 5.35 GiB |
| Narrow | 55.932 s | 3.340 s | 2.588 s | 5.30 GiB |
| Empty | 54.628 s | 2.096 s | 0.818 s | 5.05 GiB |

Indexes and source files were on verified read-only Linux volumes. File cache was
retained, the model stayed loaded and queries were freshly encoded. All resource
checks passed with no OOM increases. Final comparison manifest digest:
`0efa0a21215a66cab53e102e48dbbe8ed237c1650c95688f8b5f9ee77865a480`.

**Earlier failed runs.** Where files are stored affects speed:

| Earlier configuration | Startup | First request | Later requests | Result |
| --- | ---: | ---: | --- | --- |
| Read-only Mac bind mounts | 388.334 s | 122.065 s | First repeat 110.580 s | Stopped on purpose; incomplete |
| Managed indexes, raw Parquet on Mac bind; discard policy | 139.509 s | 15.098 s | Max 24.418 s; 35 above 10 s | Completed, failed timing gates |
| Optimized source `51bd422`, full fixture | 96.652 s | 12.716 s | Max 9.648 s | Full fixture passed timing; fresh empty startup failed at 138.474 s |

- No OOM in the first two runs. Paired sampled app/database maxima: 3,172,130,816
  and 3,452,321,792 bytes.
- The managed-index run read about 2,683,293,696 bytes per unfiltered warm request
  with dense cache discard.
- A retained-catalogue component check built 511,958 entries in 13.227 s and
  avoided new metadata scans for three resolver wrappers. That alone is not full
  acceptance.

**Not established:** current full-application latency, answer accuracy, cold
permission-cache timing, or fair agent-versus-fixed comparisons. See
[Status](STATUS.md).

<a id="measure-the-existing-loader"></a>

## 14. Advanced: profile the legacy loader

`scripts/profile_retrieval.py` measures legacy retrieval components. It is not the
acceptance harness for the current web app. Run one attempt with explicit paths
and a new output directory:

```sh
python scripts/profile_retrieval.py --mode hybrid \
  --corpus /data/documents.parquet \
  --bm25-index /data/indexes/bm25 \
  --dense-index /data/indexes/dense-bge-base \
  --output .cache/resource-profile/hybrid-attempt
```

Other modes: `inventory`, `corpus`, `model`, `bm25`, `dense`. No login, critic,
generator or judge is called. Prepare the pinned model cache first. Use a
container memory limit as well as the profiler's own watchdog. Keep raw reports
private. A resource profile is not an answer-quality score.

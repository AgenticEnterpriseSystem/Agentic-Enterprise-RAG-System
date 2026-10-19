# Baseline retrieval results

BM25+ and BGE-small searched all **511,958 documents**. Each returned its top 20
for **300 development questions**; 18 lack reference-document labels, leaving
**282 scored questions**. These are retrieval scores, not answer-quality scores.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | **0.692** | 0.747 | **0.604** |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |

Recall measures the share of reference documents found; nDCG also rewards higher
ranks. Historical query times were about 20 ms for BM25 on CPU and 130 ms for BGE
on GPU. Builds took roughly five minutes and two hours respectively. Hardware
and run settings differ, so these are not controlled latency comparisons.

## Findings

- BM25 had higher average scores in every scored category. It won on Recall@10
  for 88 questions; BGE won on 14; 180 tied. The documented mean gap was 0.22
  (95% confidence interval 0.17–0.28).
- BGE retrieved reference documents for 10 of the 72 questions where BM25 found
  none in its top 10. Both methods failed on 62 questions.
- A historical saved-ranking merge scored 0.686 with equal weights and 0.705
  with BM25 weights of 0.6–0.8 (11 questions better, four worse than BM25).
  Those weights were chosen on development; improvement needs held-out validation.
  This is not a measurement of the current hybrid pipeline and its defaults.
- Reworded questions remain difficult: Recall@10 was 0.413 for BM25 and 0.160
  for BGE. Internal IDs/codenames and closely scored similar documents are possible
  explanations, not established causes. Earlier analysis reproduced BGE scores
  on ten missed questions; reference documents ranked below returned documents.

Live hybrid/reranker and LLM answer quality remain unmeasured. **C3 is paused**;
the final 100 questions remain untouched until settings are frozen.

<details>
<summary>Detailed category scores</summary>

| Question type | Questions | BM25+ Recall@10 | BGE Recall@10 | BM25+ nDCG@10 | BGE nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 282 | **0.692** | 0.468 | **0.604** | 0.367 |
| Basic | 105 | **0.771** | 0.505 | **0.674** | 0.375 |
| Semantic (reworded) | 75 | **0.413** | 0.160 | **0.281** | 0.084 |
| Intra-document reasoning | 24 | **1.000** | 0.750 | **0.911** | 0.641 |
| Project related | 24 | **0.676** | 0.543 | **0.634** | 0.475 |
| Constrained | 18 | **0.917** | 0.861 | **0.820** | 0.705 |
| Completeness | 12 | **0.497** | 0.334 | **0.516** | 0.369 |
| Conflicting info | 12 | **0.792** | 0.625 | **0.810** | 0.512 |
| Miscellaneous | 12 | **0.917** | 0.750 | **0.886** | 0.641 |

Recall at other cutoffs:

| | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: |
| BM25+ | **0.625** | **0.692** | **0.747** |
| BGE-small | 0.382 | 0.468 | 0.525 |

Categories with 12–24 questions are small, so treat their scores as rough.

</details>

## Saved results

[BM25 results](../evaluation/results/bm25_full_development/),
[BGE results](../evaluation/results/dense_full_development/),
[titled BGE results](../evaluation/results/dense_titled_full_development/), and
[BGE-base results](../evaluation/results/dense_base_titled_full_development/) each contain:
`summary.json` (scores and slices), `predictions.jsonl` (top 20 IDs/scores), and
`upstream_answers.jsonl` (top ten in benchmark format). These files do not contain
the corpus or indexes. Setup and reproduction commands are in
[README](../README.md#quick-start); dataset provenance is in the
[Data Card](DATA_CARD.md).

## Reproducibility details

- **Dataset version:** upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`.
  Counts and a content fingerprint are in `evaluation/corpus_manifest.json`.
- **Duplicate IDs:** the upstream repository has 511,962 files, but four document
  IDs are each used by two different files. We keep the file that upstream's own
  index (`uuid_index.json`) points to.
- **BM25+ settings:** k1=1.5, b=0.75, delta=1, lowercase word tokens. Our index
  builder is tested to produce the same scores as the `bm25s` library's own
  indexer.
- **BGE settings:** model `BAAI/bge-small-en-v1.5` at revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; 500-token passages with 50-token
  overlap; half precision (fp16) on the GPU; BGE's standard query prefix; exact
  cosine search.
- **Runs:** BM25+ ran on 4 CPUs. BGE ran on a Colab T4 from commit `e924855`. The
  BGE index's corpus file hash differs from the local one because the file was
  rebuilt on Colab; the document content fingerprint is identical.

## Experiment: title on every passage

**Result: a small, not statistically clear improvement.** In the original BGE run,
only a document's first passage contains its title, so later passages of a long
email, ticket, or CRM record lose the customer or feature name.
`python -m src.retrieval.corpus_dense --title-every-passage` prefixes the title
(capped at 48 tokens) to every passage; spans still point into the original text.
It was run with `TITLE_EVERY_PASSAGE = True` in
[`notebooks/dense_baseline_colab.ipynb`](../notebooks/dense_baseline_colab.ipynb)
on a Colab High-RAM T4, from commit `2ab9ec8` (26 shards, all 511,958 documents).

| | Recall@5 | Recall@10 | Recall@20 | nDCG@10 | Reference doc ranked first |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-small, original | 0.382 | 0.468 | 0.525 | 0.367 | 86 questions |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 | 93 questions |
| BM25+ (for reference) | **0.625** | **0.692** | **0.747** | **0.604** | **155 questions** |

- Per question (Recall@10): 17 improved, 13 got worse, 252 unchanged. Mean change
  +0.013, 95% bootstrap confidence interval −0.019 to +0.045, so the gain may be noise.
- Recall@10 by type: conflicting info 0.625 → 0.708, constrained 0.861 → 0.917,
  intra-document 0.750 → 0.792, basic 0.505 → 0.533; semantic 0.160 → 0.133 and
  project related 0.543 → 0.532 dropped slightly. Most categories have 12–24
  questions, so these are rough.
- Merged with BM25+ (saved rankings, RRF k=60), Recall@10 was 0.690 with equal
  weights and 0.701 with BM25 weight 0.7, the same as with the original BGE run.
- Query time was unchanged (median 127 ms on GPU).

An earlier pilot (40 development questions, each ranking only about 40 candidate
documents) had predicted a jump from 0.367 to 0.507. Titles help BGE choose among
a few similar candidates, but on the full corpus the reference document often never
reaches BGE's top 20 at all. Small-pool pilots are useful for screening ideas but
overstate full-corpus gains. The passage format is not BGE-small's main weakness;
the model itself is.

Results: [`evaluation/results/dense_titled_full_development/`](../evaluation/results/dense_titled_full_development/).

## Experiment: BGE-base

**Result: better ranking near the top, about the same coverage; still well behind
BM25+.** `python -m src.retrieval.corpus_dense --model bge-base --title-every-passage`
builds the same index with `BAAI/bge-base-en-v1.5` (768-d vectors) at revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; searching loads the model and revision
recorded in the index `manifest.json`. It ran with `MODEL = "bge-base"` in
[`notebooks/dense_baseline_colab.ipynb`](../notebooks/dense_baseline_colab.ipynb) on a
Colab T4 with High-RAM, from commit `d11eb1b` (26 shards, all 511,958 documents). Query
time was about 250 ms on GPU, twice BGE-small's.

| | Recall@5 | Recall@10 | Recall@20 | nDCG@10 | Reference doc ranked first |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 | 93 questions |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 | 103 questions |
| BM25+ (for reference) | **0.625** | **0.692** | **0.747** | **0.604** | **155 questions** |

- Compared with titled BGE-small, Recall@5 rose by 0.050 (95% bootstrap confidence
  interval +0.017 to +0.085; 28 questions better, 11 worse). Recall@10 rose by only
  0.005 (−0.031 to +0.041; 22 better, 16 worse), which may be noise.
- Recall@10 by type: project related 0.532 → 0.603, conflicting info 0.708 → 0.792,
  miscellaneous 0.750 → 0.833, completeness 0.334 → 0.399; semantic 0.133 → 0.093 and
  constrained 0.917 → 0.861 dropped. Most categories have 12–24 questions, so these are rough.
- Of the 72 questions where BM25+ finds no reference document in its top 10, BGE-base
  finds one for 7 (titled BGE-small: 9). The larger model ranks documents better but does
  not reach many documents BM25+ misses.

**Merged with BM25+** (saved top-20 rankings, RRF k=60), Recall@10 was 0.695 with equal
weights and **0.712 with BM25 weight 0.7**: +0.020 over BM25+ alone (95% confidence
interval +0.006 to +0.036; 12 questions better, two worse). This is the first merge whose
gain over BM25+ is clearly above zero, but it is small, and the 0.7 weight was chosen on
development questions, so it must be confirmed on the calibration split. Together, the
two top-20 lists contain 77.7% of the reference documents (Recall@10 measures 69–71%),
so a re-ranker over these candidates has more room than further BGE sizes.

Results: [`evaluation/results/dense_base_titled_full_development/`](../evaluation/results/dense_base_titled_full_development/).

# Baseline retrieval results

BM25+ and BGE-small searched all **511,958 documents**. Each returned its top 20
for **300 development questions**; 18 lack reference-document labels, leaving
**282 scored questions**. These are retrieval scores, not answer-quality scores.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | **0.604** |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| **Hybrid: BM25+ and BGE-base (RRF, BM25 weight 0.7)** | **0.641** | **0.722** | **0.760** | **0.604** |

Recall measures the share of reference documents found; nDCG also rewards higher
ranks. The hybrid row is the current best setting; it was confirmed on the calibration
split (see [Hybrid search and reranking](#hybrid-search-and-reranking)). Historical query times were about 20 ms for BM25 on CPU and 130 ms for BGE
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

## Hybrid search and reranking

**Result: BM25+ combined with BGE-base beats BM25+ alone, confirmed on held-out
questions; the cross-encoder rerankers made results worse.**

`python -m src.evaluation.run_hybrid` runs the project's `HybridRetriever` over the
BM25+ and BGE-base (titled) indexes. Each question is searched once per method (top 50
documents out of all 511,958), and every variant reuses those results. It ran with
[`notebooks/hybrid_colab.ipynb`](../notebooks/hybrid_colab.ipynb) on a Colab T4 with
High-RAM, from commit `4856894`, with no errors. Both searches together took about
0.9 seconds per question.

**Development** (282 scored questions); changes are in Recall@10:

| Run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 | Change in Recall@10 |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25+ alone | 0.625 | 0.692 | 0.747 | 0.604 | — |
| Hybrid, equal weights | 0.627 | 0.700 | 0.744 | 0.579 | +0.008 vs BM25+ (−0.024 to +0.041; 21 better, 19 worse) |
| **Hybrid, BM25 weight 0.7** | **0.641** | **0.722** | **0.760** | **0.604** | **+0.029 vs BM25+ (+0.005 to +0.055; 19 better, 7 worse)** |
| Hybrid 0.7 + `bge-reranker-base` | 0.551 | 0.637 | 0.716 | 0.520 | −0.084 vs hybrid (−0.129 to −0.041; 14 better, 46 worse) |
| Hybrid 0.7 + `bge-reranker-v2-m3` | 0.602 | 0.669 | 0.715 | 0.560 | −0.053 vs hybrid (−0.093 to −0.012; 15 better, 37 worse) |

Ranges are 95% bootstrap confidence intervals over questions. The rerankers rescored the
50 fused candidates in fp16 with a 512-token limit, at revisions
`2cfc18c9415c912f9d8155881c133215df768a70` (base, +0.37 s per question) and
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (v2-m3, +1.3 s per question). With 50
candidates per method, the live hybrid scored slightly higher than the earlier estimate
from saved top-20 lists (0.712).

**Calibration** (one check of the chosen setting; 94 of 100 questions scored):

| Run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ alone | 0.653 | 0.712 | 0.792 | 0.592 |
| BGE-base alone | 0.425 | 0.487 | 0.584 | 0.389 |
| **Hybrid, BM25 weight 0.7** | **0.658** | **0.767** | **0.821** | **0.625** |

- The hybrid raised calibration Recall@10 by 0.056 (95% confidence interval +0.013 to
  +0.106; 7 questions better, two worse), confirming the development result on questions
  not used to choose it. Recall@5 barely changed (+0.004), so the gain comes from
  reference documents entering ranks 6–10.
- On development, gains came from basic (0.771 → 0.819), conflicting info (0.792 →
  0.917), completeness (0.497 → 0.567), constrained (0.917 → 0.972) and project-related
  (0.676 → 0.716) questions; semantic questions did not improve (0.413 → 0.400).
- The fusion weight was chosen on development; calibration is now used once for this
  setting and should not be used to tune it further. The final split is untouched.

**Why the rerankers hurt.** They often moved reference documents from the top five to
below rank 20. Of the 45 reference documents that the base reranker dropped from the
hybrid's top five out of its top ten, 43 are longer than 3,000 characters (one meeting
transcript has 28,600). The reranker sees only one section of each document: the best
3,000-character window, or the 500-token BGE passage when both methods found the
document, then cut to 512 tokens. BGE passages omit the title, and in about a third of
these cases the BM25 window did too. BM25 scores whole documents, so a section that
lacks the answer or the customer or project name can turn a strong match into a weak
one. Possible fixes to test on development: score several sections per document and
keep the best, always prefix the title, allow longer inputs (v2-m3 accepts more than
512 tokens), or blend the reranker score with the fusion score.

Results: `evaluation/results/hybrid_bge-base_*_full_development/`, and on calibration
`bm25_full_calibration/`, `dense_bge-base_full_calibration/` and
`hybrid_bge-base_w0.7_full_calibration/`.

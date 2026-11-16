# Experiments and results

This is the project's experiment log: every retrieval and agent experiment, what was measured,
and what was decided. **Part 1** covers retrieval (Recall and nDCG on development questions);
**Part 2** covers the live agent (answered questions, time and cost). How to run the
evaluations (splits, commands, scripts, rubric) is in [Evaluation](EVALUATION.md). Saved
results are in [`evaluation/results/`](../evaluation/README.md).

## Summary

Recall@10 is the share of a question's reference documents found in the top 10 results,
averaged over the 282 scored development questions; nDCG@10 also rewards ranking them near the
top. "Answered" in live runs means the agent's critic accepted the evidence and an answer was
written; it is not graded for correctness (see [caveats](#open-items-and-caveats)).

**In the system now**

| Change | Why | Effect | Section |
| --- | --- | --- | --- |
| Hybrid search: BM25+ (weight 0.7) + BGE-base (0.3), weighted RRF | BM25+ is strong on IDs and names; BGE finds some documents it misses | Recall@10 0.692 → 0.722 (development), 0.712 → 0.767 (calibration) | [Hybrid](#hybrid-search-and-reranking) |
| BGE-base with the title on every passage | Later passages of long documents had lost the customer or product name | Better ranking than BGE-small (Recall@5 +0.050) | [BGE-base](#experiment-bge-base) |
| 6,000-character excerpts, 48,000-character context, 32,000-token budget | The critic often couldn't see the answer in a 3,000-character slice (only 63% of questions had every answer fact visible) | +3 answered on both question sets; median time no worse; about 60–70% more cost per question | [Longer excerpts](#longer-excerpts) |
| Round 2 keeps round-1 excerpts whole | Round 2 had cut them to 1,500 characters or dropped them, losing facts the critic had already seen | Part of the 25 → 31 step | [Fixes](#fixes-adopted) |
| Output limit 1,024 → 2,048 tokens | Long answers were cut off and rejected as errors | Errors 2 → 1; qst_0118 now answered | [Fixes](#fixes-adopted) |
| Spending-cap fix | The cap counted worst-case cost and never released the unused part, so sessions ran out early | Bug fix; doesn't change answers | [Fixes](#fixes-adopted) |
| Time limit 120 → 200 s | Room for slow questions | A ceiling only; nothing gets slower | [Fixes](#fixes-adopted) |
| Reranker kept optional, off by default | It was the default briefly, then reverted after live runs | Available for GPU use with `RAG_RERANKER=bge-reranker-base` | [Reranker in the agent](#reranker-and-top-10-in-the-agent) |

**Tried and not adopted**

| Experiment | Result | Why not adopted | Section |
| --- | --- | --- | --- |
| Equal BM25+/dense weights | Recall@10 0.700 vs 0.722 at BM25 weight 0.7 | Worse | [Hybrid](#hybrid-search-and-reranking) |
| Rerankers on one 3,000-character slice per document (October) | Recall@10 fell to 0.637 (base) and 0.669 (v2-m3) | Worse: long documents lost their answer section | [Hybrid](#hybrid-search-and-reranking) |
| Arctic-embed-l-v2.0 instead of BGE-base | Better alone (0.554 vs 0.487); a tie inside the hybrid (0.723 vs 0.722) | Not worth a bigger model and a new index | [Arctic](#full-corpus-arctic-embed-l-v20) |
| Rerankers on whole-document windows, blended (`bge-reranker-base`, `bge-reranker-v2-m3`) | Retrieval up: Recall@10 0.722 → 0.744 (base, fast setting, blend 0.7), up to 0.787 (v2-m3, 100 candidates) | Live agent: about 2× slower on a laptop CPU and no more questions answered (5 vs 8 on 20) | [Reranker revisited](#reranker-revisited-whole-document-windows-and-blending), [in the agent](#reranker-and-top-10-in-the-agent) |
| Reranker order alone, without blending | 0.720 vs 0.722 for the hybrid (base, 50 candidates) | Blending is better | [Reranker revisited](#reranker-revisited-whole-document-windows-and-blending) |
| Path headers in the BM25+ index | +0.001 Recall@10 (+0.008 with the reranker: 3 of 282 questions) | Too small to justify a second index | [Path headers](#path-headers) |
| Path headers in reranker windows | −0.006 Recall@10 | Slightly worse | [Path headers](#path-headers) |
| Top 10 results instead of 5 | 6 vs 8 answered on 20 questions | No gain, about 80% more cost | [Reranker in the agent](#reranker-and-top-10-in-the-agent) |

## Part 1: Retrieval experiments

All retrieval experiments search all **511,958 documents** for the **300 development
questions**; 18 have no reference documents (`info_not_found`, `high_level`), leaving **282
scored questions**. Ranges are 95% paired bootstrap confidence intervals over questions.

### Baseline retrieval results

BM25+ and BGE-small searched all **511,958 documents**. Each returned its top 20
for **300 development questions**; 18 lack reference-document labels, leaving
**282 scored questions**. These are retrieval scores, not answer-quality scores.

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | **0.604** |
| BGE-small + FAISS | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-small, title on every passage | 0.398 | 0.481 | 0.555 | 0.385 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Arctic-embed-l-v2.0, title on every passage | 0.493 | 0.554 | 0.627 | 0.468 |
| **Hybrid: BM25+ and BGE-base (RRF, BM25 weight 0.7)** | **0.641** | 0.722 | 0.760 | 0.604 |
| Hybrid: BM25+ and Arctic (RRF, BM25 weight 0.7) | 0.638 | **0.723** | **0.768** | **0.628** |

Recall measures the share of reference documents found; nDCG also rewards higher
ranks. **The BM25+ and BGE-base hybrid is the team's chosen setting**, confirmed on the
calibration split (see [Hybrid search and reranking](#hybrid-search-and-reranking)). The
Arctic hybrid scored marginally higher, but the difference is too small to justify
switching for now (see [Full-corpus Arctic](#full-corpus-arctic-embed-l-v20)). Historical query times were about 20 ms for BM25 on CPU and 130 ms for BGE
on GPU. Builds took roughly five minutes and two hours respectively. Hardware
and run settings differ, so these are not controlled latency comparisons.

#### Findings

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
  for BGE-small. Internal IDs/codenames and closely scored similar documents are possible
  explanations, not established causes. Earlier analysis reproduced BGE scores
  on ten missed questions; reference documents ranked below returned documents.

The live hybrid and the rerankers are measured in the sections below; agent answers are in
[Part 2](#part-2-agent-experiments-live-runs). The final 100 questions remain untouched until
settings are frozen.

Per-type scores for BM25+, BGE-base and the hybrid are in
[Results by question type](#results-by-question-type).

#### Saved results

[BM25 results](../evaluation/results/bm25_full_development/),
[BGE results](../evaluation/results/dense_full_development/),
[titled BGE results](../evaluation/results/dense_titled_full_development/), and
[BGE-base results](../evaluation/results/dense_base_titled_full_development/) each contain:
`summary.json` (scores and slices), `predictions.jsonl` (top 20 IDs/scores), and
`upstream_answers.jsonl` (top ten in benchmark format). These files do not contain
the corpus or indexes. Reproduction commands are in
[Evaluation](EVALUATION.md#run-checks-and-experiments); dataset provenance is in the
[Data Card](DATA_CARD.md).

#### Reproducibility details

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

### Experiment: title on every passage

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

### Experiment: BGE-base

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

### Hybrid search and reranking

**Result: BM25+ combined with BGE-base beats BM25+ alone, confirmed on held-out
questions; the cross-encoder rerankers made results worse as first tested.** (Reading whole
documents fixed that; see [Reranker revisited](#reranker-revisited-whole-document-windows-and-blending).)

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

#### Results by question type

The tables below break the development results down by the benchmark's question types, for
BM25+ alone, BGE-base alone (title on every passage) and the hybrid (BM25 weight 0.7). They
show where each method is strong and where the hybrid helps. `info_not_found` (12) and
`high_level` (6) questions have no reference documents, so they are not scored. Types with
12–24 questions are small; treat their differences as rough.

**Recall@10** (share of reference documents in the top 10):

| Question type | Questions | BM25+ | BGE-base | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.543 | **0.819** |
| Semantic (reworded) | 75 | **0.413** | 0.093 | 0.400 |
| Intra-document reasoning | 24 | **1.000** | 0.792 | **1.000** |
| Project related | 24 | 0.676 | 0.603 | **0.716** |
| Constrained | 18 | 0.917 | 0.861 | **0.972** |
| Completeness | 12 | 0.497 | 0.399 | **0.567** |
| Conflicting info | 12 | 0.792 | 0.792 | **0.917** |
| Miscellaneous | 12 | **0.917** | 0.833 | **0.917** |
| **All scored** | 282 | 0.692 | 0.487 | **0.722** |

**nDCG@10** (also rewards ranking reference documents near the top):

| Question type | BM25+ | BGE-base | Hybrid |
| --- | ---: | ---: | ---: |
| Basic | **0.674** | 0.442 | 0.660 |
| Semantic (reworded) | **0.281** | 0.058 | 0.253 |
| Intra-document reasoning | 0.911 | 0.710 | **0.958** |
| Project related | 0.634 | 0.567 | **0.676** |
| Constrained | 0.820 | 0.726 | **0.875** |
| Completeness | 0.516 | 0.414 | **0.572** |
| Conflicting info | 0.810 | 0.730 | **0.858** |
| Miscellaneous | **0.886** | 0.751 | 0.844 |
| **All scored** | **0.604** | 0.416 | **0.604** |

**Recall@5 and Recall@20** (BM25+ / BGE-base / hybrid):

| Question type | Recall@5 | Recall@20 |
| --- | --- | --- |
| Basic | 0.743 / 0.514 / 0.733 | 0.800 / 0.619 / 0.829 |
| Semantic (reworded) | 0.307 / 0.067 / 0.320 | 0.480 / 0.173 / 0.480 |
| Intra-document reasoning | 0.958 / 0.750 / 0.958 | 1.000 / 0.875 / 1.000 |
| Project related | 0.523 / 0.462 / 0.553 | 0.816 / 0.715 / 0.822 |
| Constrained | 0.806 / 0.861 / 0.944 | 0.972 / 0.944 / 0.972 |
| Completeness | 0.393 / 0.320 / 0.412 | 0.622 / 0.436 / 0.636 |
| Conflicting info | 0.792 / 0.750 / 0.875 | 0.917 / 0.833 / 0.958 |
| Miscellaneous | 0.917 / 0.833 / 0.917 | 0.917 / 0.833 / 0.917 |
| **All scored** | 0.625 / 0.448 / 0.641 | 0.747 / 0.562 / 0.760 |

- **BM25+ matches or beats BGE-base on Recall@10 in every type,** most of all on semantic
  (reworded) questions, where BGE-base finds almost nothing (0.093). This corpus is full of project codenames and IDs,
  which exact word matching handles well.
- **The hybrid's Recall@10 gains come from five types:** basic, project related,
  constrained, completeness and conflicting info. It ties on intra-document reasoning and
  miscellaneous, and is slightly below BM25+ on semantic questions.
- **Ranking (nDCG@10) is unchanged overall (0.604):** the hybrid ranks better on five types
  but slightly worse on basic, semantic and miscellaneous, probably because BGE-base's weaker
  matches push some of BM25+'s top hits down a few places.
- **Semantic questions are the weakest type for every method** (Recall@10 at most 0.413),
  and they remain the largest group of agent questions whose documents are never found
  (see [Part 2](#remaining-failures)).

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
512 tokens), or blend the reranker score with the fusion score. The first, second and
fourth were tested next; see [Reranker revisited](#reranker-revisited-whole-document-windows-and-blending).

Results: `evaluation/results/hybrid_bge-base_*_full_development/`, and on calibration
`bm25_full_calibration/`, `dense_bge-base_full_calibration/` and
`hybrid_bge-base_w0.7_full_calibration/`.

### Embedding model screening on a sample

**Result: Arctic-embed-l-v2.0 and Qwen3-Embedding-0.6B are much stronger than BGE-base on
their own and find three times as many of BM25+'s misses; their hybrids with BM25+ improve
more modestly.** Arctic is the first candidate for a full-corpus run.

Embedding the full corpus takes hours per model, so candidates were compared on a fixed
sample of 20,364 documents built by `python -m src.data.sample`: all 431 reference
documents of the development questions, the 9,933 other documents that BM25+ or BGE-base
ranked in their top 20 for those questions on the full corpus, and 10,000 seeded random
documents (document list SHA-256 `6e6eeccc…`). No calibration or final question was used.
`python -m src.evaluation.screen_models` scored each model on the 282 scored development
questions, alone and merged with BM25+ (RRF, BM25 weight 0.7, top 50 per method). It ran with
[`notebooks/model_screening_colab.ipynb`](../notebooks/model_screening_colab.ipynb) on a
Colab T4 (fp16, 500-token passages with the title on every passage, as for BGE-base), from
commit `5bf0525`, with no errors. An earlier partial run (commit `aa7a4ac`, without Qwen3)
gave identical scores for the models in both.

| Model (parameters) | Alone R@5 | Alone R@10 | Alone R@20 | BM25+ misses found | Hybrid R@10 | Hybrid nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25+ alone | 0.635 | 0.680 | 0.774 | — | — | 0.595 |
| BGE-base (110M) | 0.448 | 0.487 | 0.562 | 8 of 74 | 0.715 | 0.605 |
| `gte-modernbert-base` (149M) | 0.535 | 0.633 | 0.705 | 18 of 74 | 0.724 | **0.635** |
| `arctic-embed-l-v2.0` (568M) | **0.571** | **0.670** | **0.752** | 24 of 74 | 0.739 | 0.632 |
| `qwen3-embedding-0.6b` (596M) | 0.533 | 0.643 | 0.711 | **25 of 74** | **0.743** | 0.627 |

"BM25+ misses found": of the 74 questions where BM25+ has no reference document in its top
10, how many the model alone finds. Ranges below are 95% bootstrap confidence intervals over
questions, paired on the same questions.

- **Alone, every candidate beats BGE-base clearly** on Recall@10: Arctic +0.183 (+0.134 to
  +0.233; 65 questions better, 13 worse), Qwen3 +0.156 (+0.100 to +0.213; 63 better, 21
  worse), GTE +0.146 (+0.098 to +0.194; 59 better, 15 worse). Arctic alone nearly matches
  BM25+ (0.670 vs 0.680). BGE-base scores 0.487 on the sample, the same as on the full
  corpus.
- **Semantic (reworded) questions** improved most: Recall@10 0.093 for BGE-base, 0.307 for
  GTE and Arctic, 0.347 for Qwen3 (BM25+: 0.387). Completeness questions stayed weak for all
  dense models (0.33–0.40), and Qwen3 dropped on project-related questions (0.470 vs 0.603
  for BGE-base).
- **Hybrids:** against BM25+ alone, Qwen3 adds +0.063 (+0.026 to +0.102) and Arctic +0.059
  (+0.024 to +0.096); BGE-base adds +0.036 (−0.003 to +0.074). Against the BGE-base hybrid
  the gains are smaller and not yet clear: Qwen3 +0.028 (−0.008 to +0.064), Arctic +0.023
  (−0.010 to +0.058), GTE +0.009. BM25+ already finds most of what the stronger models add.
- Arctic and Qwen3 find different BM25+ misses (35 between them), so a three-way hybrid may
  add more.

**Caveat:** the sample keeps BM25+'s and BGE-base's own top competitors from the full corpus
but not the new models', so the new models' standalone gains are probably overstated. Only a
full-corpus run settles them.

**Next (done):** Arctic was chosen for a full-corpus run (strongest alone, fewest questions
worse than BGE-base, no drop on project questions); see the next section. Qwen3 was the
second candidate.

Results: `evaluation/results/screening/` (one folder per run, each with the sample manifest
in `summary.json`).

### Full-corpus Arctic-embed-l-v2.0

**Result: Arctic is clearly better than BGE-base on its own, but the hybrid with BM25+ is
only marginally better. BGE-base stays the team's dense model for now.**

`arctic-embed-l-v2.0` (revision `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, `query: `
prefix) embedded all 511,958 documents with the same passages as BGE-base (500-token
windows, title on every passage), in bf16 on a Colab L4 (103 shards of 5,000 documents),
from commit `38489e0`. The hybrid ran with
[`notebooks/hybrid_colab.ipynb`](../notebooks/hybrid_colab.ipynb) from commit `62002a6`,
without a reranker. There were no errors. Both searches together took about 1.2 seconds per
question (BGE-base hybrid: about 0.9).

**Arctic alone (development, 282 questions):** Recall@5 0.493, Recall@10 0.554, Recall@20
0.627, nDCG@10 0.468. Against BGE-base alone, Recall@10 rose by 0.067 (95% confidence
interval +0.026 to +0.108; 35 questions better, 20 worse). It finds a reference document
for 11 of the 72 questions BM25+ misses (BGE-base: 7). Semantic questions rose from 0.093
to 0.160. On calibration (94 questions) Arctic alone scored 0.580 vs 0.487 for BGE-base
(+0.093, interval +0.008 to +0.181).

**The sample overstated the gain.** On the screening sample Arctic alone scored 0.670 and
found 24 BM25+ misses; on the full corpus, 0.554 and 11. BGE-base scored 0.487 in both,
because the sample contained its own top competitors but not Arctic's. Treat sample gains
as an upper bound: the model ranking held, but only about a third of the gain survived.

**Hybrids with BM25+ (RRF, top 50 per method):**

| Split | Run | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | BM25+ and BGE-base, weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Development | BM25+ and Arctic, equal weights | 0.642 | 0.703 | 0.761 | 0.612 |
| Development | BM25+ and Arctic, weight 0.7 | 0.638 | 0.723 | 0.768 | 0.628 |
| Calibration | BM25+ alone | 0.653 | 0.712 | 0.792 | 0.592 |
| Calibration | BM25+ and BGE-base, weight 0.7 | 0.658 | 0.767 | 0.821 | 0.625 |
| Calibration | BM25+ and Arctic, weight 0.7 | 0.690 | 0.788 | 0.822 | 0.655 |

- Development, Arctic hybrid vs BGE-base hybrid: Recall@10 +0.001 (−0.021 to +0.023; 9
  better, 10 worse), a tie; nDCG@10 +0.023 (+0.002 to +0.046), slightly better ranking.
- Calibration, Arctic hybrid vs BGE-base hybrid: Recall@10 +0.020 (−0.018 to +0.060) and
  nDCG@10 +0.030 (−0.014 to +0.076), the same direction but not established on 94 questions.
  Against BM25+ alone, the Arctic hybrid added +0.076 (+0.030 to +0.130).
- This was the calibration split's second and last single check (the first was the BGE-base
  hybrid). It should not be used to tune these settings further; the final split is untouched.
- Offline estimates from the saved top-20 lists matched: BM25+ and BGE-base 0.712, BM25+ and
  Arctic 0.710, all three together 0.713. Including Arctic raises the share of reference
  documents found somewhere in the combined top-20 lists from 0.775 to 0.805, so it adds
  candidates for a later step that reads them (an LLM selector, the agent, or a fixed reranker).

**Decision:** BGE-base remains the dense model; Arctic is kept as a ready alternative. Better
embeddings have stopped moving the merged top-10 recall; the next gains are expected from
steps that read the candidates.

Results: `evaluation/results/dense_arctic-embed-l-v2.0_titled_full_development/`,
`hybrid_arctic-embed-l-v2.0_*_full_development/`, `hybrid_arctic-embed-l-v2.0_w0.7_full_calibration/`
and `dense_arctic-embed-l-v2.0_full_calibration/`.

### Reranker revisited: whole-document windows and blending

**Result: reading whole documents and blending with the hybrid order turned the rerankers'
losses into clear retrieval gains. They are kept optional and off by default, because the
gains did not carry over to the live agent ([Part 2](#reranker-and-top-10-in-the-agent)).**

The October rerankers saw one 3,000-character slice per document. Three changes, all from the
list of fixes above, were tested on development only (calibration was not used again):

- **Whole-document windows (MaxP):** each candidate is scored on windows of 2,000 characters
  across its whole document, each starting with the title, and keeps its best window's score.
- **Blending:** the reranker's rank is combined with the hybrid's rank by weighted RRF instead
  of replacing it (`--rerank-weights`; the weight is the reranker's share).
- **Cross-validated weight:** the blend weight (1.0 to 0.6) is chosen by 5-fold
  cross-validation on development, so the reported score is not tuned on the same questions.

Three settings trade accuracy for speed. They differ in how many fused candidates are
reranked and how many windows of each document are read:

| Setting | Candidates reranked | Windows per document | Windows read per question | In plain terms |
| --- | --- | ---: | ---: | --- |
| Fast | 20: the hybrid's top 20 | up to 4 | about 74 | Re-sorts only the top 20, reading at most 4 chunks of each |
| Current | 50: the fused top 50, in practice BM25+'s top 50 | up to 8 | about 278 | Re-sorts the top 50, reading up to 8 chunks each |
| Deep | 100: BM25+'s 50 plus the dense model's 50 | up to 8 | about 500 | Also considers the dense model's own finds, so documents BM25+ never ranked can reach the top |

**`bge-reranker-v2-m3`** (568M parameters; cross-validated blend), with time per search
including the hybrid search itself:

| Setting | Recall@5 | No reference doc in top 5 (of 282) | Recall@10 | nDCG@10 | Recall@10 change vs hybrid | Per search, Colab GPU | Per search, laptop CPU |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Hybrid, no reranker | 0.641 | 83 | 0.722 | 0.604 | — | ~1.0 s (measured) | 14.3 s (measured) |
| Fast | 0.702 | 67 | 0.745 | 0.679 | +0.023 (+0.006 to +0.043) | ~3.0 s (est.) | 82.9 s (measured) |
| Current | 0.721 | 62 | 0.773 | 0.687 | +0.051 (+0.027 to +0.079) | ~8.3 s (est.) | ~280 s (est.) |
| Deep | 0.741 | 56 | 0.787 | 0.703 | +0.066 (+0.034 to +0.100) | ~14.2 s (est.) | ~490 s (est.) |

**`bge-reranker-base`** (278M parameters; cross-validated blend) on the same questions:

| Setting | Recall@5 | No reference doc in top 5 (of 282) | Recall@10 | nDCG@10 | Recall@10 change vs hybrid | Reranking per search, Colab GPU |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Hybrid, no reranker | 0.641 | 83 | 0.722 | 0.604 | — | — |
| Fast | 0.691 | 70 | 0.736 | 0.659 | +0.015 (−0.007 to +0.037) | ~0.26 s (est.) |
| Current | 0.696 | 69 | 0.763 | 0.665 | +0.041 (+0.014 to +0.070) | ~0.9 s (est.) |
| Deep | 0.707 | 66 | 0.760 | 0.668 | +0.039 (+0.008 to +0.071) | ~1.6 s (est.) |

GPU times are estimated from the windows each setting reads and the measured seconds per window
on Colab GPUs; laptop times are on a 16 GB laptop CPU (`scripts/time_reranker.py`). With the
web app's fixed blend weight of 0.7 rather than a cross-validated one, base at the fast setting
scored Recall@10 0.744 and nDCG@10 0.661; on a laptop CPU it adds about 20–30 seconds per
search (live runs in Part 2).

- **Both rerankers improve ranking clearly** (nDCG@10 +0.05 to +0.10, all intervals above
  zero). Recall@10 gains are clear except for base at the fast setting.
- **v2-m3 is more accurate but much slower:** on a laptop CPU even the fast setting takes about
  83 seconds per search, which rules it out for the web app without a GPU.
- **The deep setting helps v2-m3 most** (documents only the dense model found can reach the
  top), but costs roughly twice the current setting.

**Blending vs the reranker's order alone** (Recall@10 / nDCG@10):

| Reranker and setting | Hybrid | Reranker alone (weight 1.0) | Blend 0.7 | Cross-validated |
| --- | --- | --- | --- | --- |
| base, fast | 0.722 / 0.604 | 0.734 / 0.644 | 0.744 / 0.661 | 0.736 / 0.659 |
| base, current | 0.722 / 0.604 | 0.720 / 0.633 | 0.758 / 0.665 | 0.763 / 0.665 |
| v2-m3, fast | 0.722 / 0.604 | 0.745 / 0.674 | 0.745 / 0.671 | 0.745 / 0.679 |
| v2-m3, current | 0.722 / 0.604 | 0.756 / 0.678 | 0.773 / 0.687 | 0.773 / 0.687 |

- **Blending matters most for the weaker reranker:** base's order alone at 50 candidates
  matches the hybrid (0.720), while blended it reaches 0.758–0.763. The hybrid's order
  corrects base's mistakes.
- **v2-m3 needs less help:** at 20 candidates its order alone equals every blend (0.745).
- **The blend never did clearly worse than the reranker alone,** so blending is the default
  whenever a reranker is used. Cross-validation chose weights of 0.6–0.8 for base and 0.7–0.9
  for v2-m3.

Notebooks: [`reranker_colab.ipynb`](../notebooks/reranker_colab.ipynb) and
[`reranker_tradeoffs_colab.ipynb`](../notebooks/reranker_tradeoffs_colab.ipynb).

### Path headers

**Result: no useful gain. Not adopted.**

A document's file path carries context its title often lacks; a Slack thread's title is only
its channel name. `src/retrieval/headers.py` turns the path into a one-line header, for example
`slack/devex/1759999992-opaque-routing-shim-sdk-discuss.json` becomes
`Slack · #devex · 2025-10-09 · opaque routing shim sdk discuss` (platform, folders, date and the
file name's words). The test put the header in two places without re-embedding the dense index:
in the BM25+ index (header + title + content), which changes which documents become candidates,
and at the start of every reranker window, which changes how candidates are scored. All four
combinations used `bge-reranker-base` at the current setting (50 candidates, 8 windows,
cross-validated blend), with [`headers_colab.ipynb`](../notebooks/headers_colab.ipynb).

| Setup | BM25+ index | Reranker windows | Recall@10 | nDCG@10 | Recall@10 change vs hybrid |
| --- | --- | --- | ---: | ---: | --- |
| Hybrid, no reranker (current app) | plain | — | 0.722 | 0.604 | — |
| Hybrid, no reranker | **header** | — | 0.722 | 0.607 | +0.001 (+0.000 to +0.002; 2 better, 0 worse) |
| A | plain | title only | 0.763 | 0.665 | +0.041 (+0.014 to +0.070) |
| B | plain | **header** + title | 0.757 | 0.670 | +0.035 (+0.010 to +0.063) |
| C | **header** | title only | **0.770** | 0.667 | +0.049 (+0.020 to +0.079) |
| D | **header** | **header** + title | 0.761 | **0.672** | +0.040 (+0.013 to +0.069) |

- **BM25+ header:** +0.011 Recall@10 for BM25+ alone, +0.001 in the hybrid. With the reranker
  (C vs A) +0.008: three questions where the header brought the right document into the
  candidate set (+0.000 to +0.019; 3 better, 0 worse).
- **Reranker header:** −0.006 Recall@10 (B vs A; 3 better, 4 worse) and +0.005 nDCG@10, within
  noise. Most candidates for a question share a channel or project, so their headers look alike
  and do not help the reranker tell them apart.
- **Slack questions,** which should benefit most, did not improve (all intervals include zero).
- Shipping the BM25+ header would mean distributing a second index for a gain of a few
  questions, so neither was adopted. Re-embedding the dense index with headers was not tried;
  this result removed the main reason to. The header's dates may still be useful for the agent
  (see [open items](#open-items-and-caveats)).

## Part 2: Agent experiments (live runs)

These runs send development questions through the web app's own adaptive agent in live mode
(`claude-haiku-4-5`, temperature 0) with [`scripts/time_agent.py`](../scripts/time_agent.py),
on a 16 GB laptop CPU, and record each question's time, searches, how it stopped, its cost and,
from the later runs on, where its reference documents were lost. The agent was allowed 300
seconds so every question ran to its natural end; results are judged against the app's limit.
All runs are from 2026-11-16.

### What each kind of test measures

Retrieval scores and agent results answer different questions, which is why they can disagree:

| | Recall tests (Part 1) | Live agent runs (Part 2) |
| --- | --- | --- |
| Question asked | Are the reference documents in the top 5/10 results? | Did the critic accept the evidence, so an answer was produced? |
| Questions | 300 development (282 scored) | 20, 70 and 25 development questions |
| Searches | The original question only, once | The original question, then up to 4 follow-up searches |
| Judge | A fixed answer key, the same every run | A language model whose verdicts vary between runs |

Identical runs differ by about 3–4 answered questions per 70, even at temperature 0, so only
larger changes, or changes confirmed on other questions, count.

### Reranker and top 10 in the agent

**Result: neither the reranker nor 10 results instead of 5 answered more questions; the
reranker doubled the time. Neither was adopted.** Four runs on the same 20 questions (the 10 of
the [October 21 agent record](../evaluation/agent_development/20261021.json) plus 10 sampled with
seed 5980). Top 10 raised the context limit to 36,000 characters and the token budget to 32,000
so the extra results fit.

| Setup | Answered | Errors | Median time | Slowest | Over 200 s | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **No reranker, top 5** | **8** | 1 | **67 s** | 93 s | 0 | **$0.33** |
| No reranker, top 10 | 6 | 2 | 73 s | 147 s | 0 | $0.59 |
| `bge-reranker-base` (fast, blend 0.7), top 5 | 5 | 0 | 126 s | 206 s | 1 | $0.36 |
| `bge-reranker-base`, top 10 | 6 | 0 | 138 s | 243 s | 4 | $0.61 |

- **Searching took 85–90% of the time:** about 12–16 seconds per search without the reranker
  and 30–48 seconds with it. Model calls took 6–27 seconds per question.
- **11 questions were answered by no setup and 4 by every setup** (one error aside). The other
  5 changed between runs without a consistent pattern, so the answered counts are within the
  critic's run-to-run variation. Time and cost differences are not.
- **Why the reranker's retrieval gain did not show:** it raises Recall@5 by about 0.05, so on
  20 questions it changes about one; most failures happened after the right document was
  already found (next section); and the reranker chooses documents by whole-document windows,
  while the critic still reads one excerpt of each.
- The reranker was the web app default briefly before these runs and was turned back off. It
  remains available for GPU use (`RAG_RERANKER=bge-reranker-base`).

### Where unanswered questions failed

**Result: most failures happened after the critic had already seen a reference document, so
the next fixes targeted the agent rather than retrieval.** The diagnosis run used 70 development
questions (the 20 above plus 50 sampled with seed 5980), with no reranker and top 5. For each
question, `time_agent.py` checks whether its reference documents appeared in any search, in
what the critic read, and in the answer's citations.

| Outcome | Questions | Share |
| --- | ---: | ---: |
| Answered | 25 | 36% |
| Correctly unanswered (`info_not_found`: no answer exists) | 4 | 6% |
| **Shown a reference document, judged insufficient** | **28** | **40%** |
| Reference never found by any search | 9 | 13% |
| Found but not shown to the critic | 2 | 3% |
| Error | 2 | 3% |

Of the 39 failures, 28 (72%) came after the critic had seen a reference document; retrieval
accounts for 11. Intra-document reasoning (6 of 8) and project-related (5 of 7) questions
failed mostly at the critic; semantic questions accounted for 5 of the 9 never found.

**Why the critic said "insufficient"** (sorted by hand from the critic's own reasons):

| | Reason | Count | Questions | What the critic said (examples) |
| --- | --- | ---: | --- | --- |
| A | The excerpt cut off the answer | 8 | 0431, 0391, 0226, 0357, 0430, 0215, 0133, 0152 | "excerpt cuts off mid-checklist"; "the code snippet is truncated" |
| B | A specific value missing; unclear whether it is elsewhere in the document | 8 | 0384, 0126, 0304, 0420, 0339, 0105, 0330, 0066 | "specific reduction amounts"; "explicit drift percentage threshold"; "the trace job ID" |
| C | Demands explicit confirmation or authority the question may not need | 7 | 0044, 0309, 0462, 0332, 0373, 0371, 0417 | "explicit written approval from Acme"; "only a hypothesis, not a confirmed root cause"; "a consolidated, explicit list" |
| D | Conflicting sources; refused to choose | 4 | 0176, 0149, 0342, 0354 | 0176 and 0149 are near-duplicates: three different dates for the same H200 announcement |
| E | Round 2 dropped the reference it had in round 1 | 1 | 0301 | Saw the thresholds in round 1; round 2 no longer had them |

**The excerpt test.** [`scripts/excerpt_coverage.py`](../scripts/excerpt_coverage.py) checks,
offline and for every development question, whether the gold answer's specific facts (numbers,
dates, IDs, names) appear in the excerpt the critic would see, compared with the whole reference
documents. It matches exact tokens, so it undercounts paraphrased answers. Of the 269 questions
whose documents contain such facts:

| Excerpt size | Answer facts visible | Questions with every fact visible | Where this size was used |
| ---: | ---: | ---: | --- |
| 1,500 | 70% | 114 (42%) | Round 2, before the fix below |
| **3,000** | 85% | **169 (63%)** | Round 1, before the fix below |
| 4,500 | 93% | 215 (80%) | |
| 6,000 | 96% | 235 (87%) | Now |
| 9,000 | 99% | 258 (96%) | |
| 12,000 | 99% | 265 (99%) | |
| Whole document | 100% | 269 (100%) | |

By question type, project-related questions were worst (4% complete at 3,000 characters, 29% at
6,000, 79% at 9,000): they need facts spread across several long documents. Reference documents
have a median length of 6,072 characters; 21% fit in one 3,000-character excerpt. Applied to the
28 critic refusals, the facts were hidden at 3,000 characters for 15, visible at 3,000 but hidden
at round two's 1,500 for 7, and visible even at 1,500 for 6 (critic strictness or conflicts).

### Fixes adopted

**How rounds work.** The agent runs at most two rounds (`RAG_MAX_ROUNDS=2`); a round is not a
search:

| Round | What happens | Searches |
| --- | --- | --- |
| 1 | Search the question as asked; the critic judges the 5 results | 1 |
| 2 | If insufficient: run the critic's suggested follow-up queries, then the critic judges again | 2–4, depending on how many queries the critic suggests |

A `round_limit` stop means round 2 ended and the critic still judged the evidence insufficient;
the agent then writes a partial answer labelled "insufficient evidence" with what is missing. A
question with 3 or 4 searches also hit the limit; its round 2 just ran fewer queries. More
rounds would rarely help: round 2 rescued only 3 of 38 questions in the latest run.

- **Round 2 keeps round-1 evidence whole** (`src/agents/context.py`). Round 2 gave first-round
  and follow-up results half the space each and cut every excerpt to 1,500 characters, so the
  critic lost text it had already read (category E above, and 7 questions whose facts were
  visible only at 3,000). Round-1 excerpts are now kept whole and in retrieval order within two
  thirds of the space; follow-ups fill the rest.
- **Output limit 1,024 → 2,048 tokens per model call.** One answer stopped at exactly 1,024
  tokens and was rejected as invalid (qst_0118); it is now answered. Only tokens used are paid.
- **Spending-cap fix** (`src/generation/generator.py`). Each call reserved its worst-case cost
  (input plus the full output limit) against the session cap and never released the unused
  part. With the larger output limit, a run hit its $2 cap after about 52 questions with $0.86
  spent. A completed call now settles to its actual cost; failed calls keep the full reservation.
- **Time limit 120 → 200 seconds** (`RAG_MAX_SECONDS`), a ceiling for slow questions.

### Longer excerpts

A result's excerpt was a 3,000-character BM25+ slice or, when the dense model also found the
document, its ~500-token BGE passage (about 2,000–2,500 characters), which could not be extended.
`RAG_PASSAGE_CHARS` re-cuts each returned result from the whole document around the matched
passage; the critic and answer step keep excerpts up to that size, and round-2 follow-ups up to
half of it. The web app now uses 6,000 characters with a 48,000-character context and a
32,000-token budget so the longer excerpts fit (`RAG_PASSAGE_CHARS=off` restores the old ones).

**Results across the changes, same 70 questions:**

| | Original (3,000-character excerpts) | + Round-2 and output fixes | + 6,000-character excerpts (now) |
| --- | ---: | ---: | ---: |
| **Answered** | 25 (36%) | 31 (44%) | **34 (49%)** |
| Correctly unanswered (no answer exists) | 4 | 4 | 4 |
| **Handled correctly, total** | 29 (41%) | 35 (50%) | **38 (54%)** |
| Answered in round 1 (one search) | 20 | 24 | 31 |
| Reference shown, judged insufficient | 28 | 24 | 20 |
| Reference never found | 9 | 9 | 9 |
| Found but not shown | 2 | 1 | 2 |
| Errors | 2 | 1 | 1 |
| Median time per question | 69 s | — | 56 s |
| Cost for 70 questions | $1.16 | ~$1.1 | $1.81 |
| Settings | top 5, 24k context, 16k tokens | same | top 5, 6k excerpts, 48k context, 32k tokens |

The middle run was split across two sessions (questions 53–70 were rerun after the spending-cap
bug), so it has no single median time and its cost is approximate.

**Fresh questions.** The 70 were used to choose the fixes, so the excerpt size was also tested
on 25 development questions not used for tuning (`--count 95 --start 70`), with and without
longer excerpts:

| 25 fresh questions | 3,000-character excerpts | 6,000-character excerpts |
| --- | ---: | ---: |
| **Answered** | 6 (24%) | **9 (36%)** |
| Answered in round 1 | 4 | 5 |
| Reference shown, judged insufficient | 11 | 8 |
| Never found | 8 | 8 |
| Errors | 0 | 0 |
| Median time | 80 s | 74 s |
| Cost per question | $0.017 | $0.029 |

- **Original → now, on the 70:** 13 questions gained and 4 lost. A split that lopsided arises by
  chance only about 5% of the time, so the combined improvement is probably real.
- **Longer excerpts alone:** +3 on the 70 (10 gained, 7 lost) and +3 on the fresh 25 (4 gained,
  1 lost). Over 95 questions, 14 gained and 8 lost is not conclusive by itself (about 30% by
  chance), but both sets agree, more questions were answered after the first search, time did
  not rise, and the excerpt test predicted it. Cost rose by about 60–70% per question.
- **The round-2 fix on its own was mixed:** of the 25 → 31 step, six gains and all four losses
  were answered in round 2, so keeping round-1 evidence whole helped some questions and probably
  cost others room for follow-up results.
- **Side effect:** longer excerpts show more of each document, so the critic spots more
  disagreements; questions it refused over conflicting sources rose from about 4 to 7.
- **The fresh questions were harder:** 8 of 25 were never found, mostly reworded `semantic`
  questions.

### Remaining failures

In the latest run on the 70, 34 questions stopped at the round limit:

| Group | Questions | What is going on |
| --- | ---: | --- |
| No answer exists (`info_not_found`) | 4 | Correct outcome |
| Never found, or found but not shown | 11 | Retrieval: the right document never reached the critic |
| **Critic saw a reference document and still said insufficient** | **19** | Mostly the critic's rules (one more, qst_0391, stopped with no new evidence) |

The 20 critic refusals, by reason:

| Reason | Count | Examples (from the critic's own words) |
| --- | ---: | --- |
| **Conflicting sources**, refuses to pick | 7 | 0001 sees upload limits of 10, 5 and 8 MB; 0176/0149 see three H200 dates; 0342 sees two policies |
| **Has the answer, wants "explicit confirmation"** | 5 | 0314 worked out the canary start (2028-07-15) and still wants a document that states it; 0462 has the suspected cause and wants a "definitive" one; 0420 wants confirmation that the $0.09 rate applies |
| **Main answer present, a side detail missing** | 6 | 0373 asks whether alerts are suppressed during the grace period; 0454 has the cleaning date and wants the time of day |
| **Genuinely incomplete** | 2 | 0450 needs facts from 8 transcripts and found 1 |

**Note on Jane's critic changes.** The critic in all these runs already includes Jane's changes
of 25 October (commit `c1a5ea5`), so none of the results above come from an older critic. Her
changes:

- **Prompt rules:** don't certify deployment, legal or production status unless asked; separate
  questions about proposals or documented defaults from questions about signed or deployed
  settings; link excerpts from the same document through `doc_id`; don't require exact
  product-name matches or carry terms between customers; check that two sources describe the
  same parameter and scope before calling them conflicting; don't search for confirmation just
  because more documents might exist.
- **What the critic sees:** each excerpt now includes its document ID and title.
- **Final round:** the answer is written from exactly the evidence the critic approved.

The refusals above show what those rules don't cover yet:

| Refusal type | Covered by her rules? | What's still missing |
| --- | --- | --- |
| Wants "explicit confirmation" (0314, 0420, 0462) | Partly: "don't demand confirmation" | Any listed missing fact blocks the answer, even when the main ask is answered |
| Side detail missing (0373, 0454, 0372) | No | No distinction between required and optional facts |
| Conflicting sources (0001, 0176, 0149) | Partly: "check same parameter and scope" | These are true same-scope conflicts; excerpts carry no dates (the `timestamp` field is always empty), so the critic can't prefer the newest source |

Her commit noted that critic-quality improvements were still pending; the proposed critic v2
below builds on her rules.

## Open items and caveats

**Open items**, in suggested order:

| Item | Targets | Expected effect | Risk |
| --- | --- | --- | --- |
| Grade answers against gold answers (`src/evaluation/answer_metrics.py`) | All runs | Shows whether "answered" means correct, and whether some "insufficient" answers are actually right | None; costs some judge calls |
| Critic v2: required vs optional missing facts, answers derived from stated facts, document dates from file paths, prefer the newest source or cite both | ~19 critic refusals of 70 | Perhaps 8–14 more answered (estimate) | May answer when it shouldn't; test as a switchable option, with grading |
| Better follow-up queries (names, IDs and terms from round 1) | 9 never found of 70, 8 of the fresh 25, mostly reworded questions | Perhaps 1–3 more answered | Low |
| Third search round (`RAG_MAX_ROUNDS=3`) | Round-limit questions | Small: round 2 rescued only 3 of 38 | +45–60 s per failing question |
| Reranker on a GPU | Retrieval misses | Untested live on a GPU; CPU runs showed no gain | Needs a GPU |

**Caveats**

- **"Answered" isn't "correct".** None of the live runs were graded; "answered" means the critic
  accepted the evidence.
- **The 70 questions were used to choose the fixes,** so their numbers are optimistic. The 25
  fresh questions are the fairer check, and the original system was never run on them.
- **Noise:** identical runs differ by about 3–4 answered questions per 70, even at temperature
  0. A single step of +3 isn't conclusive; the overall 25 → 34 (13 gained, 4 lost) is stronger.
- **Small fresh sample:** 25 questions can catch a regression, but can't confirm a gain of a
  few questions.
- **Hardware and model:** all live timings are from a 16 GB laptop CPU with `claude-haiku-4-5`.
  A GPU or another model would differ, especially with the reranker on.
- **Retrieval metrics vs agent results:** recall comes from 282 questions with a fixed answer
  key; agent results come from a language model judging evidence, so the two can disagree, as
  the reranker showed.

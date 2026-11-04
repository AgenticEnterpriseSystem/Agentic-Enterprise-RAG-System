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
the corpus or indexes. Reproduction commands are in
[Evaluation](EVALUATION.md#run-checks-and-experiments); dataset provenance is in the
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

## Embedding model screening on a sample

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

## Full-corpus Arctic-embed-l-v2.0

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

## Measure the existing loader

`scripts/profile_retrieval.py` observes the current loader without changing its
ranking, corpus, model revision or precision. Run one attempt at a time, using
explicit artifact paths and a new output directory:

```sh
python scripts/profile_retrieval.py --mode hybrid \
  --corpus /data/documents.parquet \
  --bm25-index /data/indexes/bm25 \
  --dense-index /data/indexes/dense-bge-base \
  --output .cache/resource-profile/hybrid-attempt
```

Modes `inventory`, `corpus`, `model` and `bm25` isolate verification, Arrow loading,
query-encoder loading/encoding, and lexical retrieval. Mode `dense` measures the
existing model/FAISS component without corpus text; any completed queries are
rank-only timings, not evidence materialization or hybrid serving. Inventory checks every
declared shard and hashes the complete artifact files; it does not load vector
payloads into a search index. Hybrid uses the real factory, shared corpus,
manifest-selected model and every dense shard. Searches call the retriever
directly using three existing question-only development workloads, broad/narrow/
denied benchmark scopes, repeated queries, source filters and bounded expansion.
No generator, critic, judge or login is invoked.

Outputs preserve source fingerprints, settings, phase events, raw memory samples,
worker logs and a summary, including failed attempts. Existing output directories
are rejected. A watchdog bounds RSS, cgroup usage, VM headroom and wall time;
use an independently enforced container memory/swap ceiling for full-corpus
experiments. These limits terminate this attempt's worker only. Offline model
flags are set before imports; the pinned model must already be cached at the
source checkout's `.cache/models`. Record the exact container image, source SHA,
mounts, limits and any excluded dependencies with the experiment. Disable an
inherited HTTP healthcheck when the container runs this non-HTTP profiler.

First-query timings are process-first measurements. Hash verification reads
corpus files, and existing filesystem caches are not cleared, so these runs do
not establish cold-disk latency. Raw cgroup totals include file cache; the Docker
CLI's memory display subtracts some cache. Keep process RSS, sampled cgroup
maxima and available kernel high-water counters separate. Missing counters remain
unknown, and exit 137 alone does not establish OOM.
[Docker memory reporting](https://docs.docker.com/reference/cli/docker/container/stats/),
[container limits](https://docs.docker.com/engine/containers/resource_constraints/),
[cgroup counters](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).

Machine-specific measurements and acceptance decisions belong in ignored
`docs/local/`, with raw attempts under ignored `.cache/`. A resource profile is
not an answer-quality score or full-stack Docker acceptance.


## Bounded dense candidate search

`src/retrieval/sharded_dense.py` adds a model-free exact scanner for existing
version-1 dense artifacts. `ShardedDense.load(directory, snapshot=..., catalog=...,
block_rows=8192)` binds a trusted snapshot and ordered metadata-only catalog to
`doc_ids.json` and every declared `.npy`/`.parquet` pair. It never loads canonical
bodies or builds an accumulating FAISS index. The existing `CorpusDense` remains
the reference implementation; application/factory wiring is a later integration
step, so adding this component alone does not change the running web app.

`candidates_vector(vector, top_k, scope=..., filters=...)` returns frozen
`RankedCandidate` spans after snapshot-aligned permission and source/date filters.
Unknown dates fail date bounds. `rank_vector(vector, top_k, mask=None)` is a
trusted offline parity interface returning document IDs/scores; its optional
mask is not an HTTP permission resolver. Query encoding belongs to the adapter.
An empty permitted selection does not scan vectors.

Stored fp16/fp32 rows are copied into bounded float32 blocks and L2-normalized
with the same primitive as the reference loader. Query vectors are used as
supplied, matching the reference vector API. Every document retains its best
passage across all blocks/shards before document top-k. Exact equal scores use
document-ID ordering and then earliest `(start_char, end_char)` passage; floating
point differences between BLAS and FAISS are assessed with a numerical tolerance,
not collapsed into artificial tie groups.

Memory includes document metadata, per-document best scores/spans, the vector
block, Arrow batches, mappings and filesystem cache. For 8,192×768 float32 values,
the vector payload is 24MiB; this is a calculation, not process/cgroup peak.
The original manifest has no cryptographic per-shard digests or catalog revision
identity. Structural/order/span checks and an externally verified snapshot do
not prove that vectors were computed from the claimed text. Full-corpus parity,
combined app/database memory and the accepted120s startup/30s first/10s warm
search targets still require measured acceptance in the later integration gates.
Historical retrieval/answer quality results above are unchanged.


## Full-corpus PostgreSQL import validation

A local LinuxARM64 Docker import on2026-11-03 stored and verified all511,958
canonical documents and1,746,899 indexed passage spans from26 dense shards.
All37 null-containing bodies were preserved byte-for-byte. Ordered catalog
fingerprint: `5b27411a79bb1f04e6b67c5f86039864ac45aaf8ad312598f24f1c7bd8b6fa03`.

The operator job used a768MiB memory/swap ceiling and2 CPUs; PostgreSQL used384MiB
and1 CPU with32MiB shared buffers. Initial migration, file verification, import
and full canonical/span verification took1,334.76seconds. Linux process high-water
RSS was352,040KiB; true cgroup peak was unavailable. The job exited0 with
DockerOOMKilled=false. Its8MiB canonical-payload batch target observed8,388,490
bytes maximum; largest document42,339bytes. These are import measurements, not
combined retrieval/model serving measurements or startup-latency acceptance.

A persistent-volume restart retained every count and the complete checkpoint.
The built minimal image also verified the entire catalog through the nonowner
runtime pool; protected body count remained0 and a synthetic permitted-context
read was rejected by RLS. No production allow policy or real login was inferred
from that check. The database occupied3,527,521,971bytes after import, excluding
separate WAL/volume overhead. Cache was warmed by source verification, and other
unit checks ran during the import. No cold-disk performance claim is made.

Local evidence is ignored under `.cache/platform-validation/doc-01-20261103/`;
reproduction/import commands are in [setup](WEB_APP.md#canonical-document-import).
The web runtime still needs the later authentication/policy/integration gates.

## Store-backed retrieval adapters

`src/retrieval/store.py` provides `StoreBM25Retriever`, `StoreDenseRetriever` and
`StoreHybridRetriever` over the frozen `DocumentStore` contract. Each operation
accepts an explicit server `RequestContext` and `FreshnessGuard`; browser role
labels and legacy allowed-ID sets are not converted into authority. Source/date
filters and snapshot-aligned scope masks apply before ranking. Current context
is checked around encoding, scoring, each body read and final output.

BM25 retains its mmap score/ranking behavior and adds a metadata-only candidate
helper. Hybrid fusion applies the existing weighted RRF arithmetic to metadata
before materializing final winners (or explicit reranker candidates), retaining
document deduplication, dense-passage preference and deterministic ties. An
optional shared catalog avoids retaining duplicate dense catalog objects.

Materialization streams exact overlapping3000-character windows with the existing
query-match/tie heuristic, retaining only the current/best window. This preserves
legacy hybrid output without requiring a full Arrow body table. It can perform
many bounded reads for a long winning document; bounded requests do not imply
constant I/O or a particular backend's physical-read size. `read_more` stays in
matched authorized documents and preserves snapshot/revision identity.

Standalone dense candidates retain the full winning embedding span and ranking.
Standalone returned excerpts are now capped at3000characters, whereas the legacy
standalone dense retriever emitted its full selected span. This declared excerpt
change is separate from document-ranking parity; the hybrid selector already had
that bound. Existing legacy retrievers remain available for comparison.

Fixture checks use real tiny BM25/NumPy/Parquet indexes and compare existing
fusion/window/hybrid/expansion behavior, including revocation boundaries and
embedded null Unicode text. These checks do not establish a live PostgreSQL,
HTTP factory, full-corpus memory/latency or answer-quality result. APP-01/PARITY-01
still perform the concrete integration and measurement.

## Reproducible full-corpus parity protocol

`scripts/compare_retrieval_backends.py` runs four separate offline stages. It
does not import or call an answer provider. Use the nonowner runtime database
configuration and the isolated0600 benchmark credential file from the reviewed
ACL operation; reference reads also pass through actual PostgreSQL policy/RLS.

```sh
python -m scripts.compare_retrieval_backends encode \
  --questions /inputs/questions.jsonl --splits /app/evaluation/splits.json \
  --snapshot /inputs/snapshot.json --corpus /artifacts/documents.parquet \
  --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/shared
python -m scripts.compare_retrieval_backends reference \
  --shared /reports/shared --credentials-file /private/benchmark-credentials.json \
  --corpus /artifacts/documents.parquet --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/reference
python -m scripts.compare_retrieval_backends candidate --scope full \
  --shared /reports/shared --credentials-file /private/benchmark-credentials.json \
  --corpus /artifacts/documents.parquet --dense-directory /artifacts/indexes/dense-bge-base \
  --bm25-directory /artifacts/indexes/bm25 --output /reports/candidate-full
```

Run `candidate` again in fresh processes with `--scope narrow` and `--scope empty`
and new output paths. The default `--file-cache-policy discard` retains the
accepted scanner behavior. The explicit `--file-cache-policy retain` experiment
keeps consumed dense pages eligible for OS file-cache reuse, while still
releasing mmap pages and closing mappings, metadata readers and file descriptors.
It suppresses only POSIX_FADV_DONTNEED advice. Retained file-cache pages count
toward cgroup/VM memory, so this mode requires a new measured resource run under
the same limits; it does not establish fit. The manifest records the policy,
and all three candidate processes must use the same policy for comparison.
Every output directory is exclusive. Interrupted runs stay
incomplete; the tool does not overwrite or silently resume them. Frozen shared
question/vector/config digests and completed case digests are checked before
comparison. Package and executed source hashes are retained. An early frozen
encode snapshot may have a different driver hash; its actual query encoder
implementation must still match, and candidate encodings must be byte-equal to
the frozen vectors. Reference and final candidate source hashes must match.

The predeclared ten development IDs are qst_0001,qst_0431,qst_0412,qst_0384,
qst_0471,qst_0482,qst_0301,qst_0451,qst_0342,qst_0176. The32-document scope is
the union of their14 referenced documents and the first18 other document IDs in
lexical order. This is a reproducible access fixture, not an unbiased retrieval
quality sample. Full-scope cases also exercise the declared query source hints
and an after2020-01-01 UTC filter; unknown timestamps fail date filters.
No calibration/final selection or answer judging is performed.

The primary oracle scores every passage and retains each document's exact
maximum before top-k/deduplication. It uses an independent FAISS1.15.1 native
reference, including the global thread partitions and1024-row BLAS tiles needed
to reproduce full-index scores. Arbitrary small FlatIP blocks can switch native
kernels and cause ULP rank differences; that failed approach is retained in the
development evidence. The bounded matching helper is tested against full native
indexes with bitwise score checks, odd/even row counts, dimensions64/768,
fp16/fp32 and multiple thread counts. Unsupported dispatch settings fail.

Primary ties use exact score equality, lexical document IDs and earliest
start/end offsets. Score tolerance is atol=rtol=2e-6; all score deltas are retained,
and a rank-ID or canonical-span mismatch fails rather than being hidden as a
near-tie. Secondary exhaustive-FAISS representative differences are explicitly
reported only with exact document-maximum/tie proof and both excerpts checked
against canonical source. Native BM25 ranks feed an independent weighted RRF
oracle (weights0.7/0.3,rrf60,candidate50,no reranker). Metadata rank prefixes5,10,20
are compared; the real timed hybrid call materializes top5.

Each candidate process has one active resolver, a resident manifest-matched CPU
fp16 BGE model, mmap BM25 and bounded dense search. Timing includes actual input
verification, imports/model/backend initialization, fresh scope resolution,
adapter-invoked real query encoding, guards and PostgreSQL materialization plus
bounded adjacent reads. Authorized empty/date-filter paths skip encoding/scoring
as the accepted adapter does; the model remains resident. Shared vectors control
numerical comparison, while fresh encoding work is measured and verified.
Startup/first/warm thresholds are120/30/10seconds with three warm repeats. Only
request_index0 in each fresh scope process receives the30-second first limit;
every later request, including new questions and source/date cases, receives
the10-second warm limit. Records carry a contiguous process-wide request_index.
Startup uses Linux process age (driver-entry fallback outside procfs). These
are first-process/warm measurements, not a cold-disk claim. Reference setup with
three workload resolvers is recorded as harness overhead and is not app readiness.
Nested startup events and `startup_components` separate input verification,
database/identity/store/policy setup, imports, catalogue retrieval, dense/BM25
loading, model loading and adapter initialization. Request events and component
fields separate native BM25/dense ranking, materialization, adjacent reads and
freshness-guard calls. These durations are inclusive: guard time overlaps
materialization and adjacent reads, so component totals must not be summed.
Every guard still calls the same authoritative resolver; timing adds no cache
or authorization shortcut.

Finally, `compare --shared ... --reference ... --candidate ...` accepts the three
candidate directories, `--corpus`, three `--resources` files and a new `--output`.
Source evidence is deduplicated across repeats and verified in one bounded scan
of the selected canonical Parquet row groups. Exact Unicode text, revisions,
titles, source/date/provenance, citation offsets and adjacent excerpts are checked.
The command exits nonzero if numerical/text, matrix, latency or resource gates fail.
Missing resource evidence never becomes an acceptance result.

Each external resource report binds `scope` and `candidate_manifest_sha256` to
one completed candidate. It includes complete=true,guard_status=not_triggered,
measurement_kind=sampled_sequential_app_db_cgroup_1s,sample_count,
raw_sampler_sha256,app_supervisor_sha256,peak_aggregate_bytes,
min_vm_available_bytes and oom_events_delta. The aggregate is the maximum of
paired app/database samples, never the sum of isolated component maxima. Require
aggregate<=6.5GiB,VMavailable>=256MiB and no OOM increase. Unknown values remain
unknown; sampled counters do not establish an unavailable true kernel peak.

### I/O placement is a measured prerequisite

The initial read-only Mac bind-mount preflight has already failed the120-second
startup target:388.334seconds for full hybrid. Its first request took122.065seconds
and the first warm repeat110.580seconds, failing both serving limits. The run was
then deliberately stopped and remains an incomplete preflight. App peak RSS was
2,673,369,088bytes; sampled cgroup maximum2,807,717,888bytes; actual paired
app/database sample maximum3,172,130,816bytes. No memory guard or OOM fired.
The separate encode component took
239.623seconds, of which209.690seconds were input hashing. These are component/
incomplete preflight observations, not completed query, resource or parity
acceptance. Preserve their raw profiles and failures.

The completed managed-index run retained the raw Parquet on the Mac bind and
the default discard policy. It covered all 120 full-profile requests, with
startup 139.509 seconds, first request 15.098 seconds, and 35 later requests
above the 10-second limit (maximum 24.418 seconds). The worker exited 0 with
no memory guard and no app OOM increase. Peak RSS was 2,678,919,168 bytes,
sampled app cgroup maximum 3,068,968,960 bytes, and sampled paired app/database maximum
3,452,321,792 bytes. This is a completed candidate profile with failed serving
gates; numerical parity still requires the independent reference/comparison.
Sampled per-request I/O shows each unfiltered warm request reading
2,683,293,696 bytes from storage under unconditional dense cache-discard advice.

Before APP integration, measure verified read-only Docker volumes for both
indexes and raw Parquet, and the explicit retain-cache policy. Keep all26 dense
shards,511,958 documents, pinned model/weights and the same app/database/VM limits;
record storage headroom and prove identical digests. This is an I/O-placement
and cache-budget experiment, not corpus reduction or extra RAM. The managed
index-only placement substantially improved latency but did not meet every gate.
The next results must establish whether retaining the approximately 2.68 GB raw
dense cache improves warm latency while fitting the existing memory limits.
Input checks remain inside the
measured boundary unless a separately reviewed immutable-deployment receipt
protocol is explicitly introduced after evidence. Full numerical, serving and
resource gates remain open until a complete measured protocol passes.

# Model card: Hybrid retriever (BM25+ and BGE-base)

**Card date:** 2026-10-24 · **App version:** 0.1.0 (working alpha)

This card covers the retrieval model that finds the documents behind every answer.
The answer-writing language model is a third-party model chosen when the app starts;
it is summarized [below](#language-model-component) and covered with the rest of the
application in the [system card](system_card.md).

## Model details

| | |
| --- | --- |
| Name | Hybrid retriever: BM25+ and BGE-base, weighted reciprocal rank fusion (RRF) |
| Version | Index format v1; fusion settings as of commit `4856894` (measured runs) |
| Owners | Hira Mian, Jane Choi, Alexander Xykis (CIS 5980 capstone team) |
| Code | `src/retrieval/hybrid.py`, `fusion.py`, `corpus_bm25.py`, `corpus_dense.py` |
| License | Code: MIT. Dataset: MIT, with the authors' request that it not be used for training |

The retriever has two parts whose rankings are combined:

- **BM25+ (keyword search).** Scores whole documents (title + text) on matching words.
  Settings: k1=1.5, b=0.75, delta=1, lowercase word tokens. The index builder is tested
  against the `bm25s` library's scores.
- **BGE-base (meaning search).** `BAAI/bge-base-en-v1.5` at revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, 768-dimension normalized vectors.
  Documents are split into 500-token passages with 50-token overlap, and the title
  (up to 48 tokens) is added to every passage. A document's score is its best passage's
  score. Search is exact cosine similarity (FAISS). Queries get BGE's standard prefix.
- **Fusion.** Each part returns its top 50 documents. They are merged with weighted RRF
  (k=60, BM25+ weight 0.7, BGE weight 0.3), duplicate documents are removed, and the
  top results are passed on. No reranker is used.

We did not train or fine-tune any model. BGE-base is used as published; BM25+ is a
classic statistical method with no learned weights. The team's work is the index
build, the passage format, the fusion settings and the evaluation.

## Intended use

- Find the internal documents most likely to answer an employee-style question in the
  **EnterpriseRAG-Bench** corpus: 511,958 synthetic documents for the fictional
  company "Redwood Inference", from Slack, Gmail, Linear, Google Drive, HubSpot,
  Fireflies, GitHub, Jira and Confluence.
- Supply the evidence passages that the app's critic checks and its answer model cites.
- Serve as a measured, reproducible baseline for retrieval research.

**Use context.** Course staff, the team and other students run it on their own
computer through the web app or the evaluation scripts, with a single demo identity.
Questions are in English. Results are search candidates to be read, not answers.

## Not intended for

- **Real company data or production use.** It has only been measured on a synthetic
  corpus. Results do not carry over to real organizations without new evaluation.
- **Deciding things on its own.** Rankings are not proof that a document is correct,
  current or authoritative. Do not use them for HR, legal, financial, security or
  safety decisions.
- **Questions that need exact counts, totals or the latest version** of a record. It
  returns a few relevant documents, not complete lists, and it has no reliable dates.
- **Other languages or other corpora** without rebuilding the indexes and re-measuring.
- **Training models** on the dataset (the dataset authors ask that it not be used this way).

## Data

- **Corpus:** [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench),
  pinned to upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`. Upstream has
  511,962 files; four document IDs are each used twice, and we keep the file upstream's
  own index points to. Counts and fingerprints are in `evaluation/corpus_manifest.json`.
- **Questions:** 500 benchmark questions with labeled reference documents, split once
  (seed 5980) into **300 development / 100 calibration / 100 final**. Questions that
  share reference documents stay in the same split. The final split has not been used.
- **Skew:** Slack is 56% of documents. Basic and semantic questions are 60% of the
  questions. See the [data card](DATA_CARD.md) for full details.

## Evaluation setup

- **Data:** the full corpus (all 511,958 documents searched for every question).
  Development: 300 questions, 282 scored (18 have no reference labels). Calibration:
  100 questions, 94 scored.
- **Metrics:** Recall@k (share of reference documents found in the top k) and
  nDCG@10 (also rewards ranking them near the top), with 95% bootstrap confidence
  intervals over questions for paired comparisons.
- **Baselines:** BM25+ alone, and BGE-small and BGE-base alone.
- **How settings were chosen:** BM25 weight 0.7 had the best development Recall@10 and
  was then checked once on calibration. Calibration is now closed for further tuning.
- **Runs:** `python -m src.evaluation.run_hybrid` on a Colab T4 GPU; outputs, settings and
  code versions are saved under `evaluation/results/`.

## Performance summary

**Development (282 scored questions)**

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ alone | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small alone | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-base alone (title on every passage) | 0.448 | 0.487 | 0.562 | 0.416 |
| Hybrid, equal weights | 0.627 | 0.700 | 0.744 | 0.579 |
| **Hybrid, BM25+ weight 0.7 (this model)** | **0.641** | **0.722** | **0.760** | **0.604** |
| Hybrid 0.7 + `bge-reranker-base` | 0.551 | 0.637 | 0.716 | 0.520 |
| Hybrid 0.7 + `bge-reranker-v2-m3` | 0.602 | 0.669 | 0.715 | 0.560 |

- Versus BM25+ alone: Recall@10 **+0.029** (95% CI +0.005 to +0.055; 19 questions better,
  7 worse). Ranking quality near the top (nDCG@10) is unchanged.
- **Calibration (94 scored questions, checked once):** Recall@10 **0.767** vs 0.712 for
  BM25+ alone (+0.056, 95% CI +0.013 to +0.106).
- An Arctic-embed-l-v2.0 hybrid scored about the same (0.723 development, 0.788
  calibration); the team kept BGE-base because the gain was too small to justify switching.

**Recall@10 by question type (development)**

| Question type | Questions scored | BM25+ | Hybrid |
| --- | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.819 |
| Semantic (reworded) | 75 | 0.413 | **0.400** |
| Intra-document reasoning | 24 | 1.000 | 1.000 |
| Project related | 24 | 0.676 | 0.716 |
| Constrained | 18 | 0.917 | 0.972 |
| Completeness | 12 | 0.497 | 0.567 |
| Conflicting info | 12 | 0.792 | 0.917 |
| Miscellaneous | 12 | 0.917 | 0.917 |

Groups of 12–24 questions are small, so treat their scores as rough. "High level" and
"info not found" questions have no reference documents and are not scored.

**Speed and size.** About 0.9 s per question for both searches on a Colab T4 GPU. The web
app encodes queries on CPU, which has not been timed. The dense vectors need about 5 GiB
of memory on top of the corpus and BM25 index.

## Language model component

The app's evidence critic and answer writer call an Anthropic Claude model through the
API. The model ID, prices and budget are set by whoever runs the app (`RAG_MODEL`).
The recorded development runs (2026-10-21) used `claude-haiku-4-5-20251001` at
temperature 0 with BM25 retrieval, not this hybrid. The team did not train this model,
and its own model card is published by its provider. **Answer quality has not been
measured**; see the [system card](system_card.md) and [evaluation](EVALUATION.md).

## Ethics and safety

- **Risk: missing or misleading evidence.** On average about 28% of a question's reference
  documents are not in the top 10. If the right document is missed, the answer model may answer from a related
  but wrong document (a similar project, an old draft, another customer).
- **Risk: false confidence.** A high rank looks authoritative but only means the words or
  meaning matched.
- **Risk: uneven quality.** Reworded (semantic) questions are much weaker (Recall@10 0.400),
  so people who don't know the exact project names or IDs get worse results.
- **Risk: privacy if pointed at real data.** The corpus is synthetic, with no real people.
  On real company data, the search index would expose whatever it contains to anyone the
  app lets in.
- **Harm if misused:** decisions based on an incomplete or outdated document.

## Limitations and recommendations

| Limitation | Recommendation / mitigation in place |
| --- | --- |
| Reworded questions score poorly; the hybrid did not help them (0.413 → 0.400) | Use project names, IDs or exact terms; the adaptive agent runs follow-up searches with different wording |
| On average 28% of a question's reference documents are missed at top 10, 24% at top 20 | Show every source excerpt so users can check; answers report "insufficient evidence" when the critic finds gaps |
| No reliable dates; can't pick the latest version | Date filters exclude undated documents instead of guessing; don't rely on it for "latest" questions |
| BGE reads at most 512 tokens per passage; long documents are split | Titles added to every passage; BM25+ scores the whole document; a document's best passage is used |
| Rerankers made results worse as tested | No reranker is enabled in the app |
| Weight 0.7 was tuned on development questions | Confirmed once on calibration; final split kept untouched for the end-of-project check |
| Measured on a synthetic company only | Not for real data without re-evaluation; documented as out of scope above |
| Query encoding on CPU in the web app is untimed | Measure before any latency claims |

## Versioning and maintenance

**Changelog**

| Date | Change |
| --- | --- |
| 2026-10 | BM25+ and BGE-small baselines measured on the full corpus |
| 2026-10 | Title on every passage; BGE-base tested |
| 2026-10 | Live hybrid measured; BM25 weight 0.7 chosen and confirmed on calibration; rerankers rejected |
| 2026-10 | Arctic-embed-l-v2.0 tested; BGE-base kept |
| 2026-10-24 | First model card draft |

**Known issues:** weaker results on reworded questions; no reliable dates; full-corpus
Docker operation not verified; CPU latency not measured; the Colab and local corpus file
hashes differ even though document contents match.

**Contact:** the team above, through issues on the project's GitHub repository.

**Related documents:** [System card](system_card.md) · [Data card](DATA_CARD.md) ·
[Evaluation](EVALUATION.md) · [Retrieval results](FULL_CORPUS.md)

# Model card: Hybrid retriever (BM25+ and BGE-base)

**Updated:** November 16, 2026 · **Owners:** Hira Mian, Jane Choi and Alexander Xykis

The **retriever** finds documents that may help answer a question. It does not write the answer. This card describes the keyword/meaning search combination and its historical research results. The separate [system card](system_card.md) covers the language model, user access and application risks. [Current status](STATUS.md) distinguishes implemented behavior from unfinished evaluation.

## Model details

**Hybrid** means combining two search methods. The current app uses both within the documents the signed-in user can read:

| Part | Plain-English purpose | Configuration |
| --- | --- | --- |
| BM25+ | Find matching words, names and IDs | Whole title + text; lowercase Unicode word tokens; k1=1.5, b=0.75, delta=1 |
| BGE-base | Find passages with similar meaning | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; 768-number normalized embeddings |
| Rank fusion | Combine positions in both result lists | Weighted reciprocal rank fusion (RRF), k=60, BM25+ weight 0.7, dense weight 0.3 |
| Reranker | Optional extra model to reorder candidates | Not enabled in the serving app |

**Embeddings** are vectors of numbers representing text. Document vectors are computed ahead of time and stored once. Each new question is encoded at request time. This is indexing, not training or fine-tuning: published model weights are unchanged.

### How a search works

1. BM25+ and BGE each return up to 50 document candidates from the permitted set.
2. A dense document's score is its best matching embedded passage score.
3. RRF combines the two document rankings and removes duplicates.
4. Selected document passages are loaded for the app to read.

For a document present in both lists, the fusion score is:

```text
0.7 / (60 + keyword rank) + 0.3 / (60 + dense rank)
```

A missing list contributes zero. This combines ranks, not the raw scores from two different scoring systems. The lexical weighting is strong; a semantic match alone does not guarantee a high combined rank.

A **token** is a model text piece, not necessarily a whole word. Passage construction uses a 500-token window budget and 50-token overlap. With title-on-every-passage indexing, up to 48 title tokens are repeated and their length is subtracted from the content-window budget. The model accepts at most 512 tokens including special tokens. Queries use the configured BGE retrieval prefix. Returned excerpts retain exact document character offsets.

### Current serving backend and access checks

The authenticated app uses a bounded reader over saved dense-vector shards, with exact cosine-similarity ranking: comparing the directions of normalized vectors. It copies permitted rows into bounded float32 blocks instead of building a whole resident FAISS index. FAISS is a vector-search library; its normalization math is still used. Historical experiments below used the earlier FAISS-based path; their numbers are not a fresh benchmark of this implementation.

A **scope** is the documents this user may read; a **permission mask** marks the permitted document positions. Scope is resolved before search. Dense search checks it before question encoding and masks denied rows before similarity scoring. BM25 applies its mask before candidate selection. PostgreSQL loads only selected permitted bodies and checks access again. The shared vector files still have to be traversed: a small scope is not a guarantee of proportional disk/memory/time savings.

The managed app requests up to eight text hits per query and then selects at most eight unique documents for the shared critic/answer context, subject to smaller text/token budgets. Configurable ceilings are 5, 8 and 10. Several queries can collect a larger pool. This differs from historical retrieval scoring at top 10 or 20, and earlier `top_k=5` was not a global five-document context.

## Intended use

Find useful English-language evidence in the synthetic EnterpriseRAG-Bench corpus, supply it to the evidence critic/answer writer, and support reproducible retrieval experiments. Search results are candidates for inspection, not proof that a statement is correct, current or authoritative.

The local app supports provisioned users and explicit document grants. It does not import a real organization's identity hierarchy or native sharing rules. The code is MIT; the dataset is MIT with a separate author request not to use it for training. No team model training or fine-tuning was performed.

## Not intended for

- Real company deployment without new data, access-policy and quality evaluation.
- Making HR, legal, financial, security or safety decisions from a ranking alone.
- Proving a complete count or the latest version from a few text results. Supported native Jira counts use a separate complete authorized database query.
- Other languages/corpora without rebuilding and re-evaluating the indexes.
- Training on the benchmark against its authors' request.

## Data

The corpus has 511,958 unique synthetic documents about Redwood Inference from nine platforms. It is pinned to upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`. Four duplicated upstream IDs are resolved using upstream's canonical index. [The data card](DATA_CARD.md) explains extraction, policy/organization metadata and limitations.

There are 500 questions, split into 300 development, 100 calibration and 100 final questions with seed 5980. Questions sharing grouped reference documents/duplicate text remain together. Slack supplies about 56% of documents; basic and semantic questions are 60% of questions.

## Evaluation setup

The tables below report **historical document-retrieval experiments**, not generated-answer accuracy or a current user-permission experiment.

- Every question searched the full 511,958-document corpus under the unrestricted research protocol.
- Of 300 development questions, 282 had reference-document labels and were scored. Of 100 calibration questions, 94 were scored. Empty labels are excluded from these metrics.
- **Recall@k** measures the share of reference documents found in the first k results. For example, finding two of three expected documents gives recall 2/3 for that question.
- **nDCG@10** also rewards placing relevant documents near the top. It does not check the answer's factual claims.
- BM25 weight 0.7 was selected on development and checked once on calibration. Calibration is closed to further retrieval tuning under that protocol; final questions remain untouched.
- Runs used `python -m src.evaluation.run_hybrid` on a Colab T4 GPU. Saved files under `evaluation/results/` record settings and versions; the measured fusion revision was `4856894`.
- Reported 95% bootstrap intervals resample questions for paired comparisons. They describe those experiments, not a guaranteed improvement for other users or data.

## Performance summary

### Development: 282 scored questions

| Method | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25+ | 0.625 | 0.692 | 0.747 | 0.604 |
| BGE-small | 0.382 | 0.468 | 0.525 | 0.367 |
| BGE-base, title on every passage | 0.448 | 0.487 | 0.562 | 0.416 |
| Hybrid, equal weights | 0.627 | 0.700 | 0.744 | 0.579 |
| Hybrid, BM25+ weight 0.7 | 0.641 | 0.722 | 0.760 | 0.604 |
| Hybrid 0.7 + `bge-reranker-base` | 0.551 | 0.637 | 0.716 | 0.520 |
| Hybrid 0.7 + `bge-reranker-v2-m3` | 0.602 | 0.669 | 0.715 | 0.560 |

The recorded paired Recall@10 gain over BM25+ was +0.029, with 95% interval +0.005 to +0.055; 19 questions improved and 7 worsened. These gains use unrounded values; table means are rounded. nDCG@10 was unchanged at the displayed precision.

On the one calibration check, hybrid Recall@10 was 0.767 versus 0.712 for BM25+, with recorded unrounded gain +0.056 and 95% interval +0.013 to +0.106. The Arctic-embed-l-v2.0 hybrid scored 0.723 development and 0.788 calibration; BGE-base remained the chosen configuration. These are historical retrieval results, not evidence that the current adaptive app is better.

### Recall@10 by development question type

| Type | Scored questions | BM25+ | Hybrid |
| --- | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.819 |
| Semantic: reworded | 75 | 0.413 | 0.400 |
| Intra-document reasoning | 24 | 1.000 | 1.000 |
| Project related | 24 | 0.676 | 0.716 |
| Constrained | 18 | 0.917 | 0.972 |
| Completeness | 12 | 0.497 | 0.567 |
| Conflicting information | 12 | 0.792 | 0.917 |
| Miscellaneous | 12 | 0.917 | 0.917 |

Small categories have uncertain averages. High-level and information-not-found questions have no reference-document labels for these scores. Semantic questions remained weak: combining rankings did not improve their average over BM25+.

### Speed and memory

The older T4 experiment reported about 0.9 seconds per query for the hybrid search path. It is not a measurement of a whole current CPU web request, which also includes permissions, routing, repeated searches and model calls. Earlier dense-index memory figures around 5 GiB describe a different resident-index path; current serving reads saved blocks and still uses memory/file cache. [Current status](STATUS.md) records local functional/resource checks. No controlled current CPU latency benchmark or general memory guarantee is established.

## Language model component

The language model is separate from BGE embeddings. The managed live app uses Anthropic `claude-haiku-4-5-20251001` at temperature 0 for the router, critic and answer writer. Stage limits are 2,048 critic output tokens and 1,024 router/generator tokens, within a 16,000-token total question allowance. Model/rates/spending are explicit operator settings.

The adaptive critic checks requested facts and can request query reformulation or more text. It is not a final-answer verifier. Historical BM25/Haiku agent checks and the separate GPT-OSS/Ollama self-graded baseline use different protocols. Current UI checks demonstrate working paths, but also expose a source-supported date that disagreed with the reference. There is no established current answer-quality improvement or calibrated critic probability. See [status](STATUS.md), [evaluation](EVALUATION.md) and the [system card](system_card.md).

## Ethics and safety

- A high rank means a strong text/meaning match, not truth or authority.
- Recall below 1 means reference documents are missed. The historical 0.722 Recall@10 means about 28% of reference documents were missed on average per scored question; it does not mean 28% of answers were wrong.
- Similar projects, incidents and versions can produce a supported but wrong-target answer. Unknown dates cannot establish freshness.
- Permission checks are designed to prevent denied text from reaching the user/model. They do not validate a real enterprise access map, and shared index/metadata files must remain internal.
- Uneven performance can disadvantage users who do not know exact project names or IDs. The semantic-category result is a measured example.

## Limitations and recommendations

| Limitation | Current response or next check |
| --- | --- |
| Reworded questions remain weak | Router/critic can reformulate; improvement still needs measurement |
| Useful facts can be missed or cropped | Show exact excerpts, preserve supported passages, permit bounded read-more and report missing information |
| Near-duplicates can match a different target | Inspect the event/entity/date; valid citations alone are insufficient |
| Dates are often unavailable | Exclude undated evidence from date filters; do not guess “latest” |
| Rerankers worsened these tested settings | No reranker is enabled; this is not a claim that all rerankers are ineffective |
| Context ceilings change eligible returned evidence | Do not equate eight with better quality or claim a completed paired 5-versus-8 study |
| Historical research differs from current serving | Preserve exact data/code/index/scope/settings before comparison |
| Synthetic data and local tests | Re-evaluate on any new data, access model or deployment |

## Versioning and maintenance

Index format is version 1. Data cleanup, embedding/window changes or different model revisions require compatible rebuilt indexes and fresh measurements. The current scope-aware bounded backend must not be silently substituted for the old experiment when making a performance claim. Preserve old outcomes, including errors and unknown spend; do not rewrite them after a repair.

The final split remains reserved for frozen-setting evaluation. Independent human review of answers/scorers and critic calibration are unfinished. Track those acceptance items in [status](STATUS.md), not as a chronological list of private run receipts here.

**Contact:** the team through the project's GitHub issues.
**Related:** [System card](system_card.md) · [Data card](DATA_CARD.md) · [Evaluation](EVALUATION.md) · [Historical retrieval results](FULL_CORPUS.md)

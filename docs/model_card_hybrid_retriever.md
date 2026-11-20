# Model card: Hybrid retriever (BM25+ and BGE-base)

**Updated:** November 16, 2026 · **Owners:** Hira Mian, Jane Choi and Alexander Xykis

The **retriever** finds documents that may help answer a question. It does not write the answer. This card covers the keyword plus meaning search and its historical results. The [system card](system_card.md) covers the language model, access and app risks. [Status](STATUS.md) separates implemented behavior from unfinished evaluation.

## Model details

The app combines two search methods, both limited to documents the signed-in user can read:

| Part | Purpose | Configuration |
| --- | --- | --- |
| BM25+ | Match words, names and IDs | Title + text; lowercase Unicode word tokens; k1=1.5, b=0.75, delta=1 |
| BGE-base | Match meaning | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; 768-number normalized embeddings |
| Rank fusion | Merge both rankings | Weighted reciprocal rank fusion (RRF), k=60, BM25+ weight 0.7, dense weight 0.3 |
| Reranker | Reorder candidates | Not enabled in the app |

Document embeddings are computed ahead of time. Each question is encoded at request time. This is indexing, not training: model weights are unchanged.

### How a search works

1. BM25+ and BGE each return up to 50 candidate documents from the permitted set.
2. A dense document's score is its best-matching passage.
3. RRF merges the two rankings and removes duplicates.
4. Selected passages are loaded for the app.

For a document in both lists:

```text
0.7 / (60 + keyword rank) + 0.3 / (60 + dense rank)
```

A missing list adds zero. Because keyword weight is high, a semantic match alone does not guarantee a high rank.

Passages use a 500-token window with 50-token overlap. Up to 48 title tokens are repeated on each passage and subtracted from the window. The model accepts at most 512 tokens including special tokens. Queries use the BGE retrieval prefix. Returned excerpts keep exact document character offsets.

### Serving backend and access

The app reads saved dense-vector shards and ranks by exact cosine similarity. It copies permitted rows into bounded float32 blocks instead of loading a whole FAISS index. The historical experiments below used the earlier FAISS path, so their numbers do not benchmark the current backend.

Scope is resolved before search. Dense search masks denied rows before scoring; BM25 masks before picking candidates; PostgreSQL loads only permitted bodies and checks again. Shared vector files are still traversed, so a small scope does not promise proportional savings.

The app requests up to 8 text hits per query and passes at most 8 unique documents (configurable to 5, 8 or 10) to the critic and generator. Historical scoring used top 10 or 20.

## Intended use

Find English-language evidence in the synthetic EnterpriseRAG-Bench corpus for the critic and answer writer, and support reproducible retrieval experiments. Results are candidates to inspect, not proof that a statement is correct, current or authoritative.

The local app supports provisioned users and explicit document grants. It does not import a real organization's identity or sharing rules. No team model training or fine-tuning was done.

## Not intended for

- Real company use without new data, access-policy and quality evaluation.
- HR, legal, financial, security or safety decisions from a ranking alone.
- Proving a complete count or the latest version from a few text results. Native Jira counts use a separate full database query.
- Other languages or corpora without rebuilding and re-evaluating the indexes.
- Training on the benchmark against its authors' request.

## Data

511,958 unique synthetic documents about Redwood Inference from nine platforms, pinned to upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`. 500 questions split 300 development, 100 calibration, 100 final (seed 5980). Slack is about 56% of documents; basic and semantic questions are 60% of questions. See the [data card](DATA_CARD.md).

## Evaluation setup

The tables are **historical document-retrieval experiments**, not answer accuracy or a permission-scoped test.

- Every question searched the full 511,958-document corpus.
- Of 300 development questions, 282 had reference labels and were scored. Of 100 calibration questions, 94 were scored.
- **Recall@k** is the share of reference documents found in the first k results.
- **nDCG@10** also rewards ranking relevant documents near the top. It does not check answer claims.
- BM25 weight 0.7 was chosen on development and checked once on calibration. Calibration is closed to further tuning; final questions are untouched.
- Runs used `python -m src.evaluation.run_hybrid` on a Colab T4 GPU. Saved files in `evaluation/results/` record settings and versions; the measured fusion revision was `4856894`.
- 95% bootstrap intervals resample questions for paired comparisons.

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

Paired Recall@10 gain over BM25+: +0.029 (95% interval +0.005 to +0.055); 19 questions improved, 7 worsened. nDCG@10 was unchanged at the displayed precision.

Calibration check: hybrid Recall@10 0.767 versus 0.712 for BM25+ (gain +0.056, 95% interval +0.013 to +0.106). The Arctic-embed-l-v2.0 hybrid scored 0.723 development and 0.788 calibration; BGE-base stayed the choice. These are retrieval results, not evidence the adaptive app is better.

### Recall@10 by development question type

| Type | Scored questions | BM25+ | Hybrid |
| --- | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.819 |
| Semantic (reworded) | 75 | 0.413 | 0.400 |
| Intra-document reasoning | 24 | 1.000 | 1.000 |
| Project related | 24 | 0.676 | 0.716 |
| Constrained | 18 | 0.917 | 0.972 |
| Completeness | 12 | 0.497 | 0.567 |
| Conflicting information | 12 | 0.792 | 0.917 |
| Miscellaneous | 12 | 0.917 | 0.917 |

Small categories give uncertain averages. High-level and information-not-found questions have no labels, so they are not scored. Reworded questions stayed weak: fusion did not beat BM25+ on them.

### Speed and memory

The older T4 experiment took about 0.9 seconds per query for hybrid search. That is not a whole CPU web request, which also includes permissions, routing, repeated searches and model calls. Earlier dense-index memory of about 5 GiB was for a different, fully resident index. No controlled current CPU latency or memory benchmark exists; see [status](STATUS.md).

## Language model component

The language model is separate from BGE. The live app uses Anthropic `claude-haiku-4-5-20251001` at temperature 0 for router, critic and answer writer (limits in the [system card](system_card.md)). The critic checks requested facts and can reformulate or read more; it is not a final-answer verifier. Historical BM25/Haiku agent checks and the GPT-OSS/Ollama self-graded baseline use other protocols. No current answer-quality improvement or calibrated critic probability is established. See [status](STATUS.md) and [evaluation](EVALUATION.md).

## Ethics and safety

- A high rank means a strong match, not truth or authority.
- Recall 0.722@10 means about 28% of reference documents were missed on average per scored question. It does not mean 28% of answers were wrong.
- Similar projects, incidents and versions can give a supported but wrong-target answer. Unknown dates cannot show freshness.
- Permission checks aim to keep denied text from the user and model. They do not validate a real access map, and shared index files must stay internal.
- Users who do not know exact project names or IDs may be served worse. The semantic-category result is a measured example.

## Limitations and recommendations

| Limitation | Response or next check |
| --- | --- |
| Reworded questions are weak | Router and critic can reformulate; the gain is unmeasured |
| Useful facts can be missed or cropped | Show exact excerpts, keep supported passages, allow bounded read-more |
| Near-duplicates match the wrong target | Inspect event, entity and date; a valid citation is not enough |
| Dates often missing | Date filters exclude undated evidence; never guess "latest" |
| Rerankers hurt in tested settings | None enabled; not a claim about all rerankers |
| Context ceiling changes returned evidence | Eight is not proven better; no paired 5-versus-8 study |
| Research differs from serving | Record data, code, index, scope and settings before comparing |
| Synthetic data, local tests | Re-evaluate on new data, access models or deployments |

## Versioning and maintenance

Index format is version 1. Data cleanup, window changes or new model revisions need rebuilt indexes and new measurements. Do not swap the current backend in for the old experiment when making performance claims. Keep old outcomes, including errors and unknown spend.

The final split is reserved for frozen-setting evaluation. Human review and critic calibration are unfinished; track them in [status](STATUS.md).

**Contact:** the team through the project's GitHub issues.
**Related:** [System card](system_card.md) · [Data card](DATA_CARD.md) · [Evaluation](EVALUATION.md) · [Historical retrieval results](FULL_CORPUS.md)

# Agentic RAG for Enterprise Knowledge Search

A FastAPI + React app for answering questions over EnterpriseRAG-Bench documents
with cited sources. It combines BM25 and BGE-base retrieval using RRF. An LLM
critic checks the retrieved passages and requests follow-up searches when needed.

## Run the app

Follow [Setup](docs/WEB_APP.md) to install dependencies, build the frontend,
obtain the corpus/index bundle, and configure your local `.env`.

The corpus and indexes are not in Git. Download everything in the **`files_web_app`**
folder of the [team Google Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing): `documents.parquet`,
`indexes/bm25/`, `indexes/dense-bge-base/` and `questions.jsonl`. [Setup, step 2](docs/WEB_APP.md#get-the-artifacts)
shows where each one goes.
Then, from the repository root:

```sh
source .venv/bin/activate
python scripts/run_web.py
```

Open <http://127.0.0.1:8000> and wait for **Ready**. Stop with Ctrl+C.
Hybrid search is the default. `python scripts/run_web.py --check` prints the
settings the app would use without starting it; a BM25-only baseline needs
`--retrieval bm25`. Docker is an optional alternative covered in the setup guide.

| Mode | Behavior |
| --- | --- |
| `fixture` | Scripted demo; no artifacts or API key |
| `retrieval` | Real search; no generated answers |
| `live` | Search, evidence critique and cited answers; API key and budget required |

Select **Adaptive agent** for follow-up searches or **Fixed RAG** for one search.
Neither generates answers in retrieval-only mode.

## Status

Working alpha: full-corpus native hybrid search and a live agent request have
been verified. A first answer-quality baseline exists: 20 development questions
answered by GPT-OSS from BM25 evidence and graded 0–2 by an LLM judge (mean
correctness 1.30, citation support 1.55; see [Answer evaluation](docs/ANSWER_EVALUATION.md)).
Answer quality of the web configuration (hybrid + Claude) and full-corpus Docker
validation remain unfinished. Known issues are described in [Setup](docs/WEB_APP.md),
[System](docs/SYSTEM.md) and the [system card](docs/system_card.md).

The corpus and indexes are downloaded from the `files_web_app` folder of the
[team Google Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).
Never commit `.env` or API keys.

## Updates since Milestone 2

Milestone 2 delivered two retrieval baselines on the full 511,958-document corpus:
BM25+ (Recall@10 0.692) and BGE-small + FAISS (0.468). Since then:

- **Model testing:** Added document titles to every dense passage, tested four more
  dense embedding models, cross-encoder rerankers, and various fusion weights.
  Replaced BGE-small with BGE-base.
- **Hybrid search:** BM25+ and BGE-base (title on every passage) combined with
  weighted rank fusion, BM25+ weight 0.7. Recall@10 is 0.722 on development and
  0.767 on calibration.
- **Adaptive agent:** an LLM evidence critic that checks whether the evidence answers
  the question, up to one round of follow-up searches, then a cited answer. A
  one-search "Fixed RAG" path is kept as the comparison baseline.
- **Grounded answers:** answers may cite only the passages they were given, and
  invalid citations or out-of-scope evidence are rejected.
- **Web app:** FastAPI + React with clickable citations, source excerpts and search
  steps, in demo, search-only or live (Claude) mode, with spending caps. The data
  downloads from the team Drive ([Setup](docs/WEB_APP.md#get-the-artifacts)).
  Optional Docker setup included.
- **Answer evaluation:** a 0–2 rubric (correctness, completeness, citation support,
  abstention) graded by an LLM judge. First baseline, 20 questions with GPT-OSS
  answers from BM25 evidence: mean correctness 1.30 and citation support 1.55
  ([Answer evaluation](docs/ANSWER_EVALUATION.md)).
- **Docs and tests:** a [model card](docs/model_card_hybrid_retriever.md), a
  [system card](docs/system_card.md), and 206 automated tests in CI.

## How it works

```text
Question → hybrid search (BM25+ and BGE-base) → critic: "is this enough?"
         → follow-up searches if needed → answer with cited sources
```

## Dataset

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) is a
synthetic company, "Redwood Inference", with:

- **511,958 documents** from 9 sources (Slack, Gmail, Linear, Google Drive,
  HubSpot, Fireflies, GitHub, Jira, Confluence)
- **500 questions**, each labeled with the documents that answer it

We split the questions into **development** (300, for building and comparing),
**calibration** (100, for tuning later), and **final** (100, untouched until the
end). The split is in `evaluation/splits.json`.

## Results so far

Scored on the 300 development questions (282 have labeled answer documents):

| Method | What it does | Recall@10 | nDCG@10 |
| --- | --- | ---: | ---: |
| **BM25+** (baseline) | Keyword search: matches the question's words | 0.692 | **0.604** |
| **BGE-small + FAISS** (baseline) | Meaning search: matches similar meaning | 0.468 | 0.367 |
| BGE-base, title on every passage | Larger meaning-search model | 0.487 | 0.416 |
| **Hybrid: BM25+ and BGE-base** | Combines both rankings (RRF, more weight on BM25+) | **0.722** | **0.604** |

- **Recall@10**: share of the correct documents found in the top 10 results.
- **nDCG@10**: also rewards ranking the correct documents near the top (1.0 is perfect).
- The hybrid was confirmed once on the calibration questions: Recall@10 0.767 vs
  0.712 for BM25+ alone.
- Arctic-embed-l-v2.0 beats BGE-base on its own (Recall@10 0.554 vs 0.487), but its hybrid
  with BM25+ is only marginally better (0.723 vs 0.722), so the team keeps BGE-base for now
  (see [full-corpus Arctic](docs/FULL_CORPUS.md#full-corpus-arctic-embed-l-v20)).

What this means:

- Keyword search beats BGE in every question type, likely because this dataset is
  full of project codenames and IDs, which exact word matching handles well.
- BGE finds some documents that keyword search misses. Combining the two with more
  weight on keyword search is the best setting so far. Cross-encoder rerankers made
  results worse as tested, likely because they saw only one short section of long
  documents (see [retrieval results](docs/FULL_CORPUS.md#hybrid-search-and-reranking)).

Full results by question type: [docs/FULL_CORPUS.md](docs/FULL_CORPUS.md).

## Documentation

- [Setup](docs/WEB_APP.md)
- [System](docs/SYSTEM.md)
- [Evaluation](docs/EVALUATION.md)
- [Answer evaluation](docs/ANSWER_EVALUATION.md): the 20-question GPT-OSS generation and LLM-judge baseline
- [Retrieval results](docs/FULL_CORPUS.md)
- [Dataset](docs/DATA_CARD.md)
- [Model card: hybrid retriever](docs/model_card_hybrid_retriever.md)
- [System card](docs/system_card.md)

Team: Hira Mian, Jane Choi, Alexander Xykis. [MIT License](LICENSE).

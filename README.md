# Agentic RAG for Enterprise Knowledge Search

A course project that answers questions about a company's internal documents
(Slack, email, tickets, wikis, and more) and cites the documents it used.

**Status:** the two retrieval baselines are evaluated on the full dataset.
Hybrid retrieval, reranking, and the search agent have offline implementations;
live quality and performance validation remain pending. C3 answer evaluation is
currently on hold.

## How it will work

```text
Question → Search Agent → Retrieve documents → Search Critic ("is this enough?")
         → search again if needed → Generate answer with sources
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

Two baselines, scored on the 300 development questions (282 have labeled answer
documents):

| Baseline | What it does | Recall@10 | nDCG@10 |
| --- | --- | ---: | ---: |
| **BM25+** | Keyword search: matches the question's words | **0.692** | **0.604** |
| **BGE-small + FAISS** | Meaning search: matches similar meaning | 0.468 | 0.367 |

- **Recall@10**: share of the correct documents found in the top 10 results.
- **nDCG@10**: also rewards ranking the correct documents near the top (1.0 is perfect).

What this means:

- Keyword search wins in every question type, likely because this dataset is
  full of project codenames and IDs, which exact word matching handles well.
- BGE finds some documents that keyword search misses. Merging the two with more
  weight on keyword search raises Recall@10 slightly (0.692 → 0.705); bigger gains
  likely need re-ranking or a stronger model.

Full results by question type: [docs/FULL_CORPUS.md](docs/FULL_CORPUS.md).

## Quick start

Set up Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Direct dependencies are pinned to the tested environment. For the complete
Python 3.11 macOS environment snapshot, use `requirements.lock.txt`; package
availability may differ on other platforms. Fresh installation from the snapshot
has not been verified. New pilot/retrieval manifests record runtime versions.

**1. Build the dataset** (downloads about 5 GB, then takes about a minute):

```sh
python -m src.data.corpus --clone
```

**2. Run the BM25+ baseline** (about 5 minutes; needs about 8 GB of memory):

```sh
python -m src.retrieval.corpus_bm25
python -m src.evaluation.run_retrieval --method bm25 --split development \
  --output .cache/retrieval-runs/bm25-development-new
```

**3. Run the BGE baseline** (needs a GPU; about 2 hours on a free Colab T4):
open [`notebooks/dense_baseline_colab.ipynb`](notebooks/dense_baseline_colab.ipynb)
in Colab and run all cells.

Historical results are in `evaluation/results/`. The command above writes to
`.cache/retrieval-runs/` to preserve them; choose a fresh output directory per run.

## Everyday commands

With the existing corpus and indexes in `.cache/corpus/`, these are the main
commands. None makes paid calls or downloads models:

```sh
# Check code correctness
.venv/bin/python -m pytest -q

# Check BM25 artifact readiness
.venv/bin/python -m src.agents.pilot

# Compare 20 development queries (choose a fresh label each time)
.venv/bin/python -m src.evaluation.compare_fixes --label my-check
```

Add `--memory` to the comparison command for a separate full dense-index loading
measurement without a query model. See [evaluation guidance](docs/EVALUATION.md)
for the distinction between these checks and answer-quality evaluation.

## Pilot options

Reuse matching `.cache/corpus/documents.parquet`, `questions.jsonl`, and the entire
`indexes/bm25/` directory. Saved result JSON files cannot replace these artifacts.
Hybrid search also needs `indexes/dense/` and a query model matching its manifest.
For an fp32 index, use `--retrieval hybrid --device cpu --no-fp16`; for Hira's
Colab fp16 index, use `--retrieval hybrid --device cuda` on an NVIDIA GPU machine.
CPU fp16 support varies; hybrid setup may download BGE if it is not cached.

Live runs require `ANTHROPIC_API_KEY` and explicit model, total budget, and current
prices per million tokens. Replace the placeholders before running:

```sh
python -m src.agents.pilot --live --count 1 \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS \
  --input-price INPUT_DOLLARS_PER_MILLION \
  --output-price OUTPUT_DOLLARS_PER_MILLION
```

Paid calls are off by default. One provider shares the run's budget; SDK retries
are disabled. Use one worker; prices and token counts are estimates. Pilot outputs
in `.cache/agent-pilot/<timestamp>/` include cases, a reference review sheet,
usage summary, and a manifest with code/data/model/prompt/runtime versions.
The pilot is a debugging tool; **C3 answer evaluation remains paused**.

**Live demo:** open [`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) in Colab to
search all documents interactively and run the benchmark live.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/data/` | Building the dataset |
| `src/retrieval/` | BM25+, BGE, hybrid fusion, and optional reranking |
| `src/evaluation/` | Evaluation code: retrieval metrics, runner, splits, and offline comparisons |
| `src/generation/` | Writing answers with an LLM (paid calls are off by default) |
| `src/agents/` | Search Agent, Search Critic, routing, and development pilot |
| `src/api/` | Planned web API |
| `evaluation/` | Saved evaluation data only: frozen splits, corpus record, historical results |
| `.cache/` | Local corpus/index artifacts and temporary experiment outputs (Git-ignored) |
| `docs/` | Topic guides; start with this README for commands and navigation |
| `notebooks/` | Colab notebooks: the BGE baseline, and a live search demo |
| `tests/` | Four test modules: retrieval, generation/contracts, agents, evaluation |

## Documentation

Read these in order; each guide owns one topic:

1. [System](docs/SYSTEM.md): workflow, component roles, status and limitations.
2. [Baseline Results](docs/FULL_CORPUS.md): measured retrieval scores and exact run
   settings. Start here after README for a TA progress review.
3. [Evaluation](docs/EVALUATION.md): measurements, frozen splits and planned answer
   rubric. C3 LLM answer evaluation is paused.
4. [Data Card](docs/DATA_CARD.md): provenance, license, counts and dataset risks.

Personal implementation and merge plans are kept separately under `docs/local/`
and excluded from Git. They are historical planning records, not public status
reports or evidence that tickets are complete.

## Contributing

Workflow: personal branch → `staging` → `main`.
See [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Team

Hira Mian, Jane Choi, and Alexander Xykis.

## License

[MIT](LICENSE).

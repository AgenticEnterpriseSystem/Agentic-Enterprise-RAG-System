# Setup

Start with the [README demo](../README.md#run-the-demo). This guide covers real
data, live answers, development without Docker, and hybrid integration.

## Choose a mode

| `RAG_MODE` | What runs | Required |
| --- | --- | --- |
| `fixture` | Synthetic documents and scripted model responses | No artifacts or key |
| `retrieval` | Real document search; no critique or generated answer | Corpus and index |
| `live` | Real search plus model critique/generation | Corpus, index, API key, budget |

In retrieval mode, both workflow buttons perform one search. In live mode,
**Fixed RAG** searches once; **Adaptive agent** can perform follow-up searches.
Fixture examples: “Who pays the storage fee?” gives a partial answer;
“unavailable information” gives no evidence; “demo failure” exercises an error.
These scripted behaviors apply only in fixture mode.

## Get the artifacts

**Shared download location: not yet published.** Obtain a matching bundle from the
team's index owner, or build BM25 locally below. Results JSON files are not indexes.

```text
redwood-artifacts/
  documents.parquet
  questions.jsonl           # Needed for evaluation, not web queries
  indexes/
    bm25/                   # Complete directory, including manifest and doc_ids
    dense-bge-base/          # Optional hybrid index
```

The bundle must identify its producing code commit, corpus SHA-256 and index/model
manifests. Record an archive checksum for transfer. Corpus contents and document
order must match the indexes; even a regenerated Parquet file can have a different
file hash. Do not edit manifests to bypass compatibility checks.

### Build BM25 locally

Use Python 3.11 and Git. Skip this if you already have matching artifacts.
From the repository root, on macOS/Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.corpus --clone --manifest .cache/corpus/build_manifest.json
python -m src.retrieval.corpus_bm25
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell. The build downloads the pinned
upstream dataset (about 5 GB) and writes `.cache/corpus/`. Allow roughly 8 GB of RAM
for full-corpus BM25 plus application/OS headroom; build time depends on hardware.
Dense index building is covered by the
[BGE Colab notebook](../notebooks/dense_baseline_colab.ipynb).

## Configure Docker

### 1. Create your local settings file

The repository includes [`.env.example`](../.env.example), a template without
credentials. Each person makes their own `.env` from it. **Push `.env.example` to
GitHub, never `.env`.** Git ignores `.env`, and Docker excludes it from the image.
The scripted demo works without this file; real search and live answers need settings.

From the repository root, on macOS/Linux:

```sh
test -f .env || cp .env.example .env
```

On Windows PowerShell:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

These commands preserve an existing `.env`. Open it in your editor and change
the existing values—do not add duplicate lines. Docker Compose reads it
automatically; exported shell variables take precedence.

### 2. Connect the real documents

First [obtain or build the corpus and index](#get-the-artifacts). For real search
without model calls, set these values in `.env`:

```dotenv
RAG_MODE=retrieval
RAG_RETRIEVAL=bm25
RAG_DEMO_ACCESS=true
RAG_ALLOW_PAID=false
ARTIFACTS_DIR=./.cache/corpus
```

If using a downloaded bundle, change `ARTIFACTS_DIR` to its host folder, such as
`/data/redwood-artifacts` or `C:/data/redwood-artifacts`. Docker mounts it read-only
at `/data`. Optional `RAG_CORPUS` and index paths must refer to paths **inside** the
container, such as `/data/documents.parquet` and `/data/indexes/bm25`.

### 3. Start the app

```sh
docker compose up -d --build
```

Open **http://127.0.0.1:8000** and wait for **Ready**. Startup verifies and loads the
index once; it does not reload the corpus for every question.

### 4. Optional: enable generated answers

Live mode adds generated answers to real document search. Keep the document paths
from step 2. In **your local `.env`**, change the following values. Replace each
capitalized placeholder; do not copy your key into `.env.example`.

```dotenv
RAG_MODE=live
RAG_ALLOW_PAID=true
ANTHROPIC_API_KEY=YOUR_KEY
RAG_MODEL=YOUR_MODEL_ID
RAG_BUDGET_USD=YOUR_SESSION_CAP
RAG_QUESTION_BUDGET_USD=0.10
RAG_INPUT_PRICE=USD_PER_MILLION_INPUT_TOKENS
RAG_OUTPUT_PRICE=USD_PER_MILLION_OUTPUT_TOKENS
```

Previous live development tests used `claude-haiku-4-5-20251001`. Set prices for the
model you actually use. Keep keys server-side; never put them in frontend code or
share resolved `docker compose config` output, which can contain secrets.

Apply the new settings with `docker compose up -d --build`, then reopen
http://127.0.0.1:8000. The page should show **Live generation**. Each person uses
their own key and pays for requests made with it; your key is not included in GitHub.

The session cap is shared across requests; the question cap covers critique and
generation for one question. The backend reserves estimated maximum call costs,
so it may stop before displayed costs reach the cap. **Restarting resets the
session cap.** Track prior spending yourself; this is not an account-wide limit.
Use one backend process. A second expensive request receives a busy response.

Try a specific real-data question:
“For the ProofPoint pentest artifact audit-proxy escrow, what is the default
retention period, standard access turnaround, and payer for AuditWorks storage fees?”
A cited or partial response still needs review against its sources.

## Hybrid handoff

Use the **same code commit** for the app and handoff test. The app uses the existing
`HybridRetriever` and manifest-aware `BGEModel.for_index`; it does not implement
separate fusion logic. The index owner must supply:

- The exact corpus and compatible BM25 index, plus the complete BGE-base directory:
  `manifest.json`, `doc_ids.json`, and every `.npy` / `.parquet` shard pair.
- A complete supported manifest with model name, exact 40-character revision,
  precision, query prefix, normalization, sequence length, corpus hash and counts.
- Tested fusion settings, required hardware, and any reranker/code dependency.
  This web runtime uses weighted RRF (BM25 0.7, dense 0.3; k=60, pools of 50),
  **without a reranker**. Different winning configurations need explicit integration.

For compatible CPU artifacts, add to `.env` and rebuild:

```dotenv
INSTALL_HYBRID=true
RAG_RETRIEVAL=hybrid
RAG_DENSE_INDEX=/data/indexes/dense-bge-base
```

This installs CPU PyTorch, sentence-transformers and FAISS. Model weights may
need an initial download and persist in a Docker volume. Leave `RAG_FP16` unset
unless explicitly matching the manifest. BGE-small is not a substitute for BGE-base.

The Compose image is **CPU-only**. For a verified CUDA/fp16 setup, use native
startup on the index owner's GPU machine after installing dependencies and
building the frontend as below:

```sh
RAG_MODE=retrieval RAG_DEMO_ACCESS=true RAG_RETRIEVAL=hybrid \
RAG_CORPUS=/srv/redwood/documents.parquet \
RAG_BM25_INDEX=/srv/redwood/indexes/bm25 \
RAG_DENSE_INDEX=/srv/redwood/indexes/dense-bge-base \
RAG_DEVICE=cuda RAG_FP16=true RAG_BM25_WEIGHT=0.7 \
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Require `/api/ready` to return 200 and a query to return hybrid passages before
trying live generation. Dense vectors occupy host RAM even with CUDA encoding:
allow at least `passage_count × dimension × 4` bytes, plus corpus, BM25 and model
headroom. Full BGE-base runtime verification remains pending.

## Run without Docker

Use the Python environment above and Node **22.12+**. Install frontend dependencies
and build once:

```sh
cd frontend
npm ci
npm run build
cd ..
RAG_MODE=fixture RAG_DEMO_ACCESS=true python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Native startup does **not** load `.env`; export the settings in your shell.
Native artifact defaults are under `.cache/corpus`, not Docker's `/data`.
For UI editing, run `npm run dev` from `frontend/` in a second terminal. Vite
normally serves port 5173 and proxies `/api` to port 8000; `API_PROXY_TARGET`
changes the backend target. Rebuild for changes to appear in the production UI.
`requirements.lock.txt` is a macOS environment snapshot, not a portable Linux lock.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Page won't open | `docker compose ps` and `docker compose logs --tail=50 app` |
| Loading retriever | Wait for `/api/ready`; verify artifact paths and enough Docker RAM |
| Missing/incompatible artifacts | Supply matching complete files; no fixture fallback occurs |
| Slow answer | Expand Search steps for per-step timings; model calls are sequential |
| Budget reached | Review spending before changing the cap; restarting resets accounting |
| Port 8000 occupied | Set `WEB_PORT` in `.env` and reopen that port |
| Browser origin denied | Add the actual UI origin to `RAG_ALLOWED_ORIGINS` |
| Error while stopping | Check `docker compose ps -a`; the container may have exited afterward |

Compose allows 60 seconds for shutdown. This mitigates an observed delayed exit;
it is not a proven root-cause fix. Restarting Docker Desktop affects other projects.
`docker compose down` keeps artifacts and the model cache. The demo is bound to
localhost and uses a benchmark identity, not real authentication. See
[System](SYSTEM.md) for access controls and [Evaluation](EVALUATION.md) for tests.

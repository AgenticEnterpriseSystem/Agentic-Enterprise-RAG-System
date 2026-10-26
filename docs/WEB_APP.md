# Run the web app

This guide covers local installation, configuration, startup and shutdown.
Commands below are for macOS/Linux and run from the repository root.

<a id="build-bm25-locally"></a>

## 1. Install

Requirements: Python 3.11, Node.js 22.12+ and Git. Full-corpus native operation
has been tested on a 16 GiB Mac; leave memory available for the app.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
test -f .env || cp .env.example .env
```

If `.venv` already exists, skip creating it.

<a id="get-the-artifacts"></a>
<a id="hybrid-handoff"></a>

## 2. Add the search data

The corpus and indexes are too large for Git (about 4 GB). Download them from the
team's shared Google Drive folder:
<https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing>

1. Open the link, then open the folder **`files_web_app`**. If Drive asks for
   permission, request access from the team.
2. Download all of its content:

   | Item | What it is |
   | --- | --- |
   | `documents.parquet` | All 511,958 documents |
   | `bm25_indexes/` | BM25+ keyword index (8 files) |
   | `bge_base_indexes/` | BGE-base meaning-search index (54 files: `manifest.json`, `doc_ids.json`, and `shard_00000` to `shard_00025` as `.npy` and `.parquet`) |
   | `questions.jsonl` | The 500 benchmark questions (evaluation scripts only) |

   Select all four items, then right-click and choose **Download**. Drive zips large
   downloads and may split them into several zip files.
3. Unzip every zip file into one folder outside the repository, for example
   `~/Downloads/files_web_app`. If unzipping creates a second copy of a folder
   (such as `bge_base_indexes 2`), move its files into the first one:

   ```sh
   mv -n ~/Downloads/files_web_app/"bge_base_indexes 2"/* ~/Downloads/files_web_app/bge_base_indexes/
   ```

4. From the repository root, move the files into the layout the app expects. The
   two index folders are renamed as they move:

   ```sh
   SRC=~/Downloads/files_web_app
   mkdir -p .cache/hybrid-artifacts/indexes .cache/corpus
   mv "$SRC/documents.parquet" .cache/hybrid-artifacts/
   mv "$SRC/bm25_indexes" .cache/hybrid-artifacts/indexes/bm25
   mv "$SRC/bge_base_indexes" .cache/hybrid-artifacts/indexes/dense-bge-base
   mv "$SRC/questions.jsonl" .cache/corpus/
   ```

   Change `SRC` if you unzipped elsewhere.

5. Check the result. The two counts should be `8` and `54`:

   ```sh
   ls .cache/hybrid-artifacts/indexes/bm25 | wc -l
   ls .cache/hybrid-artifacts/indexes/dense-bge-base | wc -l
   ```

The finished layout:

```text
.cache/hybrid-artifacts/
  documents.parquet
  indexes/
    bm25/
    dense-bge-base/
.cache/corpus/
  questions.jsonl
```

Keep all index files and manifests together and unchanged: at startup the app
checks that both indexes were built from this exact `documents.parquet` and refuses
to start otherwise. Evaluation result files cannot replace the indexes. Git ignores
`.cache/`, so these files are never committed. Keep the repository out of iCloud
Drive or other synced folders, which can replace large files with placeholders.

If you instead received the older split download (five `rag-dense` folders), place
them in the repository root and assemble once:

```sh
python -m src.data.assemble_artifacts \
  --roots rag-dense "rag-dense 2" "rag-dense 3" "rag-dense 4" "rag-dense 5" \
  --output .cache/hybrid-artifacts
```

## 3. Configure

Edit the existing entries in `.env` to match these values:

```dotenv
RAG_MODE=retrieval
RAG_RETRIEVAL=hybrid
RAG_DEMO_ACCESS=true
RAG_ALLOW_PAID=false
ARTIFACTS_DIR=./.cache/hybrid-artifacts
RAG_DENSE_INDEX=/data/indexes/dense-bge-base
RAG_FP16=
```

The launcher reads `.env` and translates `/data` paths to the local artifact
folder. Hybrid is the default; a conflicting BM25 setting stops startup instead
of silently changing retrieval. Other exported shell settings take precedence.
Keep credentials in `.env`;
never commit that file.

| Mode | Result |
| --- | --- |
| `fixture` | Scripted demo; no search data or API key needed |
| `retrieval` | Real search excerpts; no generated answers |
| `live` | Real search, evidence checks and generated answers; API key required |

## 4. Start

```sh
python scripts/run_web.py
```

Leave the terminal open. Visit <http://127.0.0.1:8000> and wait for **Ready**.
Initial startup loads the indexes and may download model weights.

Use `python scripts/run_web.py --check` to inspect effective settings without
starting. For a deliberate BM25 baseline only, use `--retrieval bm25`.

Check <http://127.0.0.1:8000/api/ready>: HTTP 200 with `retrieval: hybrid` means
hybrid search is ready. HTTP 503 means loading or a configuration error.

To stop, press Ctrl+C and wait for exit. To restart, run the same command again.
Do not start a second instance while the first is running.

## 5. Enable generated answers

Once retrieval works, update these entries in `.env`. Replace placeholders with
your API key, model ID, spending cap and the model's current token prices:

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

Restart and refresh the browser. Select **Adaptive agent** for follow-up searches
or **Fixed RAG** for one search. Inspect the answer's citations and source excerpts.
Restarting resets the app's session cap, not actual provider spending.

## Docker alternative

Docker is optional. Install Docker with Compose v2, prepare the same search data
and `.env`, and add `INSTALL_HYBRID=true`. Stop any native instance first.

```sh
docker compose up -d --build
```

Use the same URL and readiness check. Apply settings changes with the same
command. Stop with `docker compose down`; this preserves the search data and
model cache. Inspect errors with `docker compose logs --tail=50 app`.

An 8 GiB Docker allocation is insufficient for the full hybrid setup: dense vectors
alone require about 5 GiB in addition to the corpus, BM25 and model. Full-corpus
Docker operation is not yet verified; use native startup if memory is limited.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Port 8000 already in use | Stop the existing native or Docker app before restarting. |
| Not ready | Read `/api/ready` and the terminal/logs; check artifact paths, missing files and available RAM. |
| Wrong mode | Check `.env` and exported `RAG_*` settings, then restart and refresh. |
| Missing frontend | Run `npm --prefix frontend run build`, then restart. |
| Busy or budget limit | Wait for the current request, or review the configured spending limits. |

See [System](SYSTEM.md) for architecture and limits, and
[Evaluation](EVALUATION.md) for tests and known quality limitations.

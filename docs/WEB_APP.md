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

1. Open the link and find the folder **`files_web_app`**. If Drive asks for
   permission, request access from the team. It contains:

   | Item | What it is |
   | --- | --- |
   | `documents.parquet` | All 511,958 documents |
   | `indexes/bm25/` | BM25+ keyword index (8 files) |
   | `indexes/dense-bge-base/` | BGE-base meaning-search index (54 files: `manifest.json`, `doc_ids.json`, and `shard_00000` to `shard_00025` as `.npy` and `.parquet`) |
   | `questions.jsonl` | The 500 benchmark questions (evaluation scripts only) |

2. Right-click `files_web_app` and choose **Download**. Drive zips the folder and may
   split it into several zip files.
3. Unzip every zip file into the same place, for example `~/Downloads`. If unzipping
   creates a second copy (such as `files_web_app 2`), merge it into the first:

   ```sh
   cp -R ~/Downloads/"files_web_app 2"/. ~/Downloads/files_web_app/
   ```

4. From the repository root, move the folder into the project as
   `.cache/hybrid-artifacts`, and copy the questions file where the evaluation
   scripts look for it:

   ```sh
   mkdir -p .cache/corpus
   mv ~/Downloads/files_web_app .cache/hybrid-artifacts
   cp .cache/hybrid-artifacts/questions.jsonl .cache/corpus/
   ```

   Change `~/Downloads` if you unzipped elsewhere.

5. Check the result. The two counts should be `8` and `54`:

   ```sh
   ls .cache/hybrid-artifacts/indexes/bm25 | wc -l
   ls .cache/hybrid-artifacts/indexes/dense-bge-base | wc -l
   ```

The finished layout:

```text
.cache/hybrid-artifacts/
  documents.parquet
  questions.jsonl
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

## Database foundation

The optional PostgreSQL service provides schema and connection infrastructure.
The current question API still uses its benchmark identity; real sign-in,
permission-policy resolution, corpus import, and application database wiring are
later integration work.

Set three different local secrets in `.env`: `RAG_DB_BOOTSTRAP_PASSWORD`,
`RAG_DB_MIGRATOR_PASSWORD`, and `RAG_DB_RUNTIME_PASSWORD`. Leave them out of Git.
Then initialize a **new, dedicated** database volume and run the migration job:

```sh
docker compose --profile database --profile migrations build migrate
docker compose --profile database --profile migrations run --rm migrate
```

The database has no published host port. It persists in `postgres-data`; the
migration job waits for database health and exits after upgrading. Repeating the
migration is safe. The init hook runs only on a new volume and rejects preexisting
application roles or a non-pristine database; changing environment secrets does
not rotate passwords in an existing volume. The bootstrap role creates separate
non-superuser migration and runtime roles. Application login accounts are not
seeded by this infrastructure setup.

The runtime pool defaults to two connections with no overflow. Server-resolved
request identity is transaction-local, and rollback plus `DISCARD ALL` clears
session state before reuse. Runtime credentials are checked against both the
actual login principal and active SQL identity; owner, superuser, replication,
role-creation, database-creation, and RLS-bypass authority are rejected. Passwords
in DSN query parameters are rejected because URL representations can expose them;
structured `RAG_DB_HOST/PORT/NAME/USER/PASSWORD` settings support special characters.
Migration credentials are used only by the separate migration job.

The schema covers accounts/sessions, groups/grants, immutable catalog/text,
tickets, runs/steps, and aggregate provenance. Protected text, ticket, run, and
aggregate tables start with RLS enabled and no allow policies. Published snapshot
content cannot change; ready state requires a complete catalog and canonical
bodies. Policy/session counters cannot decrease, and saved run bindings cannot
be reassigned. Actual account and permission adapters must pass their later gates
before authenticated queries are enabled.

Real PostgreSQL tests use explicitly configured `RAG_TEST_ADMIN_URL`,
`RAG_TEST_MIGRATION_URL`, and `RAG_TEST_RUNTIME_URL` on an isolated disposable
server with a `rag_test_*` database. They skip when those settings are absent and
refuse ordinary database names. CI starts its own loopback-only server. The
snapshot-restore check additionally uses `RAG_TEST_PG_CONTAINER`, labeled
`rag.task=DB-01`; its dump/restore commands run inside that disposable container.
These tiny fixtures do not establish full-corpus import or recovery performance.


### Canonical document import

The optional import job stores the full corpus and every matching dense passage
span in PostgreSQL. It uses the dedicated migration login and a separate bounded
connection, never elevated runtime credentials. It does not connect the web
application to this store yet. With the database passwords configured as above
and the complete unchanged artifacts in `.cache/hybrid-artifacts`, run:

```sh
docker compose --profile database --profile migrations run --build --rm migrate
docker compose --profile database --profile imports run --build --rm import-corpus
```

The importer requires the current single migration head `0002_catalog_import`.
`RAG_SNAPSHOT_ID` names its immutable snapshot. Rerunning with identical files
verifies/resumes the same import; changed source files reject. Each transaction
commits its rows and checkpoint together, and incomplete imports stay unavailable.
A published snapshot is never overwritten. Correct a bad source and use a new
snapshot ID rather than deleting existing data.

For a read-only verification of the actual source files, canonical bodies,
ordered catalog and all indexed spans:

```sh
docker compose --profile database --profile imports run --rm import-corpus \
  python -m src.data.import_catalog \
  --corpus /data/documents.parquet \
  --dense-directory /data/indexes/dense-bge-base \
  --snapshot-id enterprise-bge-base-canonical --verify-only
```

Change that ID if you configured another. Reports go to standard output; the
library/CLI also supports an exclusive `--report` file. No credential CLI flags
or model calls are used. Default batches target500 rows and8MiB of canonical
UTF-8 payload, with at most128 Arrow rows prefetched; one oversized document is
preserved and reported. These limits are not process RSS. The import container
has a768MiB ceiling and the database384MiB; full-stack serving fit is a separate
gate. The named database volume persists across container restarts.

The source contains37 documents with null characters. They use an exclusive
bounded `canonical_utf8` binary representation; other documents retain TEXT.
The original bytes, SHA256 revisions, Unicode lengths and citation coordinates
are preserved. No row or character is dropped. Binary payloads are capped at1MiB
and strictly UTF-8 validated; current source documents are at most42,339 UTF-8
bytes. Normal excerpts use SQL Unicode substrings. Rare binary excerpts fetch
and validate the bounded binary body before returning the exact requested span.

The lazy `PostgresDocumentStore` implements the accepted metadata/excerpt/
read-more interface with the nonowner runtime pool. Protected body reads still
default-deny until reviewed ACL policies land. A checked import is not actual
login, permission policy, web integration, or a completed Docker deployment.

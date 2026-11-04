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

The legacy hybrid loader accumulates roughly5GiB of float32 vectors alongside
full corpus text, BM25 and the query model. It exceeded the measured profiling
budget. Bounded dense and lazy storage components are being integrated; complete
Docker serving fit and latency still require their explicit acceptance gate.

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
The benchmark query profile retains its explicit demo identity. The local
account API below provides real sign-in; document policy and authenticated query
wiring remain gated integration work.

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
default-deny without a current registered scope and explicit reviewed grants. A checked import is not actual
login, permission policy, web integration, or a completed Docker deployment.

### Local account API

Set up the database and run migrations first. From an interactive terminal,
provision an account with the separate migration job:

```sh
docker compose --profile database --profile migrations run --build --rm migrate \
  python -m src.auth.accounts create jane --display-name "Jane" --role employee
```

Passwords are prompted twice, never accepted as command-line flags. Non-TTY
password creation/reset refuses to read input. The same CLI has `reset-password`
and `disable`; both invalidate existing sessions through the account version.
Role/department/project labels describe identity and grant no document rights.
There is no signup, email reset, SSO or default seeded login.

For explicit loopback HTTP development, configure:

```dotenv
RAG_AUTH_MODE=local
RAG_AUTH_SECURE_COOKIES=false
RAG_AUTH_ALLOW_INSECURE_LOOPBACK=true
RAG_AUTH_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

Start the already migrated database and app with
`docker compose --profile database up --build`. The app receives only runtime
credentials; account administration retains a separate owner login. HTTPS
profiles keep secure cookies enabled and list explicit HTTPS origins.
`RAG_AUTH_ALLOWED_ORIGINS` overrides the shared `RAG_ALLOWED_ORIGINS` fallback.
The optional `RAG_AUTH_ENABLED=true` alias also enables protected local mode;
invalid/conflicting mode flags fail startup, rather than silently serving demo
queries. Explicit factory-injected settings take precedence over environment.

| Endpoint | Behavior |
| --- | --- |
| `GET /api/auth/csrf` | Short-lived signed pre-login cookie and CSRF value |
| `POST /api/auth/login` | JSON login/password plus matching bootstrap CSRF header and allowed Origin; renew session token |
| `GET /api/auth/me` | Current public identity and session CSRF after authoritative checks; does not extend idle expiry |
| `POST /api/auth/logout` | Session CSRF plus allowed Origin; revoke session and expire cookie |

The CSRF header is `X-CSRF-Token`. Session cookies are host-only, HttpOnly and
SameSite=Strict; raw session tokens stay out of response JSON and PostgreSQL.
Mutations require an exact configured Origin. Authentication responses are
no-store and credential errors never echo request inputs. The HTTP body limit is
16KiB even for a misleading Content-Length.

Passwords use the pinned Argon2id RFC9106 low-memory profile (64MiB,3 iterations,
4 lanes); encoded parameters are checked before verification. A separate single
auth worker has bounded waiting admission, global/per-login/per-IP throttles and
bounded process-local counters. Timeouts/disconnects retain admission until the
actual worker exits. These controls are for the documented single-process app;
they are not a distributed rate limiter. [Argon2 parameter guidance](https://argon2-cffi.readthedocs.io/en/stable/parameters.html).

Idle and absolute lifetimes are checked after row-lock waits and before release.
They use the current server-configured policy; changing lifetime settings is an
administrative policy change, not permanent token revocation. Before increasing
limits where existing deadlines must remain fixed, invalidate the affected
sessions with the account reset/disable operation. Ordinary unchanged-policy
restarts preserve valid sessions.
Logout cannot be undone by a racing session touch. Account disable/password reset
is rechecked after password hashing without holding a database transaction across
that work. Runtime users remain SELECT-only on accounts; PostgreSQL row-lock
privileges are not broadened for authentication. [PostgreSQL locking privileges](https://www.postgresql.org/docs/17/sql-select.html).

This ticket provides the backend account API. The account UI and per-user query
integration come later. In local-auth mode, `/api/ready` stays503 without corpus
metadata and even a valid session receives403 for queries until document policy
is connected. No benchmark identity is used as a fallback; login remains
responsive while a separate RAG initialization/worker is occupied.

### Reviewed document permissions

Upgrade the operator database to `0003_document_policy` before resolving scopes.
Keep migration credentials in the separate operator job; the app uses only its
nonowner runtime pool. Provision human accounts with the interactive account CLI
above, then review an explicit policy manifest. No role, department or project
label creates a grant, and unclassified documents remain denied.

The manifest is JSON with these fields:

| Field | Required meaning |
| --- | --- |
| `schema_version` | `"1"` |
| `manifest_id` | Unique reviewed patch ID; changed content needs a new ID |
| `snapshot` | Exact SnapshotRef object from the verified import report |
| `origin` | `operator` or explicitly synthetic QA policy |
| `reviewed_by`, `provenance` | Named reviewer and nonempty object of provenance strings |
| `groups` | `{group_id, label, requires: []}`; optional flat compound prerequisites |
| `memberships` | `{user_id, group_id}` using actual provisioned IDs |
| `grants` | `{doc_ids: [...], user_id: ...}` or `{doc_ids: [...], group_id: ...}` |

For example, a synthetic `finance_exec` group can declare
`"requires": ["finance", "executive"]`. A user needs all three explicit
memberships (`finance_exec`, `finance`, `executive`). Membership in a project
group alone cannot unlock an executive grant. Base groups cannot themselves be
compound. Use real imported document IDs and the exact snapshot fingerprints;
the tool rejects missing users, groups and documents.

Run the review and apply steps with the migration job's existing credentials:

```sh
python -m src.access.repository dry-run --manifest /private/policy.json
python -m src.access.repository apply --manifest /private/policy.json
python -m src.access.repository revoke --manifest /private/policy.json
```

Apply is additive and idempotent; it never overwrites an existing group definition,
grant origin or membership owner. Revoke removes only rows originally added by
that exact manifest. It leaves group definitions/prerequisites intact and cannot
delete a pre-existing real policy through a synthetic/no-op manifest. A revoked
manifest requires a new review/ID before application. Policy mutations and their
epoch changes commit or roll back together; stale request contexts must resolve
again. Authenticated web queries remain denied until application integration.

Full-corpus retrieval experiments use a separate benchmark account. Prepare a
private JSON file containing only the exact SnapshotRef object from the import
report, then explicitly review all-document access:

```sh
python -m src.access.repository benchmark-create retrieval-benchmark \
  --snapshot /private/snapshot.json \
  --credentials-file /private/benchmark.private.json \
  --reviewed-by local-operator --provenance "Reviewed retrieval-only benchmark" \
  --all-documents
python -m src.access.repository benchmark-verify \
  --credentials-file /private/benchmark.private.json
```

These commands print no credential, raw nonce or identity. Creation requires a
new path, writes mode0600 exclusively and never replaces an existing file or
account. Verification uses the runtime credentials, reads a bounded ordinary0600
file through one non-symlink file descriptor, and produces an internal identity
for library consumers. Never place this file in source control. Benchmark kind
is immutable and is rejected by public login and `/me`, even with its correct
private password or a valid nonce; it cannot become a browser login bypass.
The `--synthetic` flag labels QA provenance explicitly. No default benchmark or
human policy is seeded into a deployment.

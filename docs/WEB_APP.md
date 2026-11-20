# Redwood reference guide

The [quick start](QUICKSTART.md) is the step-by-step setup. This page covers what
the quick start leaves out. For what has been tested, see [Current status](STATUS.md).

All commands run from the repository folder and use `--env-file .env.local`.

<a id="prepare-the-machine"></a>
<a id="run-the-local-application"></a>

## Already-running installation

If someone else runs Redwood for you, you don't need Docker. Ask the operator
for the URL and your own login. The operator creates the login and grants its
documents ([quick start steps 8 and 9](QUICKSTART.md#8-create-your-login)). Then
sign in and ask a question.

<a id="get-the-artifacts"></a>
<a id="hybrid-handoff"></a>
<a id="download-the-model-cache-yourself"></a>

## Data folders

Setup needs three folders. The quick start fills them in
([steps 3 to 5](QUICKSTART.md#3-download-the-document-bundle)).

| Folder | What goes in it |
| --- | --- |
| `.cache/hybrid-artifacts/` | `documents.parquet` and both prebuilt indexes (`bm25`, `dense-bge-base`) |
| `.cache/models/` | The embedding model cache (`models--BAAI--bge-base-en-v1.5`) |
| `.cache/local-source/` | The dataset source checkout, which supplies the original Jira fields |

If a teammate gives you a verified model cache, copy the whole
`models--BAAI--bge-base-en-v1.5` folder, including the files its snapshot links
point to. A folder of broken shortcuts won't work.

A **manifest** is a file that lists what a set of files contains. Keep every
file it names. `questions.jsonl` is only needed for evaluation. See
[Data and index details](FULL_CORPUS.md) if the Drive layout is unclear. The
dataset checkout does not provide a real company's permissions.

<a id="local-account-api"></a>
<a id="provision-local-accounts"></a>
<a id="reviewed-document-permissions"></a>
<a id="apply-reviewed-document-permissions"></a>

## Give a login a small set of documents

A new login can see no documents. A role such as `employee` or `admin` grants
nothing. To give a login all test documents, use
[quick start step 9](QUICKSTART.md#9-give-your-login-access-to-the-documents).
To give it only a few, use a manifest.

1. List five example document IDs. This reads no document text and changes
   nothing:

   ```sh
   python3 scripts/local_stack.py --env-file .env.local operator -- python -c '
   import json
   from pathlib import Path
   import sqlalchemy as sa
   from src.db import schema
   from src.db.runtime import database_url

   snapshot = json.loads(Path("/state/snapshot.json").read_text())
   engine = sa.create_engine(database_url(migration=True), hide_parameters=True)
   try:
       with engine.connect() as connection:
           with connection.begin():
               connection.exec_driver_sql("SET TRANSACTION READ ONLY")
               rows = connection.execute(sa.select(schema.documents.c.doc_id,
                   schema.documents.c.source_type).where(
                   schema.documents.c.snapshot_id == snapshot["snapshot_id"])
                   .order_by(schema.documents.c.position).limit(5)).mappings().all()
       print(json.dumps({"snapshot": snapshot, "candidates": [dict(row) for row in rows]}, indent=2))
   finally:
       engine.dispose()
   '
   ```

   A **snapshot** is one fixed, verified version of the document collection.

2. Save `.cache/local-operator/hira-access.json`. Replace every placeholder,
   including all snapshot values:

   ```json
   {
     "schema_version": "1",
     "manifest_id": "hira-reviewed-access-001",
     "snapshot": {
       "snapshot_id": "<actual snapshot_id>",
       "corpus_sha256": "<actual corpus_sha256>",
       "catalog_sha256": "<actual catalog_sha256>",
       "document_count": 0
     },
     "origin": "operator",
     "reviewed_by": "<reviewer's name>",
     "provenance": {"approval": "<reviewed access request>"},
     "groups": [],
     "memberships": [],
     "grants": [{"doc_ids": ["<reviewed document ID>"], "user_id": "<the login's user_id>"}]
   }
   ```

3. Preview, then apply:

   ```sh
   python3 scripts/local_stack.py --env-file .env.local operator -- \
     python -m src.access.repository dry-run --manifest /operator/hira-access.json
   python3 scripts/local_stack.py --env-file .env.local operator -- \
     python -m src.access.repository apply --manifest /operator/hira-access.json
   ```

With only a few documents, most benchmark questions will have no answer.

The app checks access before ranking, before reading text, around model calls
and before returning a response. PostgreSQL adds **row-level security (RLS)**:
database rules that reject rows outside the current user's permissions. No
request field or role label can bypass them.

<a id="full-benchmark-qa-access"></a>

### Full-benchmark access: extra notes

Commands: [quick start step 9](QUICKSTART.md#9-give-your-login-access-to-the-documents).

- Local testing only, after you have reviewed access to **all** fictional
  documents. The login must be a human login from step 8. Don't use
  `benchmark-create`; its account can't sign in to the web app.
- `prepare` never overwrites an earlier review. Keep the review folder.
  `dry-run`, `apply` and `revoke` refuse if a review file or the snapshot changed.
- An interrupted `apply` is safe to run again with the same review.
- `revoke` removes only grants these patches created. To apply again, start a
  new review with a new `--directory` and `--approval`.
- This access does not authorize Observability or paid model calls.

<a id="observability"></a>
<a id="use-and-restart-the-application"></a>

## Observability page

`/observability` shows request timings, stage counts, errors and known token and
cost usage. Search access does not grant it. It shows request metadata, not
document contents.

To allow a login, put its `user_id` in `.env.local`. Replace any existing line;
don't add a second copy:

```dotenv
RAG_OBSERVABILITY_ADMIN_USER_IDS=<the login's user_id>
```

Then restart the app and open <http://127.0.0.1:8000/observability>:

```sh
python3 scripts/local_stack.py --env-file .env.local up
```

An empty list denies everyone. No username, `admin` role or PostgreSQL
superuser is needed.

- History keeps at most 500 requests, 256 events per request and one hour. A
  restart clears it.
- It leaves out raw questions, document text and IDs, credentials and error text.
- Unknown usage shows as unavailable.
- Request IDs (`X-Request-ID`) are separate from the agent's `run_id`.

The app runs one expensive search at a time. A second one gets 429. Signing out
or changing the account clears the answer, sources and diagnostics on screen.

<a id="enable-live-generation"></a>

## Live mode (generated answers)

Setup is in [quick start step 11](QUICKSTART.md#11-optional-turn-on-ai-answers).
Details:

- **What leaves your computer:** your question and the selected document
  excerpts go to Anthropic for routing, critique and generation. Agree on the
  model, the excerpts, the number of questions and the spending cap first.
- **Keys:** use your own API key, never someone else's. Keep `.env.local` private
  (`chmod 600 .env.local`) and keep `RAG_DEMO_ACCESS=false`.
- **Model and prices:** these are the project's November 2026 test values, not a
  current price quote. Prices are USD per million input/output tokens. Confirm
  them before paid use.
- **Budget:** `RAG_BUDGET_USD` is kept in memory and resets when the app
  restarts. It is not a lifetime or account-wide limit. A restart does not
  refund earlier or uncertain spend. The app does not retry model calls.
- **Check answers yourself.** A cited answer can still pick the wrong incident
  or date. A safe clarification may stop before the final answer is written.
  [Current status](STATUS.md) lists what was observed.

To go back to free search, set `RAG_MODE=retrieval`, `RAG_ALLOW_PAID=false` and
`RAG_BUDGET_USD=0`, remove the API key if you no longer need it, then run `up`.
Documents, logins and grants are kept.

<a id="evidence-context-and-observability-november-16"></a>
<a id="model-request-and-database-limits"></a>

## Settings

Set these in `.env.local`, then run `up`.

| Setting | Default | What it limits |
| --- | --- | --- |
| `RAG_EVIDENCE_DOC_LIMIT` | 8 (also 5 or 10) | Unique documents given to the critic and generator |
| `RAG_TOP_K` | 5 | Native row limit |
| `RAG_MAX_CONTEXT_CHARS` | 24,000 | Evidence characters |
| `RAG_CRITIC_MAX_OUTPUT_TOKENS` | 2,048 | Critic output |
| `RAG_MODEL_MAX_OUTPUT_TOKENS` | 1,024 | Router and generator output |
| `RAG_MAX_TOKENS` | 16,000 | Total tokens for one question |
| `RAG_MAX_ROUNDS` | 2 | Search rounds |
| `RAG_MAX_SECONDS` | 120 | Agent run time |
| `RAG_REQUEST_TIMEOUT_SECONDS` | 180 | How long the HTTP request waits |
| `RAG_QUESTION_BUDGET_USD` | 0.10 | One live question |
| `RAG_BUDGET_USD` | 0.50 live, 0 in retrieval mode | Spending cap until restart |
| `RAG_DB_IDLE_TRANSACTION_TIMEOUT_MS` | 15,000 | Idle database transactions |

The database statement timeout is 5 seconds and the lock timeout is at most
2 seconds.

The document limit is a maximum, not a promise of relevance. Several passages
from one document count once. The critic and generator see the same evidence.
If kept evidence no longer fits, the agent stops instead of dropping it
silently. The page's **Evidence context** section shows what was left out or
cropped.

## Check saved data

```sh
python3 scripts/local_stack.py --env-file .env.local verify-data
```

Stop and start: see the [quick start](QUICKSTART.md#stop-and-start-again-later).
Keep the same `.env.local`; a new one doesn't change the passwords in an
existing database.

<a id="stop-and-restart"></a>
<a id="validation-and-troubleshooting"></a>

## Troubleshooting

Start with the [quick start table](QUICKSTART.md#if-something-goes-wrong). More
cases:

| What you see | What to check |
| --- | --- |
| `ready` returns 503 | Wait for startup, then check the data and model files and the logs. `health` alone doesn't mean ready. |
| Observability returns 403 | The login's `user_id` isn't in `RAG_OBSERVABILITY_ADMIN_USER_IDS`. Document grants don't count. |
| 429 busy or timeout | Wait for the running search to finish. A lost response doesn't mean no model call was made. |
| Missing or wrong model or index files | Check the manifest and the complete model cache. Never swap in a different corpus. |
| Container exits 137 | Check `OOMKilled` and memory counters. 137 alone doesn't prove it ran out of memory. |

View logs:

```sh
docker compose --env-file .env.local logs --tail 100 app verify
```

Don't post logs, resolved Compose output, passwords or connection URLs.

**Recovering an interrupted copy.** Look at the failure and any stale lock
first. Then you can run `stage --adopt-existing` or `bootstrap --adopt-existing`.
They verify the bytes already there and never overwrite different published
data. A changed corpus needs a new snapshot target.

<a id="independent-fresh-setup-verification"></a>
<a id="final-functional-evidence-and-remaining-limits"></a>

A teammate's independent fresh setup is not yet a verified result. The
[Docker test fixture](../tests/deployment/README.md) covers five fake documents
only, not full-corpus use or answer quality.

<a id="database-foundation"></a>
<a id="canonical-document-import"></a>
<a id="native-jira-feed"></a>
<a id="database-and-import-guarantees"></a>

## Advanced: database and import

- A **migration** creates or updates the database's tables. The current head is
  `0004_jira_feed`.
- Imports tie document text, metadata and revisions to one snapshot. Ready text
  is never overwritten.
- Each source record has its own ID. Native Jira keys can repeat. Missing or
  invalid fields stay unknown.
- The app uses only the restricted `rag_runtime` database login and read-only
  volumes. The operator uses `rag_migrator` for imports and for accounts and
  grants. Never give migration credentials to the app.

To investigate a failure, run the `bootstrap` steps one at a time, in this
order: `build`, `stage`, `migrate`, `import-corpus`, `import-jira`,
`prepare-state`, `up`. Each is a `local_stack.py --env-file .env.local <name>`
command.

`scripts/local_data.py` also takes `--min-free-gib` (default 8, minimum 4).
Lowering it doesn't show the stack will fit. `bootstrap` keeps the default.

## Advanced: account maintenance

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts reset-password hira
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts disable hira
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository revoke --manifest /operator/hira-access.json
```

Resetting a password or disabling an account ends its sessions. Revoking a
manifest removes only the links that manifest created. Groups must be assigned
explicitly. A label such as CEO does not put anyone in a group.

<a id="backup-restore-and-compatible-rollback"></a>

## Advanced: backup, restore and rollback

A backup holds accounts, grants and private records. Store it outside the
database volume with private permissions, a checksum, the image identity and the
matching data and snapshot receipts. Stop the app first. Never copy a running
PostgreSQL data directory.

```sh
umask 077
mkdir -p .cache/local-backups
docker compose --env-file .env.local stop app
docker compose --env-file .env.local exec -T db sh -c \
  'exec pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom --no-owner --schema=rag' \
  > .cache/local-backups/rag.dump
shasum -a 256 .cache/local-backups/rag.dump > .cache/local-backups/rag.dump.sha256
```

Restore into a **new** project and volume. Create `.env.restore` with `init`,
the same input folders, an unused project and port, and a new database name such
as `rag_restore_example`. Don't migrate or start its app before restoring.
Replace both `rag_restore_example` values with your name:

```sh
python3 scripts/local_stack.py --env-file .env.restore build
python3 scripts/local_stack.py --env-file .env.restore stage
docker compose --env-file .env.restore up -d --wait db
docker compose --env-file .env.restore exec -T db sh -c \
  'exec pg_restore --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --no-owner --role=rag_migrator --single-transaction --exit-on-error' \
  < .cache/local-backups/rag.dump
python3 scripts/local_stack.py --env-file .env.restore operator --database rag_restore_example -- \
  python -m src.db.recovery --expected-database rag_restore_example
```

The recovery command disables restored accounts and ends their old sessions. It
keeps documents, grants and history. Review the approved access, then
reactivate each login you intend to keep:

```sh
python3 scripts/local_stack.py --env-file .env.restore operator -- \
  python -m src.auth.accounts reset-password hira --reactivate \
  --reviewed-by reviewer-name --policy-reconciliation approved-access-review-001
```

That flag records your review. It does not check the review or grant documents.
Don't reactivate every user or reuse old cookies. Then run `import-corpus`
(this verifies the existing import and publishes its snapshot to the new state
volume), `prepare-state` and `verify-data`, and start the app only after the
account and permission review. Keep the original stack separate.

To roll back the app, set `RAG_APP_IMAGE` and `RAG_OPERATOR_IMAGE` to an earlier
image that supports the current schema, then run `up`. There is no migration
rollback. For an incompatible schema, restore into a new target.

<a id="build-bm25-locally"></a>
<a id="native-development-and-evaluation"></a>

## Advanced: native development

A Python server on your host needs its own database, path and login settings;
`.env.example` uses Docker names and won't work. You need Python 3.11 and Node 22:

```sh
python3 -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
```

To build a BM25 index, write to a new folder, never over a serving index:

```sh
python -m src.retrieval.corpus_bm25 \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --output .cache/bm25-new
```

For experiments see [Evaluation](EVALUATION.md) and
[Answer evaluation](ANSWER_EVALUATION.md). Reference answers are compared only
after generation and never given to the running system.

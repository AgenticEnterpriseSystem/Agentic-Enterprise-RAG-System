# Set up and use the local app

If you only want to use an already-running installation, ask its operator for
the reachable URL and your own login. You do not need Docker. The operator must
create the account and grant its documents; you can then go straight to
[Open Search](#use-and-restart-the-application).

The path below installs a separate database on your own computer. Start with the unpaid
retrieval mode: it searches documents and shows sources, but does not call a
language model or generate an answer. Enable paid answers only after that works.

The code is in Git. The large document, search-index and model files are separate.
Your accounts, passwords, document permissions and PostgreSQL database are local
to your installation. Pulling the code does not copy Jane's installation.

For what has actually been tested, read [Current status](STATUS.md). The local
checks do not prove answer correctness or that a new teammate's setup will work
without help. Hira's independent setup is still an important check.

<a id="prepare-the-machine"></a>
<a id="run-the-local-application"></a>

## 1. Check your computer

The documented setup is for macOS or Linux. You need:

- Git and Python 3.11 on the host.
- Docker with Compose v2.10 or later. Start Docker before using the commands.
- A 16 GiB machine, with about 8 GiB available to Docker's virtual machine.
- Disk space for the downloaded files, Docker's copies, images, database and a
  backup. The build checks for 12 GiB of free headroom; that is not the total
  space the project needs. Staging also checks the files still to copy plus an
  8 GiB reserve.

Docker builds the frontend, so you do not need host Node.js for this path.
The app has a 6 GiB / 2 CPU limit. PostgreSQL has a 384 MiB / 1 CPU limit.
Neither gets extra swap beyond its memory allowance.

Open a terminal in the repository folder—the one containing `compose.yaml`:

```sh
python3 --version
docker compose version
docker system df
df -h .
```

Keep the data outside folders that can replace local files with cloud-only
placeholders. Do not copy someone else's `.env.local`, database volume or cookies.

<a id="get-the-artifacts"></a>
<a id="hybrid-handoff"></a>

## 2. Get the three input directories

A search **index** is a saved structure that makes searching faster. An
**embedding model** converts a question into numbers for semantic search. Its
cached weights are different from the already indexed document vectors.

Prepare these directories before running setup:

| Directory | What goes in it |
| --- | --- |
| `.cache/hybrid-artifacts/` | Complete document file and both prebuilt indexes |
| `.cache/models/` | The exact embedding model revision named by the dense index |
| `.cache/local-source/` | The upstream dataset Git checkout used to import native Jira fields |

Download `files_web_app` from the
[team Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).
If Drive splits the download into ZIP files, combine their contents without
replacing different copies. The final layout must be:

```text
.cache/hybrid-artifacts/
  documents.parquet
  indexes/
    bm25/
      manifest.json
      ...all other index files...
    dense-bge-base/
      manifest.json
      doc_ids.json
      ...every declared .npy and .parquet shard...
```

A **manifest** records what a file set contains and the versions it uses. Keep
all files it declares. Saved evaluation results cannot replace the indexes.
`questions.jsonl` is needed for evaluation, but not for serving the app.
See [Data and index details](FULL_CORPUS.md) if the download layout is unclear.

Ask for the team's verified model cache, or use the optional download command
[later in this guide](#download-the-model-cache-yourself). Copy the complete
`models--BAAI--bge-base-en-v1.5` cache directory, including files that snapshot
links point to. A folder containing only broken shortcuts will not work.

Get the dataset source at the required commit:

```sh
git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git .cache/local-source
git -C .cache/local-source checkout --detach d36685e273713975ee20299bbf1ab64165575b3c
```

This checkout supplies original Jira fields and source history. It does not
supply an authoritative real-company user-to-document permission system.

## 3. Create your private configuration

These examples use a new Docker project named `redwood-hira` on port 8000.
Use an unused project name if another copy is already installed.

```sh
python3 scripts/local_stack.py --env-file .env.local init \
  --artifacts-dir "$PWD/.cache/hybrid-artifacts" \
  --models-dir "$PWD/.cache/models" \
  --source-repo-dir "$PWD/.cache/local-source" \
  --operator-dir "$PWD/.cache/local-operator" \
  --project redwood-hira --port 8000
python3 scripts/local_stack.py --env-file .env.local plan
```

`init` reads `.env.example` and creates a new mode-600 `.env.local`: only its
owner can read or write it. It generates three different database passwords.
It does not overwrite an existing file, start Docker, create a user or grant
access. Keep `.env.example` as a template with empty password/API-key fields.
Never commit `.env.local`.

<a id="initialize-and-import"></a>

## 4. Build, copy and import

```sh
python3 scripts/local_stack.py --env-file .env.local bootstrap
curl --fail --max-time 5 http://127.0.0.1:8000/api/health
curl --fail --max-time 5 http://127.0.0.1:8000/api/ready
```

`bootstrap` builds the images, verifies and copies the data/model files into
Docker-managed volumes, creates the database tables, imports documents and Jira
records, and starts the app. A **volume** stores data outside a container, so
recreating the container does not erase the database. Initial setup can take time.

`health` reports that the process is alive. `ready` checks current dependencies;
503 means it is starting or unavailable. Neither response proves answer quality.
If setup stops, read [Troubleshooting](#validation-and-troubleshooting) before
retrying. Do not delete volumes to get past a failure.

<a id="local-account-api"></a>
<a id="provision-local-accounts"></a>

## 5. Create Hira's login

An **operator** is the person administering this installation. The operator runs
steps 5 and 6; a test user with a provided login can skip them. If you are setting
up your own machine, you perform these steps for your own database.

Run this from an interactive terminal. Enter a private password of at least
12 characters when prompted; do not put it in a command argument.

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts create hira --display-name Hira --role employee
```

Save the returned `user_id`. This is the account's stable internal ID; it is not
the username. Hira's ID on her computer will differ from an ID on Jane's computer.

A new account has **no document access**. A role such as `employee`, `ceo` or
`admin` is descriptive and grants nothing. The next step assigns documents.

<a id="reviewed-document-permissions"></a>
<a id="apply-reviewed-document-permissions"></a>

## 6. Give the login reviewed document access

For a small test, choose the documents the user is allowed to read. This command
prints the imported snapshot and five existing candidate IDs, without reading
any document bodies or changing permissions:

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

A **snapshot** identifies one fixed, verified version of the corpus. The candidate
IDs above are examples to review, not automatic recommendations for access.

Save a reviewed file at `.cache/local-operator/hira-access.json`. Replace every
placeholder below, including all snapshot values, before running it:

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
  "grants": [{"doc_ids": ["<reviewed document ID>"], "user_id": "<Hira's returned user_id>"}]
}
```

Preview the change, check the target user and documents, then apply it:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository dry-run --manifest /operator/hira-access.json
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository apply --manifest /operator/hira-access.json
```

This file is an explicit permission patch. A small patch makes Search available
only for those documents; unrelated benchmark questions may have no answer.
For Hira's full fictional-corpus testing, use the separate, reviewed
[full-benchmark QA recipe](#full-benchmark-qa-access) below.

The app checks access before ranking, before reading text, around model calls
and before returning a response. PostgreSQL also enforces **row-level security
(RLS)**: database checks that reject rows outside the current user's permissions.
No request field or role label can bypass those checks.

<a id="use-and-restart-the-application"></a>

## 7. Open Search and, optionally, Observability

Open <http://127.0.0.1:8000>, sign in as `hira`, and wait for **Ready**.
The initial mode returns permitted sources only. It does not run the critic or
produce an AI answer. Use a question about a document granted in step 6.

Source cards show exact passages and citation IDs. Native Jira cards show actual
imported fields; missing fields stay unavailable. Jira counts describe canonical
source records, not necessarily unique issue keys. Duplicate keys may require
clarification. Source/date filters narrow the documents the user can access.

<a id="observability"></a>

The separate `/observability` page shows safe request timing, stage counts,
errors and known token/cost usage. Search permission does not grant access to it.
To authorize Hira as a diagnostics operator, explicitly put her returned user ID
in the **private** `.env.local`:

```dotenv
RAG_OBSERVABILITY_ADMIN_USER_IDS=<Hira's returned user_id>
```

Replace that setting rather than adding a second copy. Then recreate the app:

```sh
python3 scripts/local_stack.py --env-file .env.local up
```

Open <http://127.0.0.1:8000/observability>. An empty allowlist denies everyone.
A username, `admin` role label or PostgreSQL superuser is not required. This
capability reveals request metadata across the local process, not document
contents, and does not grant documents. Each read checks the current login.

The app admits one expensive search at a time; another gets 429. A timed-out
HTTP request keeps that slot until its real worker finishes. Sign out or account
changes clear the displayed answer, sources and diagnostics.

<a id="enable-live-generation"></a>

## 8. Enable generated answers only when approved

Live mode sends the question and selected permitted excerpts to Anthropic for
routing, critique and generation. Agree on the model, excerpt transfer, number
of questions and spending cap before using it. Do not reuse Jane's private key
or assume her earlier test approvals cover another installation.

Edit the existing private `.env.local`. Keep its database secrets, paths and
`RAG_DEMO_ACCESS=false`. Replace these settings; do not append duplicates:

```dotenv
RAG_MODE=live
RAG_RETRIEVAL=hybrid
RAG_ALLOW_PAID=true
ANTHROPIC_API_KEY=<private API key>
RAG_MODEL=claude-haiku-4-5-20251001
RAG_INPUT_PRICE=1
RAG_OUTPUT_PRICE=5
RAG_BUDGET_USD=0.50
RAG_QUESTION_BUDGET_USD=0.10
```

The model and prices are the project's November 2026 test configuration, not a
current provider price quotation. Confirm access and prices before paid use.
Prices are USD per million input/output tokens. Use the agreed budgets.

```sh
chmod 600 .env.local
python3 scripts/local_stack.py --env-file .env.local up
```

Only then submit approved test questions. Check the actual answer, citation
buttons, source cards, search steps and usage. A cited answer can still select
the wrong incident or date. A safe clarification may stop without invoking the
final generator. See [Current status](STATUS.md) for observed successes and limits.

The process spending counter resets on restart. It is not an account-wide or
lifetime bill limit, and a restart does not refund previous or uncertain spend.
There are no automatic model retries in the managed runtime.

To return to unpaid search, set `RAG_MODE=retrieval`, `RAG_ALLOW_PAID=false` and
`RAG_BUDGET_USD=0`, clear an unneeded API key, then run the same `up` command.
Documents, accounts and grants are preserved.

<a id="evidence-context-and-observability-november-16"></a>
<a id="model-request-and-database-limits"></a>

## Settings you may need

| Setting | Managed default | What it limits |
| --- | --- | --- |
| `RAG_EVIDENCE_DOC_LIMIT` | 8; supports 5, 8 or 10 | Unique documents in the shared critic/generator input |
| `RAG_TOP_K` | 5 | Native row limit; text search returns up to the greater of this and the document ceiling |
| `RAG_MAX_CONTEXT_CHARS` | 24,000 | Evidence characters; token reserves may reduce the selected context |
| `RAG_CRITIC_MAX_OUTPUT_TOKENS` | 2,048 | Critic output |
| `RAG_MODEL_MAX_OUTPUT_TOKENS` | 1,024 | Router/generator output |
| `RAG_MAX_TOKENS` | 16,000 | Total tokens for one question, across all stages |
| `RAG_MAX_ROUNDS` | 2 | Adaptive search rounds |
| `RAG_MAX_SECONDS` | 120 seconds | Agent execution allowance |
| `RAG_REQUEST_TIMEOUT_SECONDS` | 180 seconds | Query HTTP wait, including work before/after the agent |
| `RAG_QUESTION_BUDGET_USD` | $0.10 | One live question |
| `RAG_BUDGET_USD` | $0.50 in the reviewed live profile; 0 in retrieval mode | In-memory process reservation cap |
| `RAG_DB_IDLE_TRANSACTION_TIMEOUT_MS` | 15,000 ms managed; 5,000 ms library default | Idle database transactions |

The 180-second HTTP wait does not give the agent 180 seconds. Library callers
that omit it keep an outer deadline equal to the agent allowance. Database
statement timeout remains 5 seconds; lock timeout remains at most 2 seconds.

An evidence ceiling is a maximum, not a quota or relevance guarantee. Several
passages from one document count once. Aggregate counts stay whole rather than
pretending to reveal all underlying documents. Earlier supported passages are
retained exactly. The critic and generator use the same frozen ordered evidence.
If retained evidence no longer fits, the agent stops safely rather than silently
dropping it. The UI's **Evidence context** section reports exclusions and cropping.
The two retrieval branches still request 50 candidates each; fusion is unchanged
and the managed factory has no reranker enabled.

Observability uses server-issued `X-Request-ID` values, separate from the agent's
`run_id`. It measures HTTP and worker completion separately with monotonic clocks.
Its history is limited to 500 requests, 256 events per request and one hour; a
restart clears it. A bounded nonblocking log queue drops excess events and reports
drops. It excludes raw questions, document text/IDs, credentials and arbitrary
error messages. Unknown usage stays unavailable. Docker log retention is separate.

<a id="stop-and-restart"></a>

## Stop, restart and check saved data

```sh
python3 scripts/local_stack.py --env-file .env.local stop
python3 scripts/local_stack.py --env-file .env.local up
python3 scripts/local_stack.py --env-file .env.local verify-data
```

These commands keep the named volumes and data. Keep the same private environment
file: generating new passwords does not rotate passwords in an existing database.
Do not use `down -v` as routine shutdown.

<a id="validation-and-troubleshooting"></a>

## Troubleshooting

| What you see | What to check |
| --- | --- |
| `ready` returns 503 | Wait for startup, then check data/model files and private logs. Liveness alone is not readiness. |
| Login works but Search returns 403 | A new account has no grants, or its session/policy changed. Check the reviewed permission patch. |
| Observability returns 403 | The current immutable user ID is not allowlisted. Document grants do not authorize diagnostics. |
| 429 busy / timeout | Wait for the actual worker to finish. Do not repeatedly submit or assume a lost response means no model call. |
| Missing/incompatible model or shards | Check the manifest and complete cache. Never switch to a demo identity or substitute a different corpus to hide failure. |
| Build/staging disk error | Free enough space for both remaining copies and the stated reserve. The reserve is not a total-size estimate. |
| Container exits 137 | Inspect `OOMKilled` and memory counters. Exit 137 alone does not prove an out-of-memory failure. |

Inspect logs with the same project/configuration:

```sh
docker compose --env-file .env.local logs --tail 100 app verify
```

Do not publish logs containing secrets, resolved Compose environments, passwords
or connection URLs. After an interrupted unpublished copy, inspect its failure
and stale lock before using `stage --adopt-existing` or
`bootstrap --adopt-existing`. These options verify existing bytes; they do not
overwrite different published data. A changed corpus needs a new snapshot target.

<a id="independent-fresh-setup-verification"></a>
<a id="final-functional-evidence-and-remaining-limits"></a>

A teammate's independent fresh setup is not yet a verified release claim. Record
the commit, machine, inputs, commands, account/grant preparation and observed
result. The tiny [Docker test fixture](../tests/deployment/README.md) checks real
accounts/database behavior on five synthetic documents; it does not prove
full-corpus resource use or answer quality.

<a id="download-the-model-cache-yourself"></a>

## Optional: download the model cache yourself

Skip this if you received the complete verified cache. This command downloads
the exact model recorded by the dense index; it does not call a paid answer model:

```sh
mkdir -p .cache/models
docker build --build-arg INSTALL_HYBRID=true --target app -t redwood-rag:models .
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/.cache/hybrid-artifacts/indexes/dense-bge-base,dst=/index,readonly" \
  --mount "type=bind,src=$PWD/.cache/models,dst=/app/.cache/models" \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  redwood-rag:models python -c \
  "from pathlib import Path; from src.retrieval.corpus_dense import EmbeddingModel; model = EmbeddingModel.for_index(Path('/index'), device='cpu', batch_size=1); print(model.identity)"
```

<a id="full-benchmark-qa-access"></a>

## Optional: grant the full fictional benchmark for QA

<details>
<summary>Operator recipe: prepare, review, apply or revoke full QA access</summary>

Use this only after explicitly reviewing access to **all** imported fictional
benchmark documents. It is for local testing, not an enterprise ACL policy.
The account must already be a human login created in step 5. Do not use
`benchmark-create`: it creates a separate command-line benchmark account that
cannot sign into the web app.

The following private helper uses the existing `PolicyManifest` and
`PolicyRepository` APIs. It prepares patches of at most 1,000 IDs each, avoiding
one enormous database statement. Preparation only writes local review files;
`dry-run` previews them, and `apply` is a separate explicit step.

Save this file locally; it is not application source and must not be committed:

```sh
umask 077
cat > .cache/local-operator/qa-grants.py <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import sqlalchemy as sa

# This helper runs from /operator; the installed project is in /app.
sys.path.insert(0, "/app")

from src.access.repository import GrantSpec, PolicyManifest, PolicyRepository
from src.contracts import SnapshotRef
from src.db import schema
from src.db.runtime import database_url

os.umask(0o077)
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("prepare", "dry-run", "apply", "revoke"))
args = parser.parse_args()
folder = Path("/operator/hira-qa-grants")
snapshot = SnapshotRef.model_validate_json(Path("/state/snapshot.json").read_bytes())
try:
    if args.action == "prepare":
        user_id = input("Returned human user_id: ").strip()
        login = input("That account's login (for example hira): ").strip()
        reviewer = input("Reviewer name: ").strip()
        approval = input("Reviewed full-fictional-benchmark approval ID: ").strip()
        if not all((user_id, login, reviewer, approval)):
            raise ValueError("Review fields cannot be empty")
        engine = sa.create_engine(database_url(migration=True), hide_parameters=True)
        try:
            with engine.connect() as connection:
                account = connection.execute(sa.select(schema.users.c.user_id).where(
                    schema.users.c.user_id == user_id, schema.users.c.login == login,
                    schema.users.c.account_kind == "human", schema.users.c.is_active.is_(True))).scalar_one()
            folder.mkdir(mode=0o700)  # Refuse to overwrite an earlier review.
            prefix = hashlib.sha256((user_id + snapshot.snapshot_id + approval).encode()).hexdigest()[:24]
            after, total, files = -1, 0, {}
            while True:
                with engine.connect() as connection:
                    rows = connection.execute(sa.select(schema.documents.c.position, schema.documents.c.doc_id)
                        .where(schema.documents.c.snapshot_id == snapshot.snapshot_id,
                               schema.documents.c.position > after)
                        .order_by(schema.documents.c.position).limit(1000)).all()
                if not rows:
                    break
                number = len(files) + 1
                manifest = PolicyManifest(manifest_id=f"qa-{prefix}-{number:04d}",
                    snapshot=snapshot, origin="operator", reviewed_by=reviewer,
                    provenance={"approval": approval, "purpose": "Reviewed full fictional benchmark QA"},
                    grants=(GrantSpec(user_id=account, doc_ids=tuple(row.doc_id for row in rows)),))
                raw = manifest.model_dump_json(indent=2).encode() + b"\n"
                name = f"grant-{number:04d}.json"
                with (folder / name).open("xb") as handle:
                    handle.write(raw)
                files[name] = hashlib.sha256(raw).hexdigest()
                total += len(rows)
                after = rows[-1].position
            if total != snapshot.document_count:
                raise ValueError("Incomplete document selection")
            review = {"user_id": user_id, "snapshot": snapshot.model_dump(mode="json"),
                      "document_count": total, "files": files}
            with (folder / "review.json").open("x") as handle:
                json.dump(review, handle, indent=2)
            print(json.dumps({"prepared": True, "user_id": user_id, "documents": total, "patches": len(files)}))
        finally:
            engine.dispose()
    else:
        review = json.loads((folder / "review.json").read_text())
        if review["snapshot"] != snapshot.model_dump(mode="json"):
            raise ValueError("Snapshot changed")
        repository = PolicyRepository()
        changed = 0
        for name, digest in sorted(review["files"].items()):
            raw = (folder / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Reviewed file changed")
            manifest = PolicyManifest.model_validate_json(raw)
            if manifest.snapshot != snapshot or any(g.user_id != review["user_id"] or len(g.doc_ids) > 1000 for g in manifest.grants):
                raise ValueError("Reviewed scope changed")
            operation = repository.dry_run if args.action == "dry-run" else getattr(repository, args.action)
            changed += operation(manifest).changed_grants
        print(json.dumps({"action": args.action, "patches": len(review["files"]), "changed_grants": changed}))
except (ValueError, OSError, sa.exc.SQLAlchemyError):
    raise SystemExit("QA grant operation failed. Preserve partial files and review the private setup.") from None
PY
```

Prepare and inspect the returned user ID, document count and review files. Then
preview; only run `apply` after confirming the intended full-benchmark access:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python /operator/qa-grants.py prepare
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python /operator/qa-grants.py dry-run
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python /operator/qa-grants.py apply
```

For the current imported corpus, preparation should declare 511,958 documents.
Keep the private review folder. An interrupted apply may have committed earlier
patches; rerunning `apply` with the same unchanged review safely skips patches
already applied. To remove only grants created by these patches:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python /operator/qa-grants.py revoke
```

Pre-existing grants remain. A revoked patch requires a new review/manifest ID to
apply again. This recipe is based on the existing APIs; an independent fresh
teammate run has not yet been accepted. Full QA access still does not authorize
Observability or paid model calls.

</details>

<a id="database-foundation"></a>
<a id="canonical-document-import"></a>
<a id="native-jira-feed"></a>
<a id="database-and-import-guarantees"></a>

## Advanced: database and import details

The migration head is `0004_jira_feed`. A **migration** creates or updates the
schema—the database's table definitions. Imports bind document text, ordered
metadata, revisions and exact Unicode character offsets to one snapshot. Ready
text is not overwritten. Embedded NUL characters have a verified binary fallback.

Jira import uses the pinned source checkout. Each physical source record has its
own ID; native Jira keys may repeat. Missing or invalid fields remain unknown.
Readiness requires the complete feed when the corpus contains Jira records.

The app receives only the restricted `rag_runtime` database login and read-only
data/index/model/state volumes. The separate operator uses `rag_migrator` for
imports and account/grant administration. The readiness receipt is a verification
record, not a permission grant. Never put migration credentials in the app.

You can run setup steps separately when investigating a failure:

```sh
python3 scripts/local_stack.py --env-file .env.local build
python3 scripts/local_stack.py --env-file .env.local stage
python3 scripts/local_stack.py --env-file .env.local migrate
python3 scripts/local_stack.py --env-file .env.local import-corpus
python3 scripts/local_stack.py --env-file .env.local import-jira
python3 scripts/local_stack.py --env-file .env.local prepare-state
python3 scripts/local_stack.py --env-file .env.local up
```

The staging CLI `scripts/local_data.py` also accepts `--min-free-gib` (default 8,
minimum 4). Its check includes bytes still to copy plus that reserve. Changing
this does not establish that the full stack fits. `bootstrap` keeps the default.

## Advanced: account and permission maintenance

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts reset-password hira
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts disable hira
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository revoke --manifest /operator/hira-access.json
```

Password resets and disabling an account invalidate its old sessions. Revoking a
patch removes only links it created, not all possible access. Groups must be
explicitly assigned. A compound group requires membership in that group and all
its declared base groups; labels such as CEO do not imply membership.

<a id="backup-restore-and-compatible-rollback"></a>

## Advanced: backup, restore and rollback

A database backup contains accounts, grants and private records. Store it outside
the source database volume with private permissions, its checksum, the image
identity and matching data/snapshot receipts. Stop the app before this backup;
do not copy a live PostgreSQL data directory.

```sh
umask 077
mkdir -p .cache/local-backups
docker compose --env-file .env.local stop app
docker compose --env-file .env.local exec -T db sh -c \
  'exec pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom --no-owner --schema=rag' \
  > .cache/local-backups/rag.dump
shasum -a 256 .cache/local-backups/rag.dump > .cache/local-backups/rag.dump.sha256
```

Restore into a **new** project and volume. Create `.env.restore` with `init`, the
same verified input directories, an unused project/port, and a database name such
as `rag_restore_example`. Do not migrate or start its app before restoring:

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

Replace both `rag_restore_example` values with the exact new name in the config.
The recovery command quarantines restored accounts: it disables them, advances
account versions and revokes old sessions. It keeps documents, grants and history.
Reconcile approved access before resetting/reactivating each intended account:

```sh
python3 scripts/local_stack.py --env-file .env.restore operator -- \
  python -m src.auth.accounts reset-password hira --reactivate \
  --reviewed-by reviewer-name --policy-reconciliation approved-access-review-001
```

This attestation does not itself verify the policy review or grant documents.
Do not reactivate every restored user or reuse old cookies. With identical data,
run `import-corpus` to verify the existing import and publish its snapshot to the
new state volume, then `prepare-state` and `verify-data`. Start the app only after
quarantine and permission/account review. Keep the original stack separate.

An application rollback needs a previous immutable image that supports the
current schema and data contracts. Set compatible `RAG_APP_IMAGE` and
`RAG_OPERATOR_IMAGE` values, then use `up`. There is no destructive migration
rollback. Use a new restore target for an incompatible schema.

<a id="build-bm25-locally"></a>
<a id="native-development-and-evaluation"></a>

## Advanced: native development and evaluation

A host Python server needs its own database/path/auth settings. The Docker
`.env.example` uses container service names and volume paths; it is not a native
host configuration. Native development needs Python 3.11, dependencies from
`requirements.txt`, Node 22 and the frontend build:

```sh
python3 -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
```

To build a BM25 index in that prepared environment, choose a new output directory:

```sh
python -m src.retrieval.corpus_bm25 \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --output .cache/bm25-new
```

Do not overwrite a verified serving index. See [Evaluation](EVALUATION.md) for
approved experiment inputs and [Answer evaluation](ANSWER_EVALUATION.md) for
scoring. Reference answers are joined only after generation, never passed into
the running answer system. Current results and open limits are in [Status](STATUS.md).

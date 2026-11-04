# Run the local application

The default Docker deployment uses real local accounts, PostgreSQL permissions,
the full corpus, and hybrid retrieval. It starts in retrieval-only mode: no paid
model calls or generated answers. Commands run from the repository root on
macOS/Linux. This is a single-process local deployment, not an SSO or public
multi-server service.

## Prepare the machine

Install Git, Python 3.11 and Docker with Compose v2.10 or later. Docker builds the React
frontend; a host Node installation is needed only for frontend development.
The target machine has 16 GiB RAM and about 8 GiB available to the Docker VM.
Compose caps the app at 6 GiB/2 CPUs and PostgreSQL at 384 MiB/1 CPU, with no
extra swap allowance. Build/import jobs run separately. Leave space for images,
the original downloads, Linux-volume copies, PostgreSQL, and a separate recovery
copy. Build/staging headroom checks are minimum reserves, not total capacity estimates.

```sh
docker compose version
docker system df
df -h .
```

Keep large files out of cloud-synced folders that may replace them with online
placeholders. Do not commit `.env*`, downloaded data, operator receipts or tokens.

<a id="get-the-artifacts"></a>
<a id="hybrid-handoff"></a>

## Get the artifacts

Download the complete `files_web_app` bundle from the
[team Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).
Merge all split ZIP downloads into one directory without replacing different
files. Use this layout:

```text
.cache/hybrid-artifacts/
  documents.parquet
  indexes/
    bm25/
    dense-bge-base/
```

Keep every manifest, document-ID file, and declared dense `.npy`/`.parquet` shard.
Use the manifest inventory, not a hard-coded file count. `questions.jsonl` is
needed for evaluation, not serving. Setup checks corpus hashes, index inventories
and completeness; result JSON or partial downloads cannot substitute for indexes.
If you have the older five-part download, assemble it first in a prepared Python
environment:

```sh
python -m src.data.assemble_artifacts \
  --roots rag-dense "rag-dense 2" "rag-dense 3" "rag-dense 4" "rag-dense 5" \
  --output .cache/hybrid-artifacts
```

The native Jira adapter also needs the source tree at the corpus's pinned commit.
It imports actual fields and Git provenance; it does not infer fields from prose.

```sh
git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git .cache/local-source
git -C .cache/local-source checkout --detach d36685e273713975ee20299bbf1ab64165575b3c
```

Prepare a model cache matching the dense manifest before staging. An existing
verified team cache can be placed at `.cache/models`. To populate it yourself,
this explicit preparation job downloads the registered pinned embedding model.
It does not call a paid generation provider. Serving later uses the cache offline.

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

## Initialize and import

The helper creates an exclusive mode-600 configuration with three different
database role secrets. It does not create an application account. Use a new
project name for a separate installation; never point another server at an
existing database volume.

```sh
python3 scripts/local_stack.py --env-file .env.local init \
  --artifacts-dir "$PWD/.cache/hybrid-artifacts" \
  --models-dir "$PWD/.cache/models" \
  --source-repo-dir "$PWD/.cache/local-source" \
  --operator-dir "$PWD/.cache/local-operator" \
  --project redwood --port 8000
python3 scripts/local_stack.py --env-file .env.local plan
python3 scripts/local_stack.py --env-file .env.local bootstrap
```

`bootstrap` builds the lean operator image, hashes/copies artifacts and models
into named Linux volumes, builds the application image, migrates the database,
imports canonical documents and native Jira records, publishes a verified
deployment receipt, and starts the app. It never substitutes a demo identity.
Staging checks source bytes and independently hashes destination bytes. Published
data is immutable; receipts and locks live in the separate state volume.

The app reads corpus/index/model/state volumes read-only and receives only the
nonowner `rag_runtime` login. Migration/import/account jobs use the separate
`rag_migrator` login. PostgreSQL has no published host port. Loopback HTTP is the
explicit local cookie exception; HTTPS requires secure cookies, disabling the
loopback exception, and exact HTTPS origins.

The individual steps are also available:

```sh
python3 scripts/local_stack.py --env-file .env.local build
python3 scripts/local_stack.py --env-file .env.local stage
python3 scripts/local_stack.py --env-file .env.local migrate
python3 scripts/local_stack.py --env-file .env.local import-corpus
python3 scripts/local_stack.py --env-file .env.local import-jira
python3 scripts/local_stack.py --env-file .env.local prepare-state
python3 scripts/local_stack.py --env-file .env.local up
```

After an interrupted unpublished copy, investigate the failed job and any stale
lock first. `stage --adopt-existing` explicitly verifies existing bytes; it does
not overwrite a published different bundle. A changed corpus needs a separate
snapshot/data target. Editing `.env.local` secrets does not rotate roles inside
an existing PostgreSQL volume.

The artifact/model staging CLI `scripts/local_data.py` accepts an explicit
`--min-free-gib` reserve: default8 GiB, minimum4 GiB. Its disk check requires the
bytes still to copy **plus** that floor. A reviewed one-shot Compose command
override can pass a different reserve to the two staging jobs; `bootstrap` keeps
the default. The reserve is not a total-capacity promise: budget separately for
models/indexes, PostgreSQL and WAL, images, recovery copies and other activity.
The independent host-build and import headroom checks remain in force. Reducing a
reserve does not delete data or make a full serving/resource check pass.

<a id="database-foundation"></a>
<a id="canonical-document-import"></a>
<a id="native-jira-feed"></a>

## Database and import guarantees

The forward migration head is `0004_jira_feed`. Canonical imports bind the raw
corpus, ordered catalogue, immutable revisions and exact Unicode offsets to one
snapshot. Source changes or incomplete catalogues fail closed. Rare embedded NUL
characters use a checked binary fallback without altering hashes or offsets.
Repeated identical imports can resume; ready text is not overwritten.

Native Jira records form a separate complete feed bound to the same snapshot and
pinned Git tree. A source-record ID identifies a physical record; Jira keys can
be duplicated. Missing or invalid native fields remain unavailable. Readiness
requires a complete ready feed when the catalogue includes Jira documents; it
does not claim that every native field is populated.

The deployment receipt is published by a runtime-credential job only after live
schema/snapshot/feed checks. It is an integrity declaration, not an access grant.
Application startup still validates the real indexes, shards and embedding model.

<a id="local-account-api"></a>

## Provision local accounts

Create a real human account through the operator service. Password entry needs
an interactive terminal and confirmation; passwords never go in command flags.

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts create jane --display-name Jane --role employee
```

Keep the returned user ID for the permission manifest. Role, department and
project labels grant no access. A new account can sign in but cannot read any
documents until an operator applies explicit grants. Reset and disable invalidate
existing sessions through the account version:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts reset-password jane
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts disable jane
```

<a id="reviewed-document-permissions"></a>

## Apply reviewed document permissions

Place an operator-reviewed `policy.json` in `.cache/local-operator/`. It contains
the exact imported `SnapshotRef`, explicit existing document/user IDs, provenance,
and review identity. Do not invent IDs or derive permissions from document text.
The operator container reads `/state/snapshot.json`; `/operator` is the private
host exchange directory. The manifest shape is:

```json
{
  "schema_version": "1",
  "manifest_id": "reviewed-project-access-001",
  "snapshot": {"snapshot_id": "<actual>", "corpus_sha256": "<actual>", "catalog_sha256": "<actual>", "document_count": 0},
  "origin": "operator",
  "reviewed_by": "<reviewer>",
  "provenance": {"approval": "<reviewed access request>"},
  "groups": [],
  "memberships": [],
  "grants": [{"doc_ids": ["<actual authorized document ID>"], "user_id": "<created user ID>"}]
}
```

Replace every placeholder, including `document_count`, with actual snapshot
values. Preview before applying:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository dry-run --manifest /operator/policy.json
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository apply --manifest /operator/policy.json
```

Revoking this manifest removes only links it originally created, preserving
pre-existing grants. Removing one patch does not prove the subject has no other
access; review all relevant current policy.

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.repository revoke --manifest /operator/policy.json
```

A compound group such as `finance_exec` requires explicit membership in the
compound group **and** every declared base group. Neither a CEO label nor base
memberships alone imply compound access. Runtime restrictions only narrow grants.
PostgreSQL RLS independently checks live grants, session, policy revision,
snapshot and the registered scope bitmap.

## Use and restart the application

Open <http://127.0.0.1:8000>, sign in, and wait for Ready. `/api/health` reports
liveness; `/api/ready` returns generic 503 while dependencies or deployment
identity are unavailable. It exposes no corpus/permission counts. Readiness checks
are live and bounded; a running process alone is not readiness.

Search is scoped to the account. Source/date controls narrow that scope. Passage
cards show exact excerpts. Native record cards distinguish verified and unavailable
fields. Aggregate counts include zero and count **canonical Jira source records**,
not unique Jira keys or global absence. Duplicate keys request clarification;
unknown coverage is not converted to zero. Retrieval-only mode returns sources
without generated answers. Live generation is a separate explicit provider/model/
rate/budget opt-in; it is unnecessary for this setup.

Each request rechecks session and policy around retrieval, expansion, tools, model
dispatch and release. Sign out/account changes clear displayed evidence. The local
service admits one RAG request at a time; busy returns 429. HTTP timeout does not
free the slot while its worker is still running.

```sh
python3 scripts/local_stack.py --env-file .env.local stop
python3 scripts/local_stack.py --env-file .env.local up
python3 scripts/local_stack.py --env-file .env.local verify-data
```

Stop/start preserves named volumes, accounts, grants and unexpired sessions.
`up` reruns the deployment gate. Never remove volumes as routine shutdown. Keep
`.env.local` private and available: regenerating it does not recover/rotate an
existing database's passwords.

## Backup, restore and compatible rollback

Use the pinned PostgreSQL 17 client in the database container. Store archives
outside the database volume with restrictive permissions, alongside the exact
source commit/image digest and artifact/snapshot receipts. Stop the app before
the operator backup; do not copy a live PostgreSQL data directory.

```sh
umask 077
mkdir -p .cache/local-backups
docker compose --env-file .env.local stop app
docker compose --env-file .env.local exec -T db sh -c \
  'exec pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom --no-owner --schema=rag' \
  > .cache/local-backups/rag.dump
shasum -a 256 .cache/local-backups/rag.dump > .cache/local-backups/rag.dump.sha256
```

Keep the source volume untouched. Restore into a **new** Compose project and
fresh volume using a new private config, separate port, and
`RAG_DB_NAME=rag_restore_<nonce>`. Use `local_stack.py init` with the same verified
artifact/model/source inputs and a different `--project`; then edit only the new
config's database name. Build compatible images and stage the same inputs. Start
only its database, whose init hook creates separate roles. Do not migrate or start
the app into this empty target before restoring the schema.

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

Replace `rag_restore_example` in both places with the exact new configured name.
Quarantine refuses other names, incompatible schema/owner, and privileged/wrong
logins. It atomically disables restored accounts, increments active account
versions, and revokes every old session, including expired or future-created
sessions. It preserves documents, grants, groups and run history. A second
successful run changes zero rows.

Keep users disabled while reconciling current approved access against restored
grants. Revoke obsolete owned patches and apply reviewed current policy; preserve
provenance. Only then reset each approved account's password and explicitly
reactivate that account:

```sh
python3 scripts/local_stack.py --env-file .env.restore operator -- \
  python -m src.auth.accounts reset-password jane --reactivate \
  --reviewed-by reviewer-name --policy-reconciliation approved-access-review-001
```

This requires a disabled human account, a new confirmed password and both review
attestations. Retain its content-free completion receipt in the private operator
record. It does not grant documents or independently verify the reconciliation.
Do not enable all restored users or reuse old cookies.

With identical artifact bytes, `import-corpus` verifies the existing ready import
and republishes its snapshot into the fresh state volume. Then run `prepare-state`
and `verify-data` with the restored config. Open the app only after quarantine,
policy review and required account operations are complete. The original stack
remains available independently. The helper rejects the project's protected
authoritative source volume and a second running server on its target volume.

For application rollback, retain the previous immutable image digest and matching
source/artifact receipts. A previous image is usable only if it supports the
current migration head and data contracts. Point the config's `RAG_APP_IMAGE` and
`RAG_OPERATOR_IMAGE` at compatible local images, then run `up`. There is no
destructive migration downgrade. For incompatible schema, restore a verified
backup into a new target rather than modifying the source database.

## Validation and troubleshooting

The CI [tiny deployment fixture](../tests/deployment/README.md) uses five explicitly
synthetic documents, genuine BM25/import formats, real Argon2 human accounts,
PostgreSQL and unchanged RLS. It checks closed preparation, complete native feed,
two account scopes, DB-loss closure and restart persistence with the same cookies.
It downloads no embedding model and is not a full-corpus performance test.

Full-corpus scoped retrieval, native record/count paths, authentication and privacy
have separate recorded acceptance evidence. The new packaging/recovery gate must
also pass root-executed checks; tiny CI does not establish full-stack capacity or
model quality. Historical failing runs remain part of the record.

Inspect failed jobs and bounded app/verify logs privately. Do not print resolved
Compose environments, connection URLs, cookies or passwords. Common causes are
missing shards/model cache, changed source hashes, incomplete native feeds,
incompatible schema, insufficient disk, or incorrect role secrets. Fix the
underlying dependency and rerun the gate; never replace a failed receipt or
use demo fallback. Exit 137 alone does not establish OOM: check `OOMKilled`,
cgroup counters and disk. The historical stopped container had `OOMKilled=false`;
its failure cause remains unproven.

<a id="build-bm25-locally"></a>

## Native development and evaluation

The Docker `.env.example` uses internal service names and named-volume paths; it
is not a working native-host configuration. Native development needs Python 3.11,
`pip install -r requirements.txt`, Node 22, `npm --prefix frontend ci`, and
`npm --prefix frontend run build`, plus separately configured real database
credentials, snapshot/index/model paths and local auth settings. Never point a
native server at an incomplete import or fall back to benchmark mode for users.

To build a BM25 artifact in a prepared Python environment:

```sh
python -m src.retrieval.corpus_bm25 \
  --corpus .cache/hybrid-artifacts/documents.parquet \
  --output .cache/hybrid-artifacts/indexes/bm25
```

Build into a new unpublished target, not a sealed production directory. The
legacy benchmark/demo profile is separate from the authenticated product. Scoped
evaluation requires explicit private credentials and bound artifacts; see
[Evaluation](EVALUATION.md) and [Answer evaluation](ANSWER_EVALUATION.md). Runtime
exports exclude gold answers; an external judge consumes separate scoring input.

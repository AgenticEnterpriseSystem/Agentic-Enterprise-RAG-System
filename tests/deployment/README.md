# Tiny real-auth deployment smoke

`fixture.py` prepares five explicitly synthetic canonical documents, a genuine
BM25 index and the importer's real span artifact format. Its2D dummy arrays are
labelled import-only and cannot be used as a registered query model. CI uses
`INSTALL_HYBRID=false` and `RAG_RETRIEVAL=bm25`; no embedding/model is downloaded.

The fixture creates a genuine local Git commit for two native Jira source files,
then uses the accepted canonical and native import libraries with explicit
`synthetic_fixture` provenance. The app has real human Argon2 passwords, opaque
cookies, CSRF, PostgreSQL storage and unchanged RLS. Roles grant no rights.
No image or template contains a seeded account/password or benchmark identity.

The Compose override adds only an explicit fixture loader for empty volumes in
the generated `rag-ci-fixture-*` project. Base staging remains strict. Generated
role secrets live in the private `compose.env` outside the mounted operator
directory. Human secrets/cookies stay600 in the operator directory, which is
never mounted into the application. Do not upload that directory or diagnostics.

Run from the repository root, with the base file first. The override explicitly
extends `stage-artifacts` from the root `compose.yaml`; its reference follows the
[base-file path rule](https://docs.docker.com/compose/how-tos/multiple-compose-files/extends/),
not a path relative to `tests/deployment/`:

```sh
python tests/deployment/fixture.py prepare --directory .cache/deployment-smoke
docker compose --env-file .cache/deployment-smoke/compose.env \
  -f compose.yaml -f tests/deployment/compose.yaml \
  --profile ci-fixture --profile operator config --quiet
```

The remaining real service sequence is in `.github/workflows/ci.yml`. A successful
static path/format test does not replace the actual Compose configuration gate.

CI checks closed preflight before schema/import, complete deployment preflight,
real login and two distinct scopes, live DB-loss readiness/query503 with
liveness200, recovery, and app+database restarts with the same account/grants and
cookies. These are tiny functional checks; they do not prove full-corpus memory,
latency, backup/restore or model quality. Root performs the full-corpus gate.

The commands target Compose v2.10+ and use `up --wait`, without the newer
wait-timeout flag. CI has an outer30-minute job timeout; local_stack bounds its
application startup wait separately. Actual root-executed version compatibility
remains part of the unaccepted OPS gate until that smoke completes.

## Real-account security acceptance

`tests/test_access_end_to_end.py` runs two real Argon2 accounts through one HTTP
application backed by disposable PostgreSQL. Its explicitly synthetic Git/Jira
fixtures cover baseline and agentic modes, mixed passage/record/count inputs,
complete count lineage, provider-input isolation, rejected expansion/injection,
mid-request authority changes, and database permission failure/recovery. The
providers are local scripts; these checks make no remote model calls. Supply the
same three `RAG_TEST_*_URL` values used by the database suite, then run:

```sh
python -m pytest tests/test_access_end_to_end.py tests/test_security_operator.py
```

The opt-in full-corpus checks use an already prepared local Docker application:

- `security_operator.py` guards the reviewed `rag-ops01-full-20261104` project,
  dedicated migrator role, schema, 511,958-document snapshot, and verified native
  Jira feed. `seed` records private credentials and exact owned manifests before
  provisioning intern/executive fixture humans with one versus two explicit
  document grants. `inspect`, `revoke-intern`, and idempotent `cleanup` operate on
  that same ledger; cleanup disables only those humans and revokes their own
  grants/sessions. Corpus records and unrelated policy stay intact. Mount this
  helper read-only into the operator job and set `RAG_SEC_COMPOSE_PROJECT` to the
  independently checked project identity.
- `security_http.py before`, `after-revoke`, and `logout` use `--state` for the
  private ledger and a fresh `--reports` directory for each phase. They verify
  exact document permissions, canonical metadata/spans, forbidden text/title/ID
  markers, owner-only run metadata, permission refresh, and old-cookie replay.
- `frontend/tests/live-security.mjs` drives real Chrome with that same `--state`
  and a fresh `--outdir`. Its optional `--control` handshake lets the root
  operator stop/recover only the owned database while the browser checks live
  fail-closed behavior. The browser script itself performs no database writes.
  CLI usage and handshake phases are documented at the top of that file.

State and cookies are private 0600 files; credentials never enter command-line
arguments. Reports contain hashes/counts and sanitized outcomes. The ordinary
browser regression suite still uses its explicitly synthetic HTTP fixture;
passing it alone is distinct from real-Docker account isolation. Scripted model
input checks and retrieval-only full-corpus checks do not establish model answer
quality or a production security certification.

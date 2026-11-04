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
